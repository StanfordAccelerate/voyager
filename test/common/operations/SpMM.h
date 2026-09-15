#pragma once

#include <vector>

#include "test/common/GraphUtils.h"
#include "test/common/Utils.h"
#include "test/common/operations/Common.h"

// The accumulator feedback depth SpMMUnit is built with. It is a pipelining
// choice, so it moves with the clock period -- and because the partials it
// leaves are summed in the vector type, it changes the arithmetic. Keep in
// step with SpMMUnit.h.
#ifndef CLOCK_PERIOD
#define CLOCK_PERIOD 5.0
#endif
constexpr int FEEDBACK_DELAY =
    (CLOCK_PERIOD < 0.8 ? 9 : (CLOCK_PERIOD < 5 ? 6 : 4));

// sum_s (src/vector_unit/VectorOps.h), which add_tree instantiates: halve the
// pack, reduce each half, add the two. Not the same shape as tree_reduce's
// pairwise sweep once the length is not a power of two, and FEEDBACK_DELAY is
// 6 or 9 at the shorter clock periods.
template <int N, typename T>
inline T balanced_tree_reduce(const T* values) {
  if constexpr (N == 1) {
    return values[0];
  } else {
    return balanced_tree_reduce<N / 2>(values) +
           balanced_tree_reduce<N - N / 2>(values + N / 2);
  }
}

template <typename Meta, typename Vector, typename Weight, typename Output,
          typename Scale>
inline std::shared_ptr<Output[]> spmm_csr(
    std::any data_ptr, std::any indices_ptr, std::any indptr_ptr,
    std::any weight_ptr, std::any weight_scale_ptr,
    const voyager::PrimOp& operation, const ScalarEnv& env) {
  bool is_matmul = operation.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight_tensor = resolve(operation, weight_key, env);
  const auto indptr_array = resolve(operation, "A_indptr", env);

  // The contracted operand's trailing dim is the output width. A linear's
  // weight arrives as [K_in, N], but a matmul's `other` keeps the batch dims
  // it was traced with -- [1, 1, K_in, N] -- so dimension 1 is not N there.
  const int K = get_shape(weight_tensor).back();
  const int num_rows = get_size(indptr_array) - 1;
  const int output_size = num_rows * K;

  int block_size = 1;
  if (has_arg(operation, "block_size")) {
    block_size = arg_int(operation, "block_size", env);
  }

  Vector* data = std::any_cast<std::shared_ptr<Vector[]>&>(data_ptr).get();
  Meta* indices = std::any_cast<std::shared_ptr<Meta[]>&>(indices_ptr).get();
  Meta* indptr = std::any_cast<std::shared_ptr<Meta[]>&>(indptr_ptr).get();

  Weight* weight = std::any_cast<std::shared_ptr<Weight[]>&>(weight_ptr).get();
  Scale* scale =
      std::any_cast<std::shared_ptr<Scale[]>&>(weight_scale_ptr).get();

  std::shared_ptr<Output[]> outputs(new Output[output_size]);
  for (int i = 0; i < output_size; i++) {
    outputs[i] = 0.0;
  }

  Meta row_offset = indptr[0];

  // The unit keeps no single running sum per output. Its multiply-accumulate
  // reads the partial sum FEEDBACK_DELAY iterations back (SpMMUnit.h,
  // acc_old[FEEDBACK_LAST]), so a row's nonzeros land in FEEDBACK_DELAY
  // independent partial sums -- nonzero j into stripe j % FEEDBACK_DELAY --
  // which an adder tree folds once the row ends.
  //
  // Summing them in one sequential pass is the same value in exact
  // arithmetic, but these partials are the vector type and every add rounds.
  // The dense GEMM accumulates in a wide integer, where the order cannot
  // matter; here it does, so gold has to fold them the way the unit does.
  std::vector<Output> partials(static_cast<size_t>(K) * FEEDBACK_DELAY);

  // iterate through the sparse input tensor
  for (int r = 0; r < num_rows; r++) {
    int row_start = indptr[r] - row_offset;
    int row_end = indptr[r + 1] - row_offset;

    for (auto& partial : partials) partial = Output(0.0);

    // iterate through the non-zero elements in the row
    for (int j = row_start; j < row_end; j++) {
      int c = indices[j];
      Vector value = data[j];
      const int stripe = (j - row_start) % FEEDBACK_DELAY;

      // multiply with the corresponding weight vector
      for (int k = 0; k < K; k++) {
        Output product =
            static_cast<Output>(value) * static_cast<Output>(weight[c * K + k]);
        if (scale != nullptr) {
          product *= static_cast<Output>(scale[(c / block_size) * K + k]);
        }
        partials[static_cast<size_t>(k) * FEEDBACK_DELAY + stripe] += product;
      }
    }

    // The tree does not see the stripes in stripe order. Each accumulate
    // reads acc_old[FEEDBACK_LAST] and the array then shifts, so after nnz
    // products acc_old[k] holds stripe (nnz - 1 - k), and it is that
    // rotated arrangement the tree folds. The rotation therefore depends on
    // how many nonzeros the row had.
    const int nnz = row_end - row_start;
    Output folded[FEEDBACK_DELAY];
    for (int k = 0; k < K; k++) {
      const Output* stripes =
          &partials[static_cast<size_t>(k) * FEEDBACK_DELAY];
      for (int i = 0; i < FEEDBACK_DELAY; i++) {
        const int stripe =
            ((nnz - 1 - i) % FEEDBACK_DELAY + FEEDBACK_DELAY) % FEEDBACK_DELAY;
        folded[i] = stripes[stripe];
      }
      outputs[r * K + k] = balanced_tree_reduce<FEEDBACK_DELAY>(folded);
    }
  }

  spdlog::debug("SpMM done\n");
  return outputs;
}
