#pragma once

#include "test/common/Utils.h"
#include "test/common/operations/Common.h"
#include "test/compiler/proto/param.pb.h"

template <typename Meta, typename Vector, typename Weight, typename Output,
          typename Scale>
inline Output* spmm_csr(std::any data_ptr, std::any indices_ptr,
                        std::any indptr_ptr, std::any weight_ptr,
                        std::any weight_scale_ptr,
                        const codegen::OpOverload& operation) {
  const auto weight_tensor = operation.kwargs().at("B").tensor();
  const auto indptr_array = operation.kwargs().at("indptr").tensor();

  const int K = get_shape(weight_tensor)[1];
  const int num_rows = get_size(indptr_array) - 1;
  const int output_size = num_rows * K;

  int block_size = 1;
  if (operation.kwargs().contains("block_size")) {
    block_size = operation.kwargs().at("block_size").int_value();
  }

  Vector* data = std::any_cast<Vector*>(data_ptr);
  Meta* indices = std::any_cast<Meta*>(indices_ptr);
  Meta* indptr = std::any_cast<Meta*>(indptr_ptr);

  Weight* weight = std::any_cast<Weight*>(weight_ptr);
  Scale* scale = std::any_cast<Scale*>(weight_scale_ptr);

  Output* outputs = new Output[output_size];
  for (int i = 0; i < output_size; i++) {
    outputs[i] = 0.0;
  }

  // iterate through the sparse input tensor
  for (int r = 0; r < num_rows; r++) {
    int row_start = indptr[r];
    int row_end = indptr[r + 1];

    // iterate through the non-zero elements in the row
    for (int j = row_start; j < row_end; j++) {
      int c = indices[j];
      Vector value = data[j];

      // multiply with the corresponding weight vector
      for (int k = 0; k < K; k++) {
        Output product =
            static_cast<Output>(value) * static_cast<Output>(weight[c * K + k]);
        if (scale != nullptr) {
          product *= static_cast<Output>(scale[(c / block_size) * K + k]);
        }
        outputs[r * K + k] += product;
      }
    }
  }

  delete[] data;
  delete[] indices;
  delete[] indptr;
  delete[] weight;
  if (scale != nullptr) {
    delete[] scale;
  }

  return outputs;
}
