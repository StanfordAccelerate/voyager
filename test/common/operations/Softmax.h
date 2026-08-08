#pragma once

#include "Common.h"
#include "VectorOps.h"
#include "test/toolchain/ApproximationConstants.h"

template <typename T>
inline T* softmax(std::any input_ptr, const std::vector<int> shape) {
  T* inputs = std::any_cast<T*>(input_ptr);

  int num_rows = 1;
  for (int i = 0; i < shape.size() - 1; i++) {
    num_rows *= shape[i];
  }
  int num_cols = shape[shape.size() - 1];

  T* outputs = new T[num_rows * num_cols];

  for (int i = 0; i < num_rows; i++) {
    int offset = i * num_cols;
    T max;
    if constexpr (std::is_same<T, CFloat>::value) {
      max = -1e30f;
    } else {
      max = T::min();
    }
    for (int j = 0; j < num_cols; j++) {
      max = inputs[offset + j] > max ? inputs[offset + j] : max;
    }

    for (int j = 0; j < num_cols; j++) {
      T normalized = static_cast<T>(inputs[offset + j] - max);
      // std::exp would take the implicit operator float() and return the true
      // exponential; the pipeline's op1_exp is vexp, which is ac_exp_pwl on
      // every lane. Take the same piecewise fit the hardware takes.
      outputs[offset + j] = normalized.exponential();
    }

    T sums[2] = {0, 0};
    int index = 0;

    for (int j = 0; j < num_cols; j += REDUCER_WIDTH) {
      T buffer[REDUCER_WIDTH];
      for (int k = 0; k < REDUCER_WIDTH; k++) {
        buffer[k] = j + k < num_cols ? outputs[offset + j + k] : T(0.0);
      }
      sums[index++ % 2] += fused_tree_reduce<REDUCER_WIDTH>(buffer);
    }

    T sum = fused_tree_reduce<2>(sums);
    T divisor = sum.reciprocal();

    for (int j = 0; j < num_cols; j++) {
      outputs[offset + j] *= divisor;
    }
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* softmax(std::map<std::string, std::any> kwargs,
                  const voyager::PrimOp& op, const ScalarEnv& env) {
  const auto input = resolve(op, "input", env);
  std::any input_ptr = kwargs[input.node];
  const auto input_shape = get_shape(input);
  return softmax<T>(input_ptr, input_shape);
}
