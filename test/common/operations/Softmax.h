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
      outputs[offset + j] = poly_approx(normalized, EXP_MAXES, EXP_RANGES,
                                        EXP_CLAMP_MIN, EXP_CLAMP_MAX);
    }

    T sums[2] = {0, 0};
    int index = 0;

    for (int j = 0; j < num_cols; j += VECTOR_UNIT_WIDTH) {
      T buffer[VECTOR_UNIT_WIDTH];
      for (int k = 0; k < VECTOR_UNIT_WIDTH; k++) {
        buffer[k] = j + k < num_cols ? outputs[offset + j + k] : T(0.0);
      }
      sums[index++ % 2] += tree_reduce(buffer, VECTOR_UNIT_WIDTH);
    }

    T sum = tree_reduce(sums, 2);
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
                  const codegen::OpOverload op) {
  const auto input = op.kwargs().at("input").tensor();
  std::any input_ptr = kwargs[input.node()];
  const auto input_shape = get_shape(input);
  return softmax<T>(input_ptr, input_shape);
}
