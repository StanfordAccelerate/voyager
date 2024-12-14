#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* slice(std::any input_tensor, const codegen::SlicingOp param,
                const codegen::Tensor input) {
  const auto repeated_field = input.shape();
  const std::vector<int> shape(repeated_field.begin(), repeated_field.end());
  const auto strides = compute_strides(shape);

  const int dim = param.dim() < 0 ? param.dim() + shape.size() : param.dim();
  const int start = param.start();
  const int end = param.end();
  const int step = param.step();
  const int num_elements = (end - start) / step;

  const auto inputs = std::any_cast<T*>(input_tensor);

  const int size = get_size(input);
  T* outputs = new T[size];

  for (int i = 0; i < size; i++) {
    std::vector<int> indices(shape.size(), 0);
    int index = i;
    for (int j = shape.size() - 1; j >= 0; --j) {
      if (j == dim) {
        indices[j] = start + (index % num_elements) * step;
        index /= num_elements;
      } else {
        indices[j] = index % shape[j];
        index /= shape[j];
      }
    }

    outputs[i] = inputs[compute_linear_index(indices, strides)];
  }

  return outputs;
}

template <typename T>
inline T* slice(std::any input_tensor, const codegen::SlicingOp param) {
  return slice<T>(input_tensor, param, param.input());
}

template <typename T>
inline T* slice(std::any input_tensor, const codegen::Tensor input) {
  return slice<T>(input_tensor, input.slicing(), input);
}
