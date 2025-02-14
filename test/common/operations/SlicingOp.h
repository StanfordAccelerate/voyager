#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* slice(std::any input_ptr, const std::vector<int> shape, int dim,
                int start, int end, int step) {
  dim = dim < 0 ? dim + shape.size() : dim;
  int num_elements = (end - start) / step;

  const auto inputs = std::any_cast<T*>(input_ptr);

  const int size = get_size(shape);
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

    int flat_index = get_flat_index(indices, shape);
    outputs[i] = inputs[flat_index];
  }

  return outputs;
}

template <typename T>
inline T* slice(std::any input_ptr, const codegen::OpOverload op) {
  if (op.target() != "slice") {
    return std::any_cast<T*>(input_ptr);
  }

  const auto& input = op.kwargs().at("input").tensor();
  const auto shape = get_shape(input);

  const int start = op.kwargs().at("start").int_value();
  const int end = op.kwargs().at("end").int_value();
  const int step = op.kwargs().at("step").int_value();
  const int dim = op.kwargs().at("dim").int_value();

  return slice<T>(input_ptr, shape, dim, start, end, step);
}
