#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* slice(std::any input_ptr, const std::vector<int> shape, uint64_t dim,
                uint64_t start, uint64_t end, uint64_t step) {
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

  const auto kwargs = op.kwargs();

  const auto input = kwargs.at("input").tensor();
  const auto shape = get_shape(input);

  uint64_t dim = kwargs.at("dim").int_value();
  uint64_t start =
      kwargs.contains("start") ? kwargs.at("start").int_value() : 0;
  uint64_t end =
      kwargs.contains("end") ? kwargs.at("end").int_value() : shape[dim];
  uint64_t step = kwargs.contains("step") ? kwargs.at("step").int_value() : 1;

  dim = dim < 0 ? dim + shape.size() : dim;
  end = end > shape[dim] ? shape[dim] : end;

  return slice<T>(input_ptr, shape, dim, start, end, step);
}
