#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* permute(std::any input_ptr, const std::vector<int>& shape,
                  const std::vector<int> dims) {
  T* inputs = std::any_cast<T*>(input_ptr);

  std::vector<int> permuted_shape(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    permuted_shape[i] = shape[dims[i]];
  }

  const int size = get_size(shape);
  T* outputs = new T[size];

  for (int i = 0; i < size; ++i) {
    std::vector<int> indices = get_indices(i, shape);

    std::vector<int> permuted_indices(dims.size());
    for (size_t j = 0; j < dims.size(); ++j) {
      permuted_indices[j] = indices[dims[j]];
    }

    int permuted_index = get_flat_index(permuted_indices, permuted_shape);
    outputs[permuted_index] = inputs[i];
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* permute(std::any input_ptr, const codegen::OpOverload op) {
  if (op.target() != "permute") {
    return std::any_cast<T*>(input_ptr);
  }

  const auto input = op.kwargs().at("input").tensor();
  const auto input_shape = get_shape(input);

  const auto dims = op.kwargs().at("dims").int_list().values();
  std::vector<int> dims_vector(dims.begin(), dims.end());

  return permute<T>(input_ptr, input_shape, dims_vector);
}

template <typename T>
inline T* transpose(std::any input_ptr, const std::vector<int>& shape, int dim0,
                    int dim1) {
  const int ndim = shape.size();
  dim0 = dim0 < 0 ? dim0 + ndim : dim0;
  dim1 = dim1 < 0 ? dim1 + ndim : dim1;

  std::vector<int> dims;
  for (int i = 0; i < ndim; ++i) {
    dims.push_back(i);
  }

  std::swap(dims[dim0], dims[dim1]);

  return permute<T>(input_ptr, shape, dims);
}

template <typename T>
inline T* transpose(std::any input_ptr, const codegen::OpOverload op) {
  if (op.target() != "transpose") {
    return std::any_cast<T*>(input_ptr);
  }

  const auto input = op.kwargs().at("input").tensor();
  const auto input_shape = get_shape(input);
  const int dim0 = op.kwargs().at("dim0").int_value();
  const int dim1 = op.kwargs().at("dim1").int_value();

  return transpose<T>(input_ptr, input_shape, dim0, dim1);
}

template <typename T>
inline T* reshape_if_needed(std::any input_ptr, const codegen::OpOverload op) {
  if (op.target() == "transpose") {
    return transpose<T>(input_ptr, op);
  } else if (op.target() == "permute") {
    return permute<T>(input_ptr, op);
  } else {
    return std::any_cast<T*>(input_ptr);
  }
}
