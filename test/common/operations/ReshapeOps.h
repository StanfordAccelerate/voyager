#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* pad_tensor(std::any input_ptr, const std::vector<int>& input_shape,
                     const int pad_amount) {
  spdlog::debug("Padding tensor with amount: {}\n", pad_amount);
  T* inputs = std::any_cast<T*>(input_ptr);

  std::vector<int> output_shape(input_shape);
  output_shape[1] += pad_amount;

  // copy the input tensor to the output tensor
  T* outputs = new T[get_size(output_shape)];
  for (int i = 0; i < get_size(input_shape); i++) {
    outputs[i] = inputs[i];
  }

  // pad the first dimension of the output tensor
  for (int i = input_shape[1]; i < input_shape[1] + pad_amount; i++) {
    for (int j = 0; j < input_shape[2]; j++) {
      outputs[i * input_shape[2] + j] = T::zero();
    }
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* pad_tensor(std::any input_ptr, const codegen::OpOverload op) {
  const auto input = op.kwargs().at("input").tensor();
  const auto input_shape = get_shape(input);
  const auto pad_list = op.kwargs().at("pad").int_list().values();
  const auto pad_amount = pad_list[pad_list.size() - 1];

  return pad_tensor<T>(input_ptr, input_shape, pad_amount);
}

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
  if (op.target() == "permute") {
    const auto input = op.kwargs().at("input").tensor();
    const auto input_shape = get_shape(input);

    const auto dims = op.kwargs().at("dims").int_list().values();
    std::vector<int> dims_vector(dims.begin(), dims.end());

    return permute<T>(input_ptr, input_shape, dims_vector);
  } else if (op.target() == "transpose") {
    const auto input = op.kwargs().at("input").tensor();
    const auto input_shape = get_shape(input);

    const int dim0 = op.kwargs().at("dim0").int_value();
    const int dim1 = op.kwargs().at("dim1").int_value();

    std::vector<int> dims_vector;
    for (int i = 0; i < input_shape.size(); ++i) {
      dims_vector.push_back(i);
    }

    std::swap(dims_vector[dim0], dims_vector[dim1]);

    return permute<T>(input_ptr, input_shape, dims_vector);
  }

  return std::any_cast<T*>(input_ptr);
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
