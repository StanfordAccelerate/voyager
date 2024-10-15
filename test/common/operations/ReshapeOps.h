#pragma once

#include "test/common/operations/Common.h"

// Helper function to compute strides
inline std::vector<int> compute_strides(const std::vector<int>& shape) {
  std::vector<int> strides(shape.size());
  strides.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Helper function to compute the linear index from multi-dimensional indices
inline int compute_linear_index(const std::vector<int>& indices,
                                const std::vector<int>& strides) {
  int linear_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    linear_index += indices[i] * strides[i];
  }
  return linear_index;
}

template <typename T>
inline T* permute(std::any input_tensor, const codegen::ReshapeParam param) {
  const auto input = param.input();
  const int size = get_size(input);
  T* outputs = new T[size];

  T* inputs = std::any_cast<T*>(input_tensor);

  const auto input_shape = get_shape(input);
  std::vector<int> order;
  if (param.opcode() == "permute") {
    order.insert(order.end(), param.dims().begin(), param.dims().end());
  } else if (param.opcode() == "transpose") {
    const int size = input.shape_size();
    const int dim1 = param.dims(0) < 0 ? param.dims(0) + size : param.dims(0);
    const int dim2 = param.dims(1) < 0 ? param.dims(1) + size : param.dims(1);
    for (int i = 0; i < size; ++i) {
      order.push_back(i);
    }
    std::swap(order[dim1], order[dim2]);
  } else {
    std::cerr << "Unsupported reshape instruction: " << param.opcode()
              << std::endl;
    exit(1);
  }

  std::vector<int> permuted_shape(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    permuted_shape[i] = input_shape[order[i]];
  }
  std::vector<int> permuted_strides = compute_strides(permuted_shape);

  std::vector<int> indices(input_shape.size(), 0);
  for (int i = 0; i < size; ++i) {
    std::vector<int> permuted_indices(order.size());
    for (size_t j = 0; j < order.size(); ++j) {
      permuted_indices[j] = indices[order[j]];
    }

    int permuted_index =
        compute_linear_index(permuted_indices, permuted_strides);

    outputs[permuted_index] = inputs[i];

    // Update multi-dimensional indices
    for (int j = input_shape.size() - 1; j >= 0; --j) {
      indices[j]++;
      if (indices[j] < input_shape[j]) {
        break;
      }
      indices[j] = 0;
    }
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* permute(std::any input_tensor, const codegen::Tensor& input) {
  const codegen::Permutation& param = input.permutation();
  const std::vector<int> shape{param.input_shape().begin(), param.input_shape().end()};
  const std::vector<int> dims{param.dims().begin(), param.dims().end()};

  T* inputs = std::any_cast<T*>(input_tensor);

  std::vector<int> order;
  if (param.opcode() == "permute") {
    order.insert(order.end(), dims.begin(), dims.end());
  } else if (param.opcode() == "transpose") {
    const int size = shape.size();
    const int dim1 = dims[0] < 0 ? dims[0] + size : dims[0];
    const int dim2 = dims[1] < 0 ? dims[1] + size : dims[1];
    for (int i = 0; i < size; ++i) {
      order.push_back(i);
    }
    std::swap(order[dim1], order[dim2]);
  } else {
    std::cerr << "Unsupported reshape instruction: " << param.opcode()
              << std::endl;
    exit(1);
  }

  std::vector<int> permuted_shape(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    permuted_shape[i] = shape[order[i]];
  }
  std::vector<int> permuted_strides = compute_strides(permuted_shape);

  const int size = get_size(shape);
  T* outputs = new T[size];
  std::vector<int> indices(shape.size(), 0);
  for (int i = 0; i < size; ++i) {
    std::vector<int> permuted_indices(order.size());
    for (size_t j = 0; j < order.size(); ++j) {
      permuted_indices[j] = indices[order[j]];
    }

    int permuted_index =
        compute_linear_index(permuted_indices, permuted_strides);

    outputs[permuted_index] = inputs[i];

    // Update multi-dimensional indices
    for (int j = shape.size() - 1; j >= 0; --j) {
      indices[j]++;
      if (indices[j] < shape[j]) {
        break;
      }
      indices[j] = 0;
    }
  }

  delete[] inputs;

  return outputs;
}
