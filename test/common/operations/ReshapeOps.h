#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* permute(std::any input_tensor, const codegen::ReshapeOp param) {
  const std::vector<int> shape{param.input_sizes().begin(),
                               param.input_sizes().end()};
  const std::vector<int> dims{param.dims().begin(), param.dims().end()};

  T* inputs = std::any_cast<T*>(input_tensor);

  std::vector<int> order;
  if (param.opcode() == "permute") {
    order.insert(order.end(), dims.begin(), dims.end());
  } else if (param.opcode() == "transpose") {
    const int ndim = shape.size();
    const int dim1 = dims[0] < 0 ? dims[0] + ndim : dims[0];
    const int dim2 = dims[1] < 0 ? dims[1] + ndim : dims[1];
    for (int i = 0; i < ndim; ++i) {
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

  const int size = get_size(shape);
  T* outputs = new T[size];

  for (int i = 0; i < size; ++i) {
    std::vector<int> indices = get_indices(i, shape);

    std::vector<int> permuted_indices(order.size());
    for (size_t j = 0; j < order.size(); ++j) {
      permuted_indices[j] = indices[order[j]];
    }

    int permuted_index = get_flat_index(permuted_indices, permuted_shape);
    outputs[permuted_index] = inputs[i];
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* permute(std::any input_tensor, const codegen::Tensor& input) {
  return permute<T>(input_tensor, input.reshape());
}
