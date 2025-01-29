#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

inline std::vector<int> get_shape(const codegen::Tensor &tensor) {
  if (tensor.has_reshape()) {
    const auto &param = tensor.reshape();
    return {param.output_sizes().begin(), param.output_sizes().end()};
  }

  if (tensor.has_slicing()) {
    const auto &param = tensor.slicing();
    return {param.output_sizes().begin(), param.output_sizes().end()};
  }

  const auto repeated_field = tensor.shape();
  return {repeated_field.begin(), repeated_field.end()};
}

inline int get_size(const std::vector<int> &shape) {
  int size = 1;
  for (const auto &dim : shape) size *= dim;
  return size;
}

inline int get_size(const codegen::Tensor &tensor) {
  const auto shape = get_shape(tensor);
  return get_size(shape);
}

// Function to compute multi-dimensional indices from a flat index
inline std::vector<int> get_indices(int flat_idx,
                                    const std::vector<int> &shape) {
  int num_dims = shape.size();
  std::vector<int> indices(num_dims, 0);
  for (int i = num_dims - 1; i >= 0; --i) {
    indices[i] = flat_idx % shape[i];
    flat_idx /= shape[i];
  }
  return indices;
}

// Function to compute flat index from multi-dimensional indices
inline int get_flat_index(const std::vector<int> &indices,
                          const std::vector<int> &shape) {
  int flat_idx = 0, multiplier = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    flat_idx += indices[i] * multiplier;
    multiplier *= shape[i];
  }
  return flat_idx;
}

inline void print_shape(const std::vector<int> &shape) {
  std::cerr << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cerr << shape[i] << (i + 1 < shape.size() ? ", " : ")");
  }
  std::cerr << std::endl;
}
