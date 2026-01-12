#pragma once
#define NO_SYSC

// IWYU pragma: begin_exports
#include <any>
#include <vector>

#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"
#include "test/common/Utils.h"
#include "test/compiler/proto/param.pb.h"
// IWYU pragma: end_exports

// Function to compute multi-dimensional indices from a flat index
inline std::vector<int> get_indices(int flat_idx,
                                    const std::vector<int>& shape) {
  int num_dims = shape.size();
  std::vector<int> indices(num_dims, 0);
  for (int i = num_dims - 1; i >= 0; --i) {
    indices[i] = flat_idx % shape[i];
    flat_idx /= shape[i];
  }
  return indices;
}

// Function to compute flat index from multi-dimensional indices
inline int get_flat_index(const std::vector<int>& indices,
                          const std::vector<int>& shape) {
  int flat_idx = 0, multiplier = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    flat_idx += indices[i] * multiplier;
    multiplier *= shape[i];
  }
  return flat_idx;
}

template <typename T>
inline T tree_reduce(T* buffer, int length) {
  int depth = length;
  while (depth > 1) {
    for (int j = 0; j < depth; j += 2) {
      buffer[j / 2] = buffer[j] + buffer[j + 1];
    }
    depth = depth / 2;
  }
  return buffer[0];
}
