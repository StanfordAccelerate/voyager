#pragma once
#define NO_SYSC

// IWYU pragma: begin_exports
#include <any>
#include <vector>

// clang-format off
#include "src/datatypes/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
// IWYU pragma: end_exports

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
