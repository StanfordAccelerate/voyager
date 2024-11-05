#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

using INTERMEDIATE_DTYPE = ACCUM_DATATYPE::AccumulationDatatype;

inline std::vector<int> get_shape(const codegen::Tensor &tensor) {
  auto repeated_field = tensor.shape();
  return std::vector<int>(repeated_field.begin(), repeated_field.end());
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
