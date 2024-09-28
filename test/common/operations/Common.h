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

/***************************************************************************
 * read_tensor Functions
 *
 * These functions handle reading tensor values from different types
 * of matrices. The functions manage whether the data should be read in
 * single or double precision formats.
 ***************************************************************************/

// inline float read_tensor(const float *matrix, int index,
//                          bool double_precision) {
//   return double_precision ? matrix[2 * index] : matrix[index];
// }

// inline ACCUM_DATATYPE read_tensor(const INPUT_DATATYPE *matrix, int index,
//                                   bool double_precision) {
//   if (!double_precision) {
//     return static_cast<ACCUM_DATATYPE>(matrix[index]);
//   } else {
//     return ACCUM_DATATYPE(&matrix[2 * index]);
//   }
// }

/***************************************************************************
 * save_tensor Functions
 *
 * These functions handle saving tensor values into different types of
 * matrices. The functions manage the precision and data types that the data
 * should be stored into.
 ***************************************************************************/

// inline void save_tensor(float *matrix, int index, float value,
//                         bool double_precision) {
//   if (!double_precision) {
//     matrix[index] = value;
//   } else {
//     matrix[2 * index] = value;
//     matrix[2 * index + 1] = 0;
//   }
// }

// inline void save_tensor(INPUT_DATATYPE *matrix, int index,
//                         INTERMEDIATE_DTYPE value, bool double_precision) {
//   if (!double_precision) {
//     matrix[index] = static_cast<INPUT_DATATYPE>(value);
//   } else {
//     ACCUM_DATATYPE p16 = value;
//     p16.storeAsLowerPrecision(&matrix[2 * index]);
//   }
// }

/***************************************************************************
 * Elementwise Functions
 *
 * These functions calculating the reciprocal of various data types: `float`,
 * `INTERMEDIATE_DTYPE`, and `UniversalPositAccum`.
 ***************************************************************************/
