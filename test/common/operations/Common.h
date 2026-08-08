#pragma once
#define NO_SYSC

// IWYU pragma: begin_exports
#include <any>
#include <type_traits>
#include <vector>

#include "ac_std_float_add_tree.h"
#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"
#include "test/common/Tensor.h"
#include "test/common/Utils.h"
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

// Only a datatype with an ac_std_float representation can be folded the way
// the hardware folds one. CFloat carries a plain float and falls back.
template <typename T, typename = void>
struct has_fused_add_tree : std::false_type {};

template <typename T>
struct has_fused_add_tree<T, std::void_t<typename T::ac_float_rep>>
    : std::true_type {};

// One pack, reduced the way VectorReducer reduces it. fused_add_tree
// (src/vector_unit/VectorOps.h) hands the pack to ac_math::fadd_tree, which
// aligns every mantissa to the pack's largest exponent, sums them in fixed
// point, and rounds *once*. tree_reduce rounds at each of its log2(length)
// nodes instead. With bfloat16's 8-bit mantissa that gap compounds over a few
// thousand terms -- far enough to move a microscaling amax across an fp8
// rounding boundary, which then shifts every code in the block.
template <int length, typename T>
inline T fused_tree_reduce(const T* buffer) {
  if constexpr (has_fused_add_tree<T>::value) {
    typename T::ac_float_rep values[length];
    for (int i = 0; i < length; i++) {
      values[i] = buffer[i].float_val;
    }

    typename T::ac_float_rep sum;
    ac_math::fadd_tree(values, sum);
    return sum;
  } else {
    T values[length];
    for (int i = 0; i < length; i++) {
      values[i] = buffer[i];
    }
    return tree_reduce(values, length);
  }
}
