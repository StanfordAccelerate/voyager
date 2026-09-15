#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"

// A resolved *use* of a TensorBox.
//
// The bufferized IR names an operand with a TensorBoxRef -- a node name plus,
// for a banked buffer, a window selecting one bank. Resolving that reference
// against the buffer table and the current scalar environment yields this: a
// concrete buffer at a concrete address, which is what the kernels consume.
//
// Two operands are not backed by memory:
//   - a fusion intermediate, which names an earlier PrimOp in the same op_list
//     and never leaves the datapath (materialized == false);
//   - a config constant (codebook, qmap, scalar scale), which the compiler
//     leaves unmapped and the kernels read from tensor_files (is_constant).
struct Tensor {
  std::string node;
  std::vector<int> shape;
  std::string dtype;

  int partition = 0;
  uint64_t address = 0;

  bool materialized = true;
  bool is_constant = false;

  // A pitched window into a larger allocation: rows of shape.back() valid
  // elements sit window_pitch elements apart, the first starting window_col
  // elements into its row. 0 = contiguous.
  int64_t window_pitch = 0;
  int64_t window_col = 0;
};

inline long get_size(const std::vector<int>& shape) {
  long size = 1;
  for (const auto& dim : shape) size *= dim;
  return size;
}

inline long get_size(const Tensor& tensor) { return get_size(tensor.shape); }

inline const std::vector<int>& get_shape(const Tensor& tensor) {
  return tensor.shape;
}

inline uint64_t get_address(const Tensor& tensor) { return tensor.address; }

inline int get_partition(const Tensor& tensor) { return tensor.partition; }

inline size_t get_width(const Tensor& tensor) {
  return get_type_width<SUPPORTED_TYPES>(tensor.dtype);
}

// Bytes spanned by the tensor, rounded up (int4/int6 pack sub-byte).
inline uint64_t get_num_bytes(const Tensor& tensor) {
  return (get_size(tensor) * get_width(tensor) + 7) / 8;
}

inline std::string shape_str(const std::vector<int>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); i++) {
    s += std::to_string(shape[i]);
    if (i + 1 < shape.size()) s += ", ";
  }
  return s + "]";
}
