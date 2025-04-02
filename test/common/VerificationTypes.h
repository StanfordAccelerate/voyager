#pragma once

#define NO_SYSC

#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "test/compiler/proto/param.pb.h"

#ifndef NUM_SRAM_BANKS
#define NUM_SRAM_BANKS 1024
#endif
#define SRAM_MEMORY_SIZE (NUM_SRAM_BANKS * 1024LL * 1024LL)

// Size of the reference memory used for verification in MB
// Should be at least as large as the SRAM memory
#define REFERENCE_MEMORY_SIZE (1024 * 1024 * 8)

static const std::unordered_set<std::string> GEMM_OPS = {
    "conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx",
};

static const std::unordered_set<std::string> MEMORY_OPS = {
    "slice",
    "permute",
    "transpose",
};

enum MemorySource { SRAM, RRAM };

inline std::ostream& operator<<(std::ostream& os, MemorySource& memory) {
  os << (memory == SRAM ? "SRAM" : "RRAM");
  return os;
}

// The accelerator has several interfaces for accessing memory, but
// these interfaces aren't always standard. For example, for some layers,
// an interface may access the residual, but for other layers, access weights.
// This data structure maps the accelerator interface to the MemorySource.
typedef std::map<std::string, MemorySource> AcceleratorMemoryMap;

inline std::vector<int> get_shape(const codegen::Tensor& tensor,
                                  bool reshape = true) {
  if (tensor.has_reshape() && reshape) {
    const auto output_shape =
        tensor.reshape().kwargs().at("output_shape").int_list().values();
    return {output_shape.begin(), output_shape.end()};
  }

  const auto repeated_field = tensor.shape();
  return {repeated_field.begin(), repeated_field.end()};
}

inline long get_size(const std::vector<int>& shape) {
  long size = 1;
  for (const auto& dim : shape) size *= dim;
  return size;
}

inline long get_size(const codegen::Tensor& tensor, bool reshape = true) {
  const auto shape = get_shape(tensor, reshape);
  return get_size(shape);
}

inline void print_shape(const std::vector<int>& shape) {
  spdlog::error("(");
  for (size_t i = 0; i < shape.size(); ++i) {
    spdlog::error("{}{}", shape[i], (i + 1 < shape.size() ? ", " : ")"));
  }
  spdlog::error("\n");
}

inline std::vector<codegen::OpOverload> get_op_list(
    const codegen::Operation& param) {
  std::vector<codegen::OpOverload> ops;
  if (param.has_op()) {
    ops.push_back(param.op());
  } else {
    const auto op_list = param.fused_op().op_list();
    ops.insert(ops.end(), op_list.begin(), op_list.end());
  }
  return ops;
}

inline std::string get_op_name(const codegen::Operation& param) {
  if (param.has_op()) {
    return param.op().name();
  } else {
    return param.fused_op().name();
  }
}

inline std::vector<codegen::Tensor> get_op_outputs(
    const codegen::Operation& param) {
  std::vector<codegen::Tensor> outputs;
  if (param.has_output()) {
    outputs.push_back(param.output());
  } else {
    for (const auto& output : param.outputs().tensors()) {
      outputs.push_back(output);
    }
  }
  return outputs;
}
