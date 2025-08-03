#pragma once

#define NO_SYSC

// IWYU pragma: begin_exports
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

// clang-format off
#include "src/datatypes/DataTypes.h"
// clang-format on
#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "test/compiler/proto/param.pb.h"
// IWYU pragma: end_exports

#define DRAM_SIZE_MB (1024 * 1024LL * 1024LL)
#define SRAM_SIZE_MB (8 * 1024 * 1024)
#define REFERENCE_MEMORY_SIZE (8 * 1024 * 1024)

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

inline bool is_fc(const codegen::OpOverload& op) {
  const auto input = op.kwargs().at("input").tensor();

  int dim = 1;
  for (int i = 0; i < input.shape_size() - 1; i++) {
    dim *= input.shape(i);
  }

  return dim == 1;
}

inline float* read_constant_param(const codegen::Tensor& tensor) {
  const char* env_var = std::getenv("NETWORK");
  std::string model_name(env_var);
  std::string project_root = std::string(std::getenv("PROJECT_ROOT"));
  std::string datatype = std::string(std::getenv("DATATYPE"));
  std::string filename =
      project_root + "/" + std::string(getenv("CODEGEN_DIR")) + "/networks/" +
      model_name + "/" + datatype + "/tensor_files/" + tensor.node() + ".bin";

  const int size = get_size(tensor, false);

  float* data = new float[size];
  std::ifstream input_stream(filename, std::ios::binary);
  input_stream.read(reinterpret_cast<char*>(data), size * sizeof(float));

  return data;
}

inline std::string getenv(std::string const& name,
                          std::string const& default_value) {
  const char* val = std::getenv(name.c_str());
  return val == NULL ? default_value : std::string(val);
}

inline bool is_soc_sim() {
  const char* env = std::getenv("SOC_SIM");
  return env != nullptr && std::string(env) == "1";
}

inline uint64_t get_address(const codegen::Tensor& tensor) {
  if (is_soc_sim() && tensor.has_scratchpad()) {
    std::string offset = getenv("SOC_MEM_OFFSET", "0");
    return tensor.scratchpad().offset() + std::stoi(offset);
  } else {
    return tensor.memory().address();
  }
}
