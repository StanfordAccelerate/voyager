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
#define SRAM_SIZE_MB (16 * 1024 * 1024)
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

inline bool is_soc_sim() {
  const char* env = std::getenv("SOC_SIM");
  return env != nullptr && std::string(env) == "1";
}

inline std::vector<int> get_shape(const codegen::Tensor& tensor,
                                  bool reshape = true, bool tiled = true) {
  if (tensor.has_reshape() && reshape) {
    const auto output_shape =
        tensor.reshape().kwargs().at("output_shape").int_list().values();
    return {output_shape.begin(), output_shape.end()};
  }

  if (is_soc_sim() && tiled && tensor.tiled_shape_size()) {
    const auto repeated_field = tensor.tiled_shape();
    return {repeated_field.begin(), repeated_field.end()};
  }

  const auto repeated_field = tensor.shape();
  return {repeated_field.begin(), repeated_field.end()};
}

inline long get_size(const std::vector<int>& shape) {
  long size = 1;
  for (const auto& dim : shape) size *= dim;
  return size;
}

inline long get_size(const codegen::Tensor& tensor, bool reshape = true,
                     bool tiled = true) {
  const auto shape = get_shape(tensor, reshape, tiled);
  return get_size(shape);
}

inline void print_shape(const std::vector<int>& shape) {
  spdlog::error("(");
  for (size_t i = 0; i < shape.size(); ++i) {
    spdlog::error("{}{}", shape[i], (i + 1 < shape.size() ? ", " : ")"));
  }
  spdlog::error("\n");
}

inline int pad_shape_to_ndim(std::vector<int>& shape, const int ndim) {
  const int padding = ndim - shape.size();
  if (padding < 0) {
    throw std::invalid_argument("Number of dimensions exceeds the limit!");
  }

  for (int i = 0; i < padding; i++) {
    shape.insert(shape.begin(), 1);
  }
  return padding;
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

inline int getenv_int(const std::string& name, int default_value = 0) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    return default_value;
  }

  try {
    return std::stoi(val);
  } catch (const std::invalid_argument&) {
    // If the value isn't a valid integer, return the default
    return default_value;
  } catch (const std::out_of_range&) {
    // If the value is too large or small for int, return the default
    return default_value;
  }
}

inline uint64_t get_address(const codegen::Tensor& tensor) {
  if (is_soc_sim() && tensor.has_scratchpad()) {
    std::string offset = getenv("SOC_MEM_OFFSET", "0");
    return tensor.scratchpad().address() + std::stoi(offset);
  } else {
    return tensor.memory().address();
  }
}

inline int get_partition(const codegen::Tensor& tensor) {
  if (is_soc_sim() && tensor.has_scratchpad()) {
    return tensor.scratchpad().partition();
  } else {
    return tensor.memory().partition();
  }
}

inline std::vector<int> get_l2_tiling(codegen::Operation param) {
  const auto op_list = get_op_list(param);
  if (op_list.empty()) {
    return {};
  }

  const auto& first_op = op_list.front();
  const auto& kwargs = first_op.kwargs();

  const auto it = kwargs.find("l2_tiling");
  if (it == kwargs.end()) {
    return {};
  }

  const auto& rep = it->second.int_list().values();
  return {rep.begin(), rep.end()};
}

inline int get_num_tiles(std::vector<int> tiling) {
  int size = 1;
  for (auto d : tiling) {
    size *= d;
  }

  const char* env = std::getenv("MAX_TILES");
  if (env == nullptr) {
    return size;
  }

  int max_tiles = std::stoi(env);
  return std::min(max_tiles, size);
}

inline std::vector<int> get_tiles(std::vector<int> full_shape,
                                  std::vector<int> tiled_shape) {
  const int rank = full_shape.size();
  std::vector<int> tiles(rank);

  pad_shape_to_ndim(tiled_shape, rank);

  for (int d = 0; d < rank; ++d) {
    tiles[d] = full_shape[d] / tiled_shape[d];
  }
  return tiles;
}

inline std::vector<int> get_tile_index(std::vector<int> shape, int index) {
  const int rank = shape.size();
  std::vector<int> tile_index(rank, 0);
  for (int i = 0; i < index; i++) {
    for (int d = rank - 1; d >= 0; --d) {
      if (++tile_index[d] < shape[d]) break;
      tile_index[d] = 0;
    }
  }
  return tile_index;
}
