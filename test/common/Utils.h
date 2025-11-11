#pragma once

#define NO_SYSC

// IWYU pragma: begin_exports
#include <any>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"
#include "test/compiler/proto/param.pb.h"
// IWYU pragma: end_exports

inline bool is_gemm_op(const std::string& op_target) {
  static const std::unordered_set<std::string> GEMM_OPS = {
      "conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx",
  };
  return GEMM_OPS.find(op_target) != GEMM_OPS.end();
}

inline bool is_fc_layer(const codegen::OpOverload& op) {
  if (!is_gemm_op(op.target())) {
    return false;
  }

  const auto input = op.kwargs().at("input").tensor();

  int dim = 1;
  for (int i = 0; i < input.shape_size() - 1; i++) {
    dim *= input.shape(i);
  }

  return dim == 1;
}

inline bool is_dma_op(const std::string& op_target) {
  static const std::unordered_set<std::string> DMA_OPS = {
      "slice",
      "permute",
      "transpose",
  };
  return DMA_OPS.find(op_target) != DMA_OPS.end();
}

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

inline float get_tensor_scalar_scale(const codegen::Tensor& tensor) {
  if (!tensor.has_dequant()) {
    return 1.0f;
  }

  const auto& dequantize_op = tensor.dequant();
  const auto scale_tensor = dequantize_op.kwargs().at("scale").tensor();
  float* scale_val = read_constant_param(scale_tensor);
  return scale_val[0];
}

static void log_diff_buckets(const std::string& label, const int buckets[5],
                             size_t size) {
  spdlog::info("{}:\n", label);
  spdlog::info("< 0.001: {:8d} ({:6.2f}%)\n", buckets[0],
               static_cast<float>(buckets[0]) / size * 100.0f);
  spdlog::info("< 0.01:  {:8d} ({:6.2f}%)\n", buckets[1],
               static_cast<float>(buckets[1]) / size * 100.0f);
  spdlog::info("< 0.1:   {:8d} ({:6.2f}%)\n", buckets[2],
               static_cast<float>(buckets[2]) / size * 100.0f);
  spdlog::info("< 1:     {:8d} ({:6.2f}%)\n", buckets[3],
               static_cast<float>(buckets[3]) / size * 100.0f);
  spdlog::info("> 1:     {:8d} ({:6.2f}%)\n", buckets[4],
               static_cast<float>(buckets[4]) / size * 100.0f);
}

template <typename T1, typename T2>
float compare_arrays(std::any matrix_a, const std::string& name_a,
                     std::any matrix_b, const std::string& name_b, size_t size,
                     const std::string& filename, bool double_precision) {
  spdlog::info("Comparing {} and {} (size: {}) -> output: {}\n", name_a, name_b,
               size, filename);

  std::ofstream diff_file;
  diff_file.open(filename);

  std::ostringstream buffer;
  buffer << name_a << " vs. " << name_b << '\n';
  constexpr size_t flush_interval = 10000;  // Flush every 10k lines

  int abs_diff_buckets[5] = {0};
  int rel_diff_buckets[5] = {0};
  double sum_abs = 0.0;

  T1* matrix_a_ptr = std::any_cast<T1*>(matrix_a);
  T2* matrix_b_ptr = std::any_cast<T2*>(matrix_b);

  // For compressing repeated identical lines in output file
  std::string prev_line = "";
  size_t repeat_start_index = 0;
  size_t repeat_count = 0;
  size_t lines_written = 0;

  for (size_t i = 0; i < size; ++i) {
    float a = static_cast<float>(matrix_a_ptr[i]);
    float b = static_cast<float>(matrix_b_ptr[i]);
    sum_abs += std::abs(a) + std::abs(b);
    float abs_diff = std::abs(a - b);

    // Build the current line content
    std::ostringstream line_stream;
    line_stream << a << " vs. " << b << " ";
    for (float j = 0.001f; j < abs_diff; j *= 10.0f) {
      line_stream << '*';
    }
    std::string current_line = line_stream.str();

    if (current_line == prev_line && a == 0 && b == 0) {
      repeat_count++;
    } else {
      if (repeat_count > 0) {
        if (repeat_count == 1) {
          buffer << "[" << repeat_start_index << "] " << prev_line << '\n';
        } else {
          buffer << "[" << repeat_start_index << "-"
                 << (repeat_start_index + repeat_count - 1) << "] ("
                 << repeat_count << "x) " << prev_line << '\n';
        }
        lines_written++;

        // Periodically flush buffer to disk to avoid memory issues
        if (lines_written % flush_interval == 0) {
          diff_file << buffer.str();
          buffer.str("");
          buffer.clear();
        }
      }

      // Start tracking the new line
      prev_line = current_line;
      repeat_start_index = i;
      repeat_count = 1;
    }

    bool is_nan = std::isinf(abs_diff) || std::isnan(abs_diff);

    // Absolute difference buckets
    abs_diff_buckets[0] += abs_diff < 0.001f;
    abs_diff_buckets[1] += abs_diff < 0.01f;
    abs_diff_buckets[2] += abs_diff < 0.1f;
    abs_diff_buckets[3] += abs_diff < 1.0f;
    abs_diff_buckets[4] += abs_diff >= 1.0f && !is_nan;

    // Does not fully protect against overflow, but lets not over engineer
    if ((a == 0 && b == 0) || is_nan) {
      rel_diff_buckets[0]++;
      rel_diff_buckets[1]++;
      rel_diff_buckets[2]++;
      rel_diff_buckets[3]++;
    } else {
      // See https://en.wikipedia.org/wiki/Relative_change_and_difference
      float rel_diff = abs_diff / ((std::abs(a) + std::abs(b)) / 2.0f);
      rel_diff_buckets[0] += rel_diff < 0.001f;
      rel_diff_buckets[1] += rel_diff < 0.01f;
      rel_diff_buckets[2] += rel_diff < 0.1f;
      rel_diff_buckets[3] += rel_diff < 1.0f;
      rel_diff_buckets[4] += rel_diff >= 1.0f;
    }
  }

  // Write out the final line(s) after the loop
  if (repeat_count > 0) {
    if (repeat_count == 1) {
      buffer << "[" << repeat_start_index << "] " << prev_line << '\n';
    } else {
      buffer << "[" << repeat_start_index << "-"
             << (repeat_start_index + repeat_count - 1) << "] (" << repeat_count
             << "x) " << prev_line << '\n';
    }
  }

  // Final flush of remaining buffered content
  diff_file << buffer.str();
  diff_file.close();

  spdlog::info("-------------------------------\n");
  log_diff_buckets("Absolute Difference Count", abs_diff_buckets, size);
  spdlog::info("-------------------------------\n");
  log_diff_buckets("Relative Difference Count", rel_diff_buckets, size);
  spdlog::info("-------------------------------\n");

  if (sum_abs == 0.0) {
    spdlog::warn("WARNING: All compared values are zero!\n");
  }

  float error = (1 - (float)rel_diff_buckets[1] / size) * 0.001f +
                (1 - (float)rel_diff_buckets[2] / size) * 0.01f +
                (1 - (float)rel_diff_buckets[3] / size) * 0.1f +
                (float)rel_diff_buckets[4] / size;

  return error * 100.0f;
}
