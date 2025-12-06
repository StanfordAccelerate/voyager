#include "test/common/DataLoader.h"

#include <fstream>
#include <iostream>

#include "spdlog/spdlog.h"
#include "test/common/Utils.h"
#include "xtensor/xadapt.hpp"

DataLoader::DataLoader(MemoryInterface* memory_interface, bool is_dut)
    : memory_interface(memory_interface), is_dut(is_dut) {}

void DataLoader::load_tensor(const codegen::Tensor& tensor,
                             std::string data_dir, bool replication) {
  const auto shape = get_shape(tensor, false, false);
  const int size = get_size(shape);

  if (size == 1 && !tensor.has_memory()) {
    return;
  }

  const auto& memory = tensor.memory();
  const int partition = memory.partition();
  const uint64_t address = memory.address();

  spdlog::debug("Loading tensor: {}\n", tensor.node());
  spdlog::debug("Shape: ");
  for (const auto& dim : shape) {
    spdlog::debug("{} ", dim);
  }
  spdlog::debug("\n");
  spdlog::debug("Datatype: {}\n", tensor.dtype());
  spdlog::debug("Address: {}\n", address);
  spdlog::debug("Replication: {}\n", replication);

  std::string filename = data_dir + "/" + tensor.node() + ".bin";
  auto array_ptr = read_tensor_from_file(filename, size);
  auto array = xt::adapt(array_ptr, size, xt::no_ownership(), shape);

  // number of elements packed into a single word for replication
  const int effective_ic_dimension = IC_DIMENSION == 64 ? 32 : IC_DIMENSION;
  const int packing_factor = effective_ic_dimension / 4 * 3;
  if (replication) {
    spdlog::debug("packing factor: {}", packing_factor);
  }

  int index = 0;
  for (auto it = array.begin(); it != array.end(); ++it) {
    if (index % 1000000 == 0) {
      spdlog::debug("Writing element {} / {}\n", index, size);
    }
    memory_interface->write_value(partition, address, index, tensor.dtype(),
                                  *it);

    index++;
    if (replication && index % IC_DIMENSION == packing_factor) {
      index += IC_DIMENSION - packing_factor;
    }
  }

  delete[] array_ptr;
}

void DataLoader::load_inputs(const std::vector<Operation> operations,
                             std::string data_dir) {
  std::vector<codegen::Tensor> inputs;
  std::vector<codegen::Tensor> inputs_with_replication;
  std::vector<codegen::Tensor> outputs;

  for (const auto op : operations) {
    const auto op_list = get_op_list(op.param);
    for (const auto op : op_list) {
      for (const auto [key, value] : op.kwargs()) {
        const auto& tensor = value.tensor();
        if (value.has_tensor() &&
            (value.tensor().has_memory() || get_size(value.tensor()) == 1)) {
          bool is_conv2d = op.target().find("conv2d") != std::string::npos;
          if (is_dut && is_conv2d && tensor.shape_size() == 4 &&
              tensor.shape(3) == 3 && key == "input" &&
              op.kwargs().at("groups").int_value() == 1) {
            inputs_with_replication.push_back(value.tensor());
          } else {
            inputs.push_back(value.tensor());
          }

          const auto tensor = value.tensor();
          if (tensor.has_dequant()) {
            const auto dequant_op = tensor.dequant();
            const auto scale = dequant_op.kwargs().at("scale");
            const auto scale_tensor = scale.tensor();

            if (scale.has_tensor() && scale_tensor.has_memory()) {
              inputs.push_back(scale_tensor);
            }

            if (dequant_op.kwargs().contains("zero_point")) {
              const auto zero_point = dequant_op.kwargs().at("zero_point");
              const auto zero_point_tensor = zero_point.tensor();
              if (zero_point.has_tensor() && zero_point_tensor.has_memory()) {
                inputs.push_back(zero_point_tensor);
              }
            }
          }
        }
      }
    }

    const auto tensors = get_op_outputs(op.param);
    for (const auto& tensor : tensors) {
      outputs.push_back(tensor);
    }
  }

  for (const auto& tensor : inputs) {
    bool found = false;
    for (const auto& output : outputs) {
      if (tensor.node() == output.node()) {
        found = true;
        break;
      }
    }

    if (!found) {
      load_tensor(tensor, data_dir, false);
    }
  }

  for (const auto& tensor : inputs_with_replication) {
    bool found = false;
    for (const auto& output : outputs) {
      if (tensor.node() == output.node()) {
        found = true;
        break;
      }
    }

    if (!found) {
      load_tensor(tensor, data_dir, true);
    }
  }
}

void DataLoader::load_outputs(const codegen::Operation param,
                              std::string data_dir) {
  const auto tensors = get_op_outputs(param);
  uint64_t address = 0;

  for (auto tensor : tensors) {
    // Store output in the last memory partition
    tensor.mutable_memory()->set_partition(-1);
    tensor.mutable_memory()->set_address(address);
    tensor.clear_tiled_shape();

    load_tensor(tensor, data_dir);
    address += get_size(tensor);
  }
}

float* DataLoader::read_tensor_from_file(const std::string& filename,
                                         int size) {
  float* value_ptr = new float[size];
  std::ifstream input_stream(filename, std::ios::binary);
  if (!input_stream.good()) {
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  }
  input_stream.read(reinterpret_cast<char*>(value_ptr), size * sizeof(float));
  if (!input_stream) {
    throw std::runtime_error(
        "Failed to read the expected amount of data from the file");
  }
  return value_ptr;
}

static inline std::vector<int> rm_strides(const std::vector<int>& shape,
                                          bool replication = false) {
  const int r = (int)shape.size();
  std::vector<int> s(r, 1);
  for (int i = r - 2; i >= 0; --i) {
    s[i] = s[i + 1] * shape[i + 1];
  }
  if (replication) {
    s[r - 2] = IC_DIMENSION;
    for (int i = r - 3; i >= 0; --i) {
      s[i] = s[i + 1] * shape[i + 1];
    }
  }
  return s;
}

void DataLoader::copy_tile(
    const std::string& dtype, const std::vector<int>& full_shape,
    const std::vector<int>& tiled_shape, const std::vector<int>& tile_strides,
    const std::vector<int>& tile_index, int src_partition, uint64_t src_address,
    bool src_contiguous, int dst_partition, uint64_t dst_address,
    bool dst_contiguous, bool replication) {
  const int rank = full_shape.size();

  std::vector<int> start(rank);
  for (int d = 0; d < rank; ++d) {
    start[d] = tile_index[d] * tile_strides[d];
  }

  const int effective_ic_dimension = IC_DIMENSION == 64 ? 32 : IC_DIMENSION;
  const int packing_factor = effective_ic_dimension / 4 * 3;

  std::vector<int> tiled_shape_copy = tiled_shape;
  std::vector<int> full_shape_copy = full_shape;
  if (replication) {
    tiled_shape_copy[rank - 1] = packing_factor;
    tiled_shape_copy[rank - 2] =
        tiled_shape[rank - 2] / (packing_factor / tiled_shape[rank - 1]);
    full_shape_copy[rank - 1] = packing_factor;
    full_shape_copy[rank - 2] =
        full_shape[rank - 2] / (packing_factor / full_shape[rank - 1]);
  }

  const int size = get_size(tiled_shape);
  const auto strides = rm_strides(full_shape_copy, replication);
  std::vector<int> indices(rank, 0);

  for (int i = 0; i < size; ++i) {
    if (i % 1000000 == 0) {
      spdlog::debug("Copying element {} / {}\n", i, size);
    }
    int index = i;
    // In the case of replication, we store 8 x 3 elements in a single word
    if (replication) {
      index = ((i / packing_factor) * IC_DIMENSION) + (i % packing_factor);
    }

    int flat = 0;
    for (int d = 0; d < rank; ++d) {
      flat += (start[d] + indices[d]) * strides[d];
    }

    int src_index = src_contiguous ? index : flat;
    int dst_index = dst_contiguous ? index : flat;

    float value = memory_interface->read_value(src_partition, src_address,
                                               src_index, dtype);
    memory_interface->write_value(dst_partition, dst_address, dst_index, dtype,
                                  value);

    // increment indices (row-major)
    for (int d = rank - 1; d >= 0; --d) {
      if (++indices[d] < tiled_shape_copy[d]) break;
      indices[d] = 0;
    }
  }
}

void DataLoader::load_scratchpad(const codegen::Operation& param,
                                 const int tile_index, const int offset) {
  const auto outputs = get_op_outputs(param);
  const auto output = outputs.back();
  const auto output_full_shape = get_shape(output, false, false);
  const auto output_tiled_shape = get_shape(output, false);
  const auto output_tiles = get_tiles(output_full_shape, output_tiled_shape);
  const int k_tile = output_tiles.back();

  const auto op_list = get_op_list(param);

  for (const auto op : op_list) {
    std::unordered_map<std::string, codegen::Tensor> tensor_dict;
    std::vector<std::string> tensors_with_replication;

    for (const auto [key, value] : op.kwargs()) {
      bool is_conv2d = op.target().find("conv2d") != std::string::npos;
      const auto tensor = value.tensor();
      if (value.has_tensor() && tensor.has_memory()) {
        tensor_dict[key] = tensor;

        if (is_dut && is_conv2d && tensor.shape_size() == 4 &&
            tensor.shape(3) == 3 && op.kwargs().at("groups").int_value() == 1 &&
            key == "input") {
          tensors_with_replication.push_back(key);
        }

        if (tensor.has_dequant()) {
          const auto dequant_op = tensor.dequant();
          const auto scale = dequant_op.kwargs().at("scale");
          const auto scale_tensor = scale.tensor();

          if (scale.has_tensor() && scale_tensor.has_memory()) {
            tensor_dict["scale"] = scale_tensor;
          }

          if (dequant_op.kwargs().contains("zero_point")) {
            const auto zero_point = dequant_op.kwargs().at("zero_point");
            const auto zero_point_tensor = zero_point.tensor();
            if (zero_point.has_tensor() && zero_point_tensor.has_memory()) {
              tensor_dict["zero_point"] = zero_point_tensor;
            }
          }
        }
      }
    }

    int stack_offset = std::stoi(getenv("SOC_MEM_OFFSET", "0"));

    for (const auto& [key, tensor] : tensor_dict) {
      std::string dtype = tensor.dtype();
      const auto full_shape = get_shape(tensor, false, false);
      const auto tiled_shape = get_shape(tensor, false);
      std::vector<int> tile_strides;
      if (tensor.tile_strides_size() > 0) {
        tile_strides = {tensor.tile_strides().begin(),
                        tensor.tile_strides().end()};
      } else {
        tile_strides = tiled_shape;
      }

      const int partition = tensor.memory().partition();
      const uint64_t address = tensor.memory().address();

      const auto scratch = tensor.scratchpad();
      const int scratch_par = scratch.partition();
      const uint64_t scratch_addr = scratch.address() + offset + stack_offset;

      int curr_tile_index = tile_index;

      if (is_gemm_op(op.target())) {
        if (key == "input" || key == "input_scale") {
          curr_tile_index = tile_index / k_tile;
        } else if (key == "weight" || key == "other" || key == "weight_scale") {
          curr_tile_index = tile_index % k_tile;
        }
      }

      auto tiles = get_tiles(full_shape, tiled_shape);
      auto actual_tiles = get_tile_index(tiles, curr_tile_index);

      bool replication = std::find(tensors_with_replication.begin(),
                                   tensors_with_replication.end(),
                                   key) != tensors_with_replication.end();

      spdlog::debug("Loading scratchpad tensor: {}\n", tensor.node());
      spdlog::debug("Shape: ");
      for (const auto& dim : tiled_shape) {
        spdlog::debug("{} ", dim);
      }
      spdlog::debug("\n");
      spdlog::debug("Tile: ");
      for (const auto& dim : actual_tiles) {
        spdlog::debug("{} ", dim);
      }
      spdlog::debug("\n");
      spdlog::debug("Tile Strides: ");
      for (const auto& dim : tile_strides) {
        spdlog::debug("{} ", dim);
      }
      spdlog::debug("\n");
      spdlog::debug("Datatype: {}\n", dtype);
      spdlog::debug("Replication: {}\n", replication);
      spdlog::debug("Main Memory Address: {}\n", address);
      spdlog::debug("Scratchpad Address: {}\n", scratch_addr);

      copy_tile(dtype, full_shape, tiled_shape, tile_strides, actual_tiles,
                partition, address, false, scratch_par, scratch_addr, true,
                replication);
    }
  }
}

void DataLoader::store_scratchpad(const codegen::Operation& param,
                                  const int tile_index, const int offset) {
  const auto tensors = get_op_outputs(param);
  int stack_offset = std::stoi(getenv("SOC_MEM_OFFSET", "0"));
  for (const auto& tensor : tensors) {
    std::string dtype = tensor.dtype();
    const auto full_shape = get_shape(tensor, false, false);
    auto tiled_shape = get_shape(tensor, false);
    pad_shape_to_ndim(tiled_shape, full_shape.size());
    auto tile_strides = tiled_shape;

    const int partition = tensor.memory().partition();
    const uint64_t address = tensor.memory().address();

    const auto scratch = tensor.scratchpad();
    const int scratch_par = scratch.partition();
    const uint64_t scratch_addr = scratch.address() + offset + stack_offset;

    auto tiles = get_tiles(full_shape, tiled_shape);
    auto actual_tiles = get_tile_index(tiles, tile_index);

    spdlog::debug("Storing scratchpad tensor: {}\n", tensor.node());
    spdlog::debug("Shape: ");
    for (const auto& dim : tiled_shape) {
      spdlog::debug("{} ", dim);
    }
    spdlog::debug("\n");
    spdlog::debug("Tile: ");
    for (const auto& dim : actual_tiles) {
      spdlog::debug("{} ", dim);
    }
    spdlog::debug("\n");
    spdlog::debug("Datatype: {}\n", dtype);
    spdlog::debug("Main Memory Address: {}\n", address);
    spdlog::debug("Scratchpad Address: {}\n", scratch_addr);

    copy_tile(dtype, full_shape, tiled_shape, tile_strides, actual_tiles,
              scratch_par, scratch_addr, true, partition, address, false);
  }
}
