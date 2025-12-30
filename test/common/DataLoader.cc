#include "test/common/DataLoader.h"

#include <fstream>
#include <iostream>

#include "spdlog/spdlog.h"
#include "test/common/Utils.h"
#include "xtensor/xadapt.hpp"

DataLoader::DataLoader(MemoryInterface* memory, bool is_dut)
    : memory(memory), is_dut(is_dut) {}

void DataLoader::load_tensor(const codegen::Tensor& tensor,
                             std::string data_dir, bool replication) {
  const auto shape = get_shape(tensor, false, false);
  const int size = get_size(shape);

  if (size == 1 && !tensor.has_memory()) {
    return;
  }

  const int partition = tensor.memory().partition();
  const uint64_t address = tensor.memory().address();

  spdlog::debug("Loading tensor: {}\n", tensor.node());
  spdlog::debug("Shape: ");
  for (const auto& dim : shape) {
    spdlog::debug("{} ", dim);
  }
  spdlog::debug("\nDatatype: {}\n", tensor.dtype());
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
    if ((index + 1) % 1000000 == 0) {
      spdlog::debug("Writing element {} / {}\n", index + 1, size);
    }
    memory->write_value(partition, address, index, tensor.dtype(), *it);
    ++index;

    if (replication && index % IC_DIMENSION == packing_factor) {
      const int pad = IC_DIMENSION - packing_factor;
      for (int i = 0; i < pad; ++i) {
        // FIXME: for codebook quantization, we need to pad the index of the 0
        // value in the codebook
        memory->write_value(partition, address, index, tensor.dtype(), 0);
        ++index;
      }
    }
  }

  delete[] array_ptr;
}

void DataLoader::load_inputs(const std::vector<Operation> operations,
                             std::string data_dir) {
  std::vector<codegen::Tensor> outputs;

  // collect all output tensors first
  for (const auto& op : operations) {
    const auto out = get_op_outputs(op.param);
    outputs.insert(outputs.end(), out.begin(), out.end());
  }

  for (const auto& op : operations) {
    auto tensors = extract_tensors_from_op(op.param);

    for (auto& t : tensors) {
      bool is_output = std::any_of(
          outputs.begin(), outputs.end(),
          [&](const auto& o) { return o.node() == t.tensor.node(); });

      if (!is_output) {
        load_tensor(t.tensor, data_dir, t.needs_replication);
      }
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
    load_tensor(tensor, data_dir);

    int width = get_type_width<SUPPORTED_TYPES>(tensor.dtype());
    address += get_size(tensor, false, false) * width / 8;
  }
}

std::map<std::string, std::any> DataLoader::get_args(
    const codegen::Operation& param) {
  std::map<std::string, std::any> kwargs;

  auto tensors = extract_tensors_from_op(param);

  for (const auto& t : tensors) {
    spdlog::debug("Reading tensor: {}\n", t.tensor.node());
    kwargs[t.tensor.node()] = memory->read_tensor(t.tensor);
  }

  return kwargs;
}

std::vector<std::any> DataLoader::get_outputs(const codegen::Operation& param) {
  const auto tensors = get_op_outputs(param);
  std::vector<std::any> outputs;
  for (auto tensor : tensors) {
    if (is_soc_sim()) {
      tensor.clear_tiled_shape();
      tensor.clear_scratchpad();
    }
    outputs.push_back(memory->read_tensor(tensor));
  }
  return outputs;
}

std::vector<std::any> DataLoader::get_reference_outputs(
    const codegen::Operation& param) {
  const auto tensors = get_op_outputs(param);
  uint64_t address = 0;
  std::vector<std::any> outputs;

  for (auto tensor : tensors) {
    tensor.mutable_memory()->set_partition(-1);
    tensor.mutable_memory()->set_address(address);
    tensor.clear_tiled_shape();
    tensor.clear_scratchpad();

    outputs.push_back(memory->read_tensor(tensor));

    int width = get_type_width<SUPPORTED_TYPES>(tensor.dtype());
    address += get_size(tensor) * width / 8;
  }

  return outputs;
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

  std::vector<int> packed_tiled_shape = tiled_shape;
  std::vector<int> packed_full_shape = full_shape;
  if (replication) {
    packed_tiled_shape[rank - 1] = packing_factor;
    packed_tiled_shape[rank - 2] =
        tiled_shape[rank - 2] / (packing_factor / tiled_shape[rank - 1]);
    packed_full_shape[rank - 1] = packing_factor;
    packed_full_shape[rank - 2] =
        full_shape[rank - 2] / (packing_factor / full_shape[rank - 1]);
  }

  const int size = get_size(tiled_shape);
  const auto strides = rm_strides(packed_full_shape, replication);
  std::vector<int> indices(rank, 0);

  for (int i = 0; i < size; ++i) {
    if ((i + 1) % 1000000 == 0) {
      spdlog::debug("Copying element {} / {}\n", i + 1, size);
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

    float value =
        memory->read_value(src_partition, src_address, src_index, dtype);
    memory->write_value(dst_partition, dst_address, dst_index, dtype, value);

    // increment indices (row-major)
    for (int d = rank - 1; d >= 0; --d) {
      if (++indices[d] < packed_tiled_shape[d]) break;
      indices[d] = 0;
    }
  }
}

void DataLoader::load_scratchpad(const codegen::Operation& param,
                                 const int tile_index, const int offset) {
  const auto outputs = get_op_outputs(param);
  const auto output = outputs.back();
  const auto output_full_shape = get_shape(output, false, false);
  const auto output_tiled_shape = get_shape(output, false, true);
  const auto output_tiles = get_tiles(output_full_shape, output_tiled_shape);
  const int k_tile = output_tiles.back();

  const auto tensors = extract_tensors_from_op(param);
  const int stack_offset = getenv_int("SOC_MEM_OFFSET", 0);

  for (auto& entry : tensors) {
    const auto& key = entry.key;
    const auto& tensor = entry.tensor;
    bool replication = entry.needs_replication;
    const auto target = entry.target;

    std::string dtype = tensor.dtype();
    const auto full_shape = get_shape(tensor, false, false);
    const auto tiled_shape = get_shape(tensor, false, true);
    std::vector<int> tile_strides;
    if (tensor.tile_strides_size() > 0) {
      tile_strides = {tensor.tile_strides().begin(),
                      tensor.tile_strides().end()};
    } else {
      tile_strides = tiled_shape;
    }

    const int partition = tensor.memory().partition();
    const uint64_t address = tensor.memory().address();

    const int scratchpad_par = tensor.scratchpad().partition();
    const uint64_t scratchpad_addr =
        tensor.scratchpad().address() + offset + stack_offset;

    int curr_tile_index = tile_index;
    if (is_gemm_op(target)) {
      if (key == "input" || key == "input_scale") {
        curr_tile_index = tile_index / k_tile;
      } else if (key == "weight" || key == "other" || key == "weight_scale") {
        curr_tile_index = tile_index % k_tile;
      }
    }

    if (target == "spmm_csr" && key == "indptr") {
      curr_tile_index = tile_index / k_tile;
    }

    auto tiles = get_tiles(full_shape, tiled_shape);
    auto actual_tiles = get_tile_index(tiles, curr_tile_index);

    spdlog::debug("Loading scratchpad tensor: {}\n", tensor.node());
    spdlog::debug("Shape: ");
    for (const auto& dim : tiled_shape) {
      spdlog::debug("{} ", dim);
    }
    spdlog::debug("\nTile: ");
    for (const auto& dim : actual_tiles) {
      spdlog::debug("{} ", dim);
    }
    spdlog::debug("\nTile Strides: ");
    for (const auto& dim : tile_strides) {
      spdlog::debug("{} ", dim);
    }
    spdlog::debug("\nDatatype: {}\n", dtype);
    spdlog::debug("Replication: {}\n", replication);
    spdlog::debug("DRAM Address: {}\n", address);
    spdlog::debug("SRAM Address: {}\n", scratchpad_addr);

    copy_tile(dtype, full_shape, tiled_shape, tile_strides, actual_tiles,
              partition, address, false, scratchpad_par, scratchpad_addr, true,
              replication);
  }
}

void DataLoader::store_scratchpad(const codegen::Operation& param,
                                  const int tile_index, const int offset) {
  const auto tensors = get_op_outputs(param);
  const int stack_offset = getenv_int("SOC_MEM_OFFSET", 0);

  for (const auto& tensor : tensors) {
    std::string dtype = tensor.dtype();
    const auto full_shape = get_shape(tensor, false, false);
    auto tiled_shape = get_shape(tensor, false, true);
    pad_shape_to_ndim(tiled_shape, full_shape.size());
    auto tile_strides = tiled_shape;

    const int partition = tensor.memory().partition();
    const uint64_t address = tensor.memory().address();

    const int scratchpad_par = tensor.scratchpad().partition();
    const uint64_t scratchpad_addr =
        tensor.scratchpad().address() + offset + stack_offset;

    auto tiles = get_tiles(full_shape, tiled_shape);
    auto actual_tiles = get_tile_index(tiles, tile_index);

    spdlog::debug("Storing scratchpad tensor: {}\n", tensor.node());
    spdlog::debug("Shape: ");
    for (const auto& dim : tiled_shape) {
      spdlog::debug("{} ", dim);
    }
    spdlog::debug("\nTile: ");
    for (const auto& dim : actual_tiles) {
      spdlog::debug("{} ", dim);
    }
    spdlog::debug("\nDatatype: {}\n", dtype);
    spdlog::debug("DRAM Address: {}\n", address);
    spdlog::debug("SRAM Address: {}\n", scratchpad_addr);

    copy_tile(dtype, full_shape, tiled_shape, tile_strides, actual_tiles,
              scratchpad_par, scratchpad_addr, true, partition, address, false);
  }
}

std::vector<ExtractedTensor> DataLoader::extract_tensors_from_op(
    const codegen::Operation& param) {
  std::vector<ExtractedTensor> result;
  const auto op_list = get_op_list(param);

  for (const auto& op : op_list) {
    for (const auto& [key, value] : op.kwargs()) {
      if (!value.has_tensor()) continue;

      const auto& tensor = value.tensor();

      // skip if tensor isn't in memory and isn't scalar
      if (!tensor.has_memory() && get_size(tensor) != 1) continue;

      bool is_conv2d = op.target().find("conv2d") != std::string::npos;

      bool needs_replication = is_dut && is_conv2d &&
                               tensor.shape_size() == 4 &&
                               tensor.shape(3) == 3 && key == "input" &&
                               op.kwargs().at("groups").int_value() == 1;

      result.push_back({key, tensor, needs_replication, op.target()});

      // handle fused dequant scale / zero_point
      if (tensor.has_dequant()) {
        auto dequant = tensor.dequant();

        if (dequant.kwargs().contains("scale")) {
          const auto& scale = dequant.kwargs().at("scale");
          if (scale.has_tensor() && scale.tensor().has_memory()) {
            result.push_back({"scale", scale.tensor(), false, op.target()});
          }
        }

        if (dequant.kwargs().contains("zero_point")) {
          const auto& zp = dequant.kwargs().at("zero_point");
          if (zp.has_tensor() && zp.tensor().has_memory()) {
            result.push_back({"zero_point", zp.tensor(), false, op.target()});
          }
        }
      }
    }
  }

  return result;
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
