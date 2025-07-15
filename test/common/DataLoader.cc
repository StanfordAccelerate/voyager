#include "test/common/DataLoader.h"

#include <fstream>
#include <iostream>

#include "spdlog/spdlog.h"
#include "test/common/VerificationTypes.h"
#include "xtensor/xadapt.hpp"

DataLoader::DataLoader(MemoryInterface* memory_interface, bool is_dut)
    : memory_interface(memory_interface), is_dut(is_dut) {}

void DataLoader::load_tensor(const codegen::Tensor& tensor,
                             std::string data_dir, bool transpose,
                             bool replication) {
  const auto shape = get_shape(tensor, false);
  const int size = get_size(shape);
  const uint64_t offset = get_address(tensor);

  if (size == 1 && !tensor.has_memory()) {
    return;
  }

  spdlog::debug("Loading tensor: {}\n", tensor.node());
  spdlog::debug("Shape: ");
  for (const auto& dim : shape) {
    spdlog::debug("{} ", dim);
  }
  spdlog::debug("\n");
  spdlog::debug("Datatype: {}\n", tensor.dtype());
  spdlog::debug("Address: {}\n", offset);
  spdlog::debug("Transposed: {}\n", transpose);
  spdlog::debug("Replication: {}\n", replication);

  std::string filename = data_dir + "/" + tensor.node() + ".bin";
  auto array_ptr = read_tensor_from_file(filename, size);
  auto array = xt::adapt(array_ptr, size, xt::no_ownership(), shape);

  // Accelerator expect the data to be layed out in a different order
  if (transpose) {
    if (shape.size() == 4) {
      array = xt::transpose(array, {2, 3, 1, 0});
    } else if (shape.size() == 2) {
      array = xt::transpose(array, {1, 0});
    }
  }

  // number of elements packed into a single word for replication
  const int packing_factor = IC_DIMENSION / 4 * 3;
  if (replication) {
    spdlog::debug("packing factor: {}", packing_factor);
  }

  int address = 0;
  for (auto it = array.begin(); it != array.end(); ++it) {
    memory_interface->write_data(tensor, address, *it);

    address++;
    if (replication && address % IC_DIMENSION == packing_factor) {
      address += IC_DIMENSION - packing_factor;
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
              tensor.shape(3) == 3) {
            inputs_with_replication.push_back(value.tensor());
          } else {
            inputs.push_back(value.tensor());
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
      load_tensor(tensor, data_dir, false, false);
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
      load_tensor(tensor, data_dir, false, true);
    }
  }
}

void DataLoader::load_outputs(const codegen::Operation param,
                              std::string data_dir) {
  const auto tensors = get_op_outputs(param);

  const auto op_list = get_op_list(param);
  std::string opcode = op_list[0].target();

  uint64_t address = 0;

  for (const auto& tensor : tensors) {
    codegen::Tensor output_tensor;
    output_tensor.CopyFrom(tensor);
    // Store output in the last memory partition
    output_tensor.mutable_memory()->set_partition(-1);

    if (is_soc_sim()) {
      output_tensor.mutable_scratchpad()->set_offset(address);
    } else {
      output_tensor.mutable_memory()->set_address(address);
    }

    load_tensor(output_tensor, data_dir);
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
