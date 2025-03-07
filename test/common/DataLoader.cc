#include "test/common/DataLoader.h"

#include "spdlog/spdlog.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

DataLoader::DataLoader(MemoryInterface* memory_interface, bool is_dut,
                       bool is_cnn)
    : memory_interface(memory_interface), is_dut(is_dut), is_cnn(is_cnn) {}

void DataLoader::load_tensor(const codegen::Tensor& tensor,
                             std::string data_dir, bool transpose,
                             bool replication) {
  const auto shape = get_shape(tensor, false);
  const int size = get_size(shape);

  spdlog::debug("Loading tensor: {}\n", tensor.node());
  spdlog::debug("Shape: ");
  for (const auto& dim : shape) {
    spdlog::debug("{} ", dim);
  }
  spdlog::debug("Datatype: {}\n", tensor.dtype());
  spdlog::debug("Address: {}\n", tensor.memory().address());
  spdlog::debug("Transposed: {}\n", transpose);
  spdlog::debug("Replication: {}\n", replication);

  // if size is 1, then it is a scalar, so it should not be
  // written to memory
  if (size == 1) {
    return;
  }

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

  int partition = tensor.memory().partition();
  uint64_t offset = tensor.memory().address();

  // number of elements packed into a single word for replication
  const int packing_factor = IC_DIMENSION / 4 * 3;
  if (replication) {
    spdlog::debug("packing factor: {}", packing_factor);
  }

  int address = 0;
  for (auto it = array.begin(); it != array.end(); ++it) {
    if (tensor.dtype() == "int8") {
      memory_interface->write_value_to_memory<DataTypes::int8>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "bfloat16") {
      memory_interface->write_value_to_memory<DataTypes::bfloat16>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "int24") {
      memory_interface->write_value_to_memory<DataTypes::int24>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "int32") {
      memory_interface->write_value_to_memory<DataTypes::int32>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "fp8_e8m0") {
      memory_interface->write_value_to_memory<DataTypes::fp8_e8m0>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "fp8_e5m3") {
      memory_interface->write_value_to_memory<DataTypes::fp8_e5m3>(
          offset, partition, address, *it);
    } else {
      // if unspecified, we will assume it's INPUT_DATATYPE
      memory_interface->write_value_to_memory<INPUT_DATATYPE>(offset, partition,
                                                              address, *it);
    }

    address++;
    if (replication && address % IC_DIMENSION == packing_factor) {
      address += IC_DIMENSION - packing_factor;
    }
  }

  delete[] array_ptr;
}

void DataLoader::load_inputs(const codegen::Operation param,
                             std::string data_dir, bool random_data) {
  // convolution layer inputs/outputs need to be permuted. If the matrix
  // operation is a convolution, the following fused vector operations will
  // need to be permuted as well. This logic should be refined in the future.
  const auto op_list = get_op_list(param);

  for (const auto& op : op_list) {
    for (const auto [key, value] : op.kwargs()) {
      const auto& tensor = value.tensor();
      if (value.has_tensor() && tensor.has_memory() &&
          tensor.node().find("constant") == std::string::npos) {
        bool is_conv2d = op.target().find("conv2d") != std::string::npos;
        bool is_replication = is_conv2d && tensor.shape(1) == 3 && is_dut;
        load_tensor(value.tensor(), data_dir, is_cnn, is_replication);
      }
    }
  }
}

void DataLoader::load_outputs(const codegen::Operation param,
                              std::string data_dir) {
  const auto tensors = get_op_outputs(param);

  const auto op_list = get_op_list(param);
  std::string opcode = op_list[0].target();
  bool transpose = is_cnn && opcode != "linear" && opcode != "linear_mx";

  uint64_t address = 0;

  for (const auto& tensor : tensors) {
    codegen::Tensor output_tensor;
    output_tensor.CopyFrom(tensor);
    // Store output in the last memory partition
    output_tensor.mutable_memory()->set_partition(-1);
    output_tensor.mutable_memory()->set_address(address);

    load_tensor(output_tensor, data_dir, transpose);
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
