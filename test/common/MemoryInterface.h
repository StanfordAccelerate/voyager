#pragma once

#include <ac_int.h>

#include <any>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "test/compiler/proto/param.pb.h"

// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

// Abstract class for interfacing with memory models.
class MemoryInterface {
 public:
  MemoryInterface() {}
  virtual ~MemoryInterface() {}

  std::any read_tensor(const codegen::Tensor& tensor) {
    int partition = tensor.memory().partition();

    int size = get_size(tensor, false);

    if (size ==
        1) {  // for scalar, we get the arg from the file, not from memory
      const char* env_var = std::getenv("NETWORK");
      std::string model_name(env_var);
      std::string project_root = std::string(std::getenv("PROJECT_ROOT"));
      std::string datatype = std::string(std::getenv("DATATYPE"));
      std::string filename = project_root + "/" +
                             std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                             model_name + "/" + datatype + "/tensor_files/" +
                             tensor.node() + ".bin";

      float scalar;
      std::ifstream input_stream(filename, std::ios::binary);
      input_stream.read(reinterpret_cast<char*>(&scalar), sizeof(float));

      if (tensor.dtype() == "bfloat16" || tensor.dtype() == "float32") {
        VECTOR_DATATYPE* data = new VECTOR_DATATYPE[1];
        data[0] = scalar;
        return data;
      } else {
        spdlog::debug("Unsupported data type for scalar tensor: {}",
                      tensor.dtype());
        std::abort();
      }
    }

    if (tensor.dtype() == "bfloat16") {
      DataTypes::bfloat16* data = new DataTypes::bfloat16[size];
      read_tensor_from_memory<DataTypes::bfloat16>(tensor.memory().address(),
                                                   partition, size, data);
      return data;
    } else if (tensor.dtype() == "int8") {
      DataTypes::int8* data = new DataTypes::int8[size];
      read_tensor_from_memory<DataTypes::int8>(tensor.memory().address(),
                                               partition, size, data);
      return data;
    } else if (tensor.dtype() == "int24") {
      DataTypes::int24* data = new DataTypes::int24[size];
      read_tensor_from_memory<DataTypes::int24>(tensor.memory().address(),
                                                partition, size, data);
      return data;
    } else if (tensor.dtype() == "int32") {
      DataTypes::int32* data = new DataTypes::int32[size];
      read_tensor_from_memory<DataTypes::int32>(tensor.memory().address(),
                                                partition, size, data);
      return data;
    } else if (tensor.dtype() == "fp8_e8m0") {
      DataTypes::fp8_e8m0* data = new DataTypes::fp8_e8m0[size];
      read_tensor_from_memory<DataTypes::fp8_e8m0>(tensor.memory().address(),
                                                   partition, size, data);
      return data;
    } else if (tensor.dtype() == "fp8_e5m3") {
      DataTypes::fp8_e5m3* data = new DataTypes::fp8_e5m3[size];
      read_tensor_from_memory<DataTypes::fp8_e5m3>(tensor.memory().address(),
                                                   partition, size, data);
      return data;
    } else {
      INPUT_DATATYPE* data = new INPUT_DATATYPE[size];
      read_tensor_from_memory<INPUT_DATATYPE>(tensor.memory().address(),
                                              partition, size, data);
      return data;
    }
  }

  std::vector<std::any> get_outputs(const codegen::Operation& param) {
    const auto tensors = get_op_outputs(param);
    std::vector<std::any> outputs;
    for (const auto& tensor : tensors) {
      outputs.push_back(read_tensor(tensor));
    }
    return outputs;
  }

  std::map<std::string, std::any> get_args(const codegen::Operation& param) {
    std::map<std::string, std::any> kwargs;

    const auto op_list = get_op_list(param);

    for (const auto op : op_list) {
      for (const auto [key, value] : op.kwargs()) {
        if (value.has_tensor() && value.tensor().has_memory()) {
          spdlog::debug("Pushing tensor: {}", value.tensor().node());
          kwargs[value.tensor().node()] = read_tensor(value.tensor());
        }
      }
    }

    return kwargs;
  }

  void write_tensor(const codegen::Tensor& tensor, const std::any data) {
    const auto& tensor_memory = tensor.memory();
    const uint64_t address = tensor_memory.address();
    const int partition = tensor_memory.partition();

    int size = 1;
    for (const auto& dim : tensor.shape()) {
      size *= dim;
    }

    const auto dtype = tensor.dtype();
    if (dtype == "bfloat16") {
      write_tensor_to_memory<DataTypes::bfloat16>(
          address, partition, size, std::any_cast<DataTypes::bfloat16*>(data));
    } else if (dtype == "int8") {
      write_tensor_to_memory<DataTypes::int8>(
          address, partition, size, std::any_cast<DataTypes::int8*>(data));
    } else if (dtype == "int24") {
      write_tensor_to_memory<DataTypes::int24>(
          address, partition, size, std::any_cast<DataTypes::int24*>(data));
    } else if (dtype == "int32") {
      write_tensor_to_memory<DataTypes::int32>(
          address, partition, size, std::any_cast<DataTypes::int32*>(data));
    } else if (dtype == "fp8_e8m0") {
      write_tensor_to_memory<DataTypes::fp8_e8m0>(
          address, partition, size, std::any_cast<DataTypes::fp8_e8m0*>(data));
    } else if (dtype == "fp8_e5m3") {
      write_tensor_to_memory<DataTypes::fp8_e5m3>(
          address, partition, size, std::any_cast<DataTypes::fp8_e5m3*>(data));
    } else {
      // Default to INPUT_DATATYPE
      write_tensor_to_memory<INPUT_DATATYPE>(
          address, partition, size, std::any_cast<INPUT_DATATYPE*>(data));
    }
  }

  template <typename T>
  void read_tensor_from_memory(const long long address, const int partition,
                               const int size, T* tensor) {
    // read extra byte in case of alignment issues
    int bytes_to_read = ((size * T::width) / 8) + 1;
    char* memory_bytes = new char[bytes_to_read];
    read_bytes_from_memory(address, partition, bytes_to_read, memory_bytes);

    ac_int<(T::width / 8 + 2) * 8> bits;

    for (int i = 0; i < size; i++) {
      // Data may be unaligned and span multiple bytes. We calculate the start
      // and end byte indices and the offset within the first byte. We then read
      // the bytes into a temporary ac_int and shift it to the correct position.
      int start = i * T::width / 8;
      int end = (i + 1) * T::width / 8;
      int offset = (i * T::width) % 8;

      for (int j = start; j <= end; j++) {
        bits.set_slc((j - start) * 8,
                     static_cast<ac_int<8, false>>(memory_bytes[j]));
      }

      tensor[i].set_bits(bits >> offset);
    }

    delete[] memory_bytes;
  }

  template <typename T>
  void write_tensor_to_memory(const uint64_t address, const int partition,
                              const int size, T* tensor) {
    size_t total_bytes = (size * T::width + 7) / 8;
    constexpr size_t width_in_bytes = (T::width + 7) / 8;
    char* buffer = new char[total_bytes];

    for (size_t i = 0; i < size; i++) {
      size_t start_bit = i * T::width;

      ac_int<width_in_bytes * 8 + 1, false> value_in_bits =
          tensor[i].bits_rep();

      for (int byte_offset = 0; byte_offset < (T::width + 7) / 8;
           byte_offset++) {
        size_t bit_start = start_bit % 8;
        size_t bits_to_write =
            std::min<size_t>(8 - bit_start, T::width - byte_offset * 8);
        unsigned char mask = (1 << bits_to_write) - 1;
        buffer[(start_bit / 8) + byte_offset] &= ~(mask << bit_start);

        char data = value_in_bits.template slc<8>(byte_offset * 8) & mask;
        buffer[(start_bit / 8) + byte_offset] |= data << bit_start;
      }
    }

    write_bytes_to_memory(address, partition, total_bytes, buffer);
    delete[] buffer;
  }

  template <typename T>
  void write_value_to_memory(const uint64_t address, const int partition,
                             const int index, T value) {
    size_t bit_offset = index * T::width;
    size_t byte_address = address + bit_offset / 8;
    constexpr size_t width_in_bytes = (T::width + 7) / 8;
    char buffer[width_in_bytes];

    ac_int<width_in_bytes * 8 + 1, false> value_in_bits = value.bits_rep();

    // for non-byte aligned data, we need to read the memory first
    if (T::width % 8 != 0) {
      read_bytes_from_memory(byte_address, partition, width_in_bytes, buffer);
    }

    size_t start_bit = bit_offset;
    for (int byte_offset = 0; byte_offset < width_in_bytes; byte_offset++) {
      size_t bit_start = start_bit % 8;
      size_t bits_to_write =
          std::min<size_t>(8 - bit_start, T::width - byte_offset * 8);
      unsigned char mask = (1 << bits_to_write) - 1;
      buffer[byte_offset] &= ~(mask << bit_start);

      char data = value_in_bits.template slc<8>(byte_offset * 8) & mask;
      buffer[byte_offset] |= data << bit_start;
    }
    write_bytes_to_memory(byte_address, partition, width_in_bytes, buffer);
  }

  virtual std::vector<std::any> get_reference_outputs(
      const codegen::Operation& param) = 0;

  virtual void write_bytes_to_memory(const long long address,
                                     const int partition, const int num_bytes,
                                     const char* bytes) = 0;

  virtual void read_bytes_from_memory(const long long address,
                                      const int partition, const int num_bytes,
                                      char* bytes) = 0;
};
