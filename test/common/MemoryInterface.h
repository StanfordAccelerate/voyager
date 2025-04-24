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
#include "src/datatypes/DataTypes.h"
// clang-format on
#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

// Abstract class for interfacing with memory models.
class MemoryInterface {
 public:
  MemoryInterface() {}
  virtual ~MemoryInterface() {}

  template <typename T>
  bool read_tensor_with_type(codegen::Tensor tensor, std::any& output) {
    if (tensor.dtype() != DataTypes::TypeName<T>::name()) {
      return false;
    }

    const auto& memory = tensor.memory();
    const uint64_t address = memory.address();
    const int partition = memory.partition();
    const int size = get_size(tensor, false);

    int num_bytes = (size * T::width + 7) / 8;
    char* buffer = new char[num_bytes];
    read_bytes_from_memory(address, partition, num_bytes, buffer);

    T* results = new T[size];

    for (int i = 0; i < size; i++) {
      // Data may be unaligned and span multiple bytes. We calculate the start
      // and end byte indices and the offset within the first byte. We then read
      // the bytes into a temporary ac_int and shift it to the correct position.
      int start = i * T::width / 8;
      int end = (i + 1) * T::width / 8;
      int offset = (i * T::width) % 8;

      ac_int<(T::width / 8 + 2) * 8> bits;

      for (int j = start; j <= end; j++) {
        bits.set_slc((j - start) * 8, static_cast<ac_int<8, false>>(buffer[j]));
      }

      results[i].set_bits(bits >> offset);
    }

    delete[] buffer;
    output = results;

    return true;
  }

  template <typename... Ts>
  std::any read_tensor_helper(const codegen::Tensor& tensor) {
    std::any output;
    bool matched = (read_tensor_with_type<Ts>(tensor, output) || ...);
    if (!matched) {
      throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
    }
    return output;
  }

  std::any read_tensor(const codegen::Tensor& tensor) {
    int size = get_size(tensor, false);

    // Read scalar from the file directly
    if (size == 1) {
      if (tensor.dtype() != "bfloat16" && tensor.dtype() != "float32") {
        throw std::runtime_error(
            "Unsupported tensor dtype for scalar tensor: " + tensor.dtype());
      }

      float* array = read_constant_param(tensor);

      VECTOR_DATATYPE* data = new VECTOR_DATATYPE[1];
      data[0] = array[0];

      delete[] array;

      return data;
    }

    return read_tensor_helper<SUPPORTED_TYPES>(tensor);
  }

  template <typename T>
  bool write_tensor_with_type(codegen::Tensor tensor, std::any data) {
    if (tensor.dtype() != DataTypes::TypeName<T>::name()) {
      return false;
    }

    const auto& memory = tensor.memory();
    const uint64_t address = memory.address();
    const int partition = memory.partition();
    const int size = get_size(tensor, false);
    T* casted = std::any_cast<T*>(data);

    size_t total_bytes = (size * T::width + 7) / 8;
    char* buffer = new char[total_bytes];

    constexpr int lcm = T::width * 8 / std::gcd(T::width, 8);

    const int num_groups = size * T::width / lcm;
    const int num_value_per_group = lcm / T::width;
    const int num_bytes_per_group = lcm / 8;
    const int num_bits_remaining = size * T::width - num_groups * lcm;

    for (int i = 0; i < num_groups; i++) {
      ac_int<lcm, false> bits;
      for (int j = 0; j < num_value_per_group; j++) {
        bits.set_slc(j * T::width,
                     static_cast<ac_int<T::width, false>>(
                         casted[i * num_value_per_group + j].bits_rep()));
      }

      for (int j = 0; j < num_bytes_per_group; j++) {
        buffer[i * num_bytes_per_group + j] =
            static_cast<char>(bits.template slc<8>(j * 8));
      }
    }

    // TODO: Handle the remaining bytes
    assert(num_bits_remaining == 0);

    write_bytes_to_memory(address, partition, total_bytes, buffer);
    delete[] buffer;

    return true;
  }

  template <typename... Ts>
  void write_tensor_helper(const codegen::Tensor& tensor, std::any data) {
    bool matched = (write_tensor_with_type<Ts>(tensor, data) || ...);
    if (!matched) {
      throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
    }
  }

  void write_tensor(const codegen::Tensor& tensor, const std::any data) {
    write_tensor_helper<SUPPORTED_TYPES>(tensor, data);
  }

  template <typename T>
  bool write_value_to_memory(const codegen::Tensor& tensor, const int index,
                             T value) {
    if (tensor.dtype() != DataTypes::TypeName<T>::name()) {
      return false;
    }

    const auto& memory = tensor.memory();
    const uint64_t address = memory.address();
    const int partition = memory.partition();

    int start = index * T::width / 8;
    int end = (index + 1) * T::width / 8;
    int offset = (index * T::width) % 8;
    int num_bytes = (end - start + 1) * 8;

    int bits_remaining = num_bytes * 8 - T::width - offset;
    num_bytes = num_bytes - bits_remaining / 8;

    constexpr int buf_width = (T::width / 8 + 2) * 8;
    ac_int<buf_width, false> bytes = value.bits_rep();
    ac_int<buf_width, false> masks = ((1UL << T::width) - 1);

    bytes = bytes << offset;
    masks = masks << offset;

    char buffer[num_bytes];

    // for non-byte aligned data, we need to read the memory first
    read_bytes_from_memory(address + start, partition, num_bytes, buffer);

    for (int i = 0; i < num_bytes; i++) {
      char mask = masks.template slc<8>(i * 8);
      char new_data = bytes.template slc<8>(i * 8) & mask;
      char orig_data = buffer[i] & ~mask;
      buffer[i] = new_data | orig_data;
    }

    write_bytes_to_memory(address + start, partition, num_bytes, buffer);

    return true;
  }

  template <typename... Ts>
  void write_data_helper(const codegen::Tensor& tensor, int index,
                         float value) {
    bool matched = (write_value_to_memory<Ts>(tensor, index, value) || ...);
    if (!matched) {
      throw std::runtime_error("Dataloader: Unsupported tensor dtype: " +
                               tensor.dtype());
    }
  }

  void write_data(const codegen::Tensor& tensor, int index, float value) {
    write_data_helper<SUPPORTED_TYPES>(tensor, index, value);
  }

  std::map<std::string, std::any> get_args(const codegen::Operation& param) {
    std::map<std::string, std::any> kwargs;

    const auto op_list = get_op_list(param);

    for (const auto op : op_list) {
      for (const auto [key, value] : op.kwargs()) {
        if (value.has_tensor() &&
            (value.tensor().has_memory() || get_size(value.tensor()) == 1)) {
          spdlog::debug("Pushing tensor: {}\n", value.tensor().node());
          kwargs[value.tensor().node()] = read_tensor(value.tensor());
        }
      }
    }

    return kwargs;
  }

  std::vector<std::any> get_outputs(const codegen::Operation& param) {
    const auto tensors = get_op_outputs(param);
    std::vector<std::any> outputs;
    for (const auto& tensor : tensors) {
      outputs.push_back(read_tensor(tensor));
    }
    return outputs;
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
