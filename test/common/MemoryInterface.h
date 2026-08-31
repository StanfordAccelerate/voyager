#pragma once

#include <ac_int.h>

#include <any>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"
#include "test/common/Tensor.h"
#include "test/common/Utils.h"

// Abstract class for interfacing with memory models.
class MemoryInterface {
 public:
  MemoryInterface() {}
  virtual ~MemoryInterface() {}

  template <typename T>
  bool read_tensor_with_type(Tensor tensor, std::any& output) {
    if (tensor.dtype != DataTypes::TypeName<T>::name()) {
      return false;
    }

    const uint64_t address = get_address(tensor);
    const int partition = get_partition(tensor);
    const int size = get_size(tensor);

    // A pitched window gathers rows of `run` valid elements sitting `pitch`
    // elements apart; a contiguous tensor is the pitch == run case.
    const int64_t run = tensor.window_pitch > 0 ? tensor.shape.back() : size;
    const int64_t pitch = tensor.window_pitch > 0 ? tensor.window_pitch : run;
    const int64_t span = size > 0 ? (size / run - 1) * pitch + run : 0;

    const int64_t num_bytes = (span * T::width + 7) / 8;
    char* buffer = new char[num_bytes];
    read_bytes_from_memory(address, partition, num_bytes, buffer);

    std::shared_ptr<T[]> results(new T[size]);

    for (int i = 0; i < size; i++) {
      const int64_t u = (i / run) * pitch + i % run;
      // Data may be unaligned and span multiple bytes. We calculate the start
      // and end byte indices and the offset within the first byte. We then read
      // the bytes into a temporary ac_int and shift it to the correct position.
      const int64_t start = u * T::width / 8;
      const int64_t end = ((u + 1) * T::width - 1) / 8;
      const int offset = (u * T::width) % 8;

      ac_int<(T::width / 8 + 2) * 8> bits;

      for (int64_t j = start; j <= end; j++) {
        bits.set_slc(static_cast<int>(j - start) * 8,
                     static_cast<ac_int<8, false>>(buffer[j]));
      }

      results[i].set_bits(bits >> offset);
    }

    delete[] buffer;
    output = results;

    return true;
  }

  template <typename... Ts>
  std::any read_tensor_helper(const Tensor& tensor) {
    std::any output;
    bool matched = (read_tensor_with_type<Ts>(tensor, output) || ...);
    if (!matched) {
      throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype);
    }
    return output;
  }

  template <typename T>
  bool create_scalar_tensor(const Tensor& tensor, std::any& data,
                            float* array) {
    if (tensor.dtype != DataTypes::TypeName<T>::name()) {
      return false;
    }

    std::shared_ptr<T[]> scalar(new T[1]);
    scalar[0] = array[0];
    data = scalar;

    delete[] array;

    return true;
  }

  template <typename... Ts>
  std::any create_scalar_tensor_helper(const Tensor& tensor, float* array) {
    std::any data;
    bool matched = (create_scalar_tensor<Ts>(tensor, data, array) || ...);
    if (!matched) {
      throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype);
    }
    return data;
  }

  std::any read_tensor(const Tensor& tensor) {
    // A config constant is never mapped into memory -- the graph never DMAs it,
    // so it has no address to read from. It comes off disk instead.
    if (tensor.is_constant) {
      if (get_size(tensor) != 1) {
        throw std::runtime_error(
            "Config constant " + tensor.node +
            " is not a scalar; it cannot be materialized as a tensor operand.");
      }
      float* array = read_constant_param(tensor);
      return create_scalar_tensor_helper<SUPPORTED_TYPES>(tensor, array);
    }

    return read_tensor_helper<SUPPORTED_TYPES>(tensor);
  }

  template <typename T>
  bool write_tensor_with_type(Tensor tensor, std::any data) {
    if (tensor.dtype != DataTypes::TypeName<T>::name()) {
      return false;
    }

    const uint64_t address = get_address(tensor);
    const int partition = get_partition(tensor);
    const int size = get_size(tensor);
    T* casted = std::any_cast<std::shared_ptr<T[]>&>(data).get();

    // A pitched window scatters rows of `run` elements `pitch` apart; the
    // bytes between the runs belong to the allocation, so read the span
    // back first and modify only the window.
    if (tensor.window_pitch > 0) {
      const int64_t run = tensor.shape.back();
      const int64_t pitch = tensor.window_pitch;
      const int64_t span = (size / run - 1) * pitch + run;
      const int64_t span_bytes = (span * T::width + 7) / 8;
      char* buffer = new char[span_bytes];
      read_bytes_from_memory(address, partition, span_bytes, buffer);

      for (int i = 0; i < size; i++) {
        const int64_t u = (i / run) * pitch + i % run;
        const int64_t start = u * T::width / 8;
        const int64_t end = ((u + 1) * T::width - 1) / 8;
        const int offset = (u * T::width) % 8;

        ac_int<(T::width / 8 + 2) * 8> bits;
        for (int64_t j = start; j <= end; j++) {
          bits.set_slc(static_cast<int>(j - start) * 8,
                       static_cast<ac_int<8, false>>(buffer[j]));
        }
        bits.set_slc(
            offset, static_cast<ac_int<T::width, false>>(casted[i].bits_rep()));
        for (int64_t j = start; j <= end; j++) {
          buffer[j] = static_cast<char>(
              bits.template slc<8>(static_cast<int>(j - start) * 8));
        }
      }

      write_bytes_to_memory(address, partition, span_bytes, buffer);
      delete[] buffer;
      return true;
    }

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
  void write_tensor_helper(const Tensor& tensor, std::any data) {
    bool matched = (write_tensor_with_type<Ts>(tensor, data) || ...);
    if (!matched) {
      throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype);
    }
  }

  void write_tensor(const Tensor& tensor, const std::any data) {
    write_tensor_helper<SUPPORTED_TYPES>(tensor, data);
  }

  template <typename T>
  bool read_value_with_type(const int partition, const int address,
                            const int index, std::string dtype, float& value) {
    if (dtype != DataTypes::TypeName<T>::name()) {
      return false;
    }

    int start = index * T::width / 8;
    int end = (index + 1) * T::width / 8;
    int offset = (index * T::width) % 8;
    int num_bytes = (end - start + 1) * 8;

    int bits_remaining = num_bytes * 8 - T::width - offset;
    num_bytes = num_bytes - bits_remaining / 8;

    char buffer[num_bytes];
    read_bytes_from_memory(address + start, partition, num_bytes, buffer);

    constexpr int buf_width = (T::width / 8 + 2) * 8;
    ac_int<buf_width, false> bits;
    for (int i = 0; i < num_bytes; i++) {
      bits.set_slc(i * 8, static_cast<ac_int<8, false>>(buffer[i]));
    }

    T typed;
    typed.set_bits(bits >> offset);

    value = static_cast<float>(typed);

    return true;
  }

  template <typename... Ts>
  void read_value_helper(const int partition, const int address,
                         const int index, std::string dtype, float& value) {
    bool matched =
        (read_value_with_type<Ts>(partition, address, index, dtype, value) ||
         ...);
    if (!matched) {
      throw std::runtime_error("Dataloader: Unsupported tensor dtype: " +
                               dtype);
    }
  }

  float read_value(const int partition, const int address, const int index,
                   std::string dtype) {
    float value;
    read_value_helper<SUPPORTED_TYPES>(partition, address, index, dtype, value);
    return value;
  }

  template <typename T>
  bool write_value_with_type(const int partition, const int address,
                             const int index, std::string dtype, T value) {
    if (dtype != DataTypes::TypeName<T>::name()) {
      return false;
    }

    int start = index * T::width / 8;
    int end = (index + 1) * T::width / 8;
    int num_bytes = (end - start + 1) * 8;
    int offset = (index * T::width) % 8;

    int bits_remaining = num_bytes * 8 - T::width - offset;
    num_bytes = num_bytes - bits_remaining / 8;

    constexpr int buf_width = (T::width / 8 + 2) * 8;
    ac_int<buf_width, false> bits = ((ac_int<buf_width, false>)value.bits_rep())
                                    << offset;
    ac_int<buf_width, false> masks =
        ((ac_int<buf_width, false>(1) << T::width) - 1) << offset;

    char buffer[num_bytes];

    // for non-byte aligned data, we need to read the memory first
    read_bytes_from_memory(address + start, partition, num_bytes, buffer);

    for (int i = 0; i < num_bytes; i++) {
      char mask = masks.template slc<8>(i * 8);
      char new_data = bits.template slc<8>(i * 8) & mask;
      char orig_data = buffer[i] & ~mask;
      buffer[i] = new_data | orig_data;
    }

    write_bytes_to_memory(address + start, partition, num_bytes, buffer);

    return true;
  }

  template <typename... Ts>
  void write_value_helper(const int partition, const int address,
                          const int index, std::string dtype, float value) {
    bool matched =
        (write_value_with_type<Ts>(partition, address, index, dtype, value) ||
         ...);
    if (!matched) {
      throw std::runtime_error("Dataloader: Unsupported tensor dtype: " +
                               dtype);
    }
  }

  void write_value(const int partition, const int address, const int index,
                   std::string dtype, float value) {
    write_value_helper<SUPPORTED_TYPES>(partition, address, index, dtype,
                                        value);
  }

  virtual void write_bytes_to_memory(const long long address,
                                     const int partition, const int num_bytes,
                                     const char* bytes) = 0;

  virtual void read_bytes_from_memory(const long long address,
                                      const int partition, const int num_bytes,
                                      char* bytes) = 0;
};
