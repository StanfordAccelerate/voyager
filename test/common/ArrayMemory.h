#pragma once

#define NO_SYSC

#include <ac_int.h>

#include <any>

#include "test/compiler/proto/param.pb.h"

class ArrayMemory {
 public:
  ArrayMemory(std::vector<long long>);
  ~ArrayMemory();

  std::vector<char*> memories;

  char* get_memory(const int partition);

  template <typename T>
  void read_tensor_from_memory(const long long address, const int partition,
                               const int size, T* tensor);
  template <typename T>
  void write_data_to_memory(const uint64_t address, const int partition,
                            const int index, T value);
  template <typename T>
  void write_tensor_to_memory(const uint64_t address, const int partition,
                              const int size, T* tensor);

  std::any read_tensor(const codegen::Tensor& tensor);
  void write_tensor(const codegen::Tensor& tensor, const std::any data);

  std::vector<std::any> get_args(const codegen::Operator& param);
  std::any get_output(const codegen::Operator& param);
  std::any get_reference_output(const codegen::Operator& param);
};

template <typename T>
void ArrayMemory::read_tensor_from_memory(const long long address,
                                          const int partition, const int size,
                                          T* tensor) {
  char* memory = get_memory(partition) + address;
  ac_int<(T::width / 8 + 2) * 8> bits;

  for (int i = 0; i < size; i++) {
    // Data may be unaligned and span multiple bytes. We calculate the start
    // and end byte indices and the offset within the first byte. We then read
    // the bytes into a temporary ac_int and shift it to the correct position.
    int start = i * T::width / 8;
    int end = (i + 1) * T::width / 8;
    int offset = (i * T::width) % 8;

    for (int j = start; j <= end; j++) {
      bits.set_slc((j - start) * 8, static_cast<ac_int<8>>(memory[j]));
    }

    bits = bits >> offset;
    tensor[i].set_bits(bits.template slc<T::width>(0));
  }
}

template <typename T>
void ArrayMemory::write_data_to_memory(const uint64_t address,
                                       const int partition, const int index,
                                       T value) {
  char* memory = get_memory(partition) + address;

  int start = index * T::width / 8;
  int end = (index + 1) * T::width / 8;
  int offset = (index * T::width) % 8;
  int num_bytes = (end - start + 1) * 8;

  int bits_remaining = num_bytes * 8 - T::width - offset;
  num_bytes = num_bytes - bits_remaining / 8;

  constexpr int buf_width = (T::width / 8 + 2) * 8;
  ac_int<buf_width> bytes = value.bits_rep();
  ac_int<buf_width> masks = ((1 << T::width) - 1);

  bytes = bytes << offset;
  masks = masks << offset;

  for (int i = 0; i < num_bytes; i++) {
    char mask = masks.template slc<8>(i * 8);
    char new_data = bytes.template slc<8>(i * 8) & mask;
    char orig_data = memory[start + i] & ~mask;
    memory[start + i] = new_data | orig_data;
  }
}

template <typename T>
void ArrayMemory::write_tensor_to_memory(const uint64_t address,
                                         const int partition, const int size,
                                         T* tensor) {
  for (int index = 0; index < size; index++) {
    write_data_to_memory<T>(address, partition, index, tensor[index]);
  }
}
