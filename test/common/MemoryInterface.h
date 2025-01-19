#pragma once

#include <ac_int.h>

// Abstract class for interfacing with memory models.
class MemoryInterface {
 public:
  // Write a value to memory at the given address.
  template <typename T>
  void write_to_memory(const long long base_address, const int index,
                       const float value, const int partition) {
    T value_t = static_cast<T>(value);

    int start = index * T::width / 8;
    int end = (index + 1) * T::width / 8;
    int offset = (index * T::width) % 8;
    int num_bytes = (end - start + 1) * 8;

    int bits_remaining = num_bytes * 8 - T::width - offset;
    num_bytes = num_bytes - bits_remaining / 8;

    char bytes[num_bytes];
    char masks[num_bytes];

    ac_int<(T::width / 8 + 2) * 8> bits = value_t.bits_rep();
    ac_int<(T::width / 8 + 2) * 8> mask = ((1 << T::width) - 1);

    bits = bits << offset;
    mask = mask << offset;

    for (int i = 0; i < num_bytes; i++) {
      bytes[i] = bits.template slc<8>(i * 8);
      masks[i] = mask.template slc<8>(i * 8);
    }

    write_bytes_to_memory(base_address + start, partition, num_bytes, bytes,
                          masks);
  }

  virtual void write_bytes_to_memory(const long long address,
                                     const int partition, const int size,
                                     const char* bytes, const char* masks) = 0;

  virtual ~MemoryInterface() = default;
};
