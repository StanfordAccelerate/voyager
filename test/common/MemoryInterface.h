#pragma once

#include <ac_int.h>

// Abstract class for interfacing with memory models.
class MemoryInterface {
 public:
  // Write a value to memory at the given address.
  template <typename T>
  void write_to_memory(const long long baseAddress, const int index,
                       const float value, const int partition) {
    T typedValue = static_cast<T>(value);
    ac_int<T::width> bits = typedValue.bits_rep();

    char bytes[T::width / 8];
    assert(T::width % 8 == 0);
    for (int i = 0; i < T::width / 8; i++) {
      bytes[i] = bits.template slc<8>(i * 8);
    }

    write_bytes_to_memory(baseAddress + index * T::width / 8, partition,
                          T::width / 8, bytes);
  }

  virtual void write_bytes_to_memory(const long long address,
                                     const int partition, const int size,
                                     const char* bytes) = 0;

  virtual ~MemoryInterface() = default;
};
