#pragma once

// Abstract class for interfacing with memory models.
class MemoryInterface {
 public:
  // Write a value to memory at the given address.
  virtual void write_to_memory(const int address, const float value,
                               const int partition, bool double_precision) = 0;

  virtual ~MemoryInterface() = default;
};
