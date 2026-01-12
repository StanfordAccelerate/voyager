#pragma once

#define NO_SYSC

#include <ac_int.h>

#include <any>

#include "test/common/MemoryInterface.h"
#include "test/compiler/proto/param.pb.h"

class ArrayMemory : public MemoryInterface {
 public:
  ArrayMemory(std::vector<uint64_t>);
  ~ArrayMemory();

  std::vector<char*> memories;

  char* get_memory(int partition);

  void write_bytes_to_memory(const long long address, const int partition,
                             const int num_bytes, const char* bytes) override;

  void read_bytes_from_memory(const long long address, const int partition,
                              const int num_bytes, char* bytes) override;
};
