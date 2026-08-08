#pragma once

#define NO_SYSC

#include <ac_int.h>

#include <any>
#include <map>

#include "test/common/MemoryInterface.h"
#include "test/common/Tensor.h"

class ArrayMemory : public MemoryInterface {
 public:
  ArrayMemory(std::vector<uint64_t>);
  ~ArrayMemory();

  std::vector<char*> memories;

  char* get_memory(int partition);

  // Records which bytes the run wrote as merged half-open ranges. A dense mask
  // is too expensive because the emulated DRAM can be hundreds of megabytes
  // and regressions run many gold models concurrently.
  void track_writes();
  bool tracking() const { return !written.empty(); }
  bool was_written(int partition, uint64_t address) const;
  bool any_written(int partition, uint64_t address, uint64_t num_bytes) const;

 private:
  std::vector<std::map<uint64_t, uint64_t>> written;
  std::vector<uint64_t> sizes;

  size_t partition_index(int partition) const;
  size_t checked_range(int partition, long long address, int num_bytes) const;

  void write_bytes_to_memory(const long long address, const int partition,
                             const int num_bytes, const char* bytes) override;

  void read_bytes_from_memory(const long long address, const int partition,
                              const int num_bytes, char* bytes) override;
};
