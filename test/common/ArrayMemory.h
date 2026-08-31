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
  // Virtual so the SoC testbench's memory can report its VPI-backed
  // scratchpad partition as always-covered: the DUT's own writes to the SRAM
  // macros are invisible to host-side tracking.
  virtual bool was_written(int partition, uint64_t address) const;
  virtual bool any_written(int partition, uint64_t address,
                           uint64_t num_bytes) const;

 protected:
  // Protected, not private: the SoC testbench's memory subclasses this and
  // routes the scratchpad partition through the simulator's VPI backdoor
  // while delegating the host-side partitions here.
  void write_bytes_to_memory(const long long address, const int partition,
                             const int num_bytes, const char* bytes) override;

  void read_bytes_from_memory(const long long address, const int partition,
                              const int num_bytes, char* bytes) override;

 private:
  std::vector<std::map<uint64_t, uint64_t>> written;
  std::vector<uint64_t> sizes;

  size_t partition_index(int partition) const;
  size_t checked_range(int partition, long long address, int num_bytes) const;
};
