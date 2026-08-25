#pragma once

#include <acc_user.h>
#include <svdpi.h>
#include <vcs_acc_user.h>
#include <vcsuser.h>
#include <vpi_user.h>

#include <string>
#include <vector>

#include "test/common/ArrayMemory.h"

// The accelerator simulator's memory for SoC RTL simulation: the DRAM and
// reference partitions are ordinary host arrays (inherited), while the
// scratchpad partition is the DUT's physical SRAM macros, reached byte-wise
// through the VPI backdoor. Writing a tile "to memory" therefore deposits it
// straight into the simulated hardware, and reading a result reads the
// hardware back -- which is what makes gold-vs-accelerator grading honest.
class SoCMemory : public ArrayMemory {
 public:
  explicit SoCMemory(const std::vector<uint64_t>& sizes);

  // The DUT writes the scratchpad macros directly; host-side write tracking
  // cannot observe that, so the VPI-backed partition reports as covered and
  // the value comparison against the SRAM readback is the real grade.
  bool was_written(int partition, uint64_t address) const override;
  bool any_written(int partition, uint64_t address,
                   uint64_t num_bytes) const override;

  // Diagnostic: every testbench write to the SRAM partition is mirrored into
  // the inherited host array. Reading the SRAM back at end-of-sim and diffing
  // against that shadow reveals any region the DUT overwrote ("stomped")
  // during the run. Returns the number of differing bytes.
  uint64_t verify_shadow();

 protected:
  void write_bytes_to_memory(const long long address, const int partition,
                             const int num_bytes, const char* bytes) override;

  void read_bytes_from_memory(const long long address, const int partition,
                              const int num_bytes, char* bytes) override;

 private:
  int cache_size;
  int num_banks;
  int num_subbanks;
  int bank_width;
  int macro_width;
  int macro_depth;
  std::string memory_suffix;
  // Byte intervals of the SRAM partition the testbench wrote (unmerged).
  std::vector<std::pair<uint64_t, uint64_t>> shadow_ranges_;

  vpiHandle get_handle(const uint64_t address);
  uint8_t read_byte(uint64_t addr);
  void write_byte(uint64_t addr, uint8_t byte);

  static std::string get_env(const char* name, std::string default_value) {
    const char* env = std::getenv(name);
    return env ? std::string(env) : default_value;
  }

  static int get_env_int(const char* name, int default_value) {
    const char* env = std::getenv(name);
    return env ? std::stoi(env) : default_value;
  }
};
