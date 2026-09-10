#pragma once

#include <string>
#include <vector>

#include "test/common/ArrayMemory.h"
#include "test/common/Simulation.h"

// Grades a full-JTAG run from the scratchpad dump GDB read back after the
// firmware finished: the dump becomes the accelerator simulator's scratchpad
// and its scratchpad-resident results are compared against gold's, which
// ran the same bounded program in place. The DRAM side is not graded: the
// chip has no DRAM, and the store-backs that would fill it are exercised
// by the testbench-driven modes.
class JtagSimChecker : public Simulation {
 public:
  explicit JtagSimChecker(const std::string& dump_path);

  // Returns the error count check_outputs reports.
  int grade();

 protected:
  ArrayMemory* make_memory(const std::string& sim,
                           const std::vector<uint64_t>& sizes) override;

 private:
  std::string dump_path_;
};
