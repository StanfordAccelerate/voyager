#pragma once
#define NO_SYSC

#include <any>
#include <map>
#include <string>
#include <vector>

#include "test/common/Backend.h"
#include "test/common/GraphUtils.h"
#include "test/common/MemoryInterface.h"

// Runs one accelerator operation bit-accurately, in the same datatypes the RTL
// uses. `kwargs` maps a buffer's node name to its contents, already fetched out
// of memory by the caller.
std::vector<std::any> run_gold_model(const voyager::Operation& operation,
                                     const ScalarEnv& env,
                                     std::map<std::string, std::any> kwargs);

// Fetches an operation's operands out of memory, runs the kernels, and writes
// the results back to their destinations.
//
// The harness uses this too: an operation the compiler tagged "cpu" runs on the
// control processor rather than the datapath, and still needs a kernel.
void run_host_operation(const voyager::Operation& operation,
                        const ScalarEnv& env, MemoryInterface* memory);

// The backend that executes an accelerator operation with the gold kernels
// rather than handing it to the DUT.
class GoldBackend : public Backend {
 public:
  explicit GoldBackend(MemoryInterface* memory) : memory_(memory) {}

  void execute(const voyager::Operation& op, const ScalarEnv& env) override;

 private:
  MemoryInterface* memory_;
};
