#pragma once

#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

#define NO_SYSC
// clang-format off
#include "src/PositTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/SimpleMemoryModel.h"
#include "test/common/UniversalPosit.h"

void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
            MemoryMap memoryMap);

class Simulation {
 public:
  Simulation(int argc, char* argv[]);

  void loadMemory();
  void run();
  int checkOutput();

 private:
  std::vector<Workload> workloads;
  std::vector<std::string> sims;
  std::string out_dir;
  std::string model;
  float tolerance;

  SimpleMemoryModel<INPUT_DATATYPE>* acceleratorMemory;
  SimpleMemoryModel<INPUT_DATATYPE>* positMemory;
  SimpleMemoryModel<float>* floatMemory;
  SimpleMemoryModel<UniversalPosit>* universalPositMemory;

  // Return environment variable
  std::string get_env_var(std::string const& name);
  void print_help();
};
