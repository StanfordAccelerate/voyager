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
            INPUT_DATATYPE *sramMemory, INPUT_DATATYPE *rramMemory,
            MemoryMap memoryMap);

class Simulation {
 public:
  Simulation();

  void loadMemory();
  void run();
  int checkOutput();
  void print_help();

 protected:
  std::vector<Workload> workloads;
  std::vector<std::string> sims;
  std::string out_dir;
  std::string model;
  float tolerance = 0.1;

  SimpleMemoryModel<INPUT_DATATYPE> *acceleratorMemory;
  SimpleMemoryModel<INPUT_DATATYPE> *positMemory;
  SimpleMemoryModel<float> *floatMemory;
  SimpleMemoryModel<UniversalPosit> *universalPositMemory;

 private:
  std::string get_env_var(std::string const &name);

  template <typename T>
  void split_string(const std::string &in_string, char delim, T result);
};
