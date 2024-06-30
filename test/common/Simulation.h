#pragma once

#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

#define NO_SYSC
// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/PytorchMemoryModelImpl.h"
#include "test/common/PyTorchModel.h"
#include "test/common/SimpleMemoryModel.h"
#include "test/common/UniversalPosit.h"
#include "test/compiler/proto/param.pb.h"

void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE *sramMemory, INPUT_DATATYPE *rramMemory,
            std::vector<MemoryMap> memoryMap);

void run_pytorch_op(std::vector<codegen::AcceleratorParam> params,
                    INPUT_DATATYPE *sramMemory, INPUT_DATATYPE *rramMemory);

class Simulation {
 public:
  Simulation();

  void loadMemory();
  void run();
  int checkOutput();
  void print_help();

 protected:
  std::vector<Workload> workloads;
  std::map<std::string, MemoryModel *> memory_models;
  std::vector<std::string> sims;
  std::string out_dir;
  std::string modelName;
  std::string tests;
  std::string task;
  float tolerance = 0.1;

  std::vector<codegen::AcceleratorParam> params;
  std::map<std::string, PyTorchMemoryModel *> memories;

 private:
  std::string get_env_var(std::string const &name);

  template <typename T>
  void split_string(const std::string &in_string, char delim, T result);
};
