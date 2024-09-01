#pragma once

#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

#define NO_SYSC
// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/MemoryModelImpl.h"
#include "test/common/UniversalPosit.h"
#include "test/compiler/proto/param.pb.h"

void run_gold_model(std::vector<codegen::AcceleratorParam> params,
                    INPUT_DATATYPE *sramMemory, INPUT_DATATYPE *rramMemory);

class Simulation {
 public:
  Simulation();

  void load_data();
  void run();
  int check_outputs();
  void print_help();

 protected:
  std::vector<std::string> sims;
  std::string model;
  std::string tests;
  std::string out_dir;
  float tolerance = 0.1;

  std::vector<codegen::AcceleratorParam> params;
  std::map<std::string, MemoryModel *> memories;

 private:
  std::string get_env_var(std::string const &name);

  template <typename T>
  void split_string(const std::string &in_string, char delim, T result);
};
