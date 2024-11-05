#pragma once

#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

#define NO_SYSC
// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/compiler/proto/param.pb.h"

void run_accelerator(std::vector<codegen::AcceleratorParam> params,
                     char *memory);

class Simulation {
 public:
  Simulation();
  ~Simulation();

  void load_data();
  void run();
  int check_outputs();
  void print_help();
  int get_ideal_runtime(const codegen::AcceleratorParam &param);

 protected:
  std::vector<std::string> sims;
  std::string model;
  std::string tests;
  std::string out_dir;
  float tolerance = 0.1;

  std::vector<codegen::AcceleratorParam> params;
  std::map<std::string, MemoryInterface *> memories;
  std::map<std::string, DataLoader *> dataLoaders;

 private:
  std::string get_env_var(std::string const &name);

  template <typename T>
  void split_string(const std::string &in_string, char delim, T result);
};
