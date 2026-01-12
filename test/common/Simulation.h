#pragma once
#define NO_SYSC

#include <string>
#include <vector>

#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/common/Network.h"
#include "test/common/operations/Common.h"
#include "test/compiler/proto/param.pb.h"

void run_accelerator(std::vector<Operation> params, DataLoader* memory);

class Simulation {
 public:
  Simulation();
  ~Simulation();

  void load_data();
  void run();
  int check_outputs();
  void print_help();
  void print_ideal_runtime(const Operation op);

 protected:
  std::string model;
  std::string out_dir;
  float tolerance = 0.1;
  std::vector<std::string> tests;
  std::vector<std::string> sims;
  Network* network;
  std::vector<Operation> operations;
  std::map<std::string, MemoryInterface*> memories;
  std::map<std::string, DataLoader*> dataloaders;

 private:
  std::string get_env_var(std::string const& name);

  template <typename T>
  void split_string(const std::string& in_string, char delim, T result);
};
