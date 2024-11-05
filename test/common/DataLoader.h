#pragma once
#include <fstream>
#include <iostream>
#include <random>

#define NO_SYSC

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/MemoryInterface.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

class DataLoader {
 public:
  DataLoader(MemoryInterface*, bool);

  void load_inputs(const codegen::AcceleratorParam param, std::string data_dir,
                   bool random_data = false);
  void load_weights(const codegen::AcceleratorParam param, std::string data_dir,
                    bool random_data = false);
  void load_outputs(const codegen::AcceleratorParam param,
                    std::string data_dir);
  void load_tensor(const codegen::Tensor& tensor, std::string data_dir,
                   bool transpose = false, bool replication = false);

  bool is_double_precision(const codegen::Tensor& tensor);
  float* read_tensor_from_file(const std::string& filename, int size);

 private:
  MemoryInterface* memory_interface;
  // special addressing is sometimes needed for DUT memory (ex. replication)
  bool is_dut;
};
