#pragma once

#include <string>

struct Files {
  std::string inputs_file;
  std::string weights_file;
  std::string bias_file;
  std::string outputs_file;
  std::string residual_file;
};

enum MemorySource { SRAM, RRAM };

struct MemoryMap {
  MemorySource inputs;
  MemorySource weights;
  MemorySource bias;
  MemorySource residual;
  MemorySource outputs;
};
