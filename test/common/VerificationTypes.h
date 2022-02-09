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

struct SimplifiedParams {
  int INPUT_OFFSET;
  int WEIGHT_OFFSET;
  int OUTPUT_OFFSET;
  bool TRANSPOSE;

  // tiling
  int loops[2][6];
  int inputXLoopIndex[2];
  int inputYLoopIndex[2];
  int reductionLoopIndex[2];
  int weightLoopIndex[2];
  int fxIndex;
  int fyIndex;
  int weightReuseIndex[2];
  int STRIDE;
  bool REPLICATION;

  bool RELU;
  bool BIAS;
  int BIAS_OFFSET;
  bool RESIDUAL;
  int RESIDUAL_OFFSET;

  // pooling
  bool MAXPOOL;
  bool AVGPOOL;

  // special vector ops
  bool SOFTMAX;
  bool FC;
  bool NO_NORM;
};
