#pragma once
#include <iostream>
#include <string>

struct Files {
  std::string inputs_file;
  std::string weights_file;
  std::string bias_file;
  std::string outputs_file;
  std::string residual_file;
};

enum MemorySource { SRAM, RRAM };

inline std::ostream& operator<<(std::ostream& os, MemorySource& memory) {
  switch (memory) {
    case SRAM:
      os << "SRAM";
      break;
    case RRAM:
      os << "RRAM";
      break;
  }
  return os;
}

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

  bool WEIGHT;
  bool ATTENTION_SCALING;
  bool INPUT_TRANSPOSE;
};

struct MemoryOffsets {
  int INPUT_OFFSET;
  int WEIGHT_OFFSET;
  int OUTPUT_OFFSET;
  int BIAS_OFFSET;
  int RESIDUAL_OFFSET;
};

const int HEAD_SIZE = 128 * 32 + 32;
const int HIDDEN_SIZE = 128 * 128 + 128;
const int INTERMEDIATE_SIZE = 4 * HIDDEN_SIZE;
const int PER_LAYER_HIDDEN_SIZE = 128 * 128 + 1;
const int PER_LAYER_INTERMEDIATE_SIZE = 128 * 512 + 1;
const int HIDDEN_BIAS_SIZE = 128 + 1;
const int INTERMEDIATE_BIAS_SIZE = 512 + 1;
