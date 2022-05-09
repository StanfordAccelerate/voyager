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
  bool STORE_IN_ACC;
  bool ACC_FROM_ACC;
  bool INPUT_TRANSPOSE;
  bool SPLIT_HEAD;
  bool CONCAT_HEAD;
  bool SOFTMAX_GRAD;
  bool NO_OP;
};

struct MemoryOffsets {
  int INPUT_OFFSET;
  int WEIGHT_OFFSET;
  int OUTPUT_OFFSET;
  int BIAS_OFFSET;
  int RESIDUAL_OFFSET;
};

#ifdef INPUT_SCALING
const int HEAD_SIZE = 128 * 32 + 32;
const int HIDDEN_SIZE = 128 * 128 + 128;
const int INTERMEDIATE_SIZE = 4 * HIDDEN_SIZE;
#else
const int HEAD_SIZE = 128 * 32;
const int HIDDEN_SIZE = 128 * 128;
const int INTERMEDIATE_SIZE = 4 * HIDDEN_SIZE;
#endif

#ifdef WEIGHT_SCALING
const int WEIGHT_HIDDEN_SIZE = 128 * 128 + 1;
const int WEIGHT_INTERMEDIATE_SIZE = 128 * 512 + 1;
const int BIAS_HIDDEN_SIZE = 128 + 1;
const int BIAS_INTERMEDIATE_SIZE = 512 + 1;
#else
const int WEIGHT_HIDDEN_SIZE = 128 * 128; // 16384
const int WEIGHT_INTERMEDIATE_SIZE = 128 * 512; // 65536
const int BIAS_HIDDEN_SIZE = 128;
const int BIAS_INTERMEDIATE_SIZE = 512;
#endif

const int ENCODER_SIZE = 12 * WEIGHT_INTERMEDIATE_SIZE +
                         7 * BIAS_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
                         24 * BIAS_HIDDEN_SIZE;
