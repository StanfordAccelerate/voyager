#pragma once
#include <iostream>
#include <string>

struct Files {
  std::string inputs_file;
  std::string weights_file;
  std::string bias_file;
  std::string outputs_file;
  std::string residual_file;
  std::string weight_grad_file;
  std::string bias_grad_file;
};

enum MemorySource { SRAM, RRAM };

inline std::ostream& operator<<(std::ostream& os, MemorySource& memory) {
  os << (memory == SRAM ? "SRAM" : "RRAM");
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
  bool WEIGHT_TRANSPOSE;

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

  bool WEIGHT;

  bool STORE_IN_ACC;
  bool ACC_FROM_ACC;

  // special vector ops
  bool SOFTMAX;
  bool ATTENTION_MASK;
  bool ATTENTION_SCALING;
  bool FC;
  bool NO_NORM;

  bool SOFTMAX_GRAD;
  bool FC_GRAD;
  bool NO_NORM_GRAD;
  bool RELU_GRAD;
  bool BIAS_GRAD;
  bool CROSS_ENTROPY_GRAD;
  bool MSE_GRAD;
  bool BCE_WITH_LOGITS_GRAD;

  // permutation ops
  bool INPUT_TRANSPOSE;
  bool CONCAT_INPUT;
  bool CONCAT_WEIGHT;
  bool SPLIT_OUTPUT;

  bool GRAD_CLIPPING;
  bool GRAD_CLIPPING_UNIT_TEST;

  bool WEIGHT_SPLITTING;
  int WEIGHT_GRADIENT_OFFSET;
  int BIAS_GRADIENT_OFFSET;
  float learningRate;

  bool ACC_T_INPUT;
  bool ACC_T_WEIGHT;
  bool ACC_T_OUTPUT;

  int inputExpBias;
  int weightExpBias;
  int outputExpBias;
  int residualExpBias;
};

struct MemoryOffsets {
  int INPUT_OFFSET;
  int WEIGHT_OFFSET;
  int OUTPUT_OFFSET;
  int BIAS_OFFSET;
  int RESIDUAL_OFFSET;
};

const int HEAD_SIZE = 128 * 32;
const int HIDDEN_SIZE = 128 * 128;
const int INTERMEDIATE_SIZE = 4 * HIDDEN_SIZE;

const int WEIGHT_HIDDEN_SIZE = 128 * 128;        // 16384
const int WEIGHT_INTERMEDIATE_SIZE = 128 * 512;  // 65536
const int BIAS_HIDDEN_SIZE = 128 * 2;
const int BIAS_INTERMEDIATE_SIZE = 512 * 2;

// const int ENCODER_ACTIVATION_SIZE = 4 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE;
// const int ENCODER_WEIGHT_SIZE = 8 * WEIGHT_INTERMEDIATE_SIZE +
//                                 5 * BIAS_INTERMEDIATE_SIZE +
//                                 3 * WEIGHT_HIDDEN_SIZE + 18 *
//                                 BIAS_HIDDEN_SIZE;
const int ENCODER_ACTIVATION_SIZE = 4 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE;
const int ENCODER_WEIGHT_SIZE =
    6 * WEIGHT_INTERMEDIATE_SIZE + 4 * WEIGHT_HIDDEN_SIZE +
    5 * BIAS_INTERMEDIATE_SIZE + 15 * BIAS_HIDDEN_SIZE;

// SRAM Memory Offsets
const int STACK_SIZE = 1024 * 1024;
const int ACTIVATION_OFFSET = STACK_SIZE + 128;
const int GRADIENT_OFFSET =
    ACTIVATION_OFFSET + 24 * ENCODER_ACTIVATION_SIZE + INTERMEDIATE_SIZE + 16;
const int ERROR_OFFSET = GRADIENT_OFFSET + 24 * ENCODER_WEIGHT_SIZE +
                         16 * BIAS_INTERMEDIATE_SIZE + 32;
