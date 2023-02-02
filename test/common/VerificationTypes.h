#pragma once
#include <iostream>
#include <string>

// By default we have 2MB of SRAM per MINOTAUR SoC
// organized as 8x 256KB Banks with 2x 128KB Macros each
#ifndef NUM_SRAM_BANKS
#define NUM_SRAM_BANKS 8
#endif
#define SRAM_MEMORY_SIZE (NUM_SRAM_BANKS * 256 * 1024)

// By default we have 12MB of RRAM per MINOTAUR SoC
// organized as 12x 1MB Banks with 4x 256KB Macros each
// 3x Superbanks and 4x Subbanks per Superbank
#ifndef NUM_RRAM_BANKS
#define NUM_RRAM_BANKS 12
#endif
#define RRAM_MEMORY_SIZE (NUM_RRAM_BANKS * 1024 * 1024)

// Size of the reference memory used for verification in MB
// Should be at least as large as the SRAM memory
#define REFERENCE_MEMORY_SIZE (1024 * 1024 * 8)

// Bandwidth-mode for a RRAM memory bank
// QUAD -> 1 clock cycle acces using 4 banks
// DOUBLE -> 2 clock cycle acces using 2 banks
// SINGLE -> 4 clock cycle acces using 1 bank
enum BandwidthMode { QUAD, DOUBLE, SINGLE };

// Power-state for a memory bank
// OFF -> whole macro if off (SRAM loses/RRAM retains state)
// ON -> whole macro is on (both retain)
// RETENTION -> control-logic is off, cells are still powered (both retain)
enum PowerState { OFF, ON, RETENTION };

struct Files {
  std::string inputs_file;
  std::string weights_file;
  std::string bias_file;
  std::string outputs_file;
  std::string residual_file;
  std::string weight_grad_file;
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
  int WEIGHT_RESIDUAL_OFFSET;
  float learningRate;

  bool ACC_T_INPUT;
  bool ACC_T_WEIGHT;
  bool ACC_T_OUTPUT;
  bool ACC_T_RESIDUAL;

  // int inputExpBias;
  // int weightExpBias;
  int outputExpBias;
  // int residualExpBias;

  bool WEIGHT_UPDATE;
  bool ERROR_FEEDBACK;

  // Depthwise convolution
  bool depthwise;

  // Power state of memory banks
  PowerState sram_banks[NUM_SRAM_BANKS];
  PowerState rram_banks[NUM_RRAM_BANKS];

  // Bandwidth-aware (RRAM only)
  BandwidthMode bandwidth_mode;
  int bank_offsets[NUM_RRAM_BANKS];
};

struct MemoryOffsets {
  int INPUT_OFFSET;
  int WEIGHT_OFFSET;
  int OUTPUT_OFFSET;
  int BIAS_OFFSET;
  int RESIDUAL_OFFSET;
};

const int ATTENTION_HEAD_SIZE = 128 * 32;
const int INTRA_BOTTLENECK_SIZE = 128 * 128;
const int INTRA_BOTTLENECK_BIAS_SIZE = 128 * 2;
const int INTERMEDIATE_SIZE = 128 * 512;
const int INTERMEDIATE_BIAS_SIZE = 512 * 2;

const int ENCODER_ACTIVATION_SIZE =
    4 * INTERMEDIATE_SIZE + 18 * INTRA_BOTTLENECK_SIZE;
const int ENCODER_WEIGHT_SIZE =
    8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
    3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE;

// SRAM Memory Offsets
const int STACK_SIZE = 0.25 * 1024 * 1024;
const int ACTIVATION_OFFSET = STACK_SIZE + 16;
const int GRADIENT_OFFSET =
    ACTIVATION_OFFSET + 24 * ENCODER_ACTIVATION_SIZE + INTERMEDIATE_SIZE + 16;
const int ERROR_OFFSET =
    GRADIENT_OFFSET + 24 * ENCODER_WEIGHT_SIZE + 512 * 16 + 2 * 16;
const int WEIGHT_OFFSET = 128 * 2;

struct Workload {
  std::string name;
  SimplifiedParams params;
  Files files;
  MemoryMap memoryMap;
  bool loadWeight = true;
};
