#pragma once
#include <iostream>
#include <map>
#include <string>

#include "test/compiler/proto/param.pb.h"

// By default we have 2MB of SRAM per MINOTAUR SoC
// organized as 8x 256KB Banks with 2x 128KB Macros each
// FIXME: use a larger memory partition for testing
#ifndef NUM_SRAM_BANKS
#define NUM_SRAM_BANKS 32
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

// The accelerator has several interfaces for accessing memory, but
// these interfaces aren't always standard. For example, for some layers,
// an interface may access the residual, but for other layers, access weights.
// This data structure maps the accelerator interface to the MemorySource.
typedef std::map<std::string, MemorySource> AcceleratorMemoryMap;

inline std::ostream& operator<<(std::ostream& os, MemoryMap& memoryMap) {
  os << "Inputs: " << memoryMap.inputs << std::endl;
  os << "Weights: " << memoryMap.weights << std::endl;
  os << "Bias: " << memoryMap.bias << std::endl;
  os << "Residual: " << memoryMap.residual << std::endl;
  os << "Outputs: " << memoryMap.outputs << std::endl;
  return os;
}

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

  // Layer name
  std::string name;

  bool ATTENTION_MASK;
  bool MERGE_LORA_WEIGHT;
  bool QUANTIZE_TO_P8;
  bool MAX_REDUCE;
  bool ELWISE_ADD;
  bool NOP;
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
// Memory looks like:
// [STACK]
// [LABEL]
// [ATTENTION_MASK]
// [ACTIVATION]
const int STACK_SIZE = 0.5 * 1024 * 1024;
const int ATTENTION_MASK_OFFSET = STACK_SIZE + 16;
const int ACTIVATION_OFFSET = ATTENTION_MASK_OFFSET + 128 * 2;
const int GRADIENT_OFFSET =
    ACTIVATION_OFFSET + 24 * ENCODER_ACTIVATION_SIZE + INTERMEDIATE_SIZE + 16;
const int ERROR_OFFSET =
    GRADIENT_OFFSET + 24 * ENCODER_WEIGHT_SIZE + 512 * 16 + 2 * 16;
const int WEIGHT_OFFSET = 0;

struct Workload {
  std::string name;
  SimplifiedParams params;
  Files files;
  MemoryMap memoryMap;
  AcceleratorMemoryMap acceleratorMemoryMap;
  bool loadWeight = true;
};

struct Tiling {
  int loops[2][6];
  int x_loop_index[2];
  int y_loop_index[2];
  int reduction_loop_index[2];
  int weight_loop_index[2];
  int fx_index;
  int fy_index;
  int weight_reuse_index[2];
  int stride;
  bool replication;
};

inline std::ostream& operator<<(std::ostream& os, const Tiling& tiling) {
  os << "Loops: " << std::endl;
  for (int i = 0; i < 2; i++) {
    os << "  " << i << ": ";
    for (int j = 0; j < 6; j++) {
      os << tiling.loops[i][j] << " ";
    }
    os << std::endl;
  }
  os << "X Loop Index: " << tiling.x_loop_index[0] << " "
     << tiling.x_loop_index[1] << std::endl;
  os << "Y Loop Index: " << tiling.y_loop_index[0] << " "
     << tiling.y_loop_index[1] << std::endl;
  os << "Reduction Loop Index: " << tiling.reduction_loop_index[0] << " "
     << tiling.reduction_loop_index[1] << std::endl;
  os << "Weight Loop Index: " << tiling.weight_loop_index[0] << " "
     << tiling.weight_loop_index[1] << std::endl;
  os << "FX Index: " << tiling.fx_index << std::endl;
  os << "FY Index: " << tiling.fy_index << std::endl;
  os << "Weight Reuse Index: " << tiling.weight_reuse_index[0] << " "
     << tiling.weight_reuse_index[1] << std::endl;
  os << "Stride: " << tiling.stride << std::endl;
  os << "Replication: " << tiling.replication << std::endl;
  return os;
}

inline Tiling get_conv_tiling(codegen::MatrixParam param) {
  // input shape = (B, C, H, W)
  // weight shape = (OC, IC, KH, KW)
  int IC = param.input().shape(1);
  int IH = param.input().shape(2);
  int IW = param.input().shape(3);
  int OC = param.weight().shape(0);
  int KH = param.weight().shape(2);
  int KW = param.weight().shape(3);
  int STRIDE = param.stride(0);

  // TODO: loop tilings are hardcoded for now, ideally we would integrate with
  // a DSE tool like Interstellar

  if (IH == 224 && IW == 224 && IC == 3 && KH == 7 && KW == 7 &&
      OC == 64) {  // conv1
    return {
        .loops = {{7, 7, 2, 1, 1, 1}, {1, 2, 7, 2, 16, 16}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
        .replication = true,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 64) {  // layer1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_0_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer1_x_conv3
    return {
        .loops = {{2, 2, 16, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_x_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_0_downsample (resnet18)
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {4, 2, 1, 1, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_0_downsample (resnet50)
    return {
        .loops = {{2, 2, 16, 1, 1, 1}, {16, 2, 1, 1, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128 && STRIDE == 2) {  // layer2_0_conv1
    return {
        .loops = {{4, 4, 4, 1, 1, 1}, {4, 3, 3, 2, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 128 && STRIDE == 1) {  // layer2_0_conv1
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_x_conv1
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {32, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2_x_conv2
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {8, 1, 3, 3, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_x_conv3
    return {
        .loops = {{1, 1, 32, 1, 1, 1}, {8, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_downsample (resnet18)
    return {
        .loops = {{2, 2, 2, 1, 1, 1}, {8, 1, 1, 8, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 1024 && STRIDE == 2) {  // layer3_0_downsample (resnet50)
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {32, 1, 1, 8, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 256 && STRIDE == 2) {  // layer3_0_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_0_conv2
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {32, 1, 1, 4, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_x_conv1
    return {
        .loops = {{1, 1, 4, 1, 1, 1}, {64, 1, 1, 4, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_x_conv2
    return {
        .loops = {{1, 1, 4, 1, 1, 1}, {16, 3, 3, 4, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 1024) {  // layer3_x_conv3
    return {
        .loops = {{2, 2, 16, 1, 1, 1}, {16, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer3
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 2) {  // layer4_0_downsample (resnet18)
    return {
        .loops = {{1, 1, 2, 1, 1, 1}, {16, 1, 1, 16, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 2048 && STRIDE == 2) {  // layer4_0_downsample (resnet50)
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {64, 1, 1, 16, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 512 && STRIDE == 2) {  // layer4_0_conv1
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 1) {  // layer4_0_conv1
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {64, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 7 && IW == 7 && IC == 2048 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 1) {  // layer4_x_conv1
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {128, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512 && STRIDE == 2) {  // layer4_x_conv2
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 2048) {  // layer4_x_conv3
    return {
        .loops = {{1, 1, 32, 1, 1, 1}, {32, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512) {  // layer4
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else {
    std::cout << "Tiling has not been hardcoded for this layer yet"
              << std::endl;
    std::cout << "IH: " << IH << " IW: " << IW << " IC: " << IC << " KH: " << KH
              << " KW: " << KW << " OC: " << OC << " STRIDE: " << STRIDE
              << std::endl;
    exit(1);
  }
}

inline Tiling get_linear_tiling(codegen::MatrixParam param) {
  auto repeated_field = param.input().shape();
  std::vector<int> input_shape(repeated_field.begin(), repeated_field.end());
  repeated_field = param.weight().shape();
  std::vector<int> weight_shape(repeated_field.begin(), repeated_field.end());

  if (input_shape[1] == 128 && input_shape[2] == 512 &&
      weight_shape[0] == 128) {
    return {
        .loops = {{2, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 64}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[1] == 128 && input_shape[2] == 128 &&
             weight_shape[0] == 128) {
    return {
        .loops = {{2, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 64}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[1] == 128 && input_shape[2] == 128 &&
             weight_shape[0] == 512) {
    return {
        .loops = {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else {
    std::cerr << "Unsupported input dimensions" << std::endl;
    std::cerr << "Input shape: ";
    for (int dim : input_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    std::cerr << "Weight shape: ";
    for (int dim : weight_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    exit(1);
  }
}

inline Tiling get_matmul_tiling(codegen::MatrixParam param) {
  auto repeated_field = param.input().shape();
  std::vector<int> input_shape(repeated_field.begin(), repeated_field.end());
  repeated_field = param.weight().shape();
  std::vector<int> weight_shape(repeated_field.begin(), repeated_field.end());

  if (input_shape[0] == 128 && input_shape[1] == 32 && weight_shape[1] == 128) {
    return {
        .loops = {{4, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 32}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[0] == 128 && input_shape[1] == 128 &&
             weight_shape[1] == 32) {
    return {
        .loops = {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else {
    std::cerr << "Unsupported input dimensions" << std::endl;
    std::cerr << "Input shape: ";
    for (int dim : input_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    std::cerr << "Weight shape: ";
    for (int dim : weight_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    exit(1);
  }
}
