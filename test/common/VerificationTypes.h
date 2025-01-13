#pragma once

#define NO_SYSC

#include <iostream>
#include <map>
#include <string>

// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/compiler/proto/param.pb.h"

#ifndef NUM_SRAM_BANKS
#define NUM_SRAM_BANKS 1024
#endif
#define SRAM_MEMORY_SIZE (NUM_SRAM_BANKS * 1024LL * 1024LL)

// Size of the reference memory used for verification in MB
// Should be at least as large as the SRAM memory
#define REFERENCE_MEMORY_SIZE (1024 * 1024 * 8)

enum MemorySource { SRAM, RRAM };

inline std::ostream& operator<<(std::ostream& os, MemorySource& memory) {
  os << (memory == SRAM ? "SRAM" : "RRAM");
  return os;
}

// The accelerator has several interfaces for accessing memory, but
// these interfaces aren't always standard. For example, for some layers,
// an interface may access the residual, but for other layers, access weights.
// This data structure maps the accelerator interface to the MemorySource.
typedef std::map<std::string, MemorySource> AcceleratorMemoryMap;

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

inline Tiling get_conv2d_tiling(codegen::Operator param) {
  const auto matrix_op = param.matrix_op();

  const auto input = matrix_op.has_mx_input() ? matrix_op.mx_input().input()
                                              : matrix_op.input();
  const auto input_shape =
      input.has_reshape() ? input.reshape().output_sizes() : input.shape();

  const auto weight_shape = matrix_op.has_mx_weight()
                                ? matrix_op.mx_weight().input().shape()
                                : matrix_op.weight().shape();

  const auto output_shape = param.output().shape();

  // input shape = (B, C, H, W)
  // weight shape = (OC, IC, KH, KW)
  int IC = input_shape[1];
  int IH = input_shape[2];
  int IW = input_shape[3];
  int OC = weight_shape[0];
  int KH = weight_shape[2];
  int KW = weight_shape[3];

  // TODO: we should use OC_DIMENSION and IC_DIMENSION instead
  const int oc_dim = 16;
  const int ic_dim = 16;

  Tiling tiling;

  if (IH == 224 && IW == 224 && IC == 3 && KH == 7 && KW == 7 &&
      OC == 64) {  // conv1

    int fx;
    if (IC_DIMENSION == 4) {
      fx = 7;
    } else if (IC_DIMENSION == 8) {
      fx = 4;
    } else if (IC_DIMENSION == 16) {
      fx = 2;
    } else if (IC_DIMENSION == 32) {
      fx = 1;
    } else {
      throw std::runtime_error("replication not supported for IC_DIMENSION=" +
                               std::to_string(IC_DIMENSION));
    }

    tiling = {.loops = {{7, 7, 2, 1, 1, 1}, {1, 2, 7, fx, 16, 16}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = true};
  } else if (IH == 224 && IW == 224 && IC == 3 && KH == 16 && KW == 16 &&
             OC == 768) {  // ViT embeddings

    int fx;
    if (IC_DIMENSION == 4) {
      fx = 16;
    } else if (IC_DIMENSION == 8) {
      fx = 8;
    } else if (IC_DIMENSION == 16) {
      fx = 4;
    } else if (IC_DIMENSION == 32) {
      fx = 2;
    } else {
      throw std::runtime_error("replication not supported for IC_DIMENSION=" +
                               std::to_string(IC_DIMENSION));
    }

    tiling = {.loops = {{1, 1, 6, 1, 1, 1}, {1, 8, 16, fx, 14, 14}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = true};

  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 64) {  // layer1
    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_0_conv1
    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer1_x_conv3
    tiling = {.loops = {{2, 2, 16, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_x_conv1

    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_0_downsample (resnet18)
    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {4, 2, 1, 1, 14, 14}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_0_downsample (resnet50)
    tiling = {.loops = {{2, 2, 16, 1, 1, 1}, {16, 2, 1, 1, 14, 14}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128 && matrix_op.stride(0) == 2) {  // layer2_0_conv1

    tiling = {.loops = {{4, 4, 4, 1, 1, 1}, {4, 3, 3, 2, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2

    tiling = {.loops = {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 128 && matrix_op.stride(0) == 1) {  // layer2_0_conv1

    tiling = {.loops = {{2, 2, 8, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_x_conv1

    tiling = {.loops = {{1, 1, 8, 1, 1, 1}, {32, 1, 1, 1, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 56 && IW == 56 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2_x_conv2

    tiling = {.loops = {{2, 2, 8, 1, 1, 1}, {8, 1, 3, 3, 14, 14}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_x_conv3

    tiling = {.loops = {{1, 1, 32, 1, 1, 1}, {8, 1, 1, 1, 28, 28}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 1},
              .fx_index = 3,
              .fy_index = 2,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_downsample (resnet18)

    tiling = {.loops = {{2, 2, 2, 1, 1, 1}, {8, 1, 1, 8, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 1024 &&
             matrix_op.stride(0) == 2) {  // layer3_0_downsample (resnet50)

    tiling = {.loops = {{2, 2, 8, 1, 1, 1}, {32, 1, 1, 8, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 256 && matrix_op.stride(0) == 2) {  // layer3_0_conv1

    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_0_conv2

    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_conv1

    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {32, 1, 1, 4, 14, 14}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_x_conv1

    tiling = {.loops = {{1, 1, 4, 1, 1, 1}, {64, 1, 1, 4, 14, 14}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_x_conv2

    tiling = {.loops = {{1, 1, 4, 1, 1, 1}, {16, 3, 3, 4, 14, 14}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 1024) {  // layer3_x_conv3

    tiling = {.loops = {{2, 2, 16, 1, 1, 1}, {16, 1, 1, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 28 && IW == 28 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer3

    tiling = {.loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512 &&
             matrix_op.stride(0) == 2) {  // layer4_0_downsample (resnet18)

    tiling = {.loops = {{1, 1, 2, 1, 1, 1}, {16, 1, 1, 16, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 2048 &&
             matrix_op.stride(0) == 2) {  // layer4_0_downsample (resnet50)

    tiling = {.loops = {{1, 1, 8, 1, 1, 1}, {64, 1, 1, 16, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 512 && matrix_op.stride(0) == 2) {  // layer4_0_conv1

    tiling = {.loops = {{1, 1, 8, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 512 && matrix_op.stride(0) == 1) {  // layer4_0_conv1

    tiling = {.loops = {{2, 2, 8, 1, 1, 1}, {64, 1, 1, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 7 && IW == 7 && IC == 2048 && KH == 1 && KW == 1 &&
             OC == 512 && matrix_op.stride(0) == 1) {  // layer4_x_conv1

    tiling = {.loops = {{1, 1, 8, 1, 1, 1}, {128, 1, 1, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 14 && IW == 14 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512 && matrix_op.stride(0) == 2) {  // layer4_x_conv2

    tiling = {.loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 2048) {  // layer4_x_conv3

    tiling = {.loops = {{1, 1, 32, 1, 1, 1}, {32, 1, 1, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512) {  // layer4

    tiling = {.loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
              .x_loop_index = {0, 5},
              .y_loop_index = {1, 4},
              .reduction_loop_index = {3, 0},
              .weight_loop_index = {2, 3},
              .fx_index = 2,
              .fy_index = 1,
              .weight_reuse_index = {4, 5},
              .stride = matrix_op.stride(0),
              .replication = false};

  } else {
    std::cout << "Using generated tiling" << std::endl;

    int x1 = 1, y1 = 1, k1 = 1;
    int x0 = output_shape[2];
    int y0 = output_shape[3];
    int k0 = weight_shape[0] / oc_dim;
    int c0 = weight_shape[1] / ic_dim;
    int fx = weight_shape[2];
    int fy = weight_shape[3];

    // Reduce OC0 to meet weight buffer constraint
    while (fx * fy * k0 * ic_dim > WEIGHT_BUFFER_SIZE) {
      if (k0 % 2 == 0) {
        k0 /= 2;
        k1 *= 2;
      } else {
        std::cerr << "Weight buffer is too small" << std::endl;
        exit(1);
      }
    }

    // Reduce X0 and Y0 to meet input buffer constraint. We are not counting
    // stride here because of the hardware implementation
    const int stride = matrix_op.stride(0);
    while (true) {
      int ix = x0 * stride + fx - 1;
      int iy = y0 * stride + fy - 1;
      if (ix * iy <= INPUT_BUFFER_SIZE) {
        break;
      }
      if (x0 % 2 == 0 && y0 % 2 == 0) {
        x0 /= 2;
        x1 *= 2;
        y0 /= 2;
        y1 *= 2;
      } else {
        std::cerr << "Input buffer is too small" << std::endl;
        exit(1);
      }
    }

    // Reduce either OC0, or OX0 and OY0, to meet accumulation buffer constraint
    const int max_k0 = k0;
    while (x0 * y0 * k0 > ACCUM_BUFFER_SIZE) {
      if (k0 % 2 == 0) {
        k0 /= 2;
        k1 *= 2;
      } else if (x0 % 2 == 0 && y0 % 2 == 0) {
        x0 /= 2;
        x1 *= 2;
        y0 /= 2;
        y1 *= 2;
        // Since we are reducing both x0 and y0, there is a chance we can
        // increase k0
        if (k0 * 2 <= max_k0) {
          k0 *= 2;
          k1 /= 2;
        }
      } else {
        std::cerr << "Accumulation buffer is too small" << std::endl;
        exit(1);
      }
    }

    tiling = {
        .loops = {{x1, y1, k1, 1, 1, 1}, {c0, k0, fy, fx, y0, x0}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = matrix_op.stride(0),
    };
  }

  return tiling;
}

inline void adjust_tiling_for_dimension(Tiling& tiling) {
  // adjust loop counters for dimension != 16
  if (!tiling.replication) {
    if (IC_DIMENSION < 16) {
      tiling.loops[1][tiling.reduction_loop_index[1]] *= (16 / IC_DIMENSION);
    } else if (IC_DIMENSION > 16) {
      tiling.loops[1][tiling.reduction_loop_index[1]] /= (IC_DIMENSION / 16);
    }

    // adjust loop counters for weight buffer constraint
    const int buffer_depth = tiling.loops[1][tiling.fx_index] *
                             tiling.loops[1][tiling.fy_index] * IC_DIMENSION;
    while (buffer_depth * tiling.loops[1][tiling.weight_loop_index[1]] >
               WEIGHT_BUFFER_SIZE &&
           tiling.loops[1][tiling.weight_loop_index[1]] > 1) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= 2;
      tiling.loops[0][tiling.weight_loop_index[0]] *= 2;
    }
  }

  if (OC_DIMENSION < 16) {
    tiling.loops[0][tiling.weight_loop_index[0]] *= (16 / OC_DIMENSION);
  } else if (OC_DIMENSION > 16) {
    int div_factor = OC_DIMENSION / 16;
    // cut down the outer loop first
    while (tiling.loops[0][tiling.weight_loop_index[0]] > 1 && div_factor > 1) {
      tiling.loops[0][tiling.weight_loop_index[0]] /= 2;
      div_factor /= 2;
    }
    while (tiling.loops[1][tiling.weight_loop_index[1]] > 1 && div_factor > 1) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= 2;
      div_factor /= 2;
    }
  }
}

inline Tiling get_linear_tiling(codegen::Operator param) {
  const auto matrix_op = param.matrix_op();

  const auto input = matrix_op.has_mx_input() ? matrix_op.mx_input().input()
                                              : matrix_op.input();
  const auto input_shape =
      input.has_reshape() ? input.reshape().output_sizes() : input.shape();

  const auto weight = matrix_op.has_mx_weight() ? matrix_op.mx_weight().input()
                                                : matrix_op.weight();
  const auto weight_shape =
      weight.has_reshape() ? weight.reshape().output_sizes() : weight.shape();

  // TODO: we should use OC_DIMENSION and IC_DIMENSION instead
  const int oc_dim = 16;
  const int ic_dim = 16;

  int x1 = 1, k1 = 1;

  int x0 = 1;
  for (int i = 0; i < input_shape.size() - 1; i++) {
    x0 *= input_shape[i];
  }

  int k0 = weight_shape[0] / oc_dim;
  int c0 = weight_shape[1] / ic_dim;

  // torch.matmul weight is also an activation
  if (matrix_op.opcode() == "matmul") {
    int size = weight_shape.size();
    c0 = weight_shape[size - 2] / ic_dim;
    k0 = weight_shape[size - 1] / oc_dim;
  }

  // Loop indices cannot exceed 1024 (10b)
  while (x0 >= 1024) {
    if (x0 % 2 == 0) {
      x0 /= 2;
      x1 *= 2;
    } else {
      std::cerr << "Input size is not divisible by 2" << std::endl;
      exit(1);
    }
  }

  while (k0 * ic_dim > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else {
      std::cerr << "Weight buffer is too small" << std::endl;
      exit(1);
    }
  }

  while (x0 * k0 > ACCUM_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else if (x0 % 2 == 0) {
      x0 /= 2;
      x1 *= 2;
    } else {
      std::cerr << "Accumulation buffer is too small" << std::endl;
      exit(1);
    }
  }

  Tiling tiling = {
      .loops = {{x1, 1, k1, 1, 1, 1}, {c0, k0, 1, 1, 1, x0}},
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

  std::cerr << "Tiling: " << tiling << std::endl;

  return tiling;
}

inline Tiling get_pooling_tiling(codegen::Operator param) {
  const auto pooling_op = param.pooling_op();
  const auto input_shape = pooling_op.input().shape();
  const auto output_shape = param.output().shape();

  int X = input_shape[2];
  int Y = input_shape[3];
  int C = pooling_op.input().shape(1);
  int K = param.output().shape(1);

  int x0, y0, stride;
  if (pooling_op.output_size_size() > 0) {
    int output_h = pooling_op.output_size(0);
    int output_w = pooling_op.output_size(1);
    stride = X / output_h;
    x0 = X - (output_h - 1) * stride;
    y0 = Y - (output_w - 1) * stride;
  } else {
    x0 = pooling_op.kernel_size(0);
    y0 = pooling_op.kernel_size(1);
    stride = pooling_op.stride(0);
  }

  int x1 = X / x0;
  int y1 = Y / y0;
  int c0 = C / IC_DIMENSION;
  int k0 = K / OC_DIMENSION;

  return {
      .loops = {{x1, y1, 1, 1, 1, 1}, {c0, k0, 1, 1, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = stride,
      .replication = false,
  };
}
