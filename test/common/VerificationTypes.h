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

// By default we have 2MB of SRAM per MINOTAUR SoC
// organized as 8x 256KB Banks with 2x 128KB Macros each
// FIXME: use a larger memory partition for testing
// #ifndef NUM_SRAM_BANKS
// #define NUM_SRAM_BANKS 8
// #endif
// #define SRAM_MEMORY_SIZE (NUM_SRAM_BANKS * 256 * 1024)
#ifndef NUM_SRAM_BANKS
#define NUM_SRAM_BANKS 256
#endif
#define SRAM_MEMORY_SIZE (NUM_SRAM_BANKS * 1024 * 1024)

// By default we have 12MB of RRAM per MINOTAUR SoC
// organized as 12x 1MB Banks with 4x 256KB Macros each
// 3x Superbanks and 4x Subbanks per Superbank
#ifndef NUM_RRAM_BANKS
#define NUM_RRAM_BANKS 64
#endif
#define RRAM_MEMORY_SIZE (NUM_RRAM_BANKS * 1024 * 1024)

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

inline Tiling get_conv2d_tiling(codegen::AcceleratorParam param) {
  const auto matrix_param = param.matrix_param();
  const auto input_shape = matrix_param.input().shape();
  const auto weight_shape = matrix_param.weight().shape();
  const auto output_shape = param.output().shape();

  // TODO: we should use OC_DIMENSION and IC_DIMENSION instead
  const int oc_dim = 16;
  const int ic_dim = 16;

  int x1, y1, k1, x0, y0, k0, c0, fx, fy;
  bool replication = weight_shape[1] < 16;
  if (replication) {
    x1 = 7, y1 = 7, k1 = 2;
    c0 = 1, k0 = 2, fy = 7, y0 = 16, x0 = 16;
    if (IC_DIMENSION == 16) {
      fx = 2;
    } else if (IC_DIMENSION == 32) {
      fx = 1;
    } else {
      fx = 7;
    }
  } else {
    x1 = 1, y1 = 1, k1 = 1;
    x0 = output_shape[2];
    y0 = output_shape[3];
    k0 = weight_shape[0] / oc_dim;
    c0 = weight_shape[1] / ic_dim;
    fx = weight_shape[2];
    fy = weight_shape[3];

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
    const int stride = matrix_param.stride(0);
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
    while (x0 * y0 * k0 > ACCUMULATION_BUFFER_SIZE) {
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
  }

  Tiling tiling = {
      .loops = {{x1, y1, k1, 1, 1, 1}, {c0, k0, fy, fx, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = matrix_param.stride(0),
      .replication = replication,
  };
  return tiling;
}

inline Tiling get_linear_tiling(codegen::AcceleratorParam param) {
  const auto matrix_param = param.matrix_param();
  const auto input_shape = matrix_param.input().shape();
  const auto weight_shape = matrix_param.weight().shape();

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
  if (matrix_param.opcode() == "matmul") {
    int size = matrix_param.weight().shape_size();
    c0 = weight_shape[size - 2] / ic_dim;
    k0 = weight_shape[size - 1] / oc_dim;
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

  while (x0 * k0 > ACCUMULATION_BUFFER_SIZE) {
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
  return tiling;
}

inline Tiling get_pooling_tiling(codegen::AcceleratorParam param) {
  const auto pooling_param = param.pooling_param();
  const auto input_shape = pooling_param.input().shape();
  const auto output_shape = param.output().shape();

  int X = input_shape[2];
  int Y = input_shape[3];
  int C = pooling_param.input().shape(1);
  int K = param.output().shape(1);

  int x0, y0, stride;
  if (pooling_param.output_size_size() > 0) {
    int output_h = pooling_param.output_size(0);
    int output_w = pooling_param.output_size(1);
    stride = X / output_h;
    x0 = X - (output_h - 1) * stride;
    y0 = Y - (output_w - 1) * stride;
  } else {
    x0 = pooling_param.kernel_size(0);
    y0 = pooling_param.kernel_size(1);
    stride = pooling_param.stride(0);
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
