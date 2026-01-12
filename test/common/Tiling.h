#pragma once

#include <iostream>

#include "test/common/Network.h"
#include "test/compiler/proto/param.pb.h"
#include "test/compiler/proto/tiling.pb.h"

struct Tiling {
  int loops[2][6];
  int x_loop_idx[2];
  int y_loop_idx[2];
  int reduction_loop_idx[2];
  int weight_loop_idx[2];
  int fx_loop_idx;
  int fy_loop_idx[2];
  int weight_reuse_idx[2];
  int stride;
  int padding;
  bool resnet_replication;
  bool generic_replication;
  int num_channels;
  int fx_unrolling;
  int input_x;
  int input_y;
};

std::ostream& operator<<(std::ostream& os, const Tiling& tiling);
Tiling get_interstellar_tiling(const voyager::Tiling& tiling);
Tiling get_tiling(const Operation& operation);

Tiling get_conv2d_tiling(const codegen::OpOverload param);
Tiling get_linear_tiling(const codegen::OpOverload param);
Tiling get_pool2d_tiling(const codegen::OpOverload param);
void adjust_tiling_for_dimension(Tiling& tiling);
