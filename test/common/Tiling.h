#pragma once

#include <iostream>

#include "test/common/Network.h"
#include "test/compiler/proto/param.pb.h"
#include "test/compiler/proto/tiling.pb.h"

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
  int padding;
  bool replication;
};

std::ostream& operator<<(std::ostream& os, const Tiling& tiling);
Tiling get_interstellar_tiling(const voyager::Tiling& tiling);
Tiling get_tiling(const Operation& operation);

Tiling get_conv2d_tiling(const codegen::OpOverload param);
Tiling get_linear_tiling(const codegen::OpOverload param);
Tiling get_pool2d_tiling(const codegen::OpOverload param);
void adjust_tiling_for_dimension(Tiling& tiling);
