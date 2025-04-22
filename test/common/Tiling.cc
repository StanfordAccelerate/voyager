#include "test/common/Tiling.h"

#include "spdlog/spdlog.h"
#include "test/common/VerificationTypes.h"

std::ostream& operator<<(std::ostream& os, const Tiling& tiling) {
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
  os << "Padding: " << tiling.padding << std::endl;
  os << "Replication: " << tiling.replication << std::endl;
  return os;
}

Tiling get_tiling(const Operation& operation) {
  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto first_op = op_list[0];

  // get environment variable
  const char* env_var = std::getenv("MANUAL_TILING");
  bool manual_tiling = env_var ? std::stoi(env_var) : false;

  Tiling tiling;
  if (manual_tiling || !operation.has_valid_tiling) {
    if (first_op.target() == "conv2d" || first_op.target() == "conv2d_mx") {
      tiling = get_conv2d_tiling(first_op);
    } else {
      tiling = get_linear_tiling(first_op);
    }
  } else {
    tiling = get_interstellar_tiling(operation.tiling);
    if (first_op.kwargs().contains("stride")) {
      auto stride = first_op.kwargs().at("stride").int_list().values();
      tiling.stride = stride[0];
    } else {
      tiling.stride = 1;
    }
  }

  return tiling;
}

Tiling get_interstellar_tiling(const voyager::Tiling& tiling) {
  Tiling accelerator_tiling;

  // Interstellar does not emit tilings with replication
  accelerator_tiling.replication = false;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      accelerator_tiling.loops[i][j] = 1;
    }
  }

  accelerator_tiling.fx_index = -1;
  accelerator_tiling.fy_index = -1;
  for (int i = 0; i < 2; i++) {
    accelerator_tiling.x_loop_index[i] = -1;
    accelerator_tiling.y_loop_index[i] = -1;
    accelerator_tiling.reduction_loop_index[i] = -1;
    accelerator_tiling.weight_loop_index[i] = -1;
  }

  int loop_index = 5;
  // L1 level
  for (int i = 0; i < tiling.level_tilings(0).loop_bounds_size(); i++) {
    if (tiling.level_tilings(0).loop_bounds(i).loop() == voyager::Loop::IC) {
      // set this later
      accelerator_tiling.loops[1][0] =
          tiling.level_tilings(0).loop_bounds(i).bound();
      accelerator_tiling.reduction_loop_index[1] = 0;
    } else {
      // all other loops need to be set in reverse order
      accelerator_tiling.loops[1][loop_index] =
          tiling.level_tilings(0).loop_bounds(i).bound();
      if (tiling.level_tilings(0).loop_bounds(i).loop() == voyager::Loop::FX) {
        accelerator_tiling.fx_index = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::FY) {
        accelerator_tiling.fy_index = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::OC) {
        accelerator_tiling.weight_loop_index[1] = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::OX) {
        accelerator_tiling.x_loop_index[1] = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::OY) {
        accelerator_tiling.y_loop_index[1] = loop_index;
      }

      loop_index--;
    }
  }

  // set any unset loop indices
  while (loop_index >= 1) {
    if (accelerator_tiling.fx_index == -1) {
      accelerator_tiling.fx_index = loop_index;
    } else if (accelerator_tiling.fy_index == -1) {
      accelerator_tiling.fy_index = loop_index;
    } else if (accelerator_tiling.weight_loop_index[1] == -1) {
      accelerator_tiling.weight_loop_index[1] = loop_index;
    } else if (accelerator_tiling.x_loop_index[1] == -1) {
      accelerator_tiling.x_loop_index[1] = loop_index;
    } else if (accelerator_tiling.y_loop_index[1] == -1) {
      accelerator_tiling.y_loop_index[1] = loop_index;
    }
    loop_index--;
  }

  // if reduction loop is not set, set it to 0
  if (accelerator_tiling.reduction_loop_index[1] == -1) {
    accelerator_tiling.reduction_loop_index[1] = 0;
  }

  for (int i = loop_index; i < 6; i++) {
    if (accelerator_tiling.fx_index == -1) {
      accelerator_tiling.fx_index = 5 - i;
    } else if (accelerator_tiling.fy_index == -1) {
      accelerator_tiling.fy_index = 5 - i;
    } else if (accelerator_tiling.weight_loop_index[1] == -1) {
      accelerator_tiling.weight_loop_index[1] = 5 - i;
    } else if (accelerator_tiling.x_loop_index[1] == -1) {
      accelerator_tiling.x_loop_index[1] = 5 - i;
    } else if (accelerator_tiling.y_loop_index[1] == -1) {
      accelerator_tiling.y_loop_index[1] = 5 - i;
    } else if (accelerator_tiling.reduction_loop_index[1] == -1) {
      accelerator_tiling.reduction_loop_index[1] = 5 - i;
    }
  }

  // set weight reuse loop index depending on if x or y are the innermost loops
  if (accelerator_tiling.x_loop_index[1] == 5 ||
      accelerator_tiling.y_loop_index[1] == 5) {
    accelerator_tiling.weight_reuse_index[1] = 5;
    accelerator_tiling.weight_reuse_index[0] = 5;
  }
  if (accelerator_tiling.x_loop_index[1] == 4 ||
      accelerator_tiling.y_loop_index[1] == 4) {
    accelerator_tiling.weight_reuse_index[0] = 4;
  }

  // L2 level
  int offset = 0;
  if (tiling.level_tilings(1).loop_bounds_size() > 0 &&
      tiling.level_tilings(1).loop_bounds(0).loop() != voyager::Loop::IC) {
    // if the first loop is not IC, then we need to manually set the IC loop to
    // 1
    accelerator_tiling.loops[0][3] = 1;
    accelerator_tiling.reduction_loop_index[0] = 3;
    offset = 1;
  }

  for (int i = 0; i < tiling.level_tilings(1).loop_bounds_size(); i++) {
    accelerator_tiling.loops[0][3 - offset - i] =
        tiling.level_tilings(1).loop_bounds(i).bound();
    if (tiling.level_tilings(1).loop_bounds(i).loop() == voyager::Loop::OC) {
      accelerator_tiling.weight_loop_index[0] = 3 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::Loop::OX) {
      accelerator_tiling.x_loop_index[0] = 3 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::Loop::OY) {
      accelerator_tiling.y_loop_index[0] = 3 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::Loop::IC) {
      accelerator_tiling.reduction_loop_index[0] = 3 - offset - i;
    }
  }

  // set any unset loop indices
  for (int i = tiling.level_tilings(1).loop_bounds_size() - 1; i < 3 - offset;
       i++) {
    accelerator_tiling.loops[0][2 - i - offset] = 1;
    if (accelerator_tiling.weight_loop_index[0] == -1) {
      accelerator_tiling.weight_loop_index[0] = 2 - i - offset;
    } else if (accelerator_tiling.x_loop_index[0] == -1) {
      accelerator_tiling.x_loop_index[0] = 2 - i - offset;
    } else if (accelerator_tiling.y_loop_index[0] == -1) {
      accelerator_tiling.y_loop_index[0] = 2 - i - offset;
    } else if (accelerator_tiling.reduction_loop_index[0] == -1) {
      accelerator_tiling.reduction_loop_index[0] = 2 - i - offset;
    }
  }

  return accelerator_tiling;
}

Tiling get_conv2d_tiling(const codegen::OpOverload param) {
  const auto kwargs = param.kwargs();

  const auto input = kwargs.at("input").tensor();
  const auto weight = kwargs.at("weight").tensor();
  const auto padding = kwargs.at("padding").int_list().values();
  const auto dilation = kwargs.at("dilation").int_list().values();
  const auto strides = kwargs.at("stride").int_list().values();

  const auto input_shape = get_shape(input);
  const auto weight_shape = get_shape(weight);

  const int output_height = (input_shape[2] + 2 * padding[0] -
                             dilation[0] * (weight_shape[2] - 1) - 1) /
                                strides[0] +
                            1;
  const int output_width = (input_shape[3] + 2 * padding[1] -
                            dilation[1] * (weight_shape[3] - 1) - 1) /
                               strides[1] +
                           1;

  std::vector<int> output_shape = {input_shape[0], weight_shape[0],
                                   output_height, output_width};

  const int oc_unroll = OC_DIMENSION;
  const int ic_unroll = IC_DIMENSION;

  int x1 = 1, y1 = 1, k1 = 1;
  int x0 = output_shape[2];
  int y0 = output_shape[3];
  int k0 = weight_shape[0] / oc_unroll;
  int c0 = weight_shape[1] / ic_unroll;
  int fx = weight_shape[2];
  int fy = weight_shape[3];
  int stride = strides[0];

  // conv1
  if (input_shape[1] == 3 && input_shape[2] == 224 && input_shape[3] == 224 &&
      weight_shape[0] == 64 && weight_shape[2] == 7 && weight_shape[3] == 7) {
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

    Tiling tiling = {
        .loops = {{7, 7, 2, 1, 1, 1}, {1, 2, 7, fx, 16, 16}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = stride,
        .replication = true,
    };

    if (IC_DIMENSION < 16) {
      tiling.loops[1][5] /= 2;
      tiling.loops[0][0] *= 2;
    }

    if (OC_DIMENSION < 16) {
      tiling.loops[0][tiling.weight_loop_index[0]] *= (16 / OC_DIMENSION);
    } else if (OC_DIMENSION > 16) {
      int div_factor = OC_DIMENSION / 16;
      while (tiling.loops[0][tiling.weight_loop_index[0]] > 1 &&
             div_factor > 1) {
        tiling.loops[0][tiling.weight_loop_index[0]] /= 2;
        div_factor /= 2;
      }
      while (tiling.loops[1][tiling.weight_loop_index[1]] > 1 &&
             div_factor > 1) {
        tiling.loops[1][tiling.weight_loop_index[1]] /= 2;
        div_factor /= 2;
      }

      if (div_factor > 1) {
        spdlog::error("OC_DIMENSION is not a multiple of 16\n");
        exit(1);
      }
    }

    return tiling;
  }

  // Reduce OC0 to meet weight buffer constraint
  while (fx * fy * k0 * ic_unroll > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else {
      spdlog::error("Weight buffer is too small\n");
      exit(1);
    }
  }

  // Reduce OC0 to meet weight buffer constraint
  while (fx * fy * k0 * ic_unroll > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else {
      spdlog::error("Weight buffer is too small\n");
      exit(1);
    }
  }

  // Reduce X0 and Y0 to meet input buffer constraint. We are not counting
  // stride here because of the hardware implementation
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
      spdlog::error("Input buffer is too small\n");
      exit(1);
    }
  }

  // Reduce either OC0, or OX0 and OY0, to meet accumulation buffer
  // constraint
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
      spdlog::error("Accumulation buffer is too small\n");
      exit(1);
    }
  }

  return {
      .loops = {{x1, y1, k1, c0, 1, 1}, {1, k0, fy, fx, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = stride,
  };
}

Tiling get_linear_tiling(const codegen::OpOverload op) {
  const auto kwargs = op.kwargs();
  const auto input_shape = get_shape(kwargs.at("input").tensor());

  bool is_matmul = op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight_shape = get_shape(kwargs.at(weight_key).tensor());

  // Generate tiling using unroll factor of 16
  const int oc_unroll = OC_DIMENSION;
  const int ic_unroll = IC_DIMENSION;

  int x1 = 1, k1 = 1, c1 = 1;
  int x0 = get_size(input_shape) / input_shape.back();
  int k0 = weight_shape[0] / oc_unroll;
  int c0 = weight_shape[1] / ic_unroll;

  // torch.matmul weight is also an activation, thus does not need to be
  // transposed
  if (op.target() == "matmul" || op.target() == "matmul_mx") {
    int size = weight_shape.size();
    c0 = weight_shape[size - 2] / ic_unroll;
    k0 = weight_shape[size - 1] / oc_unroll;
  }

  // Loop indices cannot exceed 1024 (10-bit)
  while (x0 >= 1024 || x0 * c0 > INPUT_BUFFER_SIZE) {
    if (x0 % 2 == 0) {
      x0 /= 2;
      x1 *= 2;
    } else if (c0 % 2 == 0) {
      c0 /= 2;
      c1 *= 2;
    } else {
      spdlog::error("Input buffer is too small\n");
      exit(1);
    }
  }

  while (k0 * c0 * ic_unroll > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else if (c0 % 2 == 0) {
      c0 /= 2;
      c1 *= 2;
    } else {
      spdlog::error("Weight buffer is too small\n");
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
      spdlog::error("Accumulation buffer is too small\n");
      exit(1);
    }
  }

  return {
      .loops = {{x1, 1, k1, c1, 1, 1}, {c0, k0, 1, 1, 1, x0}},
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
}

Tiling get_pool2d_tiling(const codegen::OpOverload op) {
  const auto kwargs = op.kwargs();
  const auto input_shape = get_shape(kwargs.at("input").tensor());

  int K = input_shape[1];
  int X = input_shape[2];
  int Y = input_shape[3];

  int x0, y0, stride, padding;

  if (kwargs.contains("output_size")) {
    const auto output_size = kwargs.at("output_size").int_list().values();
    int output_h = output_size[0];
    int output_w = output_size[1];

    stride = X / output_h;
    x0 = X - (output_h - 1) * stride;
    y0 = Y - (output_w - 1) * stride;
  } else {
    const auto kernel_size = kwargs.at("kernel_size").int_list().values();
    const auto strides = kwargs.at("stride").int_list().values();
    const auto paddings = kwargs.at("padding").int_list().values();

    x0 = kernel_size[0];
    y0 = kernel_size[1];
    stride = strides[0];
    padding = paddings[0];
  }

  // calculate ouptut dimension (ignoring padding, which will be handled in hw)
  int x1 = (X + 2 * padding - x0) / stride + 1;
  int y1 = (Y + 2 * padding - y0) / stride + 1;
  // pytorch assumes padding on all direction, not all of the values are used
  int actual_padding = (x1 - 1) * stride + x0 - X;
  int k0 = K / OC_DIMENSION;

  return {
      .loops = {{x1, y1, 1, 1, 1, 1}, {1, k0, 1, 1, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = stride,
      .padding = actual_padding,
      .replication = false,
  };
}
