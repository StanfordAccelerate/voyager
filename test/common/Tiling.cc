#include "test/common/Tiling.h"

#include "spdlog/spdlog.h"
#include "test/common/Utils.h"

// Used only inside this file: get_tiling picks between them.
namespace {
Tiling get_interstellar_tiling(const voyager::Tiling& tiling);
Tiling get_conv2d_tiling(const voyager::PrimOp& param, const ScalarEnv& env);
Tiling get_linear_tiling(const voyager::PrimOp& op, const ScalarEnv& env);
}  // namespace

std::ostream& operator<<(std::ostream& os, const Tiling& tiling) {
  os << "Loops: " << std::endl;
  for (int i = 0; i < 2; i++) {
    os << "  " << i << ": ";
    for (int j = 0; j < 6; j++) {
      os << tiling.loops[i][j] << " ";
    }
    os << std::endl;
  }
  os << "X Loop Index: " << tiling.x_loop_idx[0] << " " << tiling.x_loop_idx[1]
     << std::endl;
  os << "Y Loop Index: " << tiling.y_loop_idx[0] << " " << tiling.y_loop_idx[1]
     << std::endl;
  os << "Reduction Loop Index: " << tiling.reduction_loop_idx[0] << " "
     << tiling.reduction_loop_idx[1] << std::endl;
  os << "Weight Loop Index: " << tiling.weight_loop_idx[0] << " "
     << tiling.weight_loop_idx[1] << std::endl;
  os << "FX Index: " << tiling.fx_loop_idx << std::endl;
  os << "FY Index: " << tiling.fy_loop_idx[0] << " " << tiling.fy_loop_idx[1]
     << std::endl;
  os << "Weight Reuse Index: " << tiling.weight_reuse_idx[0] << " "
     << tiling.weight_reuse_idx[1] << std::endl;
  os << "Stride: " << tiling.stride << std::endl;
  os << "Padding: " << tiling.padding << std::endl;
  os << "Resnet Replication: " << tiling.resnet_replication << std::endl;
  os << "Generic Replication: " << tiling.generic_replication << std::endl;
  os << "Num Channels: " << tiling.num_channels << std::endl;
  os << "FX Unrolling: " << tiling.fx_unrolling << std::endl;
  os << "Padded Input X: " << tiling.input_x << std ::endl;
  os << "Padded Input Y: " << tiling.input_y << std::endl;
  return os;
}

Tiling get_tiling(const voyager::Operation& operation, const ScalarEnv& env) {
  const auto& first_op = *get_prim_ops(operation)[0];

  // get environment variable
  const char* env_var = std::getenv("MANUAL_TILING");
  bool manual_tiling = env_var ? std::stoi(env_var) : false;

  const bool is_conv = strip_namespace(first_op.target()) == "conv2d" ||
                       strip_namespace(first_op.target()) == "conv2d_mx";

  Tiling tiling;
  if (manual_tiling || !operation.has_tiling()) {
    spdlog::info("Using manual tiling for operation {} with target {}\n",
                 operation.name(), strip_namespace(first_op.target()));
    if (is_conv) {
      tiling = get_conv2d_tiling(first_op, env);
    } else {
      tiling = get_linear_tiling(first_op, env);
    }
  } else {
    tiling = get_interstellar_tiling(operation.tiling());
    if (has_arg(first_op, "stride")) {
      auto stride = arg_ints(first_op, "stride", env);
      tiling.stride = stride[0];
    } else {
      tiling.stride = 1;
    }

    if (has_arg(first_op, "padding")) {
      auto padding = arg_ints(first_op, "padding", env);
      tiling.padding = padding[0];
    } else {
      tiling.padding = 0;
    }
  }

  const auto input = resolve(first_op, "input", env);
  const auto input_shape = get_shape(input);

  if (strip_namespace(first_op.target()) == "conv2d" ||
      strip_namespace(first_op.target()) == "conv2d_mx") {
    tiling.input_y = input_shape[1];
    tiling.input_x = input_shape[2];
  } else {
    tiling.input_y = 1;
    tiling.input_x = get_size(input_shape) / input_shape.back();
  }

  return tiling;
}

namespace {
Tiling get_interstellar_tiling(const voyager::Tiling& tiling) {
  // `= {}` value-initializes every field to 0/false. A plain `Tiling t;` would
  // leave the scalar fields indeterminate, since Tiling is an aggregate with no
  // constructor. Fields not assigned below therefore read as 0, not garbage.
  Tiling accelerator_tiling = {};

  // Interstellar does not emit tilings with replication
  accelerator_tiling.resnet_replication = false;
  accelerator_tiling.generic_replication = false;
  accelerator_tiling.fx_unrolling = 1;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      accelerator_tiling.loops[i][j] = 1;
    }
  }

  accelerator_tiling.fx_loop_idx = -1;
  for (int i = 0; i < 2; i++) {
    accelerator_tiling.fy_loop_idx[i] = -1;
    accelerator_tiling.x_loop_idx[i] = -1;
    accelerator_tiling.y_loop_idx[i] = -1;
    accelerator_tiling.reduction_loop_idx[i] = -1;
    accelerator_tiling.weight_loop_idx[i] = -1;
  }

  int loop_index = 5;
  // L1 level
  for (int i = 0; i < tiling.level_tilings(0).loop_bounds_size(); i++) {
    if (tiling.level_tilings(0).loop_bounds(i).loop() == voyager::LOOP_IC) {
      // set this later
      accelerator_tiling.loops[1][0] =
          tiling.level_tilings(0).loop_bounds(i).bound();
      accelerator_tiling.reduction_loop_idx[1] = 0;
    } else {
      // all other loops need to be set in reverse order
      accelerator_tiling.loops[1][loop_index] =
          tiling.level_tilings(0).loop_bounds(i).bound();
      if (tiling.level_tilings(0).loop_bounds(i).loop() == voyager::LOOP_FX) {
        accelerator_tiling.fx_loop_idx = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::LOOP_FY) {
        accelerator_tiling.fy_loop_idx[1] = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::LOOP_OC) {
        accelerator_tiling.weight_loop_idx[1] = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::LOOP_OX) {
        accelerator_tiling.x_loop_idx[1] = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::LOOP_OY) {
        accelerator_tiling.y_loop_idx[1] = loop_index;
      }

      loop_index--;
    }
  }

  // set any unset loop indices
  while (loop_index >= 1) {
    if (accelerator_tiling.fx_loop_idx == -1) {
      accelerator_tiling.fx_loop_idx = loop_index;
    } else if (accelerator_tiling.fy_loop_idx[1] == -1) {
      accelerator_tiling.fy_loop_idx[1] = loop_index;
    } else if (accelerator_tiling.weight_loop_idx[1] == -1) {
      accelerator_tiling.weight_loop_idx[1] = loop_index;
    } else if (accelerator_tiling.x_loop_idx[1] == -1) {
      accelerator_tiling.x_loop_idx[1] = loop_index;
    } else if (accelerator_tiling.y_loop_idx[1] == -1) {
      accelerator_tiling.y_loop_idx[1] = loop_index;
    }
    loop_index--;
  }

  // if reduction loop is not set, set it to 0
  if (accelerator_tiling.reduction_loop_idx[1] == -1) {
    accelerator_tiling.reduction_loop_idx[1] = 0;
  }

  for (int i = loop_index; i < 6; i++) {
    if (accelerator_tiling.fx_loop_idx == -1) {
      accelerator_tiling.fx_loop_idx = 5 - i;
    } else if (accelerator_tiling.fy_loop_idx[1] == -1) {
      accelerator_tiling.fy_loop_idx[1] = 5 - i;
    } else if (accelerator_tiling.weight_loop_idx[1] == -1) {
      accelerator_tiling.weight_loop_idx[1] = 5 - i;
    } else if (accelerator_tiling.x_loop_idx[1] == -1) {
      accelerator_tiling.x_loop_idx[1] = 5 - i;
    } else if (accelerator_tiling.y_loop_idx[1] == -1) {
      accelerator_tiling.y_loop_idx[1] = 5 - i;
    } else if (accelerator_tiling.reduction_loop_idx[1] == -1) {
      accelerator_tiling.reduction_loop_idx[1] = 5 - i;
    }
  }

  // set weight reuse loop index depending on if x or y are the innermost loops
  if (accelerator_tiling.x_loop_idx[1] == 5 ||
      accelerator_tiling.y_loop_idx[1] == 5) {
    accelerator_tiling.weight_reuse_idx[1] = 5;
    accelerator_tiling.weight_reuse_idx[0] = 5;
  }
  if (accelerator_tiling.x_loop_idx[1] == 4 ||
      accelerator_tiling.y_loop_idx[1] == 4) {
    accelerator_tiling.weight_reuse_idx[0] = 4;
  }

  // L2 level
  accelerator_tiling.loops[0][4] = 1;
  accelerator_tiling.fy_loop_idx[0] = 4;

  int offset = 1;
  if (tiling.level_tilings(1).loop_bounds_size() == 0 ||
      tiling.level_tilings(1).loop_bounds(0).loop() != voyager::LOOP_IC) {
    // The outer level has no IC (reduction) loop, either because it is empty or
    // because its first loop is something else. Reserve position 3 for the
    // (trivial) reduction loop so the output loops Y/X/OC land in the outer
    // three positions (0..2).
    accelerator_tiling.loops[0][3] = 1;
    accelerator_tiling.reduction_loop_idx[0] = 3;
    offset++;
  }

  for (int i = 0; i < tiling.level_tilings(1).loop_bounds_size(); i++) {
    accelerator_tiling.loops[0][4 - offset - i] =
        tiling.level_tilings(1).loop_bounds(i).bound();
    if (tiling.level_tilings(1).loop_bounds(i).loop() == voyager::LOOP_OC) {
      accelerator_tiling.weight_loop_idx[0] = 4 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::LOOP_OX) {
      accelerator_tiling.x_loop_idx[0] = 4 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::LOOP_OY) {
      accelerator_tiling.y_loop_idx[0] = 4 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::LOOP_IC) {
      accelerator_tiling.reduction_loop_idx[0] = 4 - offset - i;
    }
  }

  // set any unset loop indices
  for (int i = tiling.level_tilings(1).loop_bounds_size() - 1; i < 4 - offset;
       i++) {
    accelerator_tiling.loops[0][3 - i - offset] = 1;
    if (accelerator_tiling.weight_loop_idx[0] == -1) {
      accelerator_tiling.weight_loop_idx[0] = 3 - i - offset;
    } else if (accelerator_tiling.x_loop_idx[0] == -1) {
      accelerator_tiling.x_loop_idx[0] = 3 - i - offset;
    } else if (accelerator_tiling.y_loop_idx[0] == -1) {
      accelerator_tiling.y_loop_idx[0] = 3 - i - offset;
    } else if (accelerator_tiling.reduction_loop_idx[0] == -1) {
      accelerator_tiling.reduction_loop_idx[0] = 3 - i - offset;
    } else if (accelerator_tiling.fy_loop_idx[0] == -1) {
      accelerator_tiling.fy_loop_idx[0] = 3 - i - offset;
    }
  }

  return accelerator_tiling;
}
}  // namespace

namespace {
Tiling get_conv2d_tiling(const voyager::PrimOp& param, const ScalarEnv& env) {
  const auto input = resolve(param, "input", env);
  const auto weight = resolve(param, "weight", env);
  const auto paddings = arg_ints(param, "padding", env);
  const auto dilation = arg_ints(param, "dilation", env);
  const auto strides = arg_ints(param, "stride", env);

  const auto input_shape = get_shape(input);
  const auto weight_shape = get_shape(weight);

  const int output_height = (input_shape[1] + 2 * paddings[0] -
                             dilation[0] * (weight_shape[0] - 1) - 1) /
                                strides[0] +
                            1;
  const int output_width = (input_shape[2] + 2 * paddings[1] -
                            dilation[1] * (weight_shape[1] - 1) - 1) /
                               strides[1] +
                           1;

  std::vector<int> output_shape = {
      output_height,
      output_width,
      input_shape[3],
      weight_shape[3],
  };

  int x1 = 1, y1 = 1, k1 = 1;
  int x0 = output_shape[2];
  int y0 = output_shape[1];
  int k0 = weight_shape[3] / OC_DIMENSION;
  int c0 = weight_shape[2] / IC_DIMENSION;
  int fx = weight_shape[1];
  int fy = weight_shape[0];
  int stride = strides[0];
  int padding = paddings[0];

  // conv2d (vit)
  if (input_shape[3] == 3 && weight_shape[0] == 16 && weight_shape[1] == 16) {
    int fx_unrolling;
    if (IC_DIMENSION == 4) {
      fx_unrolling = 1;
    } else if (IC_DIMENSION == 8) {
      fx_unrolling = 2;
    } else if (IC_DIMENSION == 16) {
      fx_unrolling = 4;
    } else if (IC_DIMENSION == 32 || IC_DIMENSION == 64) {
      fx_unrolling = 8;
    } else {
      throw std::runtime_error("replication not supported for IC_DIMENSION=" +
                               std::to_string(IC_DIMENSION));
    }

    int k1 = weight_shape[3] / 32;
    Tiling tiling = {
        .loops = {{7, 1, k1, 1, fy, 1}, {1, 2, 1, fx / fx_unrolling, 2, 14}},
        .x_loop_idx = {1, 5},
        .y_loop_idx = {0, 4},
        .reduction_loop_idx = {3, 0},
        .weight_loop_idx = {2, 1},
        .fx_loop_idx = 3,
        .fy_loop_idx = {4, 2},
        .weight_reuse_idx = {4, 5},
        .stride = stride,
        .padding = 0,
        .resnet_replication = false,
        .generic_replication = true,
        .num_channels = 3,
        .fx_unrolling = fx_unrolling,
    };

    if (IC_DIMENSION < 16) {
      tiling.loops[1][5] /= 2;
      tiling.loops[0][0] *= 2;
    }

    if (OC_DIMENSION < 16) {
      tiling.loops[0][tiling.weight_loop_idx[0]] *= (16 / OC_DIMENSION);
    } else if (OC_DIMENSION > 16) {
      int div_factor = OC_DIMENSION / 16;
      int& k1 = tiling.loops[0][tiling.weight_loop_idx[0]];
      while (k1 > 1 && k1 % 2 == 0 && div_factor > 1) {
        k1 /= 2;
        div_factor /= 2;
      }
      int& k0 = tiling.loops[1][tiling.weight_loop_idx[1]];
      while (k0 > 1 && k0 % 2 == 0 && div_factor > 1) {
        k0 /= 2;
        div_factor /= 2;
      }

      if (div_factor > 1) {
        spdlog::error("OC_DIMENSION is not a multiple of 16\n");
        exit(1);
      }
    }

    return tiling;
  }
  // conv1
  else if (input_shape[3] == 3 && weight_shape[0] == 7 &&
           weight_shape[1] == 7) {
    int fx;
    if (IC_DIMENSION == 4) {
      fx = 7;
    } else if (IC_DIMENSION == 8) {
      fx = 4;
    } else if (IC_DIMENSION == 16) {
      fx = 2;
    } else if (IC_DIMENSION == 32 || IC_DIMENSION == 64) {
      fx = 1;
    } else {
      throw std::runtime_error("replication not supported for IC_DIMENSION=" +
                               std::to_string(IC_DIMENSION));
    }

    int K1 = weight_shape[3] / 32;
    int Y1 = output_shape[0] / 16;
    int X1 = output_shape[1] / 16;
    Tiling tiling = {
        .loops = {{X1, Y1, K1, 1, 1, 1}, {1, 2, 7, fx, 16, 16}},
        .x_loop_idx = {0, 5},
        .y_loop_idx = {1, 4},
        .reduction_loop_idx = {3, 0},
        .weight_loop_idx = {2, 1},
        .fx_loop_idx = 3,
        .fy_loop_idx = {4, 2},
        .weight_reuse_idx = {4, 5},
        .stride = stride,
        .padding = 3,
        .resnet_replication = true,
        .generic_replication = false,
        .num_channels = 3,
    };

    if (IC_DIMENSION < 16) {
      tiling.loops[1][5] /= 2;
      tiling.loops[0][0] *= 2;
    }

    if (OC_DIMENSION < 16) {
      tiling.loops[0][tiling.weight_loop_idx[0]] *= (16 / OC_DIMENSION);
    } else if (OC_DIMENSION > 16) {
      int div_factor = OC_DIMENSION / 16;
      int& k1 = tiling.loops[0][tiling.weight_loop_idx[0]];
      while (k1 > 1 && k1 % 2 == 0 && div_factor > 1) {
        k1 /= 2;
        div_factor /= 2;
      }
      int& k0 = tiling.loops[1][tiling.weight_loop_idx[1]];
      while (k0 > 1 && k0 % 2 == 0 && div_factor > 1) {
        k0 /= 2;
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
  while (fx * fy * k0 * IC_DIMENSION > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else {
      spdlog::error("Weight buffer is too small\n");
      exit(1);
    }
  }

  // Reduce OC0 to meet weight buffer constraint
  while (fx * fy * k0 * IC_DIMENSION > WEIGHT_BUFFER_SIZE) {
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
      .x_loop_idx = {0, 5},
      .y_loop_idx = {1, 4},
      .reduction_loop_idx = {3, 0},
      .weight_loop_idx = {2, 1},
      .fx_loop_idx = 3,
      .fy_loop_idx = {4, 2},
      .weight_reuse_idx = {4, 5},
      .stride = stride,
      .padding = padding,
  };
}
}  // namespace

namespace {
Tiling get_linear_tiling(const voyager::PrimOp& op, const ScalarEnv& env) {
  const auto input_shape = get_shape(resolve(op, "input", env));

  bool is_matmul =
      strip_namespace(op.target()).find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight_shape = get_shape(resolve(op, weight_key, env));

  int x1 = 1, k1 = 1, c1 = 1;
  int x0 = get_size(input_shape) / input_shape.back();
  int k0 = weight_shape[0] / OC_DIMENSION;
  int c0 = weight_shape[1] / IC_DIMENSION;

  // Manual tiling
  if (x0 == 128 && weight_shape[0] == 512 && weight_shape[1] == 2048) {
    return {
        .loops = {{1, k0 / 2, 4, c0 / 2, 1, 1}, {2, 1, 2, 1, 1, 32}},
        .x_loop_idx = {2, 5},
        .y_loop_idx = {0, 1},
        .reduction_loop_idx = {3, 0},
        .weight_loop_idx = {1, 2},
        .fx_loop_idx = 4,
        .fy_loop_idx = {4, 3},
        .weight_reuse_idx = {5, 5},
        .stride = 1,
        .resnet_replication = false,
    };
  }

  // torch.matmul weight is also an activation, thus does not need to be
  // transposed
  if (strip_namespace(op.target()) == "matmul" ||
      strip_namespace(op.target()) == "matmul_mx") {
    int size = weight_shape.size();
    c0 = weight_shape[size - 2] / IC_DIMENSION;
    k0 = weight_shape[size - 1] / OC_DIMENSION;
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

  while (k0 * c0 * IC_DIMENSION > WEIGHT_BUFFER_SIZE) {
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
      .x_loop_idx = {0, 5},
      .y_loop_idx = {1, 4},
      .reduction_loop_idx = {3, 0},
      .weight_loop_idx = {2, 1},
      .fx_loop_idx = 3,
      .fy_loop_idx = {4, 2},
      .weight_reuse_idx = {4, 5},
      .stride = 1,
      .padding = 0,
      .resnet_replication = false,
  };
}
}  // namespace

Tiling get_pool2d_tiling(const voyager::PrimOp& op, const ScalarEnv& env) {
  const auto input_shape = get_shape(resolve(op, "input", env));

  int Y = input_shape[1];
  int X = input_shape[2];
  int K = input_shape[3];

  int x0, y0, stride, padding, x1, y1, k0, actual_padding;

  Tiling tiling =
      {};  // value-initialize every field to 0; see get_interstellar_tiling

  if (has_arg(op, "output_size")) {
    const auto output_size = arg_ints(op, "output_size", env);
    int output_h = output_size[0];
    int output_w = output_size[1];

    stride = X / output_h;
    x0 = X - (output_h - 1) * stride;
    y0 = Y - (output_w - 1) * stride;

    x1 = X / x0;
    y1 = Y / y0;
    k0 = K / ACCUMULATOR_WIDTH;
    actual_padding = 0;
  } else {
    const auto kernel_size = arg_ints(op, "kernel_size", env);
    const auto strides = arg_ints(op, "stride", env);
    const auto paddings = arg_ints(op, "padding", env);

    y0 = kernel_size[0];
    x0 = kernel_size[1];
    stride = strides[0];
    padding = paddings[0];

    // calculate ouptut dimension (ignoring padding, which will be handled in
    // hw)
    x1 = (X + 2 * padding - x0) / stride + 1;
    y1 = (Y + 2 * padding - y0) / stride + 1;
    // pytorch assumes padding on all direction, not all of the values are used
    actual_padding = (x1 - 1) * stride + x0 - X;
    k0 = K / ACCUMULATOR_WIDTH;
  }

  return {
      .loops = {{x1, y1, 1, 1, 1, 1}, {1, k0, 1, 1, y0, x0}},
      .x_loop_idx = {0, 5},
      .y_loop_idx = {1, 4},
      .reduction_loop_idx = {3, 0},
      .weight_loop_idx = {2, 1},
      .fx_loop_idx = 3,
      .fy_loop_idx = {4, 2},
      .weight_reuse_idx = {4, 5},
      .stride = stride,
      .padding = actual_padding,
      .resnet_replication = false,
  };
}
