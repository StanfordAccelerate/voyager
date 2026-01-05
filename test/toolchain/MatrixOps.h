#pragma once

#include "spdlog/spdlog.h"
#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/GoldModel.h"
#include "test/common/MemoryInterface.h"
#include "test/common/Network.h"
#include "test/common/Tiling.h"
#include "test/common/Utils.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/ApproximationConstants.h"
#include "test/toolchain/Common.h"

void set_vector_fetch_1(const codegen::Tensor& tensor, const Tiling& tiling,
                        VectorParams* vector_params) {
  int nonzero_dims = 0;
  for (const int& dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  const auto memory = tensor.memory();
  vector_params->vector_fetch_1_offset = get_address(tensor);
  vector_params->vector_fetch_1_mode = true;
  vector_params->vector_fetch_1_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

  vector_params->vector_fetch_1_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype());
  const int dtype_width =
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_1_dtype);
  int fetch_width = OC_DIMENSION * dtype_width;
  vector_params->vector_fetch_1_burst_size = fetch_width / 8;
  vector_params->vector_fetch_1_num_beats =
      (fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor =
      OC_DIMENSION / VECTOR_UNIT_WIDTH;

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->vector_fetch_1_x_loop_idx[0] = tiling.x_loop_idx[0];
  vector_params->vector_fetch_1_y_loop_idx[0] = tiling.y_loop_idx[0];
  vector_params->vector_fetch_1_k_loop_idx[0] = tiling.weight_loop_idx[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_idx[1] || i == tiling.x_loop_idx[1] ||
        i == tiling.y_loop_idx[1]) {
      vector_params->vector_fetch_1_loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_idx[1]) {
        vector_params->vector_fetch_1_x_loop_idx[1] = loop_index;
      }
      if (i == tiling.y_loop_idx[1]) {
        vector_params->vector_fetch_1_y_loop_idx[1] = loop_index;
      }
      if (i == tiling.weight_loop_idx[1]) {
        vector_params->vector_fetch_1_k_loop_idx[1] = loop_index;
      }
      loop_index++;
    }
  }

  float scale_val = get_tensor_scalar_scale(tensor);
  DataTypes::bfloat16 scale = scale_val != 0 ? scale_val : 1.0;
  vector_params->vector_fetch_1_dq_scale = scale.bits_rep();
}

void set_vector_fetch_2(const codegen::Tensor& tensor, const Tiling& tiling,
                        VectorParams* vector_params) {
  int nonzero_dims = 0;
  for (const int& dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  const auto memory = tensor.memory();
  vector_params->vector_fetch_2_offset = get_address(tensor);
  vector_params->vector_fetch_2_mode = true;
  vector_params->vector_fetch_2_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

  vector_params->vector_fetch_2_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype());
  const int dtype_width =
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_2_dtype);
  int fetch_width = OC_DIMENSION * dtype_width;
  vector_params->vector_fetch_2_burst_size = fetch_width / 8;
  vector_params->vector_fetch_2_num_beats =
      (fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
  vector_params->vector_fetch_2_packing_factor =
      OC_DIMENSION / VECTOR_UNIT_WIDTH;

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_2_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->vector_fetch_2_x_loop_idx[0] = tiling.x_loop_idx[0];
  vector_params->vector_fetch_2_y_loop_idx[0] = tiling.y_loop_idx[0];
  vector_params->vector_fetch_2_k_loop_idx[0] = tiling.weight_loop_idx[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_idx[1] || i == tiling.x_loop_idx[1] ||
        i == tiling.y_loop_idx[1]) {
      vector_params->vector_fetch_2_loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_idx[1]) {
        vector_params->vector_fetch_2_x_loop_idx[1] = loop_index;
      }
      if (i == tiling.y_loop_idx[1]) {
        vector_params->vector_fetch_2_y_loop_idx[1] = loop_index;
      }
      if (i == tiling.weight_loop_idx[1]) {
        vector_params->vector_fetch_2_k_loop_idx[1] = loop_index;
      }
      loop_index++;
    }
  }

  float scale_val = get_tensor_scalar_scale(tensor);
  DataTypes::bfloat16 scale = scale_val != 0 ? scale_val : 1.0;
  vector_params->vector_fetch_2_dq_scale = scale.bits_rep();
}

void set_immediate(const float scalar, const int stage,
                   const std::string opcode, VectorInstructions& inst) {
  VECTOR_DATATYPE immediate = scalar;

  if (opcode == "div" || opcode == "div_") {
    immediate = 1.0 / immediate;
  }

  if (stage == 0) {
    inst.vector_op0_src1 = VectorInstructions::from_immediate_0;
    inst.immediate0 = immediate.bits_rep();
  } else if (stage == 2) {
    inst.vector_op2_src1 = VectorInstructions::from_immediate_1;
    inst.immediate1 = immediate.bits_rep();
  } else {
    inst.vector_op3_src1 = VectorInstructions::from_immediate_2;
    inst.immediate2 = immediate.bits_rep();
  }
}

/**
 * \brief Determine whether we should use the direct path between the matrix
 * unit and the vector unit.
 *
 * Even if we have a double-buffered accumulation buffer, it may still be
 * profitable to use the direct path. This way, we save on the latency of having
 * to fully fill the accumulation buffer before even starting to drain it.
 *
 * However, the vector unit may not be able to keep up with the matrix unit. The
 * matrix unit produces `OC_DIMENSION` elements per cycle. If the inputs take
 * more than one cycle to fetch that many elements, the vector unit (and then
 * the matrix unit) will stall. The same happens if we can't write that many
 * elements to the output per cycle.
 *
 * This function determines whether it is profitable to use the direct path for
 * this particular instruction, given the accelerator's configuration. It should
 * only be called if we have a double-buffered accumulation buffer.
 */
static bool should_use_direct_path(const VectorParams* vector_params) {
  assert(DOUBLE_BUFFERED_ACCUM_BUFFER);

  // This is how much bandwidth we have available, in bits per cycle. It
  // happens that this value is the same for both inputs and the output.
  //
  // If we want to use the direct path without stalling, it had better be the
  // case that the available bandwidth exceeds the required bandwidth of
  // producing `OC_DIMENSION` elements per cycle.
  const size_t available_bandwidth = OC_PORT_WIDTH;

  // How much bandwidth we need for each of the ports, in bits per cycle. If the
  // address generation is inactive, we don't need any bandwidth. Otherwise, it
  // depends on the type we're fetching/writing.
  //
  // Remember, we need `OC_DIMENSION` elements per cycle on each (active) port
  // to keep up with the matrix unit.
  const size_t vector_fetch_1_bw =
      (vector_params->vector_fetch_1_mode == 0)
          ? 0
          : get_type_width<VU_INPUT_TYPES>(
                vector_params->vector_fetch_1_dtype) *
                OC_DIMENSION;
  const size_t vector_fetch_2_bw =
      (vector_params->vector_fetch_2_mode == 0)
          ? 0
          : get_type_width<VU_INPUT_TYPES>(
                vector_params->vector_fetch_2_dtype) *
                OC_DIMENSION;
  const size_t output_bw =
      (vector_params->output_mode == 0)
          ? 0
          : get_type_width<OUTPUT_DATATYPES>(vector_params->output_dtype) *
                OC_DIMENSION;

  bool should_use_direct_path = true;
  should_use_direct_path &= vector_fetch_1_bw <= available_bandwidth;
  should_use_direct_path &= vector_fetch_2_bw <= available_bandwidth;
  should_use_direct_path &= output_bw <= available_bandwidth;

  return should_use_direct_path;
}

void map_matrix_operation(const Operation& operation,
                          std::deque<BaseParams*>& mapped_params) {
  MatrixParams* matrix_params;
  DwCParams* dwc_params;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto matrix_op = op_list[0];

  const auto input = matrix_op.kwargs().at("input").tensor();

  bool is_matmul = matrix_op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight = matrix_op.kwargs().at(weight_key).tensor();

  bool is_mx_op = matrix_op.target().find("mx") != std::string::npos;
  bool is_fc = is_fc_layer(matrix_op);
  bool is_dwc = false;

  if (matrix_op.target().find("conv2d") != std::string::npos &&
      matrix_op.kwargs().at("groups").int_value() > 1) {
#if SUPPORT_DWC
    is_dwc = true;
#else
    throw std::runtime_error("DWC not supported in this build");
#endif
  }

  Tiling tiling;

  if (is_dwc) {
    dwc_params = new DwCParams;

    const auto& kwargs = matrix_op.kwargs();
    const auto& bias = kwargs.at("bias").tensor();
    const auto& output = get_op_outputs(param).back();

    dwc_params->input_offset = get_address(input);
    dwc_params->weight_offset = get_address(weight);
    dwc_params->bias_offset = get_address(bias);
    dwc_params->output_offset = get_address(output);

    int Y = input.shape(1);
    int X = input.shape(2);
    int C = input.shape(3);

    dwc_params->bounds[0] = Y;
    dwc_params->bounds[1] = X;
    dwc_params->bounds[2] = C;

    if (is_mx_op) {
      int block_size = kwargs.at("block_size").int_value();
      assert(block_size % UNROLLFACTOR == 0);

      dwc_params->use_mx = 1;
      dwc_params->block_size = log2(block_size);
      assert(1 << dwc_params->block_size == block_size);

      const auto& input_scale = kwargs.at("input_scale").tensor();
      const auto& weight_scale = kwargs.at("weight_scale").tensor();
      dwc_params->input_scale_offset = get_address(input_scale);
      dwc_params->weight_scale_offset = get_address(weight_scale);
    }

    const auto paddings = kwargs.at("padding").int_list().values();
    int x_pad = paddings[1];
    int y_pad = paddings[0];

    const int stride = kwargs.at("stride").int_list().values()[0];
    dwc_params->stride = stride;
    assert(stride < 7);

    int padded_Y = Y + 2 * y_pad - 2;
    int padded_X = X + 2 * x_pad - 2;

    assert(padded_Y % stride == 0);
    assert(padded_X % stride == 0);

    int X0 = ((DWC_WIDTH - 2) / stride) * stride;
    int X1 = (input.shape(2) + x_pad + x_pad - 2 + X0 - 1) /
             X0;  // Padding lines, asym in future
    int C1 = (input.shape(3) + UNROLLFACTOR - 1) / UNROLLFACTOR;

    dwc_params->loops[0][0] = Y;
    dwc_params->loops[0][1] = X1;
    dwc_params->loops[0][2] = C1;

    dwc_params->loops[1][0] = 0;  // unused
    dwc_params->loops[1][1] = X0;
    dwc_params->loops[1][2] = UNROLLFACTOR;

    dwc_params->outloops[0] = padded_Y / stride;
    dwc_params->outloops[1] = padded_X / stride;
    dwc_params->outloops[2] = X0 / stride;

    dwc_params->padding[0][0] = y_pad;
    dwc_params->padding[0][1] = y_pad;
    dwc_params->padding[1][0] = x_pad;
    dwc_params->padding[1][1] = x_pad;

    dwc_params->fast_forward_mode = (X + 2 * x_pad - (X1 - 1) * X0) == 3;

    tiling = {
        .loops = {{output.shape(1), output.shape(2),
                   output.shape(3) / UNROLLFACTOR, 1, 1, 1},
                  {1, 1, 1, 1, 1, 1}},
        .x_loop_idx = {0, 0},
        .y_loop_idx = {0, 1},
        .reduction_loop_idx = {0, 0},
        .weight_loop_idx = {0, 2},
        .fx_loop_idx = 0,
        .fy_loop_idx = {0, 0},
        .weight_reuse_idx = {0, 0},
        .stride = 1,
    };
  } else {
    matrix_params = new MatrixParams;

    if (is_fc) {
#if !SUPPORT_MVM
      throw std::runtime_error(
          "Matrix-vector multiply not supported in this build.");
#endif
      matrix_params->is_fc = true;

      auto weight_shape = get_shape(weight);
      int K = weight_shape[0];
      int C = weight_shape[1];
      int C1 = is_mx_op ? matrix_op.kwargs().at("block_size").int_value() : 1;
      int C2 = C / C1;

      auto k_loops = split_loops({K}, MAX_LOOP_VALUE);
      k_loops = adjust_loop_indices(k_loops, OC_DIMENSION);
      pad_shape_to_ndim(k_loops, 2);

      tiling = {
          .loops = {{1, 1, k_loops[0], C2, 1, 1}, {C1, k_loops[1], 1, 1, 1, 1}},
          .x_loop_idx = {0, 5},
          .y_loop_idx = {1, 4},
          .reduction_loop_idx = {3, 0},
          .weight_loop_idx = {2, 1},
          .fx_loop_idx = 3,
          .fy_loop_idx = {4, 2},
          .weight_reuse_idx = {4, 5},
          .stride = 1,
          .resnet_replication = false,
          .generic_replication = false,
      };
    } else {
      tiling = get_tiling(operation);
    }

    std::ostringstream oss;
    oss << tiling;
    spdlog::info("Using tiling: \n{}\n", oss.str());

    // Set input fields
    matrix_params->input_offset = get_address(input);
    matrix_params->input_dtype =
        get_index_from_type_name<INPUT_DATATYPE>(input.dtype());
    matrix_params->use_input_codebook =
        matrix_op.kwargs().contains("input_code");

    if (matrix_params->use_input_codebook) {
      const auto code = matrix_op.kwargs().at("input_code").tensor();
      const auto size = get_size(code);

      float* input_code = read_constant_param(code);

      int zero_idx = -1;
      for (int i = 0; i < size; i++) {
        if (input_code[i] == 0.0) {
          zero_idx = i;
        }

        SA_INPUT_TYPE value = input_code[i];
        matrix_params->input_code[i] = value.bits_rep();
      }

      if (zero_idx == -1) {
        spdlog::warn(
            "Input codebook does not contain a zero entry. Using "
            "index 0 as the zero entry.");
        zero_idx = 0;
      }
      matrix_params->input_code_zero_idx = zero_idx;

      delete[] input_code;
    }

    int input_fetch_width;
    int input_num_packs;

    if (is_fc) {
      input_num_packs =
          get_packing_factor<MV_UNIT_WIDTH, OC_PORT_WIDTH, INPUT_DATATYPE>(
              matrix_params->input_dtype, 1, input_fetch_width);
    } else {
      int c_bound = tiling.resnet_replication
                        ? 1
                        : tiling.loops[1][tiling.reduction_loop_idx[1]];
      input_num_packs =
          get_packing_factor<IC_DIMENSION, IC_PORT_WIDTH, INPUT_DATATYPE>(
              matrix_params->input_dtype, c_bound, input_fetch_width);
    }

    matrix_params->input_burst_size = input_fetch_width / 8;
    matrix_params->input_num_beats =
        (input_fetch_width + IC_PORT_WIDTH - 1) / IC_PORT_WIDTH;
    matrix_params->input_pack_factor_lg2 = std::log2(input_num_packs);

    // Set weight fields
    matrix_params->weight_offset = get_address(weight);
    matrix_params->weight_dtype =
        get_index_from_type_name<WEIGHT_DATATYPE>(weight.dtype());
    matrix_params->use_weight_codebook =
        matrix_op.kwargs().contains("weight_code");

    if (matrix_params->use_weight_codebook) {
      const auto code = matrix_op.kwargs().at("weight_code").tensor();
      const auto size = get_size(code);

      float* weight_code = read_constant_param(code);
      for (int i = 0; i < size; i++) {
        SA_WEIGHT_TYPE value = weight_code[i];
        matrix_params->weight_code[i] = value.bits_rep();
      }

      delete[] weight_code;
    }

    int weight_fetch_width;
    int weight_num_packs;

    if (is_fc) {
      weight_num_packs =
          get_packing_factor<MV_UNIT_WIDTH, OC_PORT_WIDTH, WEIGHT_DATATYPE>(
              matrix_params->weight_dtype, 1, weight_fetch_width);
    } else {
      int k_bound = matrix_params->weight_transpose
                        ? 1
                        : tiling.loops[1][tiling.weight_loop_idx[1]];
      weight_num_packs =
          get_packing_factor<OC_DIMENSION, OC_PORT_WIDTH, WEIGHT_DATATYPE>(
              matrix_params->weight_dtype, k_bound, weight_fetch_width);
    }

    matrix_params->weight_burst_size = weight_fetch_width / 8;
    matrix_params->weight_num_beats =
        (weight_fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
    matrix_params->weight_pack_factor_lg2 = std::log2(weight_num_packs);

    // Set microscaling fields
    matrix_params->is_mx_op = is_mx_op;

    if (is_mx_op) {
      const int block_size = matrix_op.kwargs().at("block_size").int_value();
      assert(block_size == std::max(IC_DIMENSION, OC_DIMENSION));

      const auto input_scale = matrix_op.kwargs().at("input_scale").tensor();
      matrix_params->input_scale_offset = get_address(input_scale);

      const auto weight_scale = matrix_op.kwargs().at("weight_scale").tensor();
      matrix_params->weight_scale_offset = get_address(weight_scale);
    }

    // Set loop bounds
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        matrix_params->loops[i][j] = tiling.loops[i][j];
      }
      matrix_params->x_loop_idx[i] = tiling.x_loop_idx[i];
      matrix_params->y_loop_idx[i] = tiling.y_loop_idx[i];
      matrix_params->reduction_loop_idx[i] = tiling.reduction_loop_idx[i];
      matrix_params->weight_loop_idx[i] = tiling.weight_loop_idx[i];
      matrix_params->weight_reuse_idx[i] = tiling.weight_reuse_idx[i];
      matrix_params->fy_loop_idx[i] = tiling.fy_loop_idx[i];
    }
    matrix_params->fx_loop_idx = tiling.fx_loop_idx;
    matrix_params->stride = tiling.stride;
    matrix_params->padding = tiling.padding;

    // Set weight address generation fields
    for (int j = 0; j < 5; j++) {
      matrix_params->weight_addr_loops[0][j] = tiling.loops[0][j];
    }
    matrix_params->weight_addr_weight_loop_idx[0] = tiling.weight_loop_idx[0];
    matrix_params->weight_addr_reduction_loop_idx[0] =
        tiling.reduction_loop_idx[0];
    matrix_params->weight_addr_fy_idx[0] = tiling.fy_loop_idx[0];

    // if OX and OY loops are the innermost L2 loops, they are irrelevant for
    // weight address generation
    if (tiling.loops[0][tiling.reduction_loop_idx[0]] == 1) {
      if (tiling.weight_loop_idx[0] < tiling.x_loop_idx[0]) {
        matrix_params->weight_addr_loops[0][tiling.x_loop_idx[0]] = 1;
      }
      if (tiling.weight_loop_idx[0] < tiling.y_loop_idx[0]) {
        matrix_params->weight_addr_loops[0][tiling.y_loop_idx[0]] = 1;
      }
    }

    matrix_params->weight_transpose = weight.reshape().target() == "transpose";

    if (matrix_params->weight_transpose) {
      // for transpose, we need to enforce that the innermost loop is the
      // unrolled reduction loop we can just use the following loop nest: C1, K,
      // FY, FX, C0

      // C0 loop
      matrix_params->weight_addr_loops[1][4] = OC_DIMENSION;
      matrix_params->weight_addr_reduction_loop_idx[2] = 4;

      // FX loop
      matrix_params->weight_addr_loops[1][3] =
          tiling.loops[1][tiling.fx_loop_idx];
      matrix_params->weight_addr_fx_idx = 3;

      // FY loop
      matrix_params->weight_addr_loops[1][2] =
          tiling.loops[1][tiling.fy_loop_idx[1]];
      matrix_params->weight_addr_fy_idx[1] = 2;

      // K loop
      matrix_params->weight_addr_loops[1][1] =
          tiling.loops[1][tiling.weight_loop_idx[1]];
      matrix_params->weight_addr_weight_loop_idx[1] = 1;

      // C1 loop
      matrix_params->weight_addr_loops[1][0] =
          tiling.loops[1][tiling.reduction_loop_idx[1]];
      if (OC_DIMENSION > IC_DIMENSION) {
        // we can reduce the number of iterations, since we have already fetched
        // the values
        if (tiling.loops[0][tiling.reduction_loop_idx[0]] >=
            (OC_DIMENSION / IC_DIMENSION)) {
          matrix_params->weight_addr_loops[0][tiling.reduction_loop_idx[0]] =
              tiling.loops[0][tiling.reduction_loop_idx[0]] /
              (OC_DIMENSION / IC_DIMENSION);
        } else {
          matrix_params->weight_addr_loops[1][tiling.reduction_loop_idx[1]] =
              tiling.loops[1][tiling.reduction_loop_idx[1]] /
              (OC_DIMENSION / IC_DIMENSION);
        }
      }
      matrix_params->weight_addr_reduction_loop_idx[1] = 0;
    } else {  // if not transpose, then we have freedom to pick any loop order
      // K1 loop
      matrix_params->weight_addr_loops[1][4] =
          tiling.loops[1][tiling.weight_loop_idx[1]];
      matrix_params->weight_addr_weight_loop_idx[1] = 4;

      // C0 loop
      if (tiling.resnet_replication) {
        matrix_params->weight_addr_loops[1][3] = 3;
      } else if (tiling.generic_replication) {
        matrix_params->weight_addr_loops[1][3] = tiling.num_channels;
      } else {
        matrix_params->weight_addr_loops[1][3] = IC_DIMENSION;
      }
      matrix_params->weight_addr_reduction_loop_idx[2] = 3;

      // C1 loop
      matrix_params->weight_addr_loops[1][2] =
          tiling.loops[1][tiling.reduction_loop_idx[1]];
      matrix_params->weight_addr_reduction_loop_idx[1] = 2;

      // FX loop
      matrix_params->weight_addr_loops[1][1] =
          tiling.loops[1][tiling.fx_loop_idx];
      if (tiling.resnet_replication) {
        matrix_params->weight_addr_loops[1][1] = 7;
      } else if (tiling.generic_replication) {
        matrix_params->weight_addr_loops[1][1] *= tiling.fx_unrolling;
      }
      matrix_params->weight_addr_fx_idx = 1;

      // FY loop
      matrix_params->weight_addr_loops[1][0] =
          tiling.loops[1][tiling.fy_loop_idx[1]];
      matrix_params->weight_addr_fy_idx[1] = 0;
    }

    matrix_params->is_resnet_replication = tiling.resnet_replication;
    matrix_params->is_generic_replication = tiling.generic_replication;
    matrix_params->num_channels = tiling.num_channels;
    matrix_params->fx_unrolling_lg2 = std::log2(tiling.fx_unrolling);

    matrix_params->input_x = tiling.input_x;
    matrix_params->input_y = tiling.input_y;

    // Set input transpose
    if (input.has_reshape()) {
      const auto reshape_op = input.reshape();
      const auto reshape_kwargs = reshape_op.kwargs();

      if (reshape_op.target() == "permute") {
        const auto int_list = reshape_kwargs.at("dims").int_list().values();
        std::vector<int> dims(int_list.begin(), int_list.end());
        const int ndim = input.shape_size();

        if (is_transpose(dims)) {
          matrix_params->input_transpose = true;
        } else if (dims[ndim - 1] == ndim - 1) {
          matrix_params->merge_heads = true;
        } else {
          throw std::invalid_argument("Unsupported permute operation!");
        }
      } else if (reshape_op.target() == "transpose") {
        int dim0 = reshape_kwargs.at("dim0").int_value();
        int dim1 = reshape_kwargs.at("dim1").int_value();
        if (dim0 > dim1) {
          std::swap(dim0, dim1);
        }

        if (dim0 == input.shape_size() - 2 && dim1 == input.shape_size() - 1) {
          matrix_params->input_transpose = true;
        } else if (dim1 != input.shape_size() - 1) {
          matrix_params->merge_heads = true;
        } else {
          throw std::invalid_argument("Unsupported transpose operation!");
        }
      }

      if (matrix_params->merge_heads) {
        const auto input_shape = input.shape();
        double result = std::log2(input_shape[input_shape.size() - 1]);
        if (std::fmod(result, 1.0) != 0.0) {
          throw std::runtime_error("Result is not an integer!");
        }
        matrix_params->head_size_lg2 = result;
      }
    }

    // Set bias
    if (matrix_op.kwargs().contains("bias")) {
      const auto bias = matrix_op.kwargs().at("bias").tensor();
      matrix_params->has_bias = true;
      matrix_params->bias_offset = get_address(bias);
    }

    // Set weight dequantize parameters
    if (weight.has_dequant()) {
      matrix_params->weight_dequant = true;
      const auto dequant = weight.dequant();
      const auto scale = dequant.kwargs().at("scale").tensor();
      matrix_params->dq_scale_offset = get_address(scale);
      if (dequant.kwargs().contains("zero_point")) {
        const auto zero_point = dequant.kwargs().at("zero_point").tensor();
        matrix_params->dq_zero_point_offset = get_address(zero_point);
      }
    }
  }

  // vector instructions
  VectorParams* vector_params = new VectorParams;

  // Rescale tiling for vector instructions
  if (is_fc) {
    tiling.loops[1][1] /= OC_DIMENSION;
  }

  if (!is_dwc && !is_fc) {
    vector_params->vector_fetch_0_mode = 3;  // read from accumulation buffer
    // Set outer loops
    for (int i = 0; i < 3; i++) {
      vector_params->vector_fetch_0_loops[0][i] = tiling.loops[0][i];
    }
    vector_params->vector_fetch_0_y_loop_idx[0] = tiling.y_loop_idx[0];
    vector_params->vector_fetch_0_x_loop_idx[0] = tiling.x_loop_idx[0];
    vector_params->vector_fetch_0_k_loop_idx[0] = tiling.weight_loop_idx[0];

    // Set inner loops
    int loop_idx = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == tiling.weight_loop_idx[1] || i == tiling.x_loop_idx[1] ||
          i == tiling.y_loop_idx[1]) {
        vector_params->vector_fetch_0_loops[1][loop_idx] = tiling.loops[1][i];
        if (i == tiling.y_loop_idx[1]) {
          vector_params->vector_fetch_0_y_loop_idx[1] = loop_idx;
        }
        if (i == tiling.x_loop_idx[1]) {
          vector_params->vector_fetch_0_x_loop_idx[1] = loop_idx;
        }
        if (i == tiling.weight_loop_idx[1]) {
          vector_params->vector_fetch_0_k_loop_idx[1] = loop_idx;
        }
        loop_idx++;
      }
    }
  }

  const auto output = get_op_outputs(param).back();
  vector_params->vector_output_offset = get_address(output);

  // Set outer loops
  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->output_y_loop_idx[0] = tiling.y_loop_idx[0];
  vector_params->output_x_loop_idx[0] = tiling.x_loop_idx[0];
  vector_params->output_k_loop_idx[0] = tiling.weight_loop_idx[0];

  // Set inner loops
  int output_loop_idx = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_idx[1] || i == tiling.x_loop_idx[1] ||
        i == tiling.y_loop_idx[1]) {
      vector_params->output_loops[1][output_loop_idx] = tiling.loops[1][i];
      if (i == tiling.y_loop_idx[1]) {
        vector_params->output_y_loop_idx[1] = output_loop_idx;
      }
      if (i == tiling.x_loop_idx[1]) {
        vector_params->output_x_loop_idx[1] = output_loop_idx;
      }
      if (i == tiling.weight_loop_idx[1]) {
        vector_params->output_k_loop_idx[1] = output_loop_idx;
      }
      output_loop_idx++;
    }
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  // Transformer head permutation
  if (output.has_reshape()) {
    vector_params->transpose_for_scores = true;
    const auto permuted_shape =
        output.reshape().kwargs().at("output_shape").int_list().values();
    double result = std::log2(permuted_shape[permuted_shape.size() - 1]);
    if (std::fmod(result, 1.0) != 0.0) {
      throw std::runtime_error("Result is not an integer!");
    }
    vector_params->head_size_lg2 = result;
  }

  vector_params->is_dwc = is_dwc;

  const int packing_factor = OC_DIMENSION / VECTOR_UNIT_WIDTH;

  VectorInstructions inst;
  inst.op_type = VectorInstructions::vector;
  inst.inst_loop_count = tiling.loops[0][tiling.x_loop_idx[0]] *
                         tiling.loops[1][tiling.x_loop_idx[1]] *
                         tiling.loops[0][tiling.y_loop_idx[0]] *
                         tiling.loops[1][tiling.y_loop_idx[1]] *
                         tiling.loops[0][tiling.weight_loop_idx[0]] *
                         tiling.loops[1][tiling.weight_loop_idx[1]] *
                         packing_factor;

  if (is_dwc) {
    inst.inst_loop_count =
        tiling.loops[0][0] * tiling.loops[0][1] * tiling.loops[0][2];
    inst.vector_op0_src0 = VectorInstructions::from_dwc_unit;
  } else if (is_fc) {
    inst.vector_op0_src0 = VectorInstructions::from_matrix_vector_unit;
  } else {
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    inst.vector_op0_src0 = VectorInstructions::from_accumulation_buffer;
    matrix_params->write_output_to_accum_buffer = true;
#else
    inst.vector_op0_src0 = VectorInstructions::from_matrix_unit;
#endif
  }

  inst.vdest = VectorInstructions::to_output;

  auto mapping = get_vector_instruction_mapping();
  int stage = 0;

  for (int i = 1; i < op_list.size(); i++) {
    const auto op = op_list[i];
    const std::string opcode = op.target();

    // Grab kwargs that are relevant for some activation functions
    std::map<std::string, VECTOR_DATATYPE> activation_kwargs;
    if (unary_ops_with_kwargs.find(op.target()) !=
        unary_ops_with_kwargs.end()) {
      for (const auto& [key, value] : op.kwargs()) {
        if (key != "input") {
          activation_kwargs[key] = VECTOR_DATATYPE(value.float_value());
        }
      }
    }

    // Dequantization doesn't take a stage in the pipeline
    if (opcode == "dequantize") {
      inst.vdequantize = true;

      const auto other = op.kwargs().at("scale").tensor();
      assert(get_size(other) == 1);

      float* array = read_constant_param(other);
      VECTOR_DATATYPE immediate = array[0];
      inst.vector_dq_scale = immediate.bits_rep();

      delete[] array;

      continue;
    }

    // Find the stage of the operation
    for (; stage < vector_unit_stages.size(); stage++) {
      // Only the last stage has a true divider
      if (opcode == "div" && stage != 3) {
        const auto other = op.kwargs().at("other").tensor();
        if (get_size(other) > 1) {
          continue;
        }
      }

      if (vector_unit_stages[stage].find(opcode) !=
          vector_unit_stages[stage].end()) {
        break;
      }
    }

    if (stage == vector_unit_stages.size()) {
      throw std::runtime_error("Vector operation not supported!\n");
    }

    spdlog::debug("stage {} target: {}\n", stage, opcode);

    unsigned int vop = mapping[opcode];
    if (stage == 0) {
      inst.vector_op0 = opcode == "div" ? VectorInstructions::vmult : vop;
    } else if (stage == 1) {
      inst.vector_op1 = vop;
    } else if (stage == 2) {
      inst.vector_op2 = opcode == "div" ? VectorInstructions::vmult : vop;
    } else if (stage == 3) {
      inst.vector_op3 = vop;
    }

    if (opcode == "quantize_mx" || opcode == "quantize_mx_outlier") {
      float quant_max = op.kwargs().at("quant_max").float_value();
      bool force_scale_power_of_two =
          op.kwargs().at("force_scale_power_of_two").bool_value();

      if (force_scale_power_of_two) {
        inst.immediate2 = floor(log2(quant_max));
      } else {
        VECTOR_DATATYPE scale = quant_max;
        inst.immediate2 = scale.bits_rep();
      }

      const auto outputs = get_op_outputs(param);
      const int num_outputs = outputs.size();

      vector_params->quantize_output_mx = true;
      vector_params->mx_scale_offset = get_address(outputs[num_outputs - 2]);

      if (opcode == "quantize_mx_outlier") {
        vector_params->has_sparse_output = true;
        vector_params->csr_data_offset = get_address(outputs[0]);
        vector_params->csr_indices_offset = get_address(outputs[1]);
        vector_params->csr_indptr_offset = get_address(outputs[2]);

        auto& config = vector_instruction_config->outlier_filter;

        VECTOR_DATATYPE threshold = op.kwargs().at("threshold").float_value();
        config.outlier_threshold = threshold.bits_rep();

        const auto quantize_input = op.kwargs().at("input").tensor();
        const auto quantize_shape = get_shape(quantize_input);
        config.dense_input_shape[1] = quantize_shape.back() / VECTOR_UNIT_WIDTH;
        config.dense_input_shape[0] =
            get_size(quantize_input) / quantize_shape.back();
      }

      if (op.kwargs().contains("output_code")) {
        const auto code = op.kwargs().at("output_code").tensor();
        const int size = get_size(code);

        float* array = read_constant_param(code);

        for (int i = 0; i < size; i++) {
          vector_params->output_code[i] = array[i] * 2;
        }

        delete[] array;

        vector_params->use_output_codebook = true;
      }
      // Copy coefficients from ApproximationConstants.h
    } else if (opcode == "gelu" || opcode == "gelu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = GELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = GELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = GELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = GELU_CLAMP_MAX;
    } else if (opcode == "silu" || opcode == "silu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SILU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = SILU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SILU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SILU_CLAMP_MAX;
    } else if (opcode == "elu" || opcode == "elu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = ELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = ELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = ELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = ELU_CLAMP_MAX;
    } else if (opcode == "log_sigmoid" || opcode == "log_sigmoid_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = LOGSIGMOID_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              LOGSIGMOID_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = LOGSIGMOID_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = LOGSIGMOID_CLAMP_MAX;
    } else if (opcode == "tanh" || opcode == "tanh_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = TANH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = TANH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = TANH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = TANH_CLAMP_MAX;
    } else if (opcode == "tanh_1" || opcode == "tanh_1_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = TANHSHRINK_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              TANHSHRINK_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = TANHSHRINK_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = TANHSHRINK_CLAMP_MAX;
    } else if (opcode == "softplus" || opcode == "softplus_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SOFTPLUS_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              SOFTPLUS_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SOFTPLUS_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SOFTPLUS_CLAMP_MAX;
    } else if (opcode == "mish" || opcode == "mish_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = MISH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = MISH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = MISH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = MISH_CLAMP_MAX;
    } else if (opcode == "selu" || opcode == "selu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = SELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SELU_CLAMP_MAX;
    } else if (opcode == "celu" || opcode == "celu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = CELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = CELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = CELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = CELU_CLAMP_MAX;
    } else if (opcode == "sigmoid" || opcode == "sigmoid_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SIGMOID_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = SIGMOID_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SIGMOID_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SIGMOID_CLAMP_MAX;
    } else if (opcode == "hardsigmoid" || opcode == "hardsigmoid_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDSIGMOID_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDSIGMOID_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDSIGMOID_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDSIGMOID_CLAMP_MAX;
    } else if (opcode == "hardswish" || opcode == "hardswish_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDSWISH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDSWISH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDSWISH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDSWISH_CLAMP_MAX;

      // Start of activation functions with kwargs
    } else if (opcode == "hardshrink" || opcode == "hardshrink_") {
      const VECTOR_DATATYPE lambda = activation_kwargs.at("lambd");
      const VECTOR_DATATYPE HARDSHRINK_MAXES[NUM_MAXES] = {
          -lambda, 0.0, 0.0, 0.0, 0.0, lambda};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDSHRINK_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDSHRINK_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDSHRINK_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDSHRINK_CLAMP_MAX;
    } else if (opcode == "hardtanh" || opcode == "hardtanh_") {
      const VECTOR_DATATYPE min_val = activation_kwargs.at("min_val");
      const VECTOR_DATATYPE max_val = activation_kwargs.at("max_val");
      const VECTOR_DATATYPE HARDTANH_MAXES[NUM_MAXES] = {
          min_val, max_val, max_val, max_val, max_val, max_val};
      const VECTOR_DATATYPE HARDTANH_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {min_val, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
          {0.0, 1.0, 0.0},     {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
          {max_val, 0.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDTANH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDTANH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDTANH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDTANH_CLAMP_MAX;
    } else if (opcode == "leaky_relu" || opcode == "leaky_relu_") {
      const VECTOR_DATATYPE negative_slope =
          activation_kwargs.at("negative_slope");
      const VECTOR_DATATYPE LEAKYRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {0.0, negative_slope, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = LEAKYRELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              LEAKYRELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = LEAKYRELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = LEAKYRELU_CLAMP_MAX;
    } else if (opcode == "rrelu" || opcode == "rrelu_") {
      const VECTOR_DATATYPE lower = activation_kwargs.at("lower");
      const VECTOR_DATATYPE upper = activation_kwargs.at("upper");
      const VECTOR_DATATYPE alpha = (lower + upper) / VECTOR_DATATYPE(2);
      const VECTOR_DATATYPE RRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {0.0, alpha, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},   {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = RRELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = RRELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = RRELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = RRELU_CLAMP_MAX;
    } else if (opcode == "softshrink" || opcode == "softshrink_") {
      const VECTOR_DATATYPE lambda = activation_kwargs.at("lambd");
      const VECTOR_DATATYPE SOFTSHRINK_MAXES[NUM_MAXES] = {
          -lambda, 0.0, 0.0, 0.0, 0.0, lambda};
      const VECTOR_DATATYPE SOFTSHRINK_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {lambda, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},    {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {-lambda, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SOFTSHRINK_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              SOFTSHRINK_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SOFTSHRINK_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SOFTSHRINK_CLAMP_MAX;
    } else if (opcode == "threshold" || opcode == "threshold_") {
      const VECTOR_DATATYPE threshold = activation_kwargs.at("threshold");
      const VECTOR_DATATYPE value = activation_kwargs.at("value");
      const VECTOR_DATATYPE THRESHOLD_MAXES[NUM_MAXES] = {0.0, 0.0, 0.0,
                                                          0.0, 0.0, threshold};
      const VECTOR_DATATYPE THRESHOLD_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
          {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
          {0.0, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = THRESHOLD_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              THRESHOLD_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = THRESHOLD_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = THRESHOLD_CLAMP_MAX;
    } else if (op.kwargs().contains("other") || opcode == "quantize") {
      std::string other_key = opcode == "quantize" ? "scale" : "other";
      const auto other = op.kwargs().at(other_key);

      if (other.has_float_value()) {
        float scalar = other.float_value();
        set_immediate(scalar, stage, opcode, inst);
      } else if (other.has_int_value()) {
        int scalar = other.int_value();
        set_immediate(scalar, stage, opcode, inst);
      } else if (other.has_tensor() && get_size(other.tensor()) == 1) {
        float* array = read_constant_param(other.tensor());
        set_immediate(array[0], stage, opcode, inst);
        delete[] array;
      } else {
        auto self = op.kwargs().at("input").tensor();
        auto tensor = other.tensor();
        auto tensor_to_load = tensor.has_memory() ? tensor : self;

        if (stage == 0) {
          inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
          set_vector_fetch_1(tensor_to_load, tiling, vector_params);
        } else if (stage == 2) {
          inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
          set_vector_fetch_2(tensor_to_load, tiling, vector_params);
        } else {
          assert(inst.vector_op2_src1 !=
                 VectorInstructions::from_vector_fetch_2);
          inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
          set_vector_fetch_2(tensor_to_load, tiling, vector_params);
        }
      }
    }

    stage++;
  }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  if (!is_dwc && should_use_direct_path(vector_params)) {
    inst.vector_op0_src0 = VectorInstructions::from_matrix_unit;
    vector_params->vector_fetch_0_mode = 0;
    matrix_params->write_output_to_accum_buffer = false;
  }
#endif

  // total output count
  vector_instruction_config->inst[0] = inst;
  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  if (is_dwc) {
    mapped_params.push_back(dwc_params);
  } else {
    mapped_params.push_back(matrix_params);
  }

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
