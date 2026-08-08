#pragma once

#include "spdlog/spdlog.h"
#include "src/Params.h"
#include "test/common/GoldModel.h"
#include "test/common/GraphUtils.h"
#include "test/common/MemoryInterface.h"
#include "test/common/Tiling.h"
#include "test/common/Utils.h"
#include "test/toolchain/ApproximationConstants.h"
#include "test/toolchain/Common.h"
#if SUPPORT_SPMM
#include "test/toolchain/SpMM.h"
#endif

void set_vector_fetch_1(const Tensor& tensor, const Tiling& tiling,
                        VectorParams* vector_params) {
  int nonzero_dims = 0;
  for (const int& dim : tensor.shape) {
    if (dim != 1) nonzero_dims++;
  }

  vector_params->vector_fetch_1_offset = get_address(tensor);
  vector_params->vector_fetch_1_mode = true;
  vector_params->vector_fetch_1_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

  const int dtype = get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype);
  const int dtype_width = get_type_width<VU_INPUT_TYPES>(dtype);
  const int fetch_width = OC_DIMENSION * dtype_width;

  vector_params->vector_fetch_1_dtype = dtype;
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
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
}

void set_vector_fetch_2(const Tensor& tensor, const Tiling& tiling,
                        VectorParams* vector_params) {
  int nonzero_dims = 0;
  for (const int& dim : tensor.shape) {
    if (dim != 1) nonzero_dims++;
  }

  vector_params->vector_fetch_2_offset = get_address(tensor);
  vector_params->vector_fetch_2_mode = true;
  vector_params->vector_fetch_2_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

  const int dtype = get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype);
  const int dtype_width = get_type_width<VU_INPUT_TYPES>(dtype);
  const int fetch_width = OC_DIMENSION * dtype_width;

  vector_params->vector_fetch_2_dtype = dtype;
  vector_params->vector_fetch_2_stride = OC_DIMENSION;
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

  return vector_fetch_1_bw <= available_bandwidth &&
         vector_fetch_2_bw <= available_bandwidth &&
         output_bw <= available_bandwidth;
}

void map_matrix_operation(const voyager::Operation& operation,
                          const ScalarEnv& env,
                          std::deque<BaseParams*>& mapped_params) {
  MatrixParams* matrix_params;
  DwCParams* dwc_params;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  const auto& op_list = get_prim_ops(operation);
  const auto& matrix_op = get_anchor_op(operation);

  const auto input = resolve(matrix_op, "input", env);
  const auto output = resolve_outputs(operation, env).back();

  bool is_matmul = matrix_op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const Tensor anchor_weight = resolve(matrix_op, weight_key, env);
  const voyager::PrimOp* weight_producer =
      get_fused_producer(operation, env, anchor_weight);
  const voyager::PrimOp* weight_dequantize_op =
      weight_producer != nullptr &&
              strip_namespace(weight_producer->target()) == "dequantize"
          ? weight_producer
          : nullptr;
  const Tensor weight = weight_dequantize_op == nullptr
                            ? anchor_weight
                            : resolve(*weight_dequantize_op, "input", env);

  bool is_mx_op = matrix_op.target().find("mx") != std::string::npos;
  bool is_fc = is_fc_layer(matrix_op);
  bool is_dwc = false;

  if (matrix_op.target().find("conv2d") != std::string::npos &&
      arg_int(matrix_op, "groups", env) > 1) {
#if SUPPORT_DWC
    is_dwc = true;
#else
    throw std::runtime_error("DWC not supported in this build");
#endif
  }

  Tiling tiling;

  if (is_dwc) {
    dwc_params = new DwCParams;
    const auto bias = resolve(matrix_op, "bias", env);
    const auto output = resolve_outputs(operation, env).back();

    dwc_params->input_offset = get_address(input);
    dwc_params->weight_offset = get_address(weight);
    dwc_params->bias_offset = get_address(bias);
    dwc_params->output_offset = get_address(output);

    int Y = input.shape[1];
    int X = input.shape[2];
    int C = input.shape[3];

    dwc_params->bounds[0] = Y;
    dwc_params->bounds[1] = X;
    dwc_params->bounds[2] = C;

    if (is_mx_op) {
      int block_size = arg_int(matrix_op, "block_size", env);
      assert(block_size % UNROLLFACTOR == 0);

      dwc_params->use_mx = 1;
      dwc_params->block_size = log2(block_size);
      assert(1 << dwc_params->block_size == block_size);

      const auto& input_scale = resolve(matrix_op, "input_scale", env);
      const auto& weight_scale = resolve(matrix_op, "weight_scale", env);
      dwc_params->input_scale_offset = get_address(input_scale);
      dwc_params->weight_scale_offset = get_address(weight_scale);
    }

    const auto paddings = arg_ints(matrix_op, "padding", env);
    int x_pad = paddings[1];
    int y_pad = paddings[0];

    const int stride = arg_ints(matrix_op, "stride", env)[0];
    dwc_params->stride = stride;
    assert(stride < 7);

    int padded_Y = Y + 2 * y_pad - 2;
    int padded_X = X + 2 * x_pad - 2;

    assert(padded_Y % stride == 0);
    assert(padded_X % stride == 0);

    int X0 = ((DWC_WIDTH - 2) / stride) * stride;
    int X1 = (input.shape[2] + x_pad + x_pad - 2 + X0 - 1) /
             X0;  // Padding lines, asym in future
    int C1 = (input.shape[3] + UNROLLFACTOR - 1) / UNROLLFACTOR;

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
        .loops = {{output.shape[1], output.shape[2],
                   output.shape[3] / UNROLLFACTOR, 1, 1, 1},
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
      if (weight_shape.size() < 2) {
        throw std::runtime_error(
            "MVM weight must have at least two dimensions.");
      }
      int K = weight_shape[weight_shape.size() - 2];
      int C = weight_shape.back();
      int C1 = is_mx_op ? arg_int(matrix_op, "block_size", env) : 1;
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
      tiling = get_tiling(operation, env);
    }

    std::ostringstream oss;
    oss << tiling;
    spdlog::info("Using tiling: \n{}\n", oss.str());

    // Set input fields
    matrix_params->input_offset = get_address(input);
    matrix_params->input_dtype =
        get_index_from_type_name<INPUT_DATATYPE>(input.dtype);
    matrix_params->use_input_codebook = has_arg(matrix_op, "input_code");

    if (matrix_params->use_input_codebook) {
      const auto code = resolve(matrix_op, "input_code", env);

      int size;
      float* input_code = read_constant_param(code, &size);

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
        get_index_from_type_name<WEIGHT_DATATYPE>(weight.dtype);
    matrix_params->use_weight_codebook = has_arg(matrix_op, "weight_code");

    if (matrix_params->use_weight_codebook) {
      const auto code = resolve(matrix_op, "weight_code", env);

      int size;
      float* weight_code = read_constant_param(code, &size);
      for (int i = 0; i < size; i++) {
        SA_WEIGHT_TYPE value = weight_code[i];
        matrix_params->weight_code[i] = value.bits_rep();
      }

      delete[] weight_code;
    }

    if (weight_dequantize_op != nullptr) {
      matrix_params->weight_dequant = true;
      matrix_params->dq_scale_offset =
          get_address(resolve(*weight_dequantize_op, "scale", env));
      if (has_arg(*weight_dequantize_op, "zero_point")) {
        matrix_params->dq_zero_point_offset =
            get_address(resolve(*weight_dequantize_op, "zero_point", env));
      }
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
      const int block_size = arg_int(matrix_op, "block_size", env);
      assert(block_size == std::max(IC_DIMENSION, OC_DIMENSION));

      const auto input_scale = resolve(matrix_op, "input_scale", env);
      matrix_params->input_scale_offset = get_address(input_scale);

      const auto weight_scale = resolve(matrix_op, "weight_scale", env);
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

    // We switch to use DMA to load transposed weight instead of transposing
    // inside the accelerator.
    matrix_params->weight_transpose = false;

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

    // Set bias
    if (has_arg(matrix_op, "bias")) {
      const auto bias = resolve(matrix_op, "bias", env);
      matrix_params->has_bias = true;
      matrix_params->bias_offset = get_address(bias);
    }
  }

  // If there are no vector operations, we don't need to setup the vector
  // instruction config
  if (!is_dwc && !is_fc && !has_fused_spmm(matrix_op) && op_list.size() == 1) {
    matrix_params->output_to_memory = true;
    matrix_params->output_offset = get_address(output);
    matrix_params->output_dtype =
        get_index_from_type_name<MU_OUTPUT_TYPES>(output.dtype);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    const size_t output_bw =
        get_type_width<MU_OUTPUT_TYPES>(matrix_params->output_dtype) *
        OC_DIMENSION;
    matrix_params->write_output_to_accum_buffer = OC_PORT_WIDTH < output_bw;
#endif
    mapped_params.push_back(matrix_params);
    return;
  }

  // fused spmm operation
  if (has_fused_spmm(matrix_op)) {
#if !SUPPORT_SPMM
    throw std::runtime_error(
        "Sparse matrix operations not supported in this build.");
#else
    map_spmm(operation, env, mapped_params, true);
#endif
  }

  // vector instructions
  VectorParams* vector_params = new VectorParams;

  // Rescale tiling for vector instructions
  if (is_fc) {
    tiling.loops[1][1] /= OC_DIMENSION;
  }

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
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype);

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
  inst.vdest = VectorInstructions::to_output;

  if (is_dwc) {
    inst.inst_loop_count =
        tiling.loops[0][0] * tiling.loops[0][1] * tiling.loops[0][2];
    inst.vector_op0_src0 = VectorInstructions::from_dwc_unit;
  } else if (is_fc) {
    inst.vector_op0_src0 = VectorInstructions::from_matrix_vector_unit;
  } else {
    inst.vector_op0_src0 = VectorInstructions::from_matrix_unit;
    if (has_fused_spmm(matrix_op)) {
      inst.vector_op0_src1 = VectorInstructions::from_spmm_unit;
      inst.vector_op0 = VectorInstructions::op0_add;
    }
  }

  std::vector<const voyager::PrimOp*> vector_ops;
  bool after_anchor = false;
  for (const voyager::PrimOp* op : op_list) {
    if (!after_anchor) {
      if (op->name() == matrix_op.name()) after_anchor = true;
      continue;
    }

    // A fused head-split transpose is realized by the output controller's
    // address relayout, not by a pipeline stage.
    if (strip_namespace(op->target()) == "transpose") {
      const Tensor& transpose_input = resolve(*op, "input", env);
      const int ndim = get_shape(transpose_input).size();
      int dim0 = arg_int(*op, "dim0", env);
      int dim1 = arg_int(*op, "dim1", env);
      if (dim0 < 0) dim0 += ndim;
      if (dim1 < 0) dim1 += ndim;
      if (std::max(dim0, dim1) == ndim - 1) {
        throw std::runtime_error(
            "Fused transpose of the last dimension cannot be mapped to the "
            "output relayout.");
      }

      const int head_size = get_shape(transpose_input).back();
      const double head_size_lg2 = std::log2(head_size);
      if (std::fmod(head_size_lg2, 1.0) != 0.0) {
        throw std::runtime_error("Head size is not a power of two.");
      }

      vector_params->transpose_for_scores = true;
      vector_params->head_size_lg2 = head_size_lg2;
      continue;
    }

    vector_ops.push_back(op);
  }

  auto map_tensor_operand = [&](const voyager::PrimOp& op, const ScalarEnv& env,
                                const std::string& other_key, int stage,
                                VectorInstructions& pipeline_inst) {
    const Tensor& self = resolve(op, "input", env);
    const Tensor& tensor = resolve(op, other_key, env);

    // An operand a dequantize produced inside this fusion is not itself
    // materialized, so fetch what that dequantize reads; the shared pipeline
    // mapper applies the dequantize scale through the stage mac.
    const voyager::PrimOp* producer =
        get_fused_producer(operation, env, tensor);
    const bool from_dequantize =
        producer != nullptr &&
        strip_namespace(producer->target()) == "dequantize";
    const Tensor tensor_to_load = from_dequantize
                                      ? resolve(*producer, "input", env)
                                      : (tensor.materialized ? tensor : self);

    if (stage == 0) {
      pipeline_inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
      set_vector_fetch_1(tensor_to_load, tiling, vector_params);
    } else if (stage == 2) {
      pipeline_inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
      set_vector_fetch_2(tensor_to_load, tiling, vector_params);
    } else {
      if (pipeline_inst.vector_op2_src1 ==
          VectorInstructions::from_vector_fetch_2) {
        throw std::runtime_error(
            "Vector pipeline stages 2 and 3 cannot fetch different side "
            "operands.");
      }
      pipeline_inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
      set_vector_fetch_2(tensor_to_load, tiling, vector_params);
    }
  };

  auto is_stage_available = [&](const voyager::PrimOp& op, const ScalarEnv& env,
                                int stage) {
    return !((strip_namespace(op.target()) == "add" ||
              strip_namespace(op.target()) == "add_") &&
             stage == 0 && has_fused_spmm(matrix_op));
  };

  map_vector_pipeline_ops(operation, env, vector_ops, vector_params,
                          vector_instruction_config, inst, mapped_params,
                          map_tensor_operand, is_stage_available);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  if (!is_dwc && !is_fc) {
    matrix_params->write_output_to_accum_buffer =
        !should_use_direct_path(vector_params);
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
