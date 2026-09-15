#pragma once

#include "test/common/Tiling.h"
#include "test/toolchain/Common.h"

void map_pool2d(const voyager::Operation& operation, const ScalarEnv& env,
                std::deque<BaseParams*>& mapped_params) {
  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  const auto& op_list = get_prim_ops(operation);
  const auto& pooling_op = get_anchor_op(operation);

  const auto input = resolve(pooling_op, "input", env);
  const auto output = resolve_outputs(operation, env).back();

  const auto output_shape = get_shape(output);
  const int output_dim = output_shape[3];

  const auto tiling = get_pool2d_tiling(pooling_op, env);

  // input
  int input_dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype);
  int input_dtype_width = get_type_width<VU_INPUT_TYPES>(input_dtype);
  int input_fetch_width = ACCUMULATOR_WIDTH * input_dtype_width;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 1;
  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = ACCUMULATOR_WIDTH;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = 1;

  for (int i = 0; i < 2; i++) {
    vector_params->vector_fetch_0_loops[i][0] =
        tiling.loops[i][tiling.weight_loop_idx[i]];
    vector_params->vector_fetch_0_loops[i][1] =
        tiling.loops[i][tiling.y_loop_idx[i]];
    vector_params->vector_fetch_0_loops[i][2] =
        tiling.loops[i][tiling.x_loop_idx[i]];

    vector_params->vector_fetch_0_x_loop_idx[i] = 2;
    vector_params->vector_fetch_0_y_loop_idx[i] = 1;
    vector_params->vector_fetch_0_k_loop_idx[i] = 0;
  }

  // striding only applied to x and y dimensions
  vector_params->stride[1] = tiling.stride;    // x
  vector_params->stride[0] = tiling.stride;    // y
  vector_params->padding[1] = tiling.padding;  // x
  vector_params->padding[0] = tiling.padding;  // y

  // output
  vector_params->vector_output_offset = get_address(output);

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = tiling.loops[0][tiling.y_loop_idx[0]];
  vector_params->output_loops[1][1] = tiling.loops[0][tiling.x_loop_idx[0]];
  vector_params->output_loops[1][2] = output_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->output_y_loop_idx[i] = 0;
    vector_params->output_x_loop_idx[i] = 1;
    vector_params->output_k_loop_idx[i] = 2;
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype);

  const int reduce_count = tiling.loops[1][tiling.y_loop_idx[1]] *
                           tiling.loops[1][tiling.x_loop_idx[1]];

  const int inst_loop_count = tiling.loops[0][tiling.y_loop_idx[0]] *
                              tiling.loops[0][tiling.x_loop_idx[0]] *
                              output_dim / ACCUMULATOR_WIDTH;

  bool is_max_pool = pooling_op.target().find("max") != std::string::npos;
  vector_params->is_maxpool = is_max_pool;

  // perform max/sum accumulation
  VectorInstructions inst0;
  inst0.op_type = VectorInstructions::accumulation;
  inst0.inst_loop_count = inst_loop_count;
  inst0.reduce_count = reduce_count;
  inst0.reduce_op =
      is_max_pool ? VectorInstructions::rmax : VectorInstructions::radd;
  inst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst0;

  // feed accumulator
  VectorInstructions inst1;
  inst1.op_type = VectorInstructions::vector;
  inst1.inst_loop_count = inst_loop_count * reduce_count;
  inst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst1.vdest = VectorInstructions::to_accumulate;

  if (!is_max_pool) {
    inst1.vector_op2 = VectorInstructions::op2_mul;
    inst1.vector_op2_src1 = VectorInstructions::from_immediate_1;
    int kernel_size = tiling.loops[1][tiling.x_loop_idx[1]];
    VECTOR_DATATYPE scale = 1.0 / (kernel_size * kernel_size);
    inst1.immediate1 = scale.bits_rep();
  }

  vector_instruction_config->inst[1] = inst1;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
