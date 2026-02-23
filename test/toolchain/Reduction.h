#pragma once

#include "test/toolchain/Common.h"

void map_reducer_op(const codegen::Operation& param,
                    std::deque<BaseParams*>& mapped_params) {
  const auto op_list = get_op_list(param);
  const auto reduction_op = op_list[0];

  const auto kwargs = reduction_op.kwargs();
  const auto input = kwargs.at("input").tensor();
  const auto output = get_op_outputs(param).back();

  auto input_shape = get_shape(input);
  const int rcount = input_shape.back() / REDUCER_WIDTH;

  input_shape = {get_size(input_shape) / OC_DIMENSION};
  input_shape = split_loops(input_shape, 1024);
  pad_shape_to_ndim(input_shape, 3);

  auto output_shape = get_shape(output);
  output_shape = {get_size(output_shape) / OC_DIMENSION};
  output_shape = split_loops(output_shape, 1024);
  pad_shape_to_ndim(output_shape, 3);

  const auto target = reduction_op.target();

  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  int input_dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());
  int input_dtype_width = get_type_width<VU_INPUT_TYPES>(input_dtype);
  int input_fetch_width = OC_DIMENSION * input_dtype_width;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 1;
  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = 1;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }

  vector_params->vector_fetch_0_loops[1][0] = input_shape[0];
  vector_params->vector_fetch_0_loops[1][1] = input_shape[1];
  vector_params->vector_fetch_0_loops[1][2] = input_shape[2];

  for (int i = 0; i < 2; i++) {
    vector_params->vector_fetch_0_y_loop_idx[i] = 0;
    vector_params->vector_fetch_0_x_loop_idx[i] = 1;
    vector_params->vector_fetch_0_k_loop_idx[i] = 2;
  }

  vector_params->vector_output_offset = get_address(output);
  vector_params->output_mode = 1;

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }

  vector_params->output_loops[1][0] = output_shape[0];
  vector_params->output_loops[1][1] = output_shape[1];
  vector_params->output_loops[1][2] = output_shape[2];

  for (int i = 0; i < 2; i++) {
    vector_params->output_y_loop_idx[i] = 0;
    vector_params->output_x_loop_idx[i] = 1;
    vector_params->output_k_loop_idx[i] = 2;
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  VectorInstructions inst0;
  inst0.op_type = VectorInstructions::reduction;
  inst0.inst_loop_count = get_size(output) / REDUCER_WIDTH;
  if (target == "amax") {
    inst0.reduce_op = VectorInstructions::rmax;
  } else if (target == "sum") {
    inst0.reduce_op = VectorInstructions::radd;
  } else {
    assert(false && "Unsupported reduction operation!");
  }
  inst0.reduce_count = rcount;
  inst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst0;

  VectorInstructions inst1;
  inst1.op_type = VectorInstructions::vector;
  inst1.inst_loop_count = get_size(input) / OC_DIMENSION;
  inst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = inst1;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}

void map_reduction(const codegen::Operation& param,
                   std::deque<BaseParams*>& mapped_params) {
  const auto op_list = get_op_list(param);
  const auto reduction_op = op_list[0];
  const auto dim = reduction_op.kwargs().at("dim").int_list().values();
  assert(dim.size() == 1 &&
         "Only single dimension reduction is supported in this function!");

  const auto input = reduction_op.kwargs().at("input").tensor();
  const auto input_shape = get_shape(input);

  int reduction_dim = dim[0];
  if (reduction_dim < 0) {
    reduction_dim += input_shape.size();
  }

  if (reduction_dim == input_shape.size() - 1) {
    map_reducer_op(param, mapped_params);
  } else {
    spdlog::error(
        "Only reduction on the last dimension is supported in this function!");
    std::abort();
  }
}
