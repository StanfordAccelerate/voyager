#pragma once

#include "test/toolchain/Common.h"

void map_microscaling(const codegen::Operation& param,
                      std::deque<BaseParams*>& mapped_params) {
  const auto op_list = get_op_list(param);
  const auto quantize_mx_op = op_list[0];

  const auto input = quantize_mx_op.kwargs().at("input").tensor();
  const int axis = quantize_mx_op.kwargs().at("axes").int_list().values()[0];
  const int block_size = quantize_mx_op.kwargs().at("block_size").int_value();
  const float quant_max = quantize_mx_op.kwargs().at("quant_max").float_value();
  const bool force_scale_power_of_two =
      quantize_mx_op.kwargs().at("force_scale_power_of_two").bool_value();

  codegen::Tensor output;
  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  // Microscaling on the last dimension can be handled by vector operations
  auto input_shape = get_shape(input);
  if (axis != -2 && axis != input_shape.size() - 2) {
    throw std::invalid_argument(
        "Microscaling only supported on second to last dimension");
  }

  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 1;
  vector_params->vector_fetch_0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  int input_width =
      VECTOR_UNIT_WIDTH *
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_0_dtype);
  vector_params->vector_fetch_0_burst_size = input_width / 8;
  vector_params->vector_fetch_0_num_beats = input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = 1;

  input_shape = squeeze_shape(input_shape);
  input_shape = split_loops(input_shape, MAX_LOOP_VALUE);
  pad_shape_to_ndim(input_shape, 3);

  vector_params->vector_fetch_0_loops[0][0] = input_shape[0];
  vector_params->vector_fetch_0_loops[0][1] = input_shape[1] / block_size;
  vector_params->vector_fetch_0_loops[0][2] =
      input_shape[2] / VECTOR_UNIT_WIDTH;
  vector_params->vector_fetch_0_loops[1][0] = 1;
  vector_params->vector_fetch_0_loops[1][1] = block_size;
  vector_params->vector_fetch_0_loops[1][2] = 1;

  for (int i = 0; i < 2; i++) {
    vector_params->vector_fetch_0_y_loop_idx[i] = 0;
    vector_params->vector_fetch_0_x_loop_idx[i] = 1;
    vector_params->vector_fetch_0_k_loop_idx[i] = 2;
  }

  vector_params->vector_output_offset = get_address(output);
  vector_params->output_mode = 1;

  vector_params->output_loops[0][0] = input_shape[0];
  vector_params->output_loops[0][1] = input_shape[1] / block_size;
  vector_params->output_loops[0][2] = input_shape[2] / OC_DIMENSION;
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = 1;

  for (int i = 0; i < 2; i++) {
    vector_params->output_y_loop_idx[i] = 0;
    vector_params->output_x_loop_idx[i] = 1;
    vector_params->output_k_loop_idx[i] = 2;
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  // perform max accumulation
  VectorInstructions vinst0;
  vinst0.op_type = VectorInstructions::accumulation;
  vinst0.inst_loop_count = get_size(input) / block_size / VECTOR_UNIT_WIDTH;
  vinst0.reduce_op = VectorInstructions::rmax;
  vinst0.reduce_count = block_size;
  vinst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = vinst0;

  // feed accumulator
  VectorInstructions vinst1;
  vinst1.op_type = VectorInstructions::vector;
  vinst1.inst_loop_count = get_size(input) / VECTOR_UNIT_WIDTH;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vector_op0_src1 = VectorInstructions::from_immediate_0;
  VECTOR_DATATYPE immediate = force_scale_power_of_two
                                  ? 1.0 / pow(2, floor(log2(quant_max)))
                                  : 1.0 / quant_max;
  vinst1.immediate0 = immediate.bits_rep();
  vinst1.vector_op0 = VectorInstructions::vmult;
  vinst1.vector_op1 = VectorInstructions::vabs;
  vinst1.vdest = VectorInstructions::to_accumulate;
  vector_instruction_config->inst[1] = vinst1;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
