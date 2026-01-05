#pragma once

#include "test/toolchain/Common.h"

void map_matrix_vector_multiply(const codegen::Operation& param,
                                std::deque<BaseParams*>& mapped_params) {
  const auto op_list = get_op_list(param);
  const auto matrix_op = op_list[0];

  const auto input = matrix_op.kwargs().at("input").tensor();
  bool is_matmul = matrix_op.target().find("matmul") != std::string::npos;
  std::string key = is_matmul ? "other" : "weight";
  const auto weight = matrix_op.kwargs().at(key).tensor();
  bool has_bias = matrix_op.kwargs().contains("bias");

  codegen::Tensor output;
  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  const auto weight_shape = get_shape(weight);
  int output_dim = weight_shape[0];
  int reduction_dim = weight_shape[1];

  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  // round output_dim up to a multiple of DIMENSION
  output_dim = (output_dim + VECTOR_UNIT_WIDTH - 1) / VECTOR_UNIT_WIDTH *
               VECTOR_UNIT_WIDTH;

  // input is a vector of size reduction_dim
  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_broadcast = 0b010000;
  vector_params->vector_fetch_0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  int packing_factor = OC_DIMENSION / VECTOR_UNIT_WIDTH;
  int vector_fetch_0_input_width =
      OC_DIMENSION *
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_0_dtype);
  vector_params->vector_fetch_0_burst_size = vector_fetch_0_input_width / 8;
  vector_params->vector_fetch_0_num_beats =
      vector_fetch_0_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = 1;
  vector_params->vector_fetch_0_loops[1][1] = output_dim;
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // weight is a matrix of output_dim x reduction_dim
  vector_params->vector_fetch_1_offset = get_address(weight);
  vector_params->vector_fetch_1_mode = true;
  vector_params->vector_fetch_1_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(weight.dtype());

  int vector_fetch_1_input_width =
      OC_DIMENSION *
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_1_dtype);
  vector_params->vector_fetch_1_burst_size = vector_fetch_1_input_width / 8;
  vector_params->vector_fetch_1_num_beats =
      vector_fetch_1_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = 1;
  vector_params->vector_fetch_1_loops[1][1] = output_dim;
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->vector_fetch_1_y_loop_idx[i] = 0;
    vector_params->vector_fetch_1_x_loop_idx[i] = 1;
    vector_params->vector_fetch_1_k_loop_idx[i] = 2;
  }

  // bias
  if (has_bias) {
    const auto bias = matrix_op.kwargs().at("bias").tensor();
    vector_params->vector_fetch_2_offset = get_address(bias);
    vector_params->vector_fetch_2_mode = true;
    vector_params->vector_fetch_2_broadcast = 0b011;
    vector_params->vector_fetch_2_dtype =
        get_index_from_type_name<VU_INPUT_TYPES>(bias.dtype());

    int vector_fetch_2_input_width =
        OC_DIMENSION *
        get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_2_dtype);
    vector_params->vector_fetch_2_burst_size = vector_fetch_2_input_width / 8;
    vector_params->vector_fetch_2_num_beats =
        vector_fetch_2_input_width / OC_PORT_WIDTH;
    vector_params->vector_fetch_2_packing_factor = packing_factor;

    for (int i = 0; i < 3; i++) {
      vector_params->vector_fetch_2_loops[0][i] = 1;
    }
    vector_params->vector_fetch_2_loops[1][0] = 1;
    vector_params->vector_fetch_2_loops[1][1] = 1;
    vector_params->vector_fetch_2_loops[1][2] = output_dim / OC_DIMENSION;

    for (int i = 0; i < 2; i++) {
      vector_params->vector_fetch_2_y_loop_idx[i] = 0;
      vector_params->vector_fetch_2_x_loop_idx[i] = 1;
      vector_params->vector_fetch_2_k_loop_idx[i] = 2;
    }
  }

  // output
  vector_params->vector_output_offset = get_address(output);
  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = output_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->output_y_loop_idx[i] = 0;
    vector_params->output_x_loop_idx[i] = 1;
    vector_params->output_k_loop_idx[i] = 2;
  }

  // inst0 - start reduction engine
  VectorInstructions vinst0;
  vinst0.op_type = VectorInstructions::reduction;
  vinst0.reduce_count = reduction_dim / VECTOR_UNIT_WIDTH;
  vinst0.reduce_op = VectorInstructions::radd;
  vinst0.rdest =
      has_bias ? VectorInstructions::to_op0 : VectorInstructions::to_memory;
  vinst0.immediate0 = 1;
  vector_instruction_config->inst[0] = vinst0;

  // inst1 - input x weight, send to reduce
  // reduction_dim / DIMENSION to do the complete reduction, DIMENSION to fill
  // up the entire vector (this is now output_dim dimension)
  VectorInstructions vinst1;
  vinst1.op_type = VectorInstructions::vector;
  vinst1.inst_loop_count = reduction_dim;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  vinst1.vector_op0 = VectorInstructions::vmult;
  vinst1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = vinst1;

  // inst2 - add bias, write out
  if (has_bias) {
    VectorInstructions vinst2;
    vinst2.op_type = VectorInstructions::vector;
    vinst2.vector_op0_src0 = VectorInstructions::from_reduction_0;
    vinst2.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
    vinst2.vector_op2 = VectorInstructions::vadd;
    vinst2.vdest = VectorInstructions::to_output;
    vector_instruction_config->inst[2] = vinst2;
  }

  vector_instruction_config->num_inst = has_bias ? 3 : 2;
  vector_instruction_config->config_loop_count = output_dim / VECTOR_UNIT_WIDTH;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
