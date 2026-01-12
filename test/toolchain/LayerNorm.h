#pragma once

#include "test/toolchain/Common.h"

void map_layer_norm(const codegen::Operation& param,
                    std::deque<BaseParams*>& mapped_params) {
  const auto op_list = get_op_list(param);
  const auto layer_norm_op = op_list[0];

  const auto input = layer_norm_op.kwargs().at("input").tensor();

  codegen::Tensor output;

  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  VectorParams* vector_params;
  VectorInstructionConfig* vector_instruction_config;

  const auto input_shape = squeeze_shape(get_shape(input));
  const int input_size = get_size(input);
  const int reduction_dim = input_shape.back();

  std::vector<int> non_reduction_loops;
  for (int i = 0; i < input_shape.size() - 1; i++) {
    non_reduction_loops.push_back(input_shape[i]);
  }

  non_reduction_loops = split_loops(non_reduction_loops, MAX_LOOP_VALUE);
  pad_shape_to_ndim(non_reduction_loops, 2);
  const int reduced_size = get_size(non_reduction_loops);

  const int packing_factor = OC_DIMENSION / VECTOR_UNIT_WIDTH;

  int input_dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());
  int input_type_width = get_type_width<VU_INPUT_TYPES>(input_dtype);

  int vector_dtype = get_type_index<VECTOR_DATATYPE, VU_INPUT_TYPES>();
  int vector_type_width = get_type_width<VU_INPUT_TYPES>(vector_dtype);

  const int mean_scratch_address =
      get_address(input) + input_size * input_type_width / 8;
  const int var_scratch_address =
      mean_scratch_address +
      reduced_size * OC_DIMENSION * vector_type_width / 8;
  const int intermediate_address =
      var_scratch_address + reduced_size * OC_DIMENSION * vector_type_width / 8;

  // ======================================================================
  // Pass 1: compute mean
  // ======================================================================

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = input_dtype;
  set_dequantize_scale(input, vector_params);

  int vector_fetch_0_input_width = OC_DIMENSION * input_type_width;
  vector_params->vector_fetch_0_burst_size = vector_fetch_0_input_width / 8;
  vector_params->vector_fetch_0_num_beats =
      vector_fetch_0_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  vector_params->vector_output_offset = mean_scratch_address;
  vector_params->output_mode = 2;
  vector_params->output_dtype =
      get_type_index<VECTOR_DATATYPE, OUTPUT_DATATYPES>();

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = reduced_size;

  // Configure reduction engine
  VectorInstructions inst0_0;
  inst0_0.op_type = VectorInstructions::reduction;
  inst0_0.inst_loop_count = reduced_size;
  inst0_0.reduce_count = reduction_dim / OC_DIMENSION * packing_factor;
  inst0_0.reduce_op = VectorInstructions::radd;
  inst0_0.rduplicate = 1;
  inst0_0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst0_0;

  // Scale inputs by 1 / norm_size and send to the reduction engine
  VectorInstructions inst0_1;
  inst0_1.op_type = VectorInstructions::vector;
  inst0_1.inst_loop_count = input_size / OC_DIMENSION * packing_factor;
  inst0_1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst0_1.vector_op0_src1 = VectorInstructions::from_immediate_0;
  inst0_1.vector_op0 = VectorInstructions::vmult;
  VECTOR_DATATYPE immediate = 1.0 / reduction_dim;
  inst0_1.immediate0 = immediate.bits_rep();
  inst0_1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = inst0_1;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);

  // ======================================================================
  // Pass 2: compute variance
  // ======================================================================

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = input_dtype;
  set_dequantize_scale(input, vector_params);

  vector_params->vector_fetch_0_burst_size = vector_fetch_0_input_width / 8;
  vector_params->vector_fetch_0_num_beats =
      vector_fetch_0_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 1: mean
  vector_params->vector_fetch_1_offset = mean_scratch_address;
  vector_params->vector_fetch_1_mode = 1;
  vector_params->vector_fetch_1_dtype = vector_dtype;

  int vector_fetch_1_input_width = OC_DIMENSION * vector_type_width;
  vector_params->vector_fetch_1_burst_size = vector_fetch_1_input_width / 8;
  vector_params->vector_fetch_1_num_beats =
      vector_fetch_1_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = packing_factor;

  vector_params->vector_fetch_1_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_1_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  vector_params->vector_output_offset = var_scratch_address;
  vector_params->output_mode = 2;
  vector_params->output_dtype =
      get_type_index<VECTOR_DATATYPE, OUTPUT_DATATYPES>();

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = reduced_size;

  // Configure reduction unit
  VectorInstructions inst1_0;
  inst1_0.op_type = VectorInstructions::reduction;
  inst1_0.inst_loop_count = reduced_size;
  inst1_0.reduce_count = reduction_dim / OC_DIMENSION * packing_factor;
  inst1_0.reduce_op = VectorInstructions::radd;
  inst1_0.rsqrt = 1;
  inst1_0.rreciprocal = 1;
  inst1_0.rscale = 1;
  VECTOR_DATATYPE divisor = sqrt(reduction_dim);
  inst1_0.immediate1 = divisor.bits_rep();
  inst1_0.rduplicate = 1;
  inst1_0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst1_0;

  VectorInstructions inst1_1;
  inst1_1.op_type = VectorInstructions::vector;
  inst1_1.inst_loop_count = input_size / OC_DIMENSION * packing_factor;
  inst1_1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst1_1.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst1_1.vector_op0 = VectorInstructions::vsub;
  inst1_1.vector_op2 = VectorInstructions::vsquare;
  inst1_1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = inst1_1;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);

  // ======================================================================
  // Pass 3: subtract mean and divide by variance
  // ======================================================================

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = input_dtype;
  set_dequantize_scale(input, vector_params);

  vector_params->vector_fetch_0_burst_size = vector_fetch_0_input_width / 8;
  vector_params->vector_fetch_0_num_beats =
      vector_fetch_0_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 1: mean
  vector_params->vector_fetch_1_offset = mean_scratch_address;
  vector_params->vector_fetch_1_mode = 1;
  vector_params->vector_fetch_1_dtype = vector_dtype;

  vector_params->vector_fetch_1_burst_size = vector_fetch_1_input_width / 8;
  vector_params->vector_fetch_1_num_beats =
      vector_fetch_1_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = packing_factor;

  vector_params->vector_fetch_1_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_1_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 2: variance
  vector_params->vector_fetch_2_offset = var_scratch_address;
  vector_params->vector_fetch_2_mode = 1;
  vector_params->vector_fetch_2_dtype = vector_dtype;

  vector_params->vector_fetch_2_burst_size = vector_fetch_1_input_width / 8;
  vector_params->vector_fetch_2_num_beats =
      vector_fetch_1_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_2_packing_factor = packing_factor;

  vector_params->vector_fetch_2_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_2_loops[0][i] = 1;
  }
  vector_params->vector_fetch_2_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_2_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_2_loops[1][2] = reduction_dim / OC_DIMENSION;

  vector_params->vector_output_offset = intermediate_address;
  vector_params->output_mode = 2;
  vector_params->output_dtype =
      get_type_index<VECTOR_DATATYPE, OUTPUT_DATATYPES>();

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = non_reduction_loops[0];
  vector_params->output_loops[1][1] = non_reduction_loops[1];
  vector_params->output_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Multiply inputs with the inverse sqrt of the variance
  VectorInstructions inst2;
  inst2.op_type = VectorInstructions::vector;
  inst2.inst_loop_count = input_size / OC_DIMENSION * packing_factor;
  inst2.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst2.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst2.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
  inst2.vector_op0 = VectorInstructions::vsub;
  inst2.vector_op2 = VectorInstructions::vmult;
  inst2.vdest = VectorInstructions::to_output;
  vector_instruction_config->inst[0] = inst2;

  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);

  // ======================================================================
  // Pass 4: perform affine transformation
  // ======================================================================

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  // Fetch inputs
  vector_params->vector_fetch_0_offset = intermediate_address;
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = vector_dtype;

  vector_fetch_0_input_width = OC_DIMENSION * vector_type_width;
  vector_params->vector_fetch_0_burst_size = vector_fetch_0_input_width / 8;
  vector_params->vector_fetch_0_num_beats =
      vector_fetch_0_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Fetch weights
  const auto weight = layer_norm_op.kwargs().at("weight").tensor();
  vector_params->vector_fetch_1_offset = get_address(weight);
  vector_params->vector_fetch_1_mode = true;
  vector_params->vector_fetch_1_broadcast = 0b011;
  vector_params->vector_fetch_1_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(weight.dtype());

  vector_fetch_1_input_width =
      OC_DIMENSION *
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_1_dtype);
  vector_params->vector_fetch_1_burst_size = vector_fetch_1_input_width / 8;
  vector_params->vector_fetch_1_num_beats =
      vector_fetch_1_input_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_1_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Fetch bias
  const bool has_bias = layer_norm_op.kwargs().contains("bias");
  if (has_bias) {
    const auto bias = layer_norm_op.kwargs().at("bias").tensor();
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
    vector_params->vector_fetch_2_loops[1][0] = non_reduction_loops[0];
    vector_params->vector_fetch_2_loops[1][1] = non_reduction_loops[1];
    vector_params->vector_fetch_2_loops[1][2] = reduction_dim / OC_DIMENSION;
  }

  vector_params->vector_output_offset = get_address(output);
  vector_params->output_mode = 2;
  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = non_reduction_loops[0];
  vector_params->output_loops[1][1] = non_reduction_loops[1];
  vector_params->output_loops[1][2] = reduction_dim / OC_DIMENSION;

  // inputs x weights + bias
  VectorInstructions inst3;
  inst3.op_type = VectorInstructions::vector;
  inst3.inst_loop_count = get_size(output) / OC_DIMENSION * packing_factor;
  inst3.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst3.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst3.vector_op0 = VectorInstructions::vmult;
  if (has_bias) {
    inst3.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
    inst3.vector_op2 = VectorInstructions::vadd;
  }
  inst3.vdest = VectorInstructions::to_output;

  set_quantize_params(param, vector_params, inst3, vector_instruction_config);

  vector_instruction_config->inst[0] = inst3;

  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
