#pragma once

#include "test/toolchain/Common.h"

void map_layer_norm(const voyager::Operation& operation, const ScalarEnv& env,
                    std::deque<BaseParams*>& mapped_params) {
  const auto& op_list = get_prim_ops(operation);
  const auto& layer_norm_op = get_anchor_op(operation);

  const Tensor& anchor_input = resolve(layer_norm_op, "input", env);
  const voyager::PrimOp* input_producer =
      get_fused_producer(operation, env, anchor_input);
  const voyager::PrimOp* dequantize_op =
      input_producer != nullptr &&
              strip_namespace(input_producer->target()) == "dequantize"
          ? input_producer
          : nullptr;
  const Tensor& input = dequantize_op == nullptr
                            ? anchor_input
                            : resolve(*dequantize_op, "input", env);
  const auto output = resolve_outputs(operation, env).back();

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

  constexpr int vu_unit_ratio = OC_DIMENSION / VECTOR_UNIT_WIDTH;

  int input_dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype);
  int input_dtype_width = get_type_width<VU_INPUT_TYPES>(input_dtype);
  int input_fetch_width = OC_DIMENSION * input_dtype_width;

  int vector_dtype = get_type_index<VECTOR_DATATYPE, VU_INPUT_TYPES>();
  int vector_dtype_width = get_type_width<VU_INPUT_TYPES>(vector_dtype);
  int vector_fetch_width = OC_DIMENSION * vector_dtype_width;

  // The compiler reserves the scratchpad the four passes stage their
  // intermediates in, so these are read off the operation. Deriving them from
  // the input tile instead would land inside the input's own second bank.
  const int mean_scratch_address =
      get_address(resolve(layer_norm_op, "mean", env));
  const int var_scratch_address =
      get_address(resolve(layer_norm_op, "variance", env));
  const int intermediate_address =
      get_address(resolve(layer_norm_op, "normalized", env));

  // ======================================================================
  // Pass 1: compute mean
  // ======================================================================

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = vu_unit_ratio;

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
  VectorInstructions inst0;
  inst0.op_type = VectorInstructions::reduction;
  inst0.inst_loop_count = reduced_size;
  inst0.reduce_count = reduction_dim / REDUCER_WIDTH;
  inst0.reduce_op = VectorInstructions::radd;
  inst0.rduplicate = 1;
  inst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst0;

  // Scale inputs by 1 / norm_size and send to the reduction engine
  VectorInstructions inst1;
  inst1.op_type = VectorInstructions::vector;
  inst1.inst_loop_count = input_size / VECTOR_UNIT_WIDTH;
  inst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst1.vector_op0_src1 = VectorInstructions::from_immediate_0;
  inst1.vector_op0 = VectorInstructions::op0_mul;
  VECTOR_DATATYPE immediate = 1.0 / reduction_dim;
  inst1.immediate0 = immediate.bits_rep();
  inst1.vdest = VectorInstructions::to_reduce;
  set_dequantize_scale(dequantize_op, env, inst1);
  vector_instruction_config->inst[1] = inst1;

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
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = vu_unit_ratio;

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
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
  vector_params->vector_fetch_1_burst_size = vector_fetch_width / 8;
  vector_params->vector_fetch_1_num_beats = vector_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = vu_unit_ratio;

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
  VectorInstructions inst2;
  inst2.op_type = VectorInstructions::reduction;
  inst2.inst_loop_count = reduced_size;
  inst2.reduce_count = reduction_dim / REDUCER_WIDTH;
  inst2.reduce_op = VectorInstructions::radd;
  inst2.rsqrt = 1;
  inst2.rreciprocal = 1;
  inst2.rscale = 1;
  VECTOR_DATATYPE divisor = sqrt(reduction_dim);
  inst2.immediate1 = divisor.bits_rep();
  inst2.rduplicate = 1;
  inst2.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst2;

  VectorInstructions inst3;
  inst3.op_type = VectorInstructions::vector;
  inst3.inst_loop_count = input_size / VECTOR_UNIT_WIDTH;
  inst3.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst3.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst3.vector_op0 = VectorInstructions::op0_sub;
  inst3.vector_op2 = VectorInstructions::op2_sqr;
  inst3.vdest = VectorInstructions::to_reduce;
  set_dequantize_scale(dequantize_op, env, inst3);
  vector_instruction_config->inst[1] = inst3;

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

  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = vu_unit_ratio;

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
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
  vector_params->vector_fetch_1_burst_size = vector_fetch_width / 8;
  vector_params->vector_fetch_1_num_beats = vector_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = vu_unit_ratio;

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
  vector_params->vector_fetch_2_stride = OC_DIMENSION;
  vector_params->vector_fetch_2_burst_size = vector_fetch_width / 8;
  vector_params->vector_fetch_2_num_beats = vector_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_2_packing_factor = vu_unit_ratio;

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
  VectorInstructions inst4;
  inst4.op_type = VectorInstructions::vector;
  inst4.inst_loop_count = input_size / VECTOR_UNIT_WIDTH;
  inst4.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst4.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst4.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
  inst4.vector_op0 = VectorInstructions::op0_sub;
  inst4.vector_op2 = VectorInstructions::op2_mul;
  inst4.vdest = VectorInstructions::to_output;
  set_dequantize_scale(dequantize_op, env, inst4);
  vector_instruction_config->inst[0] = inst4;

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

  input_fetch_width = OC_DIMENSION * vector_dtype_width;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = vu_unit_ratio;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Fetch weights
  const auto weight = resolve(layer_norm_op, "weight", env);
  int weight_dtype = get_index_from_type_name<VU_INPUT_TYPES>(weight.dtype);
  int weight_dtype_width = get_type_width<VU_INPUT_TYPES>(weight_dtype);
  int weight_fetch_width = OC_DIMENSION * weight_dtype_width;

  vector_params->vector_fetch_1_offset = get_address(weight);
  vector_params->vector_fetch_1_mode = true;
  vector_params->vector_fetch_1_broadcast = 0b011;
  vector_params->vector_fetch_1_dtype = weight_dtype;
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
  vector_params->vector_fetch_1_burst_size = weight_fetch_width / 8;
  vector_params->vector_fetch_1_num_beats = weight_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = vu_unit_ratio;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_1_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Fetch bias
  const bool has_bias = has_arg(layer_norm_op, "bias");
  if (has_bias) {
    const auto bias = resolve(layer_norm_op, "bias", env);
    int bias_dtype = get_index_from_type_name<VU_INPUT_TYPES>(bias.dtype);
    int bias_dtype_width = get_type_width<VU_INPUT_TYPES>(bias_dtype);
    int bias_fetch_width = OC_DIMENSION * bias_dtype_width;

    vector_params->vector_fetch_2_offset = get_address(bias);
    vector_params->vector_fetch_2_mode = true;
    vector_params->vector_fetch_2_broadcast = 0b011;
    vector_params->vector_fetch_2_dtype = bias_dtype;
    vector_params->vector_fetch_2_stride = OC_DIMENSION;
    vector_params->vector_fetch_2_burst_size = bias_fetch_width / 8;
    vector_params->vector_fetch_2_num_beats = bias_fetch_width / OC_PORT_WIDTH;
    vector_params->vector_fetch_2_packing_factor = vu_unit_ratio;

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
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype);

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = non_reduction_loops[0];
  vector_params->output_loops[1][1] = non_reduction_loops[1];
  vector_params->output_loops[1][2] = reduction_dim / OC_DIMENSION;

  // inputs x weights + bias
  VectorInstructions inst5;
  inst5.op_type = VectorInstructions::vector;
  inst5.inst_loop_count = input_size / VECTOR_UNIT_WIDTH;
  inst5.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst5.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst5.vector_op0 = VectorInstructions::op0_mul;
  if (has_bias) {
    inst5.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
    inst5.vector_op2 = VectorInstructions::op2_add;
  }
  inst5.vdest = VectorInstructions::to_output;
  set_quantize_params(operation, env, vector_params, inst5,
                      vector_instruction_config, mapped_params);
  vector_instruction_config->inst[0] = inst5;

  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
