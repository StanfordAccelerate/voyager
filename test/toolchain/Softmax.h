#pragma once

#include "test/toolchain/Common.h"

void map_softmax(const voyager::Operation& operation, const ScalarEnv& env,
                 std::deque<BaseParams*>& mapped_params) {
  const auto& op_list = get_prim_ops(operation);
  const auto& softmax_op = get_anchor_op(operation);

  const Tensor& anchor_input = resolve(softmax_op, "input", env);
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

  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

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

  int input_dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype);
  int input_dtype_width = get_type_width<VU_INPUT_TYPES>(input_dtype);
  int input_fetch_width = OC_DIMENSION * input_dtype_width;

  int vector_dtype = get_type_index<VECTOR_DATATYPE, VU_INPUT_TYPES>();
  int vector_dtype_width = get_type_width<VU_INPUT_TYPES>(vector_dtype);
  int vector_fetch_width = OC_DIMENSION * vector_dtype_width;

  // The compiler reserves the scratchpad the two reduction passes stage their
  // results in, so these are read off the operation. Deriving them from the
  // input tile instead would land inside the input's own second bank.
  const int max_scratch_memory = get_address(resolve(softmax_op, "max", env));
  const int sum_scratch_memory = get_address(resolve(softmax_op, "sum", env));

  // ----------------------------------------------------------------------------
  // Pass 1: Calculate max and subtract max from tensor
  // ----------------------------------------------------------------------------

  // Address generator 0: Input tensor.
  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Output: Scratch memory for max.
  vector_params->vector_output_offset = max_scratch_memory;
  vector_params->output_mode = 2;
  vector_params->output_dtype =
      get_type_index<VECTOR_DATATYPE, OUTPUT_DATATYPES>();

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = reduced_size;

  // Instruction 0 - start reduction engine to calculate max
  VectorInstructions inst0;
  inst0.op_type = VectorInstructions::reduction;
  inst0.inst_loop_count = reduced_size;
  inst0.reduce_count = reduction_dim / REDUCER_WIDTH;
  inst0.reduce_op = VectorInstructions::rmax;
  inst0.rduplicate = 1;
  inst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst0;

  // Instruction 1 - send to reduction engine to calculate max
  VectorInstructions inst1;
  inst1.op_type = VectorInstructions::vector;
  inst1.inst_loop_count = input_size / VECTOR_UNIT_WIDTH;
  inst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst1.vdest = VectorInstructions::to_reduce;
  set_dequantize_scale(dequantize_op, env, inst1);
  vector_instruction_config->inst[1] = inst1;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);

  // ---------------------------------------------------------------------------
  // Pass 2: Find the normalization constant.
  // ---------------------------------------------------------------------------

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  // Address generator 0: Input tensor.
  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 1: Scratch max.
  vector_params->vector_fetch_1_offset = max_scratch_memory;
  vector_params->vector_fetch_1_mode = 1;
  vector_params->vector_fetch_1_dtype = vector_dtype;
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
  vector_params->vector_fetch_1_burst_size = vector_fetch_width / 8;
  vector_params->vector_fetch_1_num_beats = vector_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = packing_factor;

  vector_params->vector_fetch_1_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_1_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Output: Scratch memory for sum.
  vector_params->vector_output_offset = sum_scratch_memory;
  vector_params->output_mode = 2;
  vector_params->output_dtype =
      get_type_index<VECTOR_DATATYPE, OUTPUT_DATATYPES>();

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = reduced_size;

  // Instruction 2 - start reduction engine to calculate sum
  VectorInstructions inst2;
  inst2.op_type = VectorInstructions::reduction;
  inst2.inst_loop_count = reduced_size;
  inst2.reduce_count = reduction_dim / REDUCER_WIDTH;
  inst2.reduce_op = VectorInstructions::radd;
  inst2.rreciprocal = 1;
  inst2.rduplicate = 1;
  inst2.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst2;

  // Instruction 3 - subtract max and exp, and reduce sum
  VectorInstructions inst3;
  inst3.op_type = VectorInstructions::vector;
  inst3.inst_loop_count = input_size / VECTOR_UNIT_WIDTH;
  inst3.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst3.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst3.vector_op0 = VectorInstructions::op0_sub;
  inst3.vector_op1 = VectorInstructions::op1_exp;
  inst3.vdest = VectorInstructions::to_reduce;
  set_dequantize_scale(dequantize_op, env, inst3);
  vector_instruction_config->inst[1] = inst3;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);

  // ---------------------------------------------------------------------------
  // Pass 3: Divide by the normalization constant and apply exp.
  // ---------------------------------------------------------------------------

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  // Address generator 0: Input tensor.
  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 1: Scratch max.
  vector_params->vector_fetch_1_offset = max_scratch_memory;
  vector_params->vector_fetch_1_mode = 1;
  vector_params->vector_fetch_1_dtype = vector_dtype;
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
  vector_params->vector_fetch_1_burst_size = vector_fetch_width / 8;
  vector_params->vector_fetch_1_num_beats = vector_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = packing_factor;

  vector_params->vector_fetch_1_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_1_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 2: Scratch max.
  vector_params->vector_fetch_2_offset = sum_scratch_memory;
  vector_params->vector_fetch_2_mode = 1;
  vector_params->vector_fetch_2_dtype = vector_dtype;
  vector_params->vector_fetch_2_stride = OC_DIMENSION;
  vector_params->vector_fetch_2_burst_size = vector_fetch_width / 8;
  vector_params->vector_fetch_2_num_beats = vector_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_2_packing_factor = packing_factor;

  vector_params->vector_fetch_2_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_2_loops[0][i] = 1;
  }
  vector_params->vector_fetch_2_loops[1][0] = non_reduction_loops[0];
  vector_params->vector_fetch_2_loops[1][1] = non_reduction_loops[1];
  vector_params->vector_fetch_2_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Output
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

  // Instruction 4 - subtract max and exp, and divide by reduced value
  VectorInstructions inst4;
  inst4.op_type = VectorInstructions::vector;
  inst4.inst_loop_count = input_size / OC_DIMENSION * packing_factor;
  inst4.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst4.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst4.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
  inst4.vector_op0 = VectorInstructions::op0_sub;
  inst4.vector_op1 = VectorInstructions::op1_exp;
  inst4.vector_op2 = VectorInstructions::op2_mul;
  inst4.vdest = VectorInstructions::to_output;
  set_dequantize_scale(dequantize_op, env, inst4);
  set_quantize_params(operation, env, vector_params, inst4,
                      vector_instruction_config, mapped_params);
  vector_instruction_config->inst[0] = inst4;

  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
