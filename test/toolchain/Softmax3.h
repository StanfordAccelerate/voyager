#pragma once

#include "test/toolchain/Common.h"

void MapSoftmax(const codegen::Operation &param,
                std::deque<BaseParams *> &mappedParams,
                std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto op_list = get_op_list(param);
  const auto softmax_op = op_list[0];

  const auto input = softmax_op.kwargs().at("input").tensor();

  codegen::Tensor output;
  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto input_shape = squeeze_shape(get_shape(input));
  const int input_size = get_size(input);
  const int reduction_dim = input_shape.back();

  std::vector<int> non_reduction_loops;
  for (int i = 0; i < input_shape.size() - 1; i++) {
    non_reduction_loops.push_back(input_shape[i]);
  }

  non_reduction_loops = split_loops(non_reduction_loops, 1024);
  pad_shape_to_ndim(non_reduction_loops, 2);

  const int output_size = get_size(output);
  const int reduced_size = get_size(non_reduction_loops);

  const auto output_memory = output.memory();
  const int max_scratch_memory = output_memory.address() + 2 * output_size;
  const int sum_scratch_memory = max_scratch_memory + 2 * output_size;

  // ----------------------------------------------------------------------------
  // Pass 1: Calculate max and subtract max from tensor
  // ----------------------------------------------------------------------------

  // Address generator 0: Input tensor.
  const auto input_memory = input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->ADDRESS_GEN0_OFFSET = input_memory.address();
  vector_params->addr_gen0_mode = 2;
  vector_params->addr_gen0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen0_loops[0][i] = 1;
  }
  vector_params->addr_gen0_loops[1][0] = non_reduction_loops[0];
  vector_params->addr_gen0_loops[1][1] = non_reduction_loops[1];
  vector_params->addr_gen0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Output: Scratch memory for max.
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = max_scratch_memory;
  vector_params->output_mode = 2;
  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = reduced_size;
  vector_params->output_dtype =
      get_type_index<VECTOR_DATATYPE, OUTPUT_DATATYPES>();

  // Instruction 0 - start reduction engine to calculate max
  VectorInstructions vinst0;
  vinst0.op_type = VectorInstructions::reduction;
  vinst0.inst_count = reduced_size;
  vinst0.reduce_count = reduction_dim / OC_DIMENSION;
  vinst0.reduce_op = VectorInstructions::rmax;
  vinst0.rduplicate = 1;
  vinst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // Instruction 1 - send to reduction engine to calculate max
  VectorInstructions vinst1;
  vinst1.op_type = VectorInstructions::vector;
  vinst1.inst_count = input_size / OC_DIMENSION;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = 1;

  vector_instruction_config->instLen = 2;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);

  // ---------------------------------------------------------------------------
  // Pass 2: Find the normalization constant.
  // ---------------------------------------------------------------------------

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  // Address generator 0: Input tensor.
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->ADDRESS_GEN0_OFFSET = input_memory.address();
  vector_params->addr_gen0_mode = 2;
  vector_params->addr_gen0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen0_loops[0][i] = 1;
  }
  vector_params->addr_gen0_loops[1][0] = non_reduction_loops[0];
  vector_params->addr_gen0_loops[1][1] = non_reduction_loops[1];
  vector_params->addr_gen0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 1: Scratch max.
  vector_params->ADDRESS_GEN1_OFFSET = max_scratch_memory;
  vector_params->addr_gen1_mode = 1;
  vector_params->addr_gen1_dtype =
      get_type_index<VECTOR_DATATYPE, VU_INPUT_TYPES>();
  vector_params->addr_gen1_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen1_loops[0][i] = 1;
  }
  vector_params->addr_gen1_loops[1][0] = non_reduction_loops[0];
  vector_params->addr_gen1_loops[1][1] = non_reduction_loops[1];
  vector_params->addr_gen1_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Output: Scratch memory for sum.
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = sum_scratch_memory;
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
  VectorInstructions vinst2;
  vinst2.op_type = VectorInstructions::reduction;
  vinst2.inst_count = reduced_size;
  vinst2.reduce_count = reduction_dim / OC_DIMENSION;
  vinst2.reduce_op = VectorInstructions::radd;
  vinst2.rreciprocal = 1;
  vinst2.rduplicate = 1;
  vinst2.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = vinst2;
  vector_instruction_config->instCount[0] = 1;

  // Instruction 3 - subtract max and exp, and reduce sum
  VectorInstructions vinst3;
  vinst3.op_type = VectorInstructions::vector;
  vinst3.inst_count = input_size / OC_DIMENSION;
  vinst3.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst3.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  vinst3.vector_op0 = VectorInstructions::vsub;
  vinst3.vector_op1 = VectorInstructions::vexp;
  vinst3.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = vinst3;
  vector_instruction_config->instCount[1] = 1;

  vector_instruction_config->instLen = 2;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);

  // ---------------------------------------------------------------------------
  // Pass 3: Divide by the normalization constant and apply exp.
  // ---------------------------------------------------------------------------

  vector_params = new VectorParams;
  vector_instruction_config = new VectorInstructionConfig;

  // Address generator 0: Input tensor.
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->ADDRESS_GEN0_OFFSET = input_memory.address();
  vector_params->addr_gen0_mode = 2;
  vector_params->addr_gen0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen0_loops[0][i] = 1;
  }
  vector_params->addr_gen0_loops[1][0] = non_reduction_loops[0];
  vector_params->addr_gen0_loops[1][1] = non_reduction_loops[1];
  vector_params->addr_gen0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 1: Scratch max.
  vector_params->ADDRESS_GEN1_OFFSET = max_scratch_memory;
  vector_params->addr_gen1_mode = 1;
  vector_params->addr_gen1_dtype =
      get_type_index<VECTOR_DATATYPE, VU_INPUT_TYPES>();
  vector_params->addr_gen1_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen1_loops[0][i] = 1;
  }
  vector_params->addr_gen1_loops[1][0] = non_reduction_loops[0];
  vector_params->addr_gen1_loops[1][1] = non_reduction_loops[1];
  vector_params->addr_gen1_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Address generator 2: Scratch max.
  vector_params->ADDRESS_GEN2_OFFSET = sum_scratch_memory;
  vector_params->addr_gen2_mode = 1;
  vector_params->addr_gen2_dtype =
      get_type_index<VECTOR_DATATYPE, VU_INPUT_TYPES>();
  vector_params->addr_gen2_broadcast = 0b100;
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen2_loops[0][i] = 1;
  }
  vector_params->addr_gen2_loops[1][0] = non_reduction_loops[0];
  vector_params->addr_gen2_loops[1][1] = non_reduction_loops[1];
  vector_params->addr_gen2_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Output
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();
  vector_params->output_mode = 2;
  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());
  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = non_reduction_loops[0];
  vector_params->output_loops[1][1] = non_reduction_loops[1];
  vector_params->output_loops[1][2] = reduction_dim / OC_DIMENSION;

  // Instruction 4 - subtract max and exp, and divide by reduced value
  VectorInstructions inst4;
  inst4.op_type = VectorInstructions::vector;
  inst4.inst_count = input_size / OC_DIMENSION;
  inst4.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst4.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst4.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
  inst4.vector_op0 = VectorInstructions::vsub;
  inst4.vector_op1 = VectorInstructions::vexp;
  inst4.vector_op2 = VectorInstructions::vmult;
  inst4.vdest = VectorInstructions::to_output;

  set_quantize_params(param, vector_params, inst4);

  vector_instruction_config->inst[0] = inst4;
  vector_instruction_config->instCount[0] = 1;

  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}