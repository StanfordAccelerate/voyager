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
  const int outer_dim = input_shape.back();
  int inner_dim = 1;
  std::vector<int> non_reduction_loops;

  for (int i = 0; i < input_shape.size() - 1; i++) {
    inner_dim *= input_shape[i];
    non_reduction_loops.push_back(input_shape[i]);
  }

  non_reduction_loops = split_loops(non_reduction_loops, 1024);
  pad_shape_to_ndim(non_reduction_loops, 4);

  // inputs
  const auto input_memory = input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->vector_fetch_0_offset = input_memory.address();
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_broadcast = 0b010000;
  vector_params->vector_fetch_0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  vector_params->vector_fetch_0_loops[0][0] = non_reduction_loops[0];
  vector_params->vector_fetch_0_loops[0][1] = non_reduction_loops[1];
  vector_params->vector_fetch_0_loops[0][2] = non_reduction_loops[2];
  vector_params->vector_fetch_0_loops[1][0] = non_reduction_loops[3];
  vector_params->vector_fetch_0_loops[1][1] = 3;
  vector_params->vector_fetch_0_loops[1][2] = outer_dim / OC_DIMENSION;

  // output
  const auto output_memory = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();
  vector_params->output_mode = 2;

  vector_params->output_loops[0][0] = 1;
  vector_params->output_loops[0][1] = non_reduction_loops[0];
  vector_params->output_loops[0][2] = non_reduction_loops[1];
  vector_params->output_loops[1][0] = non_reduction_loops[2];
  vector_params->output_loops[1][1] = non_reduction_loops[3];
  vector_params->output_loops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  // Instruction 0 - start reduction engine to calculate max
  VectorInstructions vinst0;
  vinst0.op_type = VectorInstructions::reduction;
  vinst0.reduce_count = outer_dim / OC_DIMENSION;
  vinst0.reduce_op = VectorInstructions::rmax;
  vinst0.rduplicate = 1;
  // broadcast max over entire array, for 2 passes
  vinst0.rbroadcast = 1;
  vinst0.immediate0 = 2 * outer_dim / OC_DIMENSION;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // Instruction 1 - send to reduction engine to calculate max
  VectorInstructions vinst1;
  vinst1.op_type = VectorInstructions::vector;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = outer_dim / OC_DIMENSION;

  // Instruction 2 - start reduction engine to calculate sum
  VectorInstructions vinst2;
  vinst2.op_type = VectorInstructions::reduction;
  vinst2.reduce_count = outer_dim / OC_DIMENSION;
  vinst2.reduce_op = VectorInstructions::radd;
  vinst2.rreciprocal = 1;
  vinst2.rduplicate = 1;
  vinst2.rbroadcast = 1;
  vinst2.immediate0 = outer_dim / OC_DIMENSION;
  vinst2.rdest = VectorInstructions::to_op2;
  vector_instruction_config->inst[2] = vinst2;
  vector_instruction_config->instCount[2] = 1;

  // Instruction 3 - subtract max and exp, and reduce sum
  VectorInstructions vinst3;
  vinst3.op_type = VectorInstructions::vector;
  vinst3.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst3.vector_op0_src1 = VectorInstructions::from_reduction_0;
  vinst3.vector_op0 = VectorInstructions::vsub;
  vinst3.vector_op1 = VectorInstructions::vexp;
  vinst3.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[3] = vinst3;
  vector_instruction_config->instCount[3] = outer_dim / OC_DIMENSION;

  // Instruction 4 - subtract max and exp, and divide by reduced value
  VectorInstructions inst4;
  inst4.op_type = VectorInstructions::vector;
  inst4.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst4.vector_op0_src1 = VectorInstructions::from_reduction_0;
  inst4.vector_op2_src1 = VectorInstructions::from_reduction_1;
  inst4.vector_op0 = VectorInstructions::vsub;
  inst4.vector_op1 = VectorInstructions::vexp;
  inst4.vector_op2 = VectorInstructions::vmult;
  inst4.vdest = VectorInstructions::to_output;

  set_quantize_params(param, vector_params, inst4);

  vector_instruction_config->inst[4] = inst4;
  vector_instruction_config->instCount[4] = outer_dim / OC_DIMENSION;

  vector_instruction_config->instLen = 5;
  vector_instruction_config->instLoopCount = inner_dim;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}