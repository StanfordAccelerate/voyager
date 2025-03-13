#pragma once

#include "test/toolchain/Common.h"

void MapMatrixVectorMultiply(const codegen::Operation &param,
                             std::deque<BaseParams *> &mappedParams,
                             std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto op_list = get_op_list(param);
  const auto matrix_op = op_list[0];

  const auto input = matrix_op.kwargs().at("input").tensor();
  const auto weight = matrix_op.kwargs().at("weight").tensor();
  const auto bias = matrix_op.kwargs().at("bias").tensor();
  const auto output = param.output();

  int output_dim = weight.shape(0);
  int reduction_dim = weight.shape(1);

  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  AcceleratorMemoryMap accelerator_memory_map;

  // round output_dim up to a multiple of DIMENSION
  output_dim = (output_dim + OC_DIMENSION - 1) / OC_DIMENSION * OC_DIMENSION;

  // input is a vector of size reduction_dim
  const auto input_memory = input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.address();
  vector_params->addressGen0Mode = 2;
  vector_params->vec0_broadcast = 0b010000;
  vector_params->vector_input_0_type =
      get_index_from_type_name<VECTOR_INPUT_DATATYPES>(input.dtype());

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen0Loop[0][i] = 1;
  }
  vector_params->addressGen0Loop[1][0] = 1;
  vector_params->addressGen0Loop[1][1] = output_dim;
  vector_params->addressGen0Loop[1][2] = reduction_dim / OC_DIMENSION;

  // weight is a matrix of output_dim x reduction_dim
  const auto weight_memory = weight.memory();
  accelerator_memory_map["vector1"] = get_partition(weight_memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = weight_memory.address();
  vector_params->addressGen1Mode = true;
  vector_params->vector_input_1_type =
      get_index_from_type_name<VECTOR_INPUT_DATATYPES>(weight.dtype());

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen1Loops[0][i] = 1;
  }
  vector_params->addressGen1Loops[1][0] = 1;
  vector_params->addressGen1Loops[1][1] = output_dim;
  vector_params->addressGen1Loops[1][2] = reduction_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen1InputYLoopIndex[i] = 0;
    vector_params->addressGen1InputXLoopIndex[i] = 1;
    vector_params->addressGen1WeightLoopIndex[i] = 2;
  }

  // bias
  const auto bias_memory = bias.memory();
  accelerator_memory_map["vector2"] = get_partition(bias_memory.partition());
  vector_params->ADDRESS_GEN2_OFFSET = bias_memory.address();
  vector_params->addressGen2Mode = true;
  vector_params->vec2_broadcast = 0b011;
  vector_params->vector_input_2_type =
      get_index_from_type_name<VECTOR_INPUT_DATATYPES>(bias.dtype());

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen2Loops[0][i] = 1;
  }
  vector_params->addressGen2Loops[1][0] = 1;
  vector_params->addressGen2Loops[1][1] = 1;
  vector_params->addressGen2Loops[1][2] = output_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen2InputYLoopIndex[i] = 0;
    vector_params->addressGen2InputXLoopIndex[i] = 1;
    vector_params->addressGen2WeightLoopIndex[i] = 2;
  }

  // output
  const auto output_memory = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();
  vector_params->output_types =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  for (int i = 0; i < 3; i++) {
    vector_params->outputLoops[0][i] = 1;
  }
  vector_params->outputLoops[1][0] = 1;
  vector_params->outputLoops[1][1] = 1;
  vector_params->outputLoops[1][2] = output_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->outputYLoopIndex[i] = 0;
    vector_params->outputXLoopIndex[i] = 1;
    vector_params->outputWeightLoopIndex[i] = 2;
  }

  // inst0 - start reduction engine
  VectorInstructions vinst0;
  vinst0.instType = VectorInstructions::reduction;
  vinst0.rCount = reduction_dim / OC_DIMENSION;
  vinst0.rOp = VectorInstructions::radd;
  vinst0.rdest = 0;
  vinst0.immediate0 = 1;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // inst1 - input x weight, send to reduce
  VectorInstructions vinst1;
  vinst1.instType = VectorInstructions::vector;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  vinst1.vector_op0 = VectorInstructions::vmult;
  vinst1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = vinst1;

  // reduction_dim/DIMENSION to do the complete reduction
  // DIMENSION to fill up the entire vector (this is now output_dim dimension)
  vector_instruction_config->instCount[1] = reduction_dim;

  // inst2 - add bias, write out
  VectorInstructions vinst2;
  vinst2.instType = VectorInstructions::vector;
  vinst2.vector_op0_src0 = VectorInstructions::from_reduction_0;
  vinst2.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
  vinst2.vector_op2 = VectorInstructions::vadd;
  vinst2.vdest = VectorInstructions::to_output;
  vector_instruction_config->inst[2] = vinst2;
  vector_instruction_config->instCount[2] = 1;

  vector_instruction_config->instLen = 3;
  vector_instruction_config->instLoopCount = output_dim / OC_DIMENSION;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
