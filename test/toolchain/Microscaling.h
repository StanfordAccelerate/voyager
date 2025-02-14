#pragma once

#include "test/toolchain/Common.h"

void MapMXQparam(const codegen::Operation &param,
                 std::deque<BaseParams *> &mappedParams,
                 std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto op_list = get_op_list(param);
  const auto mx_qparam_op = op_list[0];

  const auto input = mx_qparam_op.kwargs().at("input").tensor();
  const auto output = param.output();

  // Not used
  int mx_axis = mx_qparam_op.kwargs().at("axis").int_value();

  // send to reduce unit, configure to calculate max exponent value
  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  AcceleratorMemoryMap accelerator_memory_map;

  // assume block size of 32
  int block_size = 32;

  int dim = get_size(input);
  int dim_factors[2];
  factorize_for_address_gen(dim / OC_DIMENSION, dim_factors);

  // inputs
  const auto input_memory = input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.address();
  vector_params->addressGen0Mode = 1;
  vector_params->addressGen0Loop[0][0] = 1;
  vector_params->addressGen0Loop[0][1] = 1;
  vector_params->addressGen0Loop[0][2] = 1;
  vector_params->addressGen0Loop[1][0] = 1;
  vector_params->addressGen0Loop[1][1] = dim_factors[0];
  vector_params->addressGen0Loop[1][2] = dim_factors[1];

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen0InputXLoopIndex[i] = 0;
    vector_params->addressGen0InputYLoopIndex[i] = 1;
    vector_params->addressGen0WeightLoopIndex[i] = 2;
  }

  vector_params->fetch_vector_type_0 =
      DataTypes::TypeName<VECTOR_DATATYPE>::name() == input.dtype();

  int output_dim_factors[2];
  factorize_for_address_gen(dim / OC_DIMENSION / block_size,
                            output_dim_factors);

  // output
  const auto output_mem = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_mem.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_mem.address();
  for (int i = 0; i < 3; i++) {
    vector_params->outputLoops[0][i] = 1;
  }
  vector_params->outputLoops[1][0] = 1;
  vector_params->outputLoops[1][1] = output_dim_factors[0];
  vector_params->outputLoops[1][2] = output_dim_factors[1];

  for (int i = 0; i < 2; i++) {
    vector_params->outputXLoopIndex[i] = 0;
    vector_params->outputYLoopIndex[i] = 1;
    vector_params->outputWeightLoopIndex[i] = 2;
  }

  vector_params->output_scale_type = true;

  // inst 0 - start reduction engine to calculate mx scale
  VectorInstructions vinst0;
  vinst0.instType = VectorInstructions::reduction;
  vinst0.rCount = block_size / OC_DIMENSION;
  vinst0.rOp = VectorInstructions::rmxscale;
  vinst0.rDest = VectorInstructions::toVectorOp0Src0;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // inst 1 - send to reduction engine
  VectorInstructions vinst1;
  vinst1.instType = VectorInstructions::vector;
  vinst1.vInput = VectorInstructions::readFromVectorFetch;
  vinst1.vAccumulatePush = VectorInstructions::nop;
  vinst1.vOp0Src1 = VectorInstructions::nop;
  vinst1.vOp0 = VectorInstructions::nop;
  vinst1.vOp1 = VectorInstructions::nop;
  vinst1.vOp2 = VectorInstructions::toReduce;
  vinst1.vOp3Src1 = VectorInstructions::nop;
  vinst1.vOp3 = VectorInstructions::nop;
  vinst1.vOp4 = VectorInstructions::nop;
  vinst1.vDest = VectorInstructions::nop;
  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = block_size;

  // inst 2 - write out
  VectorInstructions vinst2;
  vinst2.instType = VectorInstructions::vector;
  vinst2.vInput = VectorInstructions::readFromReduce;
  vinst2.vDest = VectorInstructions::vWriteOut;
  vector_instruction_config->inst[2] = vinst2;
  vector_instruction_config->instCount[2] = 1;

  vector_instruction_config->instLen = 3;
  vector_instruction_config->instLoopCount = dim / OC_DIMENSION / block_size;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}