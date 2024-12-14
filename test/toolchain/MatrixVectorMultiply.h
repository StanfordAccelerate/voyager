#pragma once

#include "test/toolchain/Common.h"

void MapMatrixVectorMultiply(const codegen::Operator &param,
                             std::deque<BaseParams *> &mappedParams,
                             std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto matrix_op = param.matrix_op();
  const auto weights = matrix_op.weight();
  int output_dim = weights.shape(0);
  int reduction_dim = weights.shape(1);

  // round output_dim up to a multiple of DIMENSION
  output_dim = (output_dim + OC_DIMENSION - 1) / OC_DIMENSION * OC_DIMENSION;

  // input is a vector of size reduction_dim
  const auto input_memory = matrix_op.input().memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 2;
  vector_params->addressGen0Broadcast = 0b010000;

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen0Loop[0][i] = 1;
  }
  vector_params->addressGen0Loop[1][0] = 1;
  vector_params->addressGen0Loop[1][1] = output_dim;
  vector_params->addressGen0Loop[1][2] = reduction_dim / OC_DIMENSION;

  vector_params->DP_VEC0 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != matrix_op.input().dtype();

  // weights is a matrix of output_dim x reduction_dim
  const auto weights_memory = weights.memory();
  accelerator_memory_map["vector1"] = get_partition(weights_memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = weights_memory.offset();
  vector_params->addressGen1Mode = 1;
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

  vector_params->DP_VEC1 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != matrix_op.weight().dtype();

  // bias
  const auto bias_memory = matrix_op.bias().memory();
  accelerator_memory_map["vector2"] = get_partition(bias_memory.partition());
  vector_params->ADDRESS_GEN2_OFFSET = bias_memory.offset();
  vector_params->addressGen2Mode = 3;
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

  vector_params->DP_VEC2 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != matrix_op.bias().dtype();

  // output
  const auto output_memory = param.output().memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.offset();
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

  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();

  // inst0 - start reduction engine
  VectorInstructions vinst0;
  vinst0.instType = VectorInstructions::reduction;
  vinst0.rCount = reduction_dim / OC_DIMENSION;
  vinst0.rOp = VectorInstructions::radd;
  vinst0.rDuplicate = 0;
  vinst0.rDest = VectorInstructions::toVectorOp0Src0;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // inst 1 - inputs x weights, send to reduce
  VectorInstructions vinst1;
  vinst1.instType = VectorInstructions::vector;
  vinst1.vInput = VectorInstructions::readFromVectorFetch;
  vinst1.vAccumulatePush = VectorInstructions::nop;
  vinst1.vOp0Src1 = VectorInstructions::readInterface;
  vinst1.vOp0 = VectorInstructions::vmult;
  vinst1.vOp1 = VectorInstructions::nop;
  vinst1.vOp2 = VectorInstructions::toReduce;
  vinst1.vOp3Src1 = VectorInstructions::nop;
  vinst1.vOp3 = VectorInstructions::nop;
  vinst1.vOp4 = VectorInstructions::nop;
  vinst1.vDest = VectorInstructions::nop;
  vector_instruction_config->inst[1] = vinst1;

  // reduction_dim/DIMENSION to do the complete reduction
  // DIMENSION to fill up the entire vector (this is now output_dim dimension)
  vector_instruction_config->instCount[1] = reduction_dim;

  // inst2 - add bias, write out
  VectorInstructions vinst2;
  vinst2.instType = VectorInstructions::vector;
  vinst2.vInput = VectorInstructions::readFromReduce;
  vinst2.vAccumulatePush = VectorInstructions::nop;
  vinst2.vOp0Src1 = VectorInstructions::nop;
  vinst2.vOp0 = VectorInstructions::nop;
  vinst2.vOp1 = VectorInstructions::nop;
  vinst2.vOp2 = VectorInstructions::nop;
  vinst2.vOp3Src1 = VectorInstructions::readNormalInterface;
  vinst2.vOp3 = VectorInstructions::vadd;
  vinst2.vOp4 = VectorInstructions::nop;
  vinst2.vDest = VectorInstructions::vWriteOut;
  vector_instruction_config->inst[2] = vinst2;
  vector_instruction_config->instCount[2] = 1;

  vector_instruction_config->instLen = 3;
  vector_instruction_config->instLoopCount = output_dim / OC_DIMENSION;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
