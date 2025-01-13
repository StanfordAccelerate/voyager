#pragma once

#include "test/toolchain/Common.h"

void MapLayerNorm(const codegen::Operator &param,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params;
  VectorInstructionConfig *vinstr_config;
  AcceleratorMemoryMap memory_map;

  const auto &matrix_op = param.matrix_op();
  const auto &input = matrix_op.input();
  const auto input_shape = squeeze_shape(get_tensor_shape(input));

  const int outer_dim = input_shape.back();
  int inner_dim = 1;
  std::vector<int> non_reduction_loops;

  for (int i = 0; i < input_shape.size() - 1; i++) {
    inner_dim *= input_shape[i];
    non_reduction_loops.push_back(input_shape[i]);
  }

  non_reduction_loops = split_loops(non_reduction_loops, 1024);
  pad_shape_to_ndim(non_reduction_loops, 4);

  // ======================================================================
  // Pass 1: calculate mean and subtract mean from tensor
  // ======================================================================
  vector_params = new VectorParams;
  vinstr_config = new VectorInstructionConfig;

  const auto input_memory = input.memory();
  memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 2;
  vector_params->addressGen0Broadcast = 0b010000;

  // Fetch inputs twice, once for calculating mean and once for subtracting mean
  vector_params->addressGen0Loop[0][0] = non_reduction_loops[0];
  vector_params->addressGen0Loop[0][1] = non_reduction_loops[1];
  vector_params->addressGen0Loop[0][2] = non_reduction_loops[2];
  vector_params->addressGen0Loop[1][0] = non_reduction_loops[3];
  vector_params->addressGen0Loop[1][1] = 2;
  vector_params->addressGen0Loop[1][2] = outer_dim / OC_DIMENSION;

  vector_params->DP_VEC0 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != input.dtype();

  // Overwrite inputs
  memory_map["outputs"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = input_memory.offset();
  vector_params->outputAddressMode = 2;

  vector_params->outputLoops[0][0] = 1;
  vector_params->outputLoops[0][1] = non_reduction_loops[0];
  vector_params->outputLoops[0][2] = non_reduction_loops[1];
  vector_params->outputLoops[1][0] = non_reduction_loops[2];
  vector_params->outputLoops[1][1] = non_reduction_loops[3];
  vector_params->outputLoops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();

  // Configure reduction engine
  VectorInstructions instr0_0;
  instr0_0.instType = VectorInstructions::reduction;
  instr0_0.rCount = outer_dim / OC_DIMENSION;
  instr0_0.rOp = VectorInstructions::radd;
  instr0_0.rDuplicate = 1;
  instr0_0.rDest = VectorInstructions::toVectorOp0Src1;
  instr0_0.rBroadcast = 1;
  instr0_0.immediate0 = outer_dim / OC_DIMENSION;
  vinstr_config->inst[0] = instr0_0;
  vinstr_config->instCount[0] = 1;

  // Scale inputs by 1 / norm_size and send to the reduction engine
  VectorInstructions instr0_1;
  instr0_1.instType = VectorInstructions::vector;
  instr0_1.vInput = VectorInstructions::readFromVectorFetch;
  instr0_1.vAccumulatePush = VectorInstructions::nop;
  instr0_1.vOp0Src1 = VectorInstructions::op0immediate;
  instr0_1.vOp0 = VectorInstructions::vmult;
  VECTOR_DATATYPE immediate = 1.0 / outer_dim;
  instr0_1.immediate0 = immediate.bits_rep();
  instr0_1.vOp1 = VectorInstructions::nop;
  instr0_1.vOp2 = VectorInstructions::toReduce;
  instr0_1.vOp3Src1 = VectorInstructions::nop;
  instr0_1.vOp3 = VectorInstructions::nop;
  instr0_1.vOp4 = VectorInstructions::nop;
  instr0_1.vDest = VectorInstructions::nop;
  vinstr_config->inst[1] = instr0_1;
  vinstr_config->instCount[1] = outer_dim / OC_DIMENSION;

  // Subtract mean from tensor
  VectorInstructions instr0_2;
  instr0_2.instType = VectorInstructions::vector;
  instr0_2.vInput = VectorInstructions::readFromVectorFetch;
  instr0_2.vAccumulatePush = VectorInstructions::nop;
  instr0_2.vOp0 = VectorInstructions::vsub;
  instr0_2.vOp0Src1 = VectorInstructions::readFromReduce;
  instr0_2.vOp1 = VectorInstructions::nop;
  instr0_2.vOp2 = VectorInstructions::nop;
  instr0_2.vOp3Src1 = VectorInstructions::nop;
  instr0_2.vOp3 = VectorInstructions::nop;
  instr0_2.vOp4 = VectorInstructions::nop;
  instr0_2.vDest = VectorInstructions::vWriteOut;
  vinstr_config->inst[2] = instr0_2;
  vinstr_config->instCount[2] = outer_dim / OC_DIMENSION;

  vinstr_config->instLen = 3;
  vinstr_config->instLoopCount = inner_dim;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vinstr_config);
  opMemoryMaps.push_back(memory_map);

  // ======================================================================
  // Pass 2: calculate variance and divide by variance
  // ======================================================================
  vector_params = new VectorParams;
  vinstr_config = new VectorInstructionConfig;

  memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 2;
  vector_params->addressGen0Broadcast = 0b010000;

  // Fetch inputs twice, once for calculating variance and once for division
  vector_params->addressGen0Loop[0][0] = non_reduction_loops[0];
  vector_params->addressGen0Loop[0][1] = non_reduction_loops[1];
  vector_params->addressGen0Loop[0][2] = non_reduction_loops[2];
  vector_params->addressGen0Loop[1][0] = non_reduction_loops[3];
  vector_params->addressGen0Loop[1][1] = 2;
  vector_params->addressGen0Loop[1][2] = outer_dim / OC_DIMENSION;

  vector_params->DP_VEC0 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != input.dtype();

  // Overwrite inputs
  memory_map["outputs"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = input_memory.offset();
  vector_params->outputAddressMode = 2;

  vector_params->outputLoops[0][0] = 1;
  vector_params->outputLoops[0][1] = non_reduction_loops[0];
  vector_params->outputLoops[0][2] = non_reduction_loops[1];
  vector_params->outputLoops[1][0] = non_reduction_loops[2];
  vector_params->outputLoops[1][1] = non_reduction_loops[3];
  vector_params->outputLoops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();

  // Configure reduction unit
  VectorInstructions instr1_0;
  instr1_0.instType = VectorInstructions::reduction;
  instr1_0.rCount = outer_dim / OC_DIMENSION;
  instr1_0.rOp = VectorInstructions::radd;
  instr1_0.rDuplicate = 1;
  instr1_0.rSqrt = 1;
  instr1_0.rReciprocal = 1;
  instr1_0.rDest = VectorInstructions::toVectorOp0Src1;
  instr1_0.rBroadcast = 1;
  instr1_0.immediate0 = outer_dim / OC_DIMENSION;
  vinstr_config->inst[0] = instr1_0;
  vinstr_config->instCount[0] = 1;

  // Perform squaring and send outputs to reduction engine
  VectorInstructions instr1_1;
  instr1_1.instType = VectorInstructions::vector;
  instr1_1.vInput = VectorInstructions::readFromVectorFetch;
  instr1_1.vAccumulatePush = VectorInstructions::nop;
  instr1_1.vOp0Src1 = VectorInstructions::nop;
  instr1_1.vOp0 = VectorInstructions::nop;
  instr1_1.vOp1 = VectorInstructions::nop;
  instr1_1.vOp2 = VectorInstructions::toReduce;
  instr1_1.vOp3Src1 = VectorInstructions::nop;
  instr1_1.vOp3 = VectorInstructions::vsquare;
  instr1_1.vOp4 = VectorInstructions::nop;
  instr1_1.vDest = VectorInstructions::nop;
  vinstr_config->inst[1] = instr1_1;
  vinstr_config->instCount[1] = outer_dim / OC_DIMENSION;

  // Multiply inputs with the inverse sqrt of the variance
  VectorInstructions instr1_2;
  instr1_2.instType = VectorInstructions::vector;
  instr1_2.vInput = VectorInstructions::readFromVectorFetch;
  instr1_2.vAccumulatePush = VectorInstructions::nop;
  instr1_2.vOp0 = VectorInstructions::vmult;
  instr1_2.vOp0Src1 = VectorInstructions::readFromReduce;
  instr1_2.vOp1 = VectorInstructions::nop;
  instr1_2.vOp2 = VectorInstructions::nop;
  instr1_2.vOp3Src1 = VectorInstructions::op3immediate;
  VECTOR_DATATYPE divisor = sqrt(outer_dim);
  instr1_2.immediate1 = divisor.bits_rep();
  instr1_2.vOp3 = VectorInstructions::vmult;
  instr1_2.vOp4 = VectorInstructions::nop;
  instr1_2.vDest = VectorInstructions::vWriteOut;
  vinstr_config->inst[2] = instr1_2;
  vinstr_config->instCount[2] = outer_dim / OC_DIMENSION;

  vinstr_config->instLen = 3;
  vinstr_config->instLoopCount = inner_dim;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vinstr_config);
  opMemoryMaps.push_back(memory_map);

  // ======================================================================
  // Pass 3: Apply LayerNorm weight and bias
  // ======================================================================
  vector_params = new VectorParams;
  vinstr_config = new VectorInstructionConfig;

  // Fetch inputs
  memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 2;

  vector_params->addressGen0Loop[0][0] = 1;
  vector_params->addressGen0Loop[0][1] = non_reduction_loops[0];
  vector_params->addressGen0Loop[0][2] = non_reduction_loops[1];
  vector_params->addressGen0Loop[1][0] = non_reduction_loops[2];
  vector_params->addressGen0Loop[1][1] = non_reduction_loops[3];
  vector_params->addressGen0Loop[1][2] = outer_dim / OC_DIMENSION;

  vector_params->DP_VEC0 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != input.dtype();

  // Fetch weights
  const auto weight_memory = matrix_op.weight().memory();
  memory_map["vector1"] = get_partition(weight_memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = weight_memory.offset();
  vector_params->addressGen1Mode = 3;

  auto param_loops = squeeze_shape(non_reduction_loops);
  pad_shape_to_ndim(param_loops, 2);

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen1InputYLoopIndex[i] = 0;
    vector_params->addressGen1InputXLoopIndex[i] = 1;
    vector_params->addressGen1WeightLoopIndex[i] = 2;
  }

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen1Loops[0][i] = 1;
  }
  vector_params->addressGen1Loops[1][0] = param_loops[0];
  vector_params->addressGen1Loops[1][1] = param_loops[1];
  vector_params->addressGen1Loops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->DP_VEC1 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != matrix_op.weight().dtype();

  // Fetch bias
  if (matrix_op.has_bias()) {
    const auto bias_memory = matrix_op.bias().memory();
    memory_map["vector2"] = get_partition(bias_memory.partition());
    vector_params->ADDRESS_GEN2_OFFSET = bias_memory.offset();
    vector_params->addressGen2Mode = 3;

    for (int i = 0; i < 2; i++) {
      vector_params->addressGen2InputYLoopIndex[i] = 0;
      vector_params->addressGen2InputXLoopIndex[i] = 1;
      vector_params->addressGen2WeightLoopIndex[i] = 2;
    }

    for (int i = 0; i < 3; i++) {
      vector_params->addressGen2Loops[0][i] = 1;
    }
    vector_params->addressGen2Loops[1][0] = param_loops[0];
    vector_params->addressGen2Loops[1][1] = param_loops[1];
    vector_params->addressGen2Loops[1][2] = outer_dim / OC_DIMENSION;

    vector_params->DP_VEC2 =
        DataTypes::TypeName<INPUT_DATATYPE>::name() != matrix_op.bias().dtype();
  }

  const auto output_memory = param.output().memory();
  memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.offset();
  vector_params->outputAddressMode = 2;

  vector_params->outputLoops[0][0] = 1;
  vector_params->outputLoops[0][1] = non_reduction_loops[0];
  vector_params->outputLoops[0][2] = non_reduction_loops[1];
  vector_params->outputLoops[1][0] = non_reduction_loops[2];
  vector_params->outputLoops[1][1] = non_reduction_loops[3];
  vector_params->outputLoops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();

  // inputs x weights + bias
  VectorInstructions instr2;
  instr2.instType = VectorInstructions::vector;
  instr2.vInput = VectorInstructions::readFromVectorFetch;
  instr2.vAccumulatePush = VectorInstructions::nop;
  instr2.vOp0Src1 = VectorInstructions::readInterface;
  instr2.vOp0 = VectorInstructions::vmult;
  instr2.vOp1 = VectorInstructions::nop;
  instr2.vOp2 = VectorInstructions::nop;
  if (matrix_op.has_bias()) {
    instr2.vOp3Src1 = VectorInstructions::readNormalInterface;
    instr2.vOp3 = VectorInstructions::vadd;
  } else {
    instr2.vOp3Src1 = VectorInstructions::nop;
    instr2.vOp3 = VectorInstructions::nop;
  }
  instr2.vOp4 = VectorInstructions::nop;
  instr2.vDest = VectorInstructions::vWriteOut;
  vinstr_config->inst[0] = instr2;
  vinstr_config->instCount[0] = inner_dim * outer_dim / OC_DIMENSION;

  vinstr_config->instLen = 1;
  vinstr_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vinstr_config);
  opMemoryMaps.push_back(memory_map);
}
