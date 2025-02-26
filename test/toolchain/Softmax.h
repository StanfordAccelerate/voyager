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
  vector_params->VECTOR_OFFSET = input_memory.address();
  vector_params->addressGen0Mode = 2;
  vector_params->vec0_broadcast = 0b010000;

  vector_params->addressGen0Loop[0][0] = non_reduction_loops[0];
  vector_params->addressGen0Loop[0][1] = non_reduction_loops[1];
  vector_params->addressGen0Loop[0][2] = non_reduction_loops[2];
  vector_params->addressGen0Loop[1][0] = non_reduction_loops[3];
  vector_params->addressGen0Loop[1][1] = 3;
  vector_params->addressGen0Loop[1][2] = outer_dim / OC_DIMENSION;

  vector_params->fetch_vector_type_0 =
      input.dtype() == DataTypes::TypeName<VECTOR_DATATYPE>::name();

  // output
  const auto output_memory = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();
  vector_params->outputAddressMode = 2;

  vector_params->outputLoops[0][0] = 1;
  vector_params->outputLoops[0][1] = non_reduction_loops[0];
  vector_params->outputLoops[0][2] = non_reduction_loops[1];
  vector_params->outputLoops[1][0] = non_reduction_loops[2];
  vector_params->outputLoops[1][1] = non_reduction_loops[3];
  vector_params->outputLoops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->output_vector_type =
      output.dtype() == DataTypes::TypeName<VECTOR_DATATYPE>::name();

  // Instruction 0 - start reduction engine to calculate max
  VectorInstructions vinst0;
  vinst0.instType = VectorInstructions::reduction;
  vinst0.rCount = outer_dim / OC_DIMENSION;
  vinst0.rOp = VectorInstructions::rmax;
  vinst0.rDuplicate = 1;
  // broadcast max over entire array, for 2 passes
  vinst0.rBroadcast = 1;
  vinst0.immediate0 = 2 * outer_dim / OC_DIMENSION;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // Instruction 1 - send to reduction engine to calculate max
  VectorInstructions vinst1;
  vinst1.instType = VectorInstructions::vector;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = outer_dim / OC_DIMENSION;

  // Instruction 2 - start reduction engine to calculate sum
  VectorInstructions vinst2;
  vinst2.instType = VectorInstructions::reduction;
  vinst2.rCount = outer_dim / OC_DIMENSION;
  vinst2.rOp = VectorInstructions::radd;
  vinst2.rReciprocal = 1;
  vinst2.rDuplicate = 1;
  vinst2.rBroadcast = 1;
  vinst2.immediate0 = outer_dim / OC_DIMENSION;
  vinst2.rdest = VectorInstructions::to_op2;
  vector_instruction_config->inst[2] = vinst2;
  vector_instruction_config->instCount[2] = 1;

  // Instruction 3 - subtract max and exp, and reduce sum
  VectorInstructions vinst3;
  vinst3.instType = VectorInstructions::vector;
  vinst3.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst3.vector_op0_src1 = VectorInstructions::from_reduction_0;
  vinst3.vector_op0 = VectorInstructions::vsub;
  vinst3.vector_op1 = VectorInstructions::vexp;
  vinst3.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[3] = vinst3;
  vector_instruction_config->instCount[3] = outer_dim / OC_DIMENSION;

  // Instruction 4 - subtract max and exp, and divide by reduced value
  VectorInstructions inst4;
  inst4.instType = VectorInstructions::vector;
  inst4.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst4.vector_op0_src1 = VectorInstructions::from_reduction_0;
  inst4.vector_op2_src1 = VectorInstructions::from_reduction_1;
  inst4.vector_op0 = VectorInstructions::vsub;
  inst4.vector_op1 = VectorInstructions::vexp;
  inst4.vector_op2 = VectorInstructions::vmult;
  inst4.vdest = VectorInstructions::to_output;
  vector_instruction_config->inst[4] = inst4;
  vector_instruction_config->instCount[4] = outer_dim / OC_DIMENSION;

  if (op_list.size() > 1) {
    const auto quantize_op = op_list[1];
    std::cerr << quantize_op.target() << std::endl;

    if (quantize_op.target() == "quantize") {
      const auto scale = quantize_op.kwargs().at("scale").tensor();
      assert(get_size(scale) == 1);

      // scalar scale factor
      VECTOR_DATATYPE immediate = read_constant_param(scale);
      vector_params->quantize_output = true;
      vector_params->output_scale = immediate.bits_rep();
    } else if (quantize_op.target() == "quantize_mx") {
      const int block_size = quantize_op.kwargs().at("block_size").int_value();
      assert(block_size == OC_DIMENSION);

      vector_params->quantize_output_mx = true;
      vector_params->SCALE_OFFSET =
          param.outputs().tensors(0).memory().address();
    } else {
      throw std::invalid_argument("Unsupported operation: " +
                                  quantize_op.target());
    }
  }

  vector_instruction_config->instLen = 5;
  vector_instruction_config->instLoopCount = inner_dim;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}