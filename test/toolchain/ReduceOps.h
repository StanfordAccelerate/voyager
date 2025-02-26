#pragma once

#include "test/toolchain/Common.h"

void MapReduce(const codegen::Operation &param,
               std::deque<BaseParams *> &mappedParams,
               std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto op_list = get_op_list(param);
  const auto quantize_mx_op = op_list[0];

  const auto input = quantize_mx_op.kwargs().at("input").tensor();
  const auto input_shape = get_shape(input);
  const int input_size = get_size(input);

  const int axis = quantize_mx_op.kwargs().at("axis").int_value();
  const int block_size = quantize_mx_op.kwargs().at("block_size").int_value();

  // Microscaling on the last dimension can be handled by vector operations
  if (axis != -1 && axis != input_shape.size() - 1) {
    return;
  }

  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *config = new VectorInstructionConfig;
  AcceleratorMemoryMap memory_map;

  const auto input_memory = input.memory();
  memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.address();
  vector_params->addressGen0Mode = 2;
  vector_params->vec0_broadcast = 0b010000;

  // Fetch inputs twice, once for calculating mean and once for subtracting mean
  vector_params->addressGen0Loop[0][0] = non_reduction_loops[0];
  vector_params->addressGen0Loop[0][1] = non_reduction_loops[1];
  vector_params->addressGen0Loop[0][2] = non_reduction_loops[2];
  vector_params->addressGen0Loop[1][0] = non_reduction_loops[3];
  vector_params->addressGen0Loop[1][1] = 2;
  vector_params->addressGen0Loop[1][2] = outer_dim / OC_DIMENSION;

  vector_params->fetch_vector_type_0 =
      input.dtype() == DataTypes::TypeName<VECTOR_DATATYPE>::name();

  // Overwrite inputs
  memory_map["outputs"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = input_memory.address();
  vector_params->outputAddressMode = 2;

  vector_params->outputLoops[0][0] = 1;
  vector_params->outputLoops[0][1] = non_reduction_loops[0];
  vector_params->outputLoops[0][2] = non_reduction_loops[1];
  vector_params->outputLoops[1][0] = non_reduction_loops[2];
  vector_params->outputLoops[1][1] = non_reduction_loops[3];
  vector_params->outputLoops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->output_vector_type =
      input.dtype() == DataTypes::TypeName<VECTOR_DATATYPE>::name();

  // perform max/sum reduction
  VectorInstructions vinst0;
  vinst0.instType = VectorInstructions::accumulation;
  vinst0.rOp = VectorInstructions::rmax;
  vinst0.rCount = block_size;
  vinst0.rDuplicate = false;
  vinst0.rSqrt = false;
  vinst0.rReciprocal = false;
  vinst0.rBroadcast = false;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = input_size / block_size;

  // feed accumulator
  VectorInstructions vinst1;
  vinst1.instType = VectorInstructions::vector;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vector_op0_src1 = VectorInstructions::nop;
  vinst1.vector_op0 = VectorInstructions::nop;
  vinst1.vector_op1 = VectorInstructions::nop;
  vinst1.vector_op2 = VectorInstructions::nop;
  vinst1.vector_op3 = VectorInstructions::nop;
  vinst1.vdest = VectorInstructions::to_accumulate;
  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = inst_count;

  // read out from max accumulator and write out
  VectorInstructions vinst2;
  vinst2.instType = VectorInstructions::vector;
  vinst2.vector_op0_src0 = VectorInstructions::from_accumulation;
  vinst2.vector_op0_src1 = VectorInstructions::nop;
  vinst2.vector_op0 = VectorInstructions::nop;
  vinst2.vector_op1 = VectorInstructions::nop;
  vinst2.vector_op2 = VectorInstructions::nop;
  vinst2.vector_op3 = VectorInstructions::nop;
  vinst2.vdest = VectorInstructions::to_output;
}
