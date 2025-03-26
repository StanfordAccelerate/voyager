#pragma once

#include "test/toolchain/Common.h"

void MapMicroscaling(const codegen::Operation &param,
                     std::deque<BaseParams *> &mapped_params,
                     std::deque<AcceleratorMemoryMap> &memory_maps) {
  const auto op_list = get_op_list(param);
  const auto quantize_mx_op = op_list[0];

  const auto input = quantize_mx_op.kwargs().at("input").tensor();
  const int axis = quantize_mx_op.kwargs().at("axis").int_value();
  const int block_size = quantize_mx_op.kwargs().at("block_size").int_value();
  const float quant_max = quantize_mx_op.kwargs().at("quant_max").float_value();
  const bool force_scale_power_of_two =
      quantize_mx_op.kwargs().at("force_scale_power_of_two").int_value();

  codegen::Tensor output;
  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  // Microscaling on the last dimension can be handled by vector operations
  auto input_shape = get_shape(input);
  if (axis != -2 && axis != input_shape.size() - 2) {
    throw std::invalid_argument(
        "Microscaling only supported on second to last dimension");
  }

  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  AcceleratorMemoryMap memory_map;

  const auto input_memory = input.memory();
  memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.address();
  vector_params->addressGen0Mode = 1;
  vector_params->vector_input_0_type =
      get_index_from_type_name<VECTOR_INPUT_DATATYPES>(input.dtype());

  input_shape = squeeze_shape(input_shape);
  input_shape = split_loops(input_shape, 1024);
  pad_shape_to_ndim(input_shape, 3);

  vector_params->addressGen0Loop[0][0] = input_shape[0];
  vector_params->addressGen0Loop[0][1] = input_shape[1] / block_size;
  vector_params->addressGen0Loop[0][2] = input_shape[2] / OC_DIMENSION;
  vector_params->addressGen0Loop[1][0] = 1;
  vector_params->addressGen0Loop[1][1] = block_size;
  vector_params->addressGen0Loop[1][2] = 1;

  vector_params->addressGen0InputYLoopIndex[0] = 0;
  vector_params->addressGen0InputXLoopIndex[0] = 1;
  vector_params->addressGen0WeightLoopIndex[0] = 2;

  const auto output_memory = output.memory();
  memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();
  vector_params->outputAddressMode = 1;

  vector_params->outputLoops[0][0] = input_shape[0];
  vector_params->outputLoops[0][1] = input_shape[1] / block_size;
  vector_params->outputLoops[0][2] = input_shape[2] / OC_DIMENSION;
  vector_params->outputLoops[1][0] = 1;
  vector_params->outputLoops[1][1] = 1;
  vector_params->outputLoops[1][2] = 1;

  vector_params->outputYLoopIndex[0] = 0;
  vector_params->outputXLoopIndex[0] = 1;
  vector_params->outputWeightLoopIndex[0] = 2;

  vector_params->output_types =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  // perform max accumulation
  VectorInstructions vinst0;
  vinst0.instType = VectorInstructions::accumulation;
  vinst0.rOp = VectorInstructions::rmax;
  vinst0.rCount = block_size;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // feed accumulator
  VectorInstructions vinst1;
  vinst1.instType = VectorInstructions::vector;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vector_op1 = VectorInstructions::vabs;
  vinst1.vdest = VectorInstructions::to_accumulate;
  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = block_size;

  // read out from max accumulator and scale by 1 / quant_max
  VectorInstructions vinst2;
  vinst2.instType = VectorInstructions::vector;
  vinst2.vector_op0_src0 = VectorInstructions::from_accumulation;
  vinst2.vector_op0_src1 = VectorInstructions::from_immediate_0;
  VECTOR_DATATYPE immediate;
  if (force_scale_power_of_two) {
    immediate = 1.0 / pow(2, floor(log2(quant_max)));
  } else {
    immediate = 1.0 / quant_max;
  }
  vinst2.immediate0 = immediate.bits_rep();
  vinst2.vector_op0 = VectorInstructions::vmult;
  vinst2.vdest = VectorInstructions::to_output;
  vector_instruction_config->inst[2] = vinst2;
  vector_instruction_config->instCount[2] = 1;

  vector_instruction_config->instLen = 3;
  vector_instruction_config->instLoopCount =
      get_size(input) / OC_DIMENSION / block_size;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
  memory_maps.push_back(memory_map);
}
