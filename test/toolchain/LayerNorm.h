#pragma once

#include "test/toolchain/Common.h"

void MapLayerNorm(const codegen::Operation &param,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto op_list = get_op_list(param);
  const auto layer_norm_op = op_list[0];

  const auto input = layer_norm_op.kwargs().at("input").tensor();
  const auto weight = layer_norm_op.kwargs().at("weight").tensor();

  codegen::Tensor output;

  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  VectorParams *vector_params;
  VectorInstructionConfig *vinstr_config;
  AcceleratorMemoryMap memory_map;

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

  // ======================================================================
  // Pass 1: calculate mean and subtract mean from tensor
  // ======================================================================
  vector_params = new VectorParams;
  vinstr_config = new VectorInstructionConfig;

  const auto input_memory = input.memory();
  memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->ADDRESS_GEN0_OFFSET = input_memory.address();
  vector_params->addr_gen0_mode = 2;
  vector_params->addr_gen0_broadcast = 0b010000;
  vector_params->addr_gen0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  // Fetch inputs twice, once for calculating mean and once for subtracting mean
  vector_params->addr_gen0_loops[0][0] = non_reduction_loops[0];
  vector_params->addr_gen0_loops[0][1] = non_reduction_loops[1];
  vector_params->addr_gen0_loops[0][2] = non_reduction_loops[2];
  vector_params->addr_gen0_loops[1][0] = non_reduction_loops[3];
  vector_params->addr_gen0_loops[1][1] = 2;
  vector_params->addr_gen0_loops[1][2] = outer_dim / OC_DIMENSION;

  // Overwrite inputs
  memory_map["outputs"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = input_memory.address();
  vector_params->output_mode = 2;

  vector_params->output_loops[0][0] = 1;
  vector_params->output_loops[0][1] = non_reduction_loops[0];
  vector_params->output_loops[0][2] = non_reduction_loops[1];
  vector_params->output_loops[1][0] = non_reduction_loops[2];
  vector_params->output_loops[1][1] = non_reduction_loops[3];
  vector_params->output_loops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(input.dtype());

  // Configure reduction engine
  VectorInstructions instr0_0;
  instr0_0.op_type = VectorInstructions::reduction;
  instr0_0.reduce_count = outer_dim / OC_DIMENSION;
  instr0_0.reduce_op = VectorInstructions::radd;
  instr0_0.rduplicate = 1;
  instr0_0.rbroadcast = 1;
  instr0_0.rdest = VectorInstructions::to_op0;
  instr0_0.immediate0 = outer_dim / OC_DIMENSION;
  vinstr_config->inst[0] = instr0_0;
  vinstr_config->instCount[0] = 1;

  // Scale inputs by 1 / norm_size and send to the reduction engine
  VectorInstructions instr0_1;
  instr0_1.op_type = VectorInstructions::vector;
  instr0_1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  instr0_1.vector_op0_src1 = VectorInstructions::from_immediate_0;
  instr0_1.vector_op0 = VectorInstructions::vmult;
  VECTOR_DATATYPE immediate = 1.0 / outer_dim;
  instr0_1.immediate0 = immediate.bits_rep();
  instr0_1.vdest = VectorInstructions::to_reduce;
  vinstr_config->inst[1] = instr0_1;
  vinstr_config->instCount[1] = outer_dim / OC_DIMENSION;

  // Subtract mean from tensor
  VectorInstructions instr0_2;
  instr0_2.op_type = VectorInstructions::vector;
  instr0_2.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  instr0_2.vector_op0_src1 = VectorInstructions::from_reduction_0;
  instr0_2.vector_op0 = VectorInstructions::vsub;
  instr0_2.vdest = VectorInstructions::to_output;
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
  vector_params->ADDRESS_GEN0_OFFSET = input_memory.address();
  vector_params->addr_gen0_mode = 2;
  vector_params->addr_gen0_broadcast = 0b010000;
  vector_params->addr_gen0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  // Fetch inputs twice, once for calculating variance and once for division
  vector_params->addr_gen0_loops[0][0] = non_reduction_loops[0];
  vector_params->addr_gen0_loops[0][1] = non_reduction_loops[1];
  vector_params->addr_gen0_loops[0][2] = non_reduction_loops[2];
  vector_params->addr_gen0_loops[1][0] = non_reduction_loops[3];
  vector_params->addr_gen0_loops[1][1] = 2;
  vector_params->addr_gen0_loops[1][2] = outer_dim / OC_DIMENSION;

  // Overwrite inputs
  memory_map["outputs"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = input_memory.address();
  vector_params->output_mode = 2;

  vector_params->output_loops[0][0] = 1;
  vector_params->output_loops[0][1] = non_reduction_loops[0];
  vector_params->output_loops[0][2] = non_reduction_loops[1];
  vector_params->output_loops[1][0] = non_reduction_loops[2];
  vector_params->output_loops[1][1] = non_reduction_loops[3];
  vector_params->output_loops[1][2] = outer_dim / OC_DIMENSION;

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(input.dtype());

  // Configure reduction unit
  VectorInstructions instr1_0;
  instr1_0.op_type = VectorInstructions::reduction;
  instr1_0.reduce_count = outer_dim / OC_DIMENSION;
  instr1_0.reduce_op = VectorInstructions::radd;
  instr1_0.rsqrt = 1;
  instr1_0.rreciprocal = 1;
  instr1_0.rduplicate = 1;
  instr1_0.rbroadcast = 1;
  instr1_0.rdest = VectorInstructions::to_op0;
  instr1_0.immediate0 = outer_dim / OC_DIMENSION;
  vinstr_config->inst[0] = instr1_0;
  vinstr_config->instCount[0] = 1;

  // Perform squaring and send outputs to reduction engine
  VectorInstructions instr1_1;
  instr1_1.op_type = VectorInstructions::vector;
  instr1_1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  instr1_1.vector_op2 = VectorInstructions::vsquare;
  instr1_1.vdest = VectorInstructions::to_reduce;
  vinstr_config->inst[1] = instr1_1;
  vinstr_config->instCount[1] = outer_dim / OC_DIMENSION;

  // Multiply inputs with the inverse sqrt of the variance
  VectorInstructions instr1_2;
  instr1_2.op_type = VectorInstructions::vector;
  instr1_2.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  instr1_2.vector_op0_src1 = VectorInstructions::from_reduction_0;
  instr1_2.vector_op2_src1 = VectorInstructions::from_immediate_1;
  instr1_2.vector_op0 = VectorInstructions::vmult;
  VECTOR_DATATYPE divisor = sqrt(outer_dim);
  instr1_2.immediate1 = divisor.bits_rep();
  instr1_2.vector_op2 = VectorInstructions::vmult;
  instr1_2.vdest = VectorInstructions::to_output;
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
  vector_params->ADDRESS_GEN0_OFFSET = input_memory.address();
  vector_params->addr_gen0_mode = 2;
  vector_params->addr_gen0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  vector_params->addr_gen0_loops[0][0] = 1;
  vector_params->addr_gen0_loops[0][1] = non_reduction_loops[0];
  vector_params->addr_gen0_loops[0][2] = non_reduction_loops[1];
  vector_params->addr_gen0_loops[1][0] = non_reduction_loops[2];
  vector_params->addr_gen0_loops[1][1] = non_reduction_loops[3];
  vector_params->addr_gen0_loops[1][2] = outer_dim / OC_DIMENSION;

  // Fetch weights
  const auto weight_memory = weight.memory();
  memory_map["vector1"] = get_partition(weight_memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = weight_memory.address();
  vector_params->addr_gen1_mode = true;
  vector_params->addr_gen1_broadcast = 0b011;
  vector_params->addr_gen1_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(weight.dtype());

  auto param_loops = squeeze_shape(non_reduction_loops);
  pad_shape_to_ndim(param_loops, 2);

  for (int i = 0; i < 2; i++) {
    vector_params->addr_gen1_y_loop_idx[i] = 0;
    vector_params->addr_gen1_x_loop_idx[i] = 1;
    vector_params->addr_gen1_k_loop_idx[i] = 2;
  }

  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen1_loops[0][i] = 1;
  }
  vector_params->addr_gen1_loops[1][0] = param_loops[0];
  vector_params->addr_gen1_loops[1][1] = param_loops[1];
  vector_params->addr_gen1_loops[1][2] = outer_dim / OC_DIMENSION;

  // Fetch bias
  const bool has_bias = layer_norm_op.kwargs().contains("bias");
  if (has_bias) {
    const auto bias = layer_norm_op.kwargs().at("bias").tensor();
    const auto bias_memory = bias.memory();
    memory_map["vector2"] = get_partition(bias_memory.partition());
    vector_params->ADDRESS_GEN2_OFFSET = bias_memory.address();
    vector_params->addr_gen2_mode = true;
    vector_params->addr_gen2_broadcast = 0b011;
    vector_params->addr_gen2_dtype =
        get_index_from_type_name<VU_INPUT_TYPES>(bias.dtype());

    for (int i = 0; i < 2; i++) {
      vector_params->addr_gen2_y_loop_idx[i] = 0;
      vector_params->addr_gen2_x_loop_idx[i] = 1;
      vector_params->addr_gen2_k_loop_idx[i] = 2;
    }

    for (int i = 0; i < 3; i++) {
      vector_params->addr_gen2_loops[0][i] = 1;
    }
    vector_params->addr_gen2_loops[1][0] = param_loops[0];
    vector_params->addr_gen2_loops[1][1] = param_loops[1];
    vector_params->addr_gen2_loops[1][2] = outer_dim / OC_DIMENSION;
  }

  const auto output_memory = output.memory();
  memory_map["outputs"] = get_partition(output_memory.partition());
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

  // inputs x weights + bias
  VectorInstructions inst2;
  inst2.op_type = VectorInstructions::vector;
  inst2.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst2.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst2.vector_op0 = VectorInstructions::vmult;
  if (has_bias) {
    inst2.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
    inst2.vector_op2 = VectorInstructions::vadd;
  }
  inst2.vdest = VectorInstructions::to_output;

  set_quantize_params(param, vector_params, inst2);

  vinstr_config->inst[0] = inst2;
  vinstr_config->instCount[0] = inner_dim * outer_dim / OC_DIMENSION;

  vinstr_config->instLen = 1;
  vinstr_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vinstr_config);
  opMemoryMaps.push_back(memory_map);
}
