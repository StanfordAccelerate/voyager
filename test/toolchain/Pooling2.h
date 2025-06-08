#pragma once

#include "test/common/Tiling.h"
#include "test/toolchain/Common.h"

void MapPoolingOperation(const codegen::Operation &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto op_list = get_op_list(param);
  const auto pooling_op = op_list.front();
  const auto tiling = get_pool2d_tiling(pooling_op);

  const auto input = pooling_op.kwargs().at("input").tensor();

  const auto output = param.output();
  const int output_dim = output.shape(1);

  // input
  const auto input_memory = input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->ADDRESS_GEN0_OFFSET = input_memory.address();
  vector_params->addr_gen0_mode = 1;
  vector_params->addr_gen0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  for (int i = 0; i < 2; i++) {
    vector_params->addr_gen0_loops[i][0] =
        tiling.loops[i][tiling.weight_loop_index[i]];
    vector_params->addr_gen0_loops[i][1] =
        tiling.loops[i][tiling.y_loop_index[i]];
    vector_params->addr_gen0_loops[i][2] =
        tiling.loops[i][tiling.x_loop_index[i]];

    vector_params->addr_gen0_x_loop_idx[i] = 2;
    vector_params->addr_gen0_y_loop_idx[i] = 1;
    vector_params->addr_gen0_k_loop_idx[i] = 0;
  }

  // striding only applied to x and y dimensions
  vector_params->stride[1] = tiling.stride;    // x
  vector_params->stride[0] = tiling.stride;    // y
  vector_params->padding[1] = tiling.padding;  // x
  vector_params->padding[0] = tiling.padding;  // y

  // output
  const auto output_memory = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = tiling.loops[0][tiling.y_loop_index[0]];
  vector_params->output_loops[1][1] = tiling.loops[0][tiling.x_loop_index[0]];
  vector_params->output_loops[1][2] = output_dim / (OC_DIMENSION);

  for (int i = 0; i < 2; i++) {
    vector_params->output_y_loop_idx[i] = 0;
    vector_params->output_x_loop_idx[i] = 1;
    vector_params->output_k_loop_idx[i] = 2;
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  const int reduce_count = tiling.loops[1][tiling.y_loop_index[1]] *
                           tiling.loops[1][tiling.x_loop_index[1]];

  const int inst_count = tiling.loops[0][tiling.y_loop_index[0]] *
                         tiling.loops[0][tiling.x_loop_index[0]] *
                         (output_dim / OC_DIMENSION);

  bool is_max_pool = pooling_op.target().find("max") != std::string::npos;
  vector_params->is_maxpool = is_max_pool;

  // perform max/sum accumulation
  VectorInstructions vinst0;
  vinst0.op_type = VectorInstructions::accumulation;
  vinst0.inst_count = inst_count;
  vinst0.reduce_count = reduce_count;
  vinst0.reduce_op =
      is_max_pool ? VectorInstructions::rmax : VectorInstructions::radd;
  vinst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // feed accumulator
  VectorInstructions vinst1;
  vinst1.op_type = VectorInstructions::vector;
  vinst1.inst_count = inst_count * reduce_count;
  vinst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  vinst1.vdest = VectorInstructions::to_accumulate;

  if (!is_max_pool) {
    vinst1.vector_op2 = VectorInstructions::vmult;
    vinst1.vector_op2_src1 = VectorInstructions::from_immediate_1;
    int kernel_size = tiling.loops[1][tiling.x_loop_index[1]];
    VECTOR_DATATYPE scale = 1.0 / (kernel_size * kernel_size);
    vinst1.immediate1 = scale.bits_rep();
  }

  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = 1;

  vector_instruction_config->instLen = 2;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
