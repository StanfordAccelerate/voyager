#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/Common.h"

void MapPoolingOperation(const codegen::AcceleratorParam &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params = new VectorParams;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto pooling_param = param.pooling_param();
  const auto tiling = get_pooling_tiling(param);
  int output_dim = param.output().shape(1);

  // input
  const auto input_memory = pooling_param.input().memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 1;
  // set double precision if the datatype is not the same as the input datatype
  vector_params->DP_VEC0 = DataTypes::TypeName<INPUT_DATATYPE>::name() !=
                           pooling_param.input().dtype();

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen0Loop[i][0] =
        tiling.loops[i][tiling.weight_loop_index[i]];
    vector_params->addressGen0Loop[i][1] =
        tiling.loops[i][tiling.y_loop_index[i]];
    vector_params->addressGen0Loop[i][2] =
        tiling.loops[i][tiling.x_loop_index[i]];

    vector_params->addressGen0InputXLoopIndex[i] = 2;
    vector_params->addressGen0InputYLoopIndex[i] = 1;
    vector_params->addressGen0WeightLoopIndex[i] = 0;
  }

  vector_params->addressGen0Broadcast = false;

  vector_params->addressGen1Mode = 0;
  vector_params->addressGen2Mode = 0;

  // output
  const auto output_memory = param.output().memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.offset();

  for (int i = 0; i < 3; i++) {
    vector_params->outputLoops[0][i] = 1;
  }
  vector_params->outputLoops[1][0] = tiling.loops[0][tiling.y_loop_index[0]];
  vector_params->outputLoops[1][1] = tiling.loops[0][tiling.x_loop_index[0]];
  vector_params->outputLoops[1][2] = output_dim / (OC_DIMENSION);

  for (int i = 0; i < 2; i++) {
    vector_params->outputYLoopIndex[i] = 0;
    vector_params->outputXLoopIndex[i] = 1;
    vector_params->outputWeightLoopIndex[i] = 2;
  }
  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();

  const int inst_count = tiling.loops[1][tiling.y_loop_index[1]] *
                         tiling.loops[1][tiling.x_loop_index[1]];

  bool is_max_pool = pooling_param.opcode().find("max") != std::string::npos;

  // perform max/sum reduction
  VectorInstructions vinst0;
  vinst0.instType = VectorInstructions::accumulation;
  vinst0.rOp =
      is_max_pool ? VectorInstructions::rmax : VectorInstructions::radd;
  vinst0.rCount = inst_count;
  vinst0.rDuplicate = false;
  vinst0.rSqrt = false;
  vinst0.rReciprocal = false;
  vinst0.rBroadcast = false;
  vinst0.rDest = VectorInstructions::toVectorOp0Src0;
  vector_instruction_config->inst[0] = vinst0;
  vector_instruction_config->instCount[0] = 1;

  // feed accumulator
  VectorInstructions vinst1;
  vinst1.instType = VectorInstructions::vector;
  vinst1.vInput = VectorInstructions::readFromVectorFetch;
  vinst1.vOp0Src1 = VectorInstructions::nop;
  vinst1.vOp0 = VectorInstructions::nop;
  vinst1.vOp1 = VectorInstructions::nop;
  vinst1.vOp2 = VectorInstructions::nop;
  vinst1.vOp3 = VectorInstructions::nop;
  vinst1.vOp4 = VectorInstructions::nop;
  vinst1.vDest = VectorInstructions::nop;
  vinst1.vAccumulatePush = true;
  vector_instruction_config->inst[1] = vinst1;
  vector_instruction_config->instCount[1] = inst_count;

  // read out from max accumulator and write out
  VectorInstructions vinst2;
  vinst2.instType = VectorInstructions::vector;
  vinst2.vInput = VectorInstructions::readFromAccumulation;
  vinst2.vOp0Src1 = VectorInstructions::nop;
  vinst2.vOp0 = VectorInstructions::nop;
  vinst2.vOp1 = VectorInstructions::nop;
  vinst2.vOp2 = VectorInstructions::nop;
  vinst2.vOp3 = VectorInstructions::nop;
  vinst2.vOp4 = VectorInstructions::nop;
  vinst2.vDest = VectorInstructions::vWriteOut;

  if (!is_max_pool) {
    vinst2.vOp3 = VectorInstructions::vmult;
    vinst2.vOp3Src1 = VectorInstructions::op3immediate;
    int kernel_size = tiling.loops[1][tiling.x_loop_index[1]];
    VECTOR_DATATYPE scale = 1.0 / (kernel_size * kernel_size);
    vinst2.immediate1 = scale.bits_rep();
  }

  vector_instruction_config->inst[2] = vinst2;
  vector_instruction_config->instCount[2] = 1;

  vector_instruction_config->instLen = 3;
  vector_instruction_config->instLoopCount =
      tiling.loops[0][tiling.y_loop_index[0]] *
      tiling.loops[0][tiling.x_loop_index[0]] * (output_dim / OC_DIMENSION);

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
