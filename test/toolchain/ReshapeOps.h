#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/GoldModel.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/Common.h"

void MapReshapeOperation(const codegen::AcceleratorParam &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params = new VectorParams;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto reshape_param = param.reshape_param();
  const auto vector_input = reshape_param.input();
  if (vector_input.shape_size() != 4 || vector_input.shape(0) != 1) {
    std::cerr << "Only 4D tensors are supported for shape permute operations"
              << std::endl;
    exit(1);
  }

  const auto input_shape = get_shape(vector_input);
  std::vector<int> order;
  if (reshape_param.opcode() == "permute") {
    order.insert(order.end(), reshape_param.dims().begin(),
                 reshape_param.dims().end());
  } else if (reshape_param.opcode() == "transpose") {
    int dim1 = reshape_param.dims(0);
    int dim2 = reshape_param.dims(1);

    int size = vector_input.shape_size();
    if (dim1 < 0) dim1 += size;
    if (dim2 < 0) dim2 += size;

    order.insert(order.end(), {0, 1, 2, 3});
    std::swap(order[dim1], order[dim2]);
  } else {
    std::cerr << "Unsupported reshape instruction: " << reshape_param.opcode()
              << std::endl;
    exit(1);
  }

  const auto input_memory = vector_input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 1;
  vector_params->addressGen0Broadcast = false;

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen0Loop[0][i] = 1;
  }
  vector_params->addressGen0Loop[1][0] = input_shape[1];
  vector_params->addressGen0Loop[1][1] = input_shape[2];
  vector_params->addressGen0Loop[1][2] = input_shape[3] / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen0InputYLoopIndex[i] = 0;
    vector_params->addressGen0InputXLoopIndex[i] = 1;
    vector_params->addressGen0WeightLoopIndex[i] = 2;
  }

  // TODO: double precision
  vector_params->DP_VEC0 = false;

  const auto output_memory = param.output().memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.offset();

  for (int i = 0; i < 3; i++) {
    vector_params->outputLoops[0][i] = 1;
  }
  vector_params->outputLoops[1][0] = input_shape[1];
  vector_params->outputLoops[1][1] = input_shape[2];
  vector_params->outputLoops[1][2] = input_shape[3] / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->outputYLoopIndex[i] = order[1] - 1;
    vector_params->outputXLoopIndex[i] = order[2] - 1;
    vector_params->outputWeightLoopIndex[i] = order[3] - 1;
  }

  // TODO: double precision
  vector_params->DP_OUTPUT = false;

  VectorInstructions vinst;
  vinst.instType = VectorInstructions::vector;
  vinst.vInput = VectorInstructions::readFromVectorFetch;
  vinst.vAccumulatePush = VectorInstructions::nop;
  vinst.vOp0 = VectorInstructions::nop;
  vinst.vOp1 = VectorInstructions::nop;
  vinst.vOp2 = VectorInstructions::nop;
  vinst.vOp3 = VectorInstructions::nop;
  vinst.vOp4 = VectorInstructions::nop;
  vinst.vDest = VectorInstructions::vWriteOut;

  // total output count
  const int size = get_size(vector_input);
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  vector_instruction_config->inst[0] = vinst;
  vector_instruction_config->instCount[0] = size / OC_DIMENSION;
  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
