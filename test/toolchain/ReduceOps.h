#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/Common.h"

void MapReduceOperation(const codegen::AcceleratorParam &param,
                        std::deque<BaseParams *> &mappedParams,
                        std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto &reduce_param = param.reduce_param();
  if (reduce_param.opcode() == "softmax") {
    VectorParams *vector_params = new VectorParams;
    VectorInstructionConfig *vector_instruction_config =
        new VectorInstructionConfig;
    AcceleratorMemoryMap accelerator_memory_map;

    const auto vector_input = reduce_param.input();
    int input_dim = 1;
    for (int i = 0; i < vector_input.shape_size() - 1; i++) {
      input_dim *= vector_input.shape(i);
    }
    int output_dim = vector_input.shape(vector_input.shape_size() - 1);

    // inputs
    const auto input_memory = vector_input.memory();
    accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
    vector_params->VECTOR_OFFSET = input_memory.offset();
    vector_params->addressGen0Mode = 2;
    vector_params->addressGen0Broadcast = false;
    vector_params->addressGen0Loop[0][0] = input_dim;
    vector_params->addressGen0Loop[0][1] = 1;
    vector_params->addressGen0Loop[0][2] = 1;
    vector_params->addressGen0Loop[1][0] = 1;
    vector_params->addressGen0Loop[1][1] = 3;
    vector_params->addressGen0Loop[1][2] = output_dim / OC_DIMENSION;

    for (int i = 0; i < 2; i++) {
      vector_params->addressGen0InputXLoopIndex[i] = 0;
      vector_params->addressGen0InputYLoopIndex[i] = 1;
      vector_params->addressGen0WeightLoopIndex[i] = 2;
    }

    // TODO: double precision
    vector_params->DP_VEC0 =
        DataTypes::TypeName<INPUT_DATATYPE>::name() != vector_input.dtype();

    // turn off address generators
    vector_params->addressGen1Mode = 0;
    vector_params->addressGen2Mode = 0;

    // output
    const auto output_mem = param.output().memory();
    accelerator_memory_map["outputs"] = get_partition(output_mem.partition());
    vector_params->VECTOR_OUTPUT_OFFSET = output_mem.offset();
    for (int i = 0; i < 3; i++) {
      vector_params->outputLoops[0][i] = 1;
    }
    vector_params->outputLoops[1][0] = 1;
    vector_params->outputLoops[1][1] = input_dim;
    vector_params->outputLoops[1][2] = output_dim / OC_DIMENSION;

    for (int i = 0; i < 2; i++) {
      vector_params->outputXLoopIndex[i] = 0;
      vector_params->outputYLoopIndex[i] = 1;
      vector_params->outputWeightLoopIndex[i] = 2;
    }

    // TODO: double precision
    vector_params->DP_OUTPUT =
        DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();
    vector_params->SPLIT_OUTPUT = false;

    // inst 0 - start reduction engine to calculate max
    VectorInstructions vinst0;
    vinst0.instType = VectorInstructions::reduction;
    vinst0.rCount = output_dim / OC_DIMENSION;
    vinst0.rOp = VectorInstructions::rmax;
    vinst0.rDuplicate = 1;
    vinst0.rDest = VectorInstructions::toVectorOp0Src1;
    vinst0.rBroadcast = 1;
    // broadcast max over entire array, for 2 passes
    vinst0.immediate0 = 2 * output_dim / OC_DIMENSION;
    vinst0.rSqrt = false;
    vinst0.rReciprocal = false;
    vector_instruction_config->inst[0] = vinst0;
    vector_instruction_config->instCount[0] = 1;

    // inst 1 - send to max
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
    vector_instruction_config->instCount[1] = output_dim / OC_DIMENSION;

    // inst 2 - start reduction engine to calculate sum
    VectorInstructions vinst2;
    vinst2.instType = VectorInstructions::reduction;
    vinst2.rCount = output_dim / OC_DIMENSION;
    vinst2.rOp = VectorInstructions::radd;
    vinst2.rDuplicate = 1;
    vinst2.rDest = VectorInstructions::toVectorOp3Src1;
    vinst2.rBroadcast = 1;
    // broadcast max over entire array
    vinst2.immediate0 = output_dim / OC_DIMENSION;
    vinst2.rSqrt = false;
    vinst2.rReciprocal = true;
    vector_instruction_config->inst[2] = vinst2;
    vector_instruction_config->instCount[2] = 1;

    // inst 3 - subtract max and exp, and reduce sum
    VectorInstructions vinst3;
    vinst3.instType = VectorInstructions::vector;
    vinst3.vInput = VectorInstructions::readFromVectorFetch;
    vinst3.vAccumulatePush = VectorInstructions::nop;
    vinst3.vOp0Src1 = VectorInstructions::readFromReduce;
    vinst3.vOp0 = VectorInstructions::vsub;
    vinst3.vOp1 = VectorInstructions::vexp;
    vinst3.vOp2 = VectorInstructions::toReduce;
    vinst3.vOp3Src1 = VectorInstructions::nop;
    vinst3.vOp3 = VectorInstructions::nop;
    vinst3.vOp4 = VectorInstructions::nop;
    vinst3.vDest = VectorInstructions::nop;
    vector_instruction_config->inst[3] = vinst3;
    vector_instruction_config->instCount[3] = output_dim / OC_DIMENSION;

    // inst 4 - subtract max and exp, and divide by reduced value
    VectorInstructions vinst4;
    vinst4.instType = VectorInstructions::vector;
    vinst4.vInput = VectorInstructions::readFromVectorFetch;
    vinst4.vAccumulatePush = VectorInstructions::nop;
    vinst4.vOp0Src1 = VectorInstructions::readFromReduce;
    vinst4.vOp0 = VectorInstructions::vsub;
    vinst4.vOp1 = VectorInstructions::vexp;
    vinst4.vOp2 = VectorInstructions::nop;
    vinst4.vOp3Src1 = VectorInstructions::readReduceInterface;
    vinst4.vOp3 = VectorInstructions::vmult;
    vinst4.vOp4 = VectorInstructions::nop;
    vinst4.vDest = VectorInstructions::vWriteOut;
    vector_instruction_config->inst[4] = vinst4;
    vector_instruction_config->instCount[4] = output_dim / OC_DIMENSION;

    vector_instruction_config->instLen = 5;
    vector_instruction_config->instLoopCount = input_dim;

    mappedParams.push_back(vector_params);
    mappedParams.push_back(vector_instruction_config);
    opMemoryMaps.push_back(accelerator_memory_map);
  } else {
    std::cerr << "Unsupported reduce instruction: " << reduce_param.opcode()
              << std::endl;
    exit(1);
  }
}