#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

void MapReduceOperation(const codegen::AcceleratorParam &param,
                        std::deque<BaseParams *> &mappedParams,
                        std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto &reduce_param = param.reduce_param();
  if (reduce_param.opcode() == "softmax") {
    const auto input = reduce_param.input();
    int x_dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      x_dim *= input.shape(i);
    }
    int y_dim = input.shape(input.shape_size() - 1);

    VectorParams *vectorParams = new VectorParams;
    VectorInstructionConfig *vectorInstructionConfig =
        new VectorInstructionConfig;
    AcceleratorMemoryMap acceleratorMemoryMap;

    const auto input_mem = reduce_param.input().memory();
    acceleratorMemoryMap["vector0"] = input_mem.partition() == 0 ? SRAM : RRAM;
    vectorParams->VECTOR_OFFSET = input_mem.offset();
    vectorParams->addressGen0Mode = true;
    vectorParams->addressGen0Broadcast = false;
    vectorParams->addressGen0Loop[0][0] = 1;
    vectorParams->addressGen0Loop[0][1] = x_dim;
    vectorParams->addressGen0Loop[0][2] = 1;
    vectorParams->addressGen0Loop[1][0] = 3;  // requires 3 passes
    vectorParams->addressGen0Loop[1][1] = 1;
    vectorParams->addressGen0Loop[1][2] = y_dim / OC_DIMENSION;
    // TODO: double precision
    vectorParams->DP_VEC0 = false;

    // turn off address generators
    vectorParams->addressGen1Mode = 0;
    vectorParams->addressGen2Mode = 0;

    const auto output_mem = param.output().memory();
    vectorParams->VECTOR_OUTPUT_OFFSET = output_mem.offset();
    vectorParams->SCALAR_OUTPUT_OFFSET = output_mem.offset();
    // vectorParams->scalarOutputCount = 0;

    // output
    acceleratorMemoryMap["outputs"] = output_mem.partition() == 0 ? SRAM : RRAM;
    for (int i = 0; i < 3; i++) {
      vectorParams->outputLoops[0][i] = 1;
    }
    vectorParams->outputXLoopIndex[0] = 0;
    vectorParams->outputYLoopIndex[0] = 1;
    vectorParams->outputWeightLoopIndex[0] = 2;

    vectorParams->outputLoops[1][0] = 1;
    vectorParams->outputLoops[1][1] = x_dim;
    vectorParams->outputLoops[1][2] = y_dim / OC_DIMENSION;
    vectorParams->outputWeightLoopIndex[1] = 2;
    vectorParams->outputYLoopIndex[1] = 1;
    vectorParams->outputXLoopIndex[1] = 0;
    // TODO: double precision
    vectorParams->DP_OUTPUT = false;
    vectorParams->SPLIT_OUTPUT = false;

    // sendSerializedParams<VectorParams, 32>(vectorParams,
    // &serialVectorParamsIn);

    // create instruction stream
    // VectorInstructionConfig vectorInstructionConfig;

    // inst 0- start reduction engine to calculate max
    VectorInstructions vInst0;
    vInst0.instType = VectorInstructions::reduction;
    vInst0.rCount = y_dim / OC_DIMENSION;
    vInst0.rOp = VectorInstructions::rmax;
    vInst0.rDuplicate = 1;
    vInst0.rDest = VectorInstructions::toVectorOp0Src1;
    vInst0.rBroadcast = 1;
    // broadcast max over entire array, for 2 passes
    ac_int<16, false> vInst0_broadcastCount = 2 * y_dim / OC_DIMENSION;
    vInst0.immediate0 = vInst0_broadcastCount.slc<8>(0);
    vInst0.immediate1 = vInst0_broadcastCount.slc<8>(8);
    vInst0.rSqrt = false;
    vInst0.rReciprocal = false;

    vectorInstructionConfig->inst[0] = vInst0;
    vectorInstructionConfig->instCount[0] = 1;

    // inst 1- send to max
    VectorInstructions vInst1;
    vInst1.instType = VectorInstructions::vector;
    vInst1.vInput = VectorInstructions::readFromVectorFetch;
    vInst1.vAccumulatePush = VectorInstructions::nop;
    vInst1.vOp0Src1 = VectorInstructions::nop;
    vInst1.vOp0 = VectorInstructions::nop;
    vInst1.vOp1 = VectorInstructions::nop;
    vInst1.vOp2 = VectorInstructions::toReduce;
    vInst1.vOp3Src1 = VectorInstructions::nop;
    vInst1.vOp3 = VectorInstructions::nop;
    vInst1.vOp4 = VectorInstructions::nop;
    vInst1.vDest = VectorInstructions::nop;
    vectorInstructionConfig->inst[1] = vInst1;
    vectorInstructionConfig->instCount[1] = y_dim / OC_DIMENSION;

    // inst 2- start reduction engine to calculate sum
    VectorInstructions vInst2;
    vInst2.instType = VectorInstructions::reduction;
    vInst2.rCount = y_dim / OC_DIMENSION;
    vInst2.rOp = VectorInstructions::radd;
    vInst2.rDuplicate = 1;
    vInst2.rDest = VectorInstructions::toVectorOp3Src1;
    vInst2.rBroadcast = 1;
    // broadcast max over entire array
    ac_int<16, false> vInst2_broadcastCount = y_dim / OC_DIMENSION;
    vInst2.immediate0 = vInst2_broadcastCount.slc<8>(0);
    vInst2.immediate1 = vInst2_broadcastCount.slc<8>(8);
    vInst2.rSqrt = false;
    vInst2.rReciprocal = true;
    vectorInstructionConfig->inst[2] = vInst2;
    vectorInstructionConfig->instCount[2] = 1;

    // inst 3- subtract max and exp, and reduce sum
    VectorInstructions vInst3;
    vInst3.instType = VectorInstructions::vector;
    vInst3.vInput = VectorInstructions::readFromVectorFetch;
    vInst3.vAccumulatePush = VectorInstructions::nop;
    vInst3.vOp0Src1 = VectorInstructions::readFromReduce;
    vInst3.vOp0 = VectorInstructions::vsub;
    vInst3.vOp1 = VectorInstructions::vexp;
    vInst3.vOp2 = VectorInstructions::toReduce;
    vInst3.vOp3Src1 = VectorInstructions::nop;
    vInst3.vOp3 = VectorInstructions::nop;
    vInst3.vOp4 = VectorInstructions::nop;
    vInst3.vDest = VectorInstructions::nop;
    vectorInstructionConfig->inst[3] = vInst3;
    vectorInstructionConfig->instCount[3] = y_dim / OC_DIMENSION;

    // inst 4- subtract max and exp, and divide by reduced value
    VectorInstructions vInst4;
    vInst4.instType = VectorInstructions::vector;
    vInst4.vInput = VectorInstructions::readFromVectorFetch;
    vInst4.vAccumulatePush = VectorInstructions::nop;
    vInst4.vOp0Src1 = VectorInstructions::readFromReduce;
    vInst4.vOp0 = VectorInstructions::vsub;
    vInst4.vOp1 = VectorInstructions::vexp;
    vInst4.vOp2 = VectorInstructions::nop;
    vInst4.vOp3Src1 = VectorInstructions::readReduceInterface;
    vInst4.vOp3 = VectorInstructions::vmult;
    vInst4.vOp4 = VectorInstructions::nop;
    vInst4.vDest = VectorInstructions::vWriteOut;
    vectorInstructionConfig->inst[4] = vInst4;
    vectorInstructionConfig->instCount[4] = y_dim / OC_DIMENSION;

    vectorInstructionConfig->instLen = 5;
    vectorInstructionConfig->instLoopCount = x_dim;

    mappedParams.push_back(vectorParams);
    mappedParams.push_back(vectorInstructionConfig);
    opMemoryMaps.push_back(acceleratorMemoryMap);
  } else {
    std::cerr << "Unsupported reduce instruction: " << reduce_param.opcode()
              << std::endl;
    exit(1);
  }
}