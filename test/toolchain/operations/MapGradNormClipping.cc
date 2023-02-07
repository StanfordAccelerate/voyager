#include "test/toolchain/operations/Operations.h"

void MapGradNormClipping(const SimplifiedParams &params, const MemoryMap &memoryMap,
                 std::deque<BaseParams *> &mappedParams,
                 std::deque<AcceleratorMemoryMap> &opMemoryMaps, int size) {
  // TODO: enable biasing exponent
  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;

  vectorParams->VECTOR_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->addressGen0Enable = true;
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[1][0] =
      2;  // two passes over tensor are required
  vectorParams->addressGen0Loop[1][1] = 1;
  vectorParams->addressGen0Loop[1][2] = size / DIMENSION;
  vectorParams->addressGen0Broadcast = false;
  vectorParams->DP_VEC0 = true;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 0;  // 2d tensor

  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = 0;  // 2d tensor

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  // vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = 0;
  vectorParams->AVGPOOL = 0;

  // output
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = 1;
  }
  vectorParams->outputXLoopIndex[0] = 0;
  vectorParams->outputYLoopIndex[0] = 1;
  vectorParams->outputWeightLoopIndex[0] = 2;

  vectorParams->outputLoops[1][0] = 1;
  vectorParams->outputLoops[1][1] = 1;
  vectorParams->outputLoops[1][2] = size / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 1;
  vectorParams->outputXLoopIndex[1] = 0;
  vectorParams->DP_OUTPUT = params.ACC_T_OUTPUT;

  // reduce entire tensor, and take inv sqrt of value
  // configure reduction unit
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::reduction;
  vInst0.rCount = size / DIMENSION;
  vInst0.rOp = VectorInstructions::radd;
  vInst0.rDuplicate = 1;
  vInst0.rInvSqrt = 1;
  vInst0.rMax1 = 1;
  vInst0.rDest = VectorInstructions::toVectorOp3Src1;
  vInst0.rBroadcast = 1;
  ac_int<16, false> size16b = size / DIMENSION;
  vInst0.immediate0 = size16b.slc<8>(0);
  vInst0.immediate1 = size16b.slc<8>(8);
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = 1;

  // send tensor squared to reduction
  VectorInstructions vInst1;
  vInst1.instType = VectorInstructions::vector;
  vInst1.vInput = VectorInstructions::readFromVectorFetch;
  vInst1.vAccumulatePush = VectorInstructions::nop;
  vInst1.vOp0Src1 = VectorInstructions::nop;
  vInst1.vOp0 = VectorInstructions::nop;
  vInst1.vOp1 = VectorInstructions::nop;
  vInst1.vOp2 = VectorInstructions::toReduce;

  vInst1.vOp3Src1 = VectorInstructions::nop;
  vInst1.vOp3 = VectorInstructions::vsquare;
  vInst1.vOp4 = VectorInstructions::nop;
  vInst1.vDest = VectorInstructions::nop;
  vectorInstructionConfig->inst[1] = vInst1;
  vectorInstructionConfig->instCount[1] = size / DIMENSION;

  // divide entire tensor by sqrt
  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::vector;
  vInst2.vInput = VectorInstructions::readFromVectorFetch;
  vInst2.vAccumulatePush = VectorInstructions::nop;
  vInst2.vOp0 = VectorInstructions::nop;
  vInst2.vOp1 = VectorInstructions::nop;
  vInst2.vOp2 = VectorInstructions::nop;
  vInst2.vOp3Src1 = VectorInstructions::readReduceInterface;
  vInst2.vOp3 = VectorInstructions::vmult;
  vInst2.vOp4 = VectorInstructions::nop;
  vInst2.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[2] = vInst2;
  vectorInstructionConfig->instCount[2] = size / DIMENSION;

  vectorInstructionConfig->instLen = 3;
  vectorInstructionConfig->instLoopCount = 1;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
}