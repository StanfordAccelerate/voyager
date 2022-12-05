#include "test/toolchain/operations/Operations.h"

void MapSoftmax(const SimplifiedParams &params,
                std::deque<BaseParams *> &mappedParams) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;

  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;

  vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen0Enable = true;
  vectorParams->addressGen0Broadcast = false;
  vectorParams->addressGen0Loop[0][0] = 1;
  vectorParams->addressGen0Loop[0][1] = X;
  vectorParams->addressGen0Loop[0][2] = 1;
  vectorParams->addressGen0Loop[1][0] = 3;  // requires 3 passes
  vectorParams->addressGen0Loop[1][1] = 1;
  vectorParams->addressGen0Loop[1][2] = Y / DIMENSION;
  vectorParams->DP_VEC0 = false;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 0;  // 2d tensor

  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = 0;  // 2d tensor

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = params.MAXPOOL;
  vectorParams->AVGPOOL = params.AVGPOOL;

  // output
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = 1;
  }
  vectorParams->outputXLoopIndex[0] = 0;
  vectorParams->outputYLoopIndex[0] = 1;
  vectorParams->outputWeightLoopIndex[0] = 2;

  vectorParams->outputLoops[1][0] = 1;
  vectorParams->outputLoops[1][1] = X;
  vectorParams->outputLoops[1][2] = Y / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 1;
  vectorParams->outputXLoopIndex[1] = 0;
  vectorParams->DP_OUTPUT = false;

  // sendSerializedParams<VectorParams, 32>(vectorParams,
  // &serialVectorParamsIn);

  // create instruction stream
  //    VectorInstructionConfig vectorInstructionConfig;

  // inst 0- start reduction engine to calculate max
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::reduction;
  vInst0.rCount = Y / DIMENSION;
  vInst0.rOp = VectorInstructions::rmax;
  vInst0.rDuplicate = 1;
  vInst0.rDest = VectorInstructions::toVectorOp0Src1;
  vInst0.rBroadcast = 1;
  // broadcast max over entire array, for 2 passes
  ac_int<16, false> vInst0_broadcastCount = 2 * Y / DIMENSION;
  vInst0.immediate0 = vInst0_broadcastCount.slc<8>(0);
  vInst0.immediate1 = vInst0_broadcastCount.slc<8>(8);
  vInst0.rInvSqrt = false;

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
  vectorInstructionConfig->instCount[1] = Y / DIMENSION;

  // inst 2- start reduction engine to calculate sum
  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::reduction;
  vInst2.rCount = Y / DIMENSION;
  vInst2.rOp = VectorInstructions::radd;
  vInst2.rDuplicate = 1;
  vInst2.rDest = VectorInstructions::toVectorOp3Src1;
  vInst2.rBroadcast = 1;
  // broadcast max over entire array
  ac_int<16, false> vInst2_broadcastCount = Y / DIMENSION;
  vInst2.immediate0 = vInst2_broadcastCount.slc<8>(0);
  vInst2.immediate1 = vInst2_broadcastCount.slc<8>(8);
  vInst2.rInvSqrt = false;

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
  vectorInstructionConfig->instCount[3] = Y / DIMENSION;

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
  vInst4.vOp3 = VectorInstructions::vdiv;
  vInst4.vOp4 = VectorInstructions::nop;
  vInst4.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[4] = vInst4;
  vectorInstructionConfig->instCount[4] = Y / DIMENSION;

  vectorInstructionConfig->instLen = 5;
  vectorInstructionConfig->instLoopCount = X;  // X

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
}
