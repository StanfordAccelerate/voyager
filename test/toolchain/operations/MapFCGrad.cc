#include "test/toolchain/operations/Operations.h"

void MapFCGrad(const SimplifiedParams &params,
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
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = 1;
  vectorParams->addressGen0Loop[1][2] = X;
  vectorParams->addressGen0Broadcast = true;
  vectorParams->addressGen0BroadcastCount = K;
  vectorParams->DP_VEC0 = false;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 2;  // 2d tensor
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen1Loops[0][i] = 1;
  }
  vectorParams->addressGen1Loops[1][0] = X;
  vectorParams->addressGen1Loops[1][1] = 1;
  vectorParams->addressGen1Loops[1][2] = K / DIMENSION;
  vectorParams->DP_VEC1 = false;

  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = 0;  // use bias mode

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  // vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = params.MAXPOOL;
  vectorParams->AVGPOOL = params.AVGPOOL;
  vectorParams->SPLIT_OUTPUT = params.SPLIT_OUTPUT;

  // output
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = 1;
  }
  vectorParams->outputXLoopIndex[0] = params.inputXLoopIndex[0];
  vectorParams->outputYLoopIndex[0] = params.inputYLoopIndex[0];
  vectorParams->outputWeightLoopIndex[0] = params.weightLoopIndex[0];

  vectorParams->outputLoops[1][0] = 1;
  vectorParams->outputLoops[1][1] = X;
  vectorParams->outputLoops[1][2] = K / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 0;
  vectorParams->outputXLoopIndex[1] = 1;
  vectorParams->DP_OUTPUT = false;

  // inst 1- (inputs x weights)
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vAccumulatePush = VectorInstructions::nop;
  vInst0.vOp0Src1 = VectorInstructions::readInterface;
  vInst0.vOp0 = VectorInstructions::vmult;
  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3Src1 = VectorInstructions::nop;
  vInst0.vOp3 = VectorInstructions::nop;
  vInst0.vOp4 = params.RELU;
  vInst0.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[0] = vInst0;

  // C/DIMENSION to do the complete reduction
  // DIMENSION to fill up the entire vector
  vectorInstructionConfig->instCount[0] = X * K / DIMENSION;

  vectorInstructionConfig->instLen = 1;
  vectorInstructionConfig->instLoopCount = 1;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
}

void MapFCGradWithNormClipping(const SimplifiedParams &params,
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
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = 1;
  vectorParams->addressGen0Loop[1][2] = X / DIMENSION;
  vectorParams->addressGen0Broadcast = true;
  vectorParams->addressGen0BroadcastCount = K;
  vectorParams->DP_VEC0 = false;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 2;  // 2d tensor
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen1Loops[0][i] = 1;
  }
  vectorParams->addressGen1Loops[1][0] = X;
  vectorParams->addressGen1Loops[1][1] = 1;
  vectorParams->addressGen1Loops[1][2] = K / DIMENSION;
  vectorParams->DP_VEC1 = false;

  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = 0;  // use bias mode

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  // vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = params.MAXPOOL;
  vectorParams->AVGPOOL = params.AVGPOOL;
  vectorParams->SPLIT_OUTPUT = params.SPLIT_OUTPUT;

  // output
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = 1;
  }
  vectorParams->outputXLoopIndex[0] = params.inputXLoopIndex[0];
  vectorParams->outputYLoopIndex[0] = params.inputYLoopIndex[0];
  vectorParams->outputWeightLoopIndex[0] = params.weightLoopIndex[0];

  vectorParams->outputLoops[1][0] = 1;
  vectorParams->outputLoops[1][1] = X;
  vectorParams->outputLoops[1][2] = K / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 0;
  vectorParams->outputXLoopIndex[1] = 1;
  vectorParams->DP_OUTPUT = true;

  // inst 1- (inputs x weights)
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vAccumulatePush = VectorInstructions::nop;
  vInst0.vOp0Src1 = VectorInstructions::readInterface;
  vInst0.vOp0 = VectorInstructions::vmult;
  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3Src1 = VectorInstructions::op3immediate0;
  vInst0.vOp3 = VectorInstructions::vscaleexp;
  vInst0.vOp4 = params.RELU;
  vInst0.immediate0 = params.outputExpBias;
  vInst0.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = X * K / DIMENSION;

  vectorInstructionConfig->instLen = 1;
  vectorInstructionConfig->instLoopCount = 1;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);

  // We need to read in the double precision scaled tensor
  VectorParams *vectorParams_reread = new VectorParams;

  vectorParams_reread->VECTOR_OFFSET = params.OUTPUT_OFFSET;
  vectorParams_reread->addressGen0Enable = true;
  for (int i = 0; i < 3; i++) {
    vectorParams_reread->addressGen0Loop[0][i] = 1;
  }
  vectorParams_reread->addressGen0Loop[1][0] = 2;
  vectorParams_reread->addressGen0Loop[1][1] = X;
  vectorParams_reread->addressGen0Loop[1][2] = K / DIMENSION;
  vectorParams_reread->addressGen0Broadcast = false;
  vectorParams_reread->DP_VEC0 = true;

  // address gen 1 (weights)
  vectorParams_reread->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams_reread->addressGen1Mode = 0;  // 2d tensor

  vectorParams_reread->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams_reread->addressGen2Mode = 0;  // use bias mode

  vectorParams_reread->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams_reread->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  // vectorParams_reread->scalarOutputCount = 0;
  vectorParams_reread->MAXPOOL = params.MAXPOOL;
  vectorParams_reread->AVGPOOL = params.AVGPOOL;
  vectorParams_reread->SPLIT_OUTPUT = params.SPLIT_OUTPUT;

  // output
  for (int i = 0; i < 3; i++) {
    vectorParams_reread->outputLoops[0][i] = 1;
  }
  vectorParams_reread->outputXLoopIndex[0] = params.inputXLoopIndex[0];
  vectorParams_reread->outputYLoopIndex[0] = params.inputYLoopIndex[0];
  vectorParams_reread->outputWeightLoopIndex[0] = params.weightLoopIndex[0];

  vectorParams_reread->outputLoops[1][0] = 1;
  vectorParams_reread->outputLoops[1][1] = X;
  vectorParams_reread->outputLoops[1][2] = K / DIMENSION;
  vectorParams_reread->outputWeightLoopIndex[1] = 2;
  vectorParams_reread->outputYLoopIndex[1] = 0;
  vectorParams_reread->outputXLoopIndex[1] = 1;
  vectorParams_reread->DP_OUTPUT = false;

  VectorInstructionConfig *vectorInstructionConfig_clip =
      new VectorInstructionConfig;

  ac_int<16, false> bcastCount = X * K / DIMENSION;

  vectorInstructionConfig_clip->inst[0].instType =
      VectorInstructions::reduction;
  vectorInstructionConfig_clip->inst[0].rCount = X * K / DIMENSION;
  vectorInstructionConfig_clip->inst[0].rOp = VectorInstructions::radd;
  vectorInstructionConfig_clip->inst[0].rDuplicate = 1;
  vectorInstructionConfig_clip->inst[0].rInvSqrt = 1;
  vectorInstructionConfig_clip->inst[0].rDest =
      VectorInstructions::toVectorOp0Src1;
  vectorInstructionConfig_clip->inst[0].rBroadcast = 1;
  vectorInstructionConfig_clip->inst[0].immediate0 = bcastCount.slc<8>(0);
  vectorInstructionConfig_clip->inst[0].immediate1 = bcastCount.slc<8>(8);
  vectorInstructionConfig_clip->instCount[0] = 1;

  // send tensor to reduction
  vectorInstructionConfig_clip->inst[1].instType = VectorInstructions::vector;
  vectorInstructionConfig_clip->inst[1].vInput =
      VectorInstructions::readFromVectorFetch;
  vectorInstructionConfig_clip->inst[1].vAccumulatePush =
      VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[1].vOp0Src1 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[1].vOp0 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[1].vOp1 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[1].vOp2 = VectorInstructions::toReduce;
  vectorInstructionConfig_clip->inst[1].vOp3Src1 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[1].vOp3 = VectorInstructions::vsquare;
  vectorInstructionConfig_clip->inst[1].vOp4 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[1].vDest = VectorInstructions::nop;
  vectorInstructionConfig_clip->instCount[1] = X * K / DIMENSION;

  // divide entire tensor by sqrt
  vectorInstructionConfig_clip->inst[2].instType = VectorInstructions::vector;
  vectorInstructionConfig_clip->inst[2].vInput =
      VectorInstructions::readFromVectorFetch;
  vectorInstructionConfig_clip->inst[2].vAccumulatePush =
      VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[2].vOp0Src1 =
      VectorInstructions::readFromReduce;
  vectorInstructionConfig_clip->inst[2].vOp0 = VectorInstructions::vmult;
  vectorInstructionConfig_clip->inst[2].vOp1 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[2].vOp2 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[2].vOp3Src1 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[2].vOp3 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[2].vOp4 = VectorInstructions::nop;
  vectorInstructionConfig_clip->inst[2].vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig_clip->instCount[2] = X * K / DIMENSION;

  vectorInstructionConfig_clip->instLen = 3;
  vectorInstructionConfig_clip->instLoopCount = 1;

  mappedParams.push_back(vectorParams_reread);
  mappedParams.push_back(vectorInstructionConfig_clip);
}