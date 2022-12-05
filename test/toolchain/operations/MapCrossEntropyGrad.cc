#include "test/toolchain/operations/Operations.h"

void MapCrossEntropyGrad(const SimplifiedParams &params,
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

  // softmax, but with an additional subtraction
  MapSoftmax(params, mappedParams);

  VectorParams *softmaxVectorParams =
      dynamic_cast<VectorParams *>(mappedParams.at(0));
  softmaxVectorParams->addressGen0Loop[0][0] = 1;
  softmaxVectorParams->addressGen0Loop[0][1] = 1;
  softmaxVectorParams->addressGen0Loop[0][2] = 1;
  softmaxVectorParams->addressGen0Loop[1][0] = 1;
  softmaxVectorParams->addressGen0Loop[1][1] = 1;
  softmaxVectorParams->addressGen0Loop[1][2] = X / DIMENSION;
  // output
  for (int i = 0; i < 3; i++) {
    softmaxVectorParams->outputLoops[0][i] = 1;
  }
  softmaxVectorParams->outputXLoopIndex[0] = 0;
  softmaxVectorParams->outputYLoopIndex[0] = 1;
  softmaxVectorParams->outputWeightLoopIndex[0] = 2;

  softmaxVectorParams->outputLoops[1][0] = 1;
  softmaxVectorParams->outputLoops[1][1] = 1;
  softmaxVectorParams->outputLoops[1][2] = X / DIMENSION;
  softmaxVectorParams->outputWeightLoopIndex[1] = 2;
  softmaxVectorParams->outputYLoopIndex[1] = 1;
  softmaxVectorParams->outputXLoopIndex[1] = 0;
  // make softmax output in DP
  softmaxVectorParams->DP_OUTPUT = true;

  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;

  vectorParams->VECTOR_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->addressGen0Enable = true;
  vectorParams->addressGen0Broadcast = false;
  vectorParams->addressGen0Loop[0][0] = 1;
  vectorParams->addressGen0Loop[0][1] = 1;
  vectorParams->addressGen0Loop[0][2] = 1;
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = 1;
  vectorParams->addressGen0Loop[1][2] = X / DIMENSION;
  vectorParams->DP_VEC0 = true;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 2;  // 2d tensor
  vectorParams->addressGen1Loops[0][0] = 1;
  vectorParams->addressGen1Loops[0][1] = 1;
  vectorParams->addressGen1Loops[0][2] = X / DIMENSION;
  vectorParams->addressGen1Loops[1][0] = 1;
  vectorParams->addressGen1Loops[1][1] = 1;
  vectorParams->addressGen1Loops[1][2] = 1;

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
  vectorParams->outputLoops[1][1] = 1;
  vectorParams->outputLoops[1][2] = X / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 1;
  vectorParams->outputXLoopIndex[1] = 0;
  vectorParams->DP_OUTPUT = false;

  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vOp0Src1 = VectorInstructions::readInterface;
  vInst0.vOp0 = VectorInstructions::vsub;
  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3Src1 = VectorInstructions::nop;
  vInst0.vOp3 = VectorInstructions::nop;
  vInst0.vOp4 = VectorInstructions::nop;
  vInst0.vAccumulatePush = 0;
  vInst0.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[0] = vInst0;

  // C/DIMENSION to do the complete reduction
  // DIMENSION to fill up the entire vector
  vectorInstructionConfig->instCount[0] = X;

  vectorInstructionConfig->instLen = 1;
  vectorInstructionConfig->instLoopCount = 1;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
}