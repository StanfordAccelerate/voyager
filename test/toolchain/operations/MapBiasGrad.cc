#include "test/toolchain/operations/Operations.h"

void MapBiasGrad(const SimplifiedParams &params,
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

  // Columnwide reduction (accumulation op)

  vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen0Enable = true;
  vectorParams->addressGen0Broadcast = false;

  vectorParams->addressGen0Loop[0][0] = 1;
  vectorParams->addressGen0Loop[0][1] = 1;
  vectorParams->addressGen0Loop[0][2] = K / DIMENSION;
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = C;
  vectorParams->addressGen0Loop[1][2] = 1;
  vectorParams->DP_VEC0 = false;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 0;  // disabled
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen1Loops[0][i] = 1;
  }

  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = 0;  // disabled

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  vectorParams->scalarOutputCount = 0;
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
  vectorParams->outputLoops[1][1] = 1;
  vectorParams->outputLoops[1][2] = K / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 0;
  vectorParams->outputXLoopIndex[1] = 1;
  vectorParams->DP_OUTPUT = true;

  // sendSerializedParams<VectorParams, 32>(vectorParams,
  // &serialVectorParamsIn);

  // create instruction stream
  //    VectorInstructionConfig vectorInstructionConfig;

  // inst 1- (inputs x weights)
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vAccumulatePush = true;
  vInst0.vOp0Src1 = VectorInstructions::nop;
  vInst0.vOp0 = VectorInstructions::nop;
  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3Src1 = VectorInstructions::nop;
  vInst0.vOp3 = VectorInstructions::nop;
  vInst0.vOp4 = params.RELU;
  vInst0.vDest = VectorInstructions::nop;
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = C * K / DIMENSION;

  VectorInstructions vInst1;
  vInst1.instType = VectorInstructions::accumulation;
  vInst1.rCount = C;
  vectorInstructionConfig->inst[1] = vInst1;
  vectorInstructionConfig->instCount[1] = 1;

  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::vector;
  vInst2.vInput = VectorInstructions::readFromAccumulation;
  vInst2.vAccumulatePush = true;
  vInst2.vOp0Src1 = VectorInstructions::nop;
  vInst2.vOp0 = VectorInstructions::nop;
  vInst2.vOp1 = VectorInstructions::nop;
  vInst2.vOp2 = VectorInstructions::nop;
  vInst2.vOp3Src1 = VectorInstructions::nop;
  vInst2.vOp3 = VectorInstructions::nop;
  vInst2.vOp4 = params.RELU;
  vInst2.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[2] = vInst2;
  vectorInstructionConfig->instCount[2] = 1;
  

  vectorInstructionConfig->instLen = 3;
  vectorInstructionConfig->instLoopCount = 1;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
}