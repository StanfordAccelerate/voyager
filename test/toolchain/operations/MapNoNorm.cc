#include "test/toolchain/operations/Operations.h"

void MapNoNorm(const SimplifiedParams &params,
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
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = X;
  vectorParams->addressGen0Loop[1][2] = K / DIMENSION;
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

  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = params.BIAS;  // use bias mode
  vectorParams->addressGen2Loops[0][0] = X;
  vectorParams->addressGen2Loops[0][1] = 1;
  vectorParams->addressGen2Loops[0][2] = 1;
  vectorParams->addressGen2Loops[1][0] = C / DIMENSION;
  vectorParams->addressGen2Loops[1][1] = 1;
  vectorParams->addressGen2Loops[1][2] = 1;
  vectorParams->addressGen2InputXLoopIndex[1] = 2;
  vectorParams->addressGen2InputYLoopIndex[1] = 1;
  vectorParams->addressGen2WeightLoopIndex[1] = 0;
  vectorParams->addressGen2WeightLoopIndex[0] = 2;

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
  vectorParams->outputLoops[1][1] = X;
  vectorParams->outputLoops[1][2] = K / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 0;
  vectorParams->outputXLoopIndex[1] = 1;
  vectorParams->DP_OUTPUT = false;

  // sendSerializedParams<VectorParams, 32>(vectorParams,
  // &serialVectorParamsIn);

  // create instruction stream
  //    VectorInstructionConfig vectorInstructionConfig;

  // inst 1- (inputs x weights) + bias
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vAccumulatePush = VectorInstructions::nop;
  vInst0.vOp0Src1 = VectorInstructions::readInterface;
  vInst0.vOp0 = VectorInstructions::vmult;
  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;
  if (params.BIAS) {
    vInst0.vOp3Src1 = VectorInstructions::readNormalInterface;
    vInst0.vOp3 = VectorInstructions::vadd;
  } else {
    vInst0.vOp3Src1 = VectorInstructions::nop;
    vInst0.vOp3 = VectorInstructions::nop;
  }
  vInst0.vOp4 = params.RELU;
  vInst0.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[0] = vInst0;

  // C/DIMENSION to do the complete reduction
  // DIMENSION to fill up the entire vector
  vectorInstructionConfig->instCount[0] = X * K / DIMENSION;

  vectorInstructionConfig->instLen = 1;
  vectorInstructionConfig->instLoopCount = 1;

  // sendSerializedParams<VectorInstructionConfig,
  // 32>(vectorInstructionConfig,
  //   &serialVectorParamsIn);

  //     vectorUnitStartSignal.SyncPop();
  //     CCS_LOG("Accelerator
  //     Layer Started.");
  //     vectorUnitDoneSignal.SyncPop();
  //     CCS_LOG("Accelerator
  //     Layer Finished.");

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
}
