#include "test/toolchain/operations/Operations.h"

void MapSoftmaxGrad(const SimplifiedParams &params,
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

  vectorParams->VECTOR_OFFSET = params.RESIDUAL_OFFSET;
  vectorParams->addressGen0Enable = true;
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = X;
  vectorParams->addressGen0Loop[1][2] = Y;
  vectorParams->addressGen0Broadcast = true;
  vectorParams->addressGen0BroadcastCount = Y;
  vectorParams->SOFTMAX_GRAD_NEGATE = true;

  vectorParams->ADDRESS_GEN1_OFFSET = params.RESIDUAL_OFFSET;
  vectorParams->addressGen1Mode = 2;  // 2d tensor
  vectorParams->addressGen1Loops[0][0] = 1;
  vectorParams->addressGen1Loops[0][1] = X;
  vectorParams->addressGen1Loops[0][2] = 1;
  vectorParams->addressGen1Loops[1][0] = Y;  // repeat
  vectorParams->addressGen1Loops[1][1] = 1;
  vectorParams->addressGen1Loops[1][2] = Y;

  vectorParams->ADDRESS_GEN2_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen2Mode = 2;  // 2d tensor
  vectorParams->addressGen2Loops[0][0] = 1;
  vectorParams->addressGen2Loops[0][1] = X;
  vectorParams->addressGen2Loops[0][2] = 1;
  vectorParams->addressGen2Loops[1][0] = Y;  // repeat
  vectorParams->addressGen2Loops[1][1] = 1;
  vectorParams->addressGen2Loops[1][2] = Y;

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = params.MAXPOOL;
  vectorParams->AVGPOOL = params.AVGPOOL;

  // output
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = params.loops[0][i];
  }
  vectorParams->outputXLoopIndex[0] = params.inputXLoopIndex[0];
  vectorParams->outputYLoopIndex[0] = params.inputYLoopIndex[0];
  vectorParams->outputWeightLoopIndex[0] = params.weightLoopIndex[0];

  vectorParams->outputLoops[1][0] = 1;
  vectorParams->outputLoops[1][1] = X;
  vectorParams->outputLoops[1][2] = Y;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 1;
  vectorParams->outputXLoopIndex[1] = 0;

  // sendSerializedParams<VectorParams, 32>(vectorParams,
  // &serialVectorParamsIn);

  // create instruction stream
  //    VectorInstructionConfig vectorInstructionConfig;

  // inst0- start reduction engine
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::reduction;
  vInst0.rCount = Y;
  vInst0.rOp = VectorInstructions::radd;
  vInst0.rDuplicate = 0;
  // vInst0.rDest = VectorInstructions::toVectorSrc0;
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = 1;

  // perform multiplication (-a_i * a_j) or (a_i * (1-a_i))
  VectorInstructions vInst1;
  vInst1.instType = VectorInstructions::vector;
  vInst1.vInput = VectorInstructions::readFromVectorFetch;
  vInst1.vAccumulatePush = VectorInstructions::nop;
  vInst1.vOp0Src1 = VectorInstructions::readInterface;
  vInst1.vOp0 = VectorInstructions::vmult;
  vInst1.vOp1 = VectorInstructions::nop;
  vInst1.vOp2 = VectorInstructions::toReduce;
  //
  vInst1.vOp3Src1 = VectorInstructions::readNormalInterface;
  vInst1.vOp3 = VectorInstructions::vmult;
  vInst1.vOp4 = VectorInstructions::nop;
  vInst1.vDest = VectorInstructions::nop;
  vectorInstructionConfig->inst[1] = vInst1;
  vectorInstructionConfig->instCount[1] = Y * DIMENSION;

  // pull out from reduce and multiply by 1/sqrt(32)
  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::vector;
  vInst2.vInput = VectorInstructions::nop;
  vInst2.vAccumulatePush = VectorInstructions::nop;
  vInst2.vOp0Src1 = VectorInstructions::nop;
  vInst2.vOp0 = VectorInstructions::vmult;
  vInst2.vOp1 = VectorInstructions::nop;
  vInst2.vOp2 = VectorInstructions::toReduce;
  // vInst2.vOp3Src0 = VectorInstructions::readReduceInterface;
  vInst2.vOp3Src1 = VectorInstructions::op3immediate0;

  vInst2.vOp3 = VectorInstructions::vmult;
  vInst2.vOp4 = VectorInstructions::nop;
  vInst2.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[2] = vInst2;
  vectorInstructionConfig->instCount[2] = 1;
  Posit<16, 1> scale = (1.0 / sqrt(32));
  vInst2.immediate0 = scale.bits;

  vectorInstructionConfig->instLen = 3;
  vectorInstructionConfig->instLoopCount = X * Y / DIMENSION;

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
