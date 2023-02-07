#include "test/toolchain/operations/Operations.h"

void MapNoNormGrad(const SimplifiedParams &params, const MemoryMap &memoryMap,
                   std::deque<BaseParams *> &mappedParams,
                   std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
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
  AcceleratorMemoryMap acceleratorMemoryMap;

  acceleratorMemoryMap["vector0"] = memoryMap.inputs;
  vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen0Enable = true;
  vectorParams->addressGen0Broadcast = false;

  vectorParams->addressGen0Loop[0][0] = 1;
  vectorParams->addressGen0Loop[0][1] = 1;
  vectorParams->addressGen0Loop[0][2] = K / DIMENSION;
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = X;
  vectorParams->addressGen0Loop[1][2] = 1;

  // address gen 1 (weights)
  acceleratorMemoryMap["vector1"] = memoryMap.weights;
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 2;  // 2d tensor
  vectorParams->addressGen1Loops[0][0] = 1;
  vectorParams->addressGen1Loops[0][1] = 1;
  vectorParams->addressGen1Loops[0][2] = K / DIMENSION;
  vectorParams->addressGen1Loops[1][0] = 1;
  vectorParams->addressGen1Loops[1][1] = X;
  vectorParams->addressGen1Loops[1][2] = 1;
  vectorParams->DP_VEC1 = true;

  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = 0;  // no bias

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  //   vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = false;
  vectorParams->AVGPOOL = false;
  vectorParams->SPLIT_OUTPUT = false;

  // output
  acceleratorMemoryMap["outputs"] = memoryMap.outputs;
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

  // sendSerializedParams<VectorParams, 32>(vectorParams,
  // &serialVectorParamsIn);

  // create instruction stream
  //    VectorInstructionConfig vectorInstructionConfig;

  // inst 0- accumulate
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::accumulation;
  vInst0.rCount = X;
  vectorInstructionConfig->instCount[0] = 1;
  vectorInstructionConfig->inst[0] = vInst0;

  // inst 1- (inputs x weights), send to accumulator
  VectorInstructions vInst1;
  vInst1.instType = VectorInstructions::vector;
  vInst1.vInput = VectorInstructions::readFromVectorFetch;
  vInst1.vOp0Src1 = VectorInstructions::readInterface;
  vInst1.vOp0 = VectorInstructions::vmult;
  vInst1.vOp1 = VectorInstructions::nop;
  vInst1.vOp2 = VectorInstructions::nop;

  vInst1.vOp3Src1 = VectorInstructions::nop;
  vInst1.vOp3 = VectorInstructions::nop;
  vInst1.vOp4 = VectorInstructions::nop;
  vInst1.vAccumulatePush = 1;
  vInst1.vDest = VectorInstructions::nop;
  vectorInstructionConfig->inst[1] = vInst1;

  // C/DIMENSION to do the complete reduction
  // DIMENSION to fill up the entire vector
  vectorInstructionConfig->instCount[1] = X;

  // inst 2- pull from accumulator
  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::vector;
  vInst2.vInput = VectorInstructions::readFromAccumulation;
  vInst2.vOp0Src1 = VectorInstructions::nop;
  vInst2.vOp0 = VectorInstructions::nop;
  vInst2.vOp1 = VectorInstructions::nop;
  vInst2.vOp2 = VectorInstructions::nop;
  vInst2.vOp3Src1 = VectorInstructions::nop;
  vInst2.vOp3 = VectorInstructions::nop;
  vInst2.vOp4 = VectorInstructions::nop;
  vInst2.vAccumulatePush = 0;
  vInst2.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[2] = vInst2;
  vectorInstructionConfig->instCount[2] = 1;

  vectorInstructionConfig->instLen = 3;
  vectorInstructionConfig->instLoopCount = C / DIMENSION;

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
  opMemoryMaps.push_back(acceleratorMemoryMap);
}
