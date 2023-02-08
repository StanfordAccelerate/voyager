#include "test/toolchain/operations/Operations.h"

void MapGradNormUnitTest(const SimplifiedParams &params,
                         const MemoryMap &memoryMap,
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
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = 1;
  vectorParams->addressGen0Loop[1][2] = X * C / DIMENSION;
  vectorParams->addressGen0Broadcast = false;
  vectorParams->DP_VEC0 = params.ACC_T_INPUT;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 0;

  acceleratorMemoryMap["vector2"] = memoryMap.residual;
  vectorParams->ADDRESS_GEN2_OFFSET = params.RESIDUAL_OFFSET;
  vectorParams->addressGen2Mode = params.RESIDUAL ? 2 : 0;
  vectorParams->addressGen2Loops[0][0] = 1;
  vectorParams->addressGen2Loops[0][1] = 1;
  vectorParams->addressGen2Loops[0][2] = X * K / DIMENSION;
  vectorParams->DP_VEC2 = params.ACC_T_RESIDUAL;

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  vectorParams->MAXPOOL = params.MAXPOOL;
  vectorParams->AVGPOOL = params.AVGPOOL;
  vectorParams->SPLIT_OUTPUT = params.SPLIT_OUTPUT;

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
  vectorParams->outputLoops[1][2] = X * C / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 0;
  vectorParams->outputXLoopIndex[1] = 1;
  vectorParams->DP_OUTPUT = true;

  // inst 1- (inputs x weights)
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vAccumulatePush = VectorInstructions::nop;
  vInst0.vOp0Src1 = VectorInstructions::nop;
  vInst0.vOp0 = VectorInstructions::nop;
  vInst0.vOp1 = VectorInstructions::vscaleexp;
  vInst0.vOp1Src1 = VectorInstructions::op1immediate0;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3 =
      params.RESIDUAL ? VectorInstructions::vadd : VectorInstructions::nop;
  vInst0.vOp3Src1 = params.RESIDUAL ? VectorInstructions::readNormalInterface
                                    : VectorInstructions::nop;
  vInst0.vOp4 = params.RELU;
  vInst0.vDest = VectorInstructions::vWriteOut;
  vInst0.immediate0 = params.outputExpBias;
  vectorInstructionConfig->inst[0] = vInst0;

  // C/DIMENSION to do the complete reduction
  // DIMENSION to fill up the entire vector
  vectorInstructionConfig->instCount[0] = X * K / DIMENSION;

  vectorInstructionConfig->instLen = 1;
  vectorInstructionConfig->instLoopCount = 1;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
  opMemoryMaps.push_back(acceleratorMemoryMap);

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
  opMemoryMaps.push_back(acceleratorMemoryMap);

  MapGradNormClipping(params, memoryMap, mappedParams, opMemoryMaps,
                      X * C / DIMENSION);
}