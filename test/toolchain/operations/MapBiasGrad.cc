#include "test/toolchain/operations/Operations.h"

// Columnwide reduction (accumulation op)
void MapBiasGrad(const SimplifiedParams &params, const MemoryMap &memoryMap,
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

  acceleratorMemoryMap["vector0"] = memoryMap.weights;
  vectorParams->VECTOR_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen0Enable = true;
  vectorParams->addressGen0Broadcast = false;
  vectorParams->addressGen0Loop[0][0] = 1;
  vectorParams->addressGen0Loop[0][1] = 1;
  vectorParams->addressGen0Loop[0][2] = K / DIMENSION;
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = C;
  vectorParams->addressGen0Loop[1][2] = 1;
  vectorParams->DP_VEC0 = params.ACC_T_WEIGHT;

  // address gen 1 (weights)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 0;  // disabled

  acceleratorMemoryMap["vector2"] = memoryMap.residual;
  vectorParams->ADDRESS_GEN2_OFFSET = params.RESIDUAL_OFFSET;
  vectorParams->addressGen2Mode = params.RESIDUAL ? 2 : 0;
  vectorParams->addressGen2Loops[0][0] = 1;
  vectorParams->addressGen2Loops[0][1] = 1;
  vectorParams->addressGen2Loops[0][2] = K / DIMENSION;
  vectorParams->DP_VEC2 = params.ACC_T_RESIDUAL;

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

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
  vectorParams->DP_OUTPUT = params.GRAD_CLIPPING ? true : params.ACC_T_OUTPUT;
  acceleratorMemoryMap["outputs"] = memoryMap.outputs;

  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::accumulation;
  vInst0.rCount = C;
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = 1;

  // inst 1- (inputs x weights)
  VectorInstructions vInst1;
  vInst1.instType = VectorInstructions::vector;
  vInst1.vInput = VectorInstructions::readFromVectorFetch;
  vInst1.vAccumulatePush = true;
  vInst1.vOp0Src1 = VectorInstructions::nop;
  vInst1.vOp0 = VectorInstructions::nop;
  vInst1.vOp1 = VectorInstructions::nop;
  vInst1.vOp2 = VectorInstructions::nop;
  vInst1.vOp3Src1 = VectorInstructions::nop;
  vInst1.vOp3 = VectorInstructions::nop;
  vInst1.vOp4 = VectorInstructions::nop;
  vInst1.vDest = VectorInstructions::nop;
  vectorInstructionConfig->inst[1] = vInst1;
  vectorInstructionConfig->instCount[1] = C;

  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::vector;
  vInst2.vInput = VectorInstructions::readFromAccumulation;
  vInst2.vOp0Src1 = VectorInstructions::nop;
  vInst2.vOp0 = VectorInstructions::nop;
  vInst2.vOp1 = VectorInstructions::vscaleexp;
  vInst2.vOp1Src1 = VectorInstructions::op1immediate0;
  vInst2.immediate0 = params.outputExpBias;
  vInst2.vOp2 = VectorInstructions::nop;
  vInst2.vOp3Src1 = VectorInstructions::nop;
  vInst2.vOp3 =
      params.RESIDUAL ? VectorInstructions::vadd : VectorInstructions::nop;
  vInst2.vOp3Src1 = params.RESIDUAL ? VectorInstructions::readNormalInterface
                                    : VectorInstructions::nop;
  vInst2.vOp4 = params.RELU;
  vInst2.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[2] = vInst2;
  vectorInstructionConfig->instCount[2] = 1;

  vectorInstructionConfig->instLen = 3;
  vectorInstructionConfig->instLoopCount = K / DIMENSION;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
  opMemoryMaps.push_back(acceleratorMemoryMap);

  if (params.GRAD_CLIPPING) {
    MapGradNormClipping(params, memoryMap, mappedParams, opMemoryMaps, K);
  }
}