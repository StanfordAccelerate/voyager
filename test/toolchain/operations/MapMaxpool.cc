#include "test/toolchain/operations/Operations.h"

void MapMaxpool(const SimplifiedParams &params, const MemoryMap &memoryMap,
                std::deque<BaseParams *> &mappedParams,
                std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * (16);
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * (16);
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;

  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;
  AcceleratorMemoryMap acceleratorMemoryMap;

  acceleratorMemoryMap["vector0"] = memoryMap.inputs;
  vectorParams->addressGen0Mode = 2;
  vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen0Loop[0][0] = 1;
  vectorParams->addressGen0WeightLoopIndex[0] = 0;

  vectorParams->addressGen0Loop[0][1] =
      params.loops[0][params.inputYLoopIndex[0]];
  vectorParams->addressGen0InputYLoopIndex[0] = 1;

  vectorParams->addressGen0Loop[0][2] =
      params.loops[0][params.inputXLoopIndex[0]];
  vectorParams->addressGen0InputXLoopIndex[0] = 2;

  vectorParams->addressGen0Loop[1][0] = C / DIMENSION;
  vectorParams->addressGen0WeightLoopIndex[1] = 0;

  vectorParams->addressGen0Loop[1][1] =
      params.loops[1][params.inputYLoopIndex[1]];
  vectorParams->addressGen0InputYLoopIndex[1] = 1;
  vectorParams->addressGen0Loop[1][2] =
      params.loops[1][params.inputXLoopIndex[1]];
  vectorParams->addressGen0InputXLoopIndex[1] = 2;

  vectorParams->addressGen0Broadcast = false;

  vectorParams->addressGen1Mode = 0;
  vectorParams->addressGen2Mode = 0;

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  // output
  acceleratorMemoryMap["outputs"] = memoryMap.outputs;
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = 1;
  }
  vectorParams->outputXLoopIndex[0] = 0;
  vectorParams->outputYLoopIndex[0] = 1;
  vectorParams->outputWeightLoopIndex[0] = 2;

  vectorParams->outputLoops[1][0] = params.loops[0][params.inputYLoopIndex[0]];
  vectorParams->outputLoops[1][1] = params.loops[0][params.inputXLoopIndex[0]];
  vectorParams->outputLoops[1][2] = K / (DIMENSION);
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 0;
  vectorParams->outputXLoopIndex[1] = 1;
  vectorParams->DP_OUTPUT = false;

  // perform max
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::accumulation;
  vInst0.rCount = params.loops[1][params.inputYLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]];
  vInst0.rOp = VectorInstructions::rmax;
  vInst0.rDuplicate = false;
  vInst0.rSqrt = false;
  vInst0.rReciprocal = false;
  vInst0.rBroadcast = false;
  vInst0.rDest = VectorInstructions::toVectorOp0Src0;
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = 1;

  // feed max accumulator
  VectorInstructions vInst1;
  vInst1.instType = VectorInstructions::vector;
  vInst1.vInput = VectorInstructions::readFromVectorFetch;
  vInst1.vOp0Src1 = VectorInstructions::nop;
  vInst1.vOp0 = VectorInstructions::nop;
  vInst1.vOp1 = VectorInstructions::nop;
  vInst1.vOp2 = VectorInstructions::nop;
  vInst1.vOp3 = VectorInstructions::nop;
  vInst1.vOp4 = VectorInstructions::nop;
  vInst1.vDest = VectorInstructions::nop;
  vInst1.vAccumulatePush = true;
  vectorInstructionConfig->inst[1] = vInst1;
  vectorInstructionConfig->instCount[1] =
      params.loops[1][params.inputYLoopIndex[1]] *
      params.loops[1][params.inputXLoopIndex[1]];

  // read out from max accumulator and write out
  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::vector;
  vInst2.vInput = VectorInstructions::readFromAccumulation;
  vInst2.vOp0Src1 = VectorInstructions::nop;
  vInst2.vOp0 = VectorInstructions::nop;
  vInst2.vOp1 = VectorInstructions::nop;
  vInst2.vOp2 = VectorInstructions::nop;
  vInst2.vOp3 = VectorInstructions::nop;
  vInst2.vOp4 = VectorInstructions::nop;
  vInst2.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[2] = vInst2;
  vectorInstructionConfig->instCount[2] = 1;

  vectorInstructionConfig->instLen = 3;
  vectorInstructionConfig->instLoopCount =
      (K / DIMENSION) * params.loops[0][params.inputYLoopIndex[0]] *
      params.loops[0][params.inputXLoopIndex[0]];

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
  opMemoryMaps.push_back(acceleratorMemoryMap);
}