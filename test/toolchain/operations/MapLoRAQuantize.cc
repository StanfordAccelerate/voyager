#include "test/toolchain/operations/Operations.h"

// Performs a P16->P8 quantization on an input
void MapLoRAQuantize(const SimplifiedParams &params, const MemoryMap &memoryMap,
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
  int size = X * Y * C * K * FX * FY;

  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;
  AcceleratorMemoryMap acceleratorMemoryMap;

  // input is a vector of size C
  acceleratorMemoryMap["vector0"] = memoryMap.inputs;
  vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen0Enable = true;
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[0][1] = 1;
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = size / 512;
  vectorParams->addressGen0Loop[1][2] = 512;
  vectorParams->addressGen0Broadcast = false;
  vectorParams->DP_VEC0 = true;

  // weights is a matrix of K x C
  acceleratorMemoryMap["vector1"] = memoryMap.weights;
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 0;  // 2d tensor
  vectorParams->DP_VEC1 = false;

  vectorParams->addressGen1Loops[0][0] = 1;
  vectorParams->addressGen1Loops[0][1] = 1;
  vectorParams->addressGen1Loops[0][2] = 1;
  vectorParams->addressGen1Loops[1][0] = 1;
  vectorParams->addressGen1Loops[1][1] = 1;
  vectorParams->addressGen1Loops[1][2] = 1;

  // bias
  acceleratorMemoryMap["vector2"] = memoryMap.residual;
  vectorParams->ADDRESS_GEN2_OFFSET = params.RESIDUAL_OFFSET;
  vectorParams->addressGen2Mode = 0;  // 2d tensor
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen2Loops[0][i] = 1;
  }
  vectorParams->addressGen2InputXLoopIndex[0] = 0;
  vectorParams->addressGen2InputYLoopIndex[0] = 1;
  vectorParams->addressGen2WeightLoopIndex[0] = 2;

  vectorParams->addressGen2Loops[1][0] = 1;
  vectorParams->addressGen2Loops[1][1] = 1;
  vectorParams->addressGen2Loops[1][2] = 1;
  vectorParams->addressGen2WeightLoopIndex[1] = 0;
  vectorParams->addressGen2InputYLoopIndex[1] = 1;
  vectorParams->addressGen2InputXLoopIndex[1] = 2;

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  // vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = params.MAXPOOL;
  vectorParams->AVGPOOL = params.AVGPOOL;

  // output
  acceleratorMemoryMap["outputs"] = memoryMap.outputs;
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = 1;
  }
  vectorParams->outputXLoopIndex[0] = 0;
  vectorParams->outputYLoopIndex[0] = 1;
  vectorParams->outputWeightLoopIndex[0] = 2;

  vectorParams->outputLoops[1][0] = 1;
  vectorParams->outputLoops[1][1] = size / 512;
  vectorParams->outputLoops[1][2] = 512;
  vectorParams->outputXLoopIndex[1] = 0;
  vectorParams->outputYLoopIndex[1] = 1;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->SPLIT_OUTPUT = false;
  vectorParams->DP_OUTPUT = false;

  // sendSerializedParams<VectorParams, 32>(vectorParams,
  // &serialVectorParamsIn);

  // create instruction stream
  //    VectorInstructionConfig vectorInstructionConfig;

  // inst0- read input as DP, write out as SP
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vAccumulatePush = VectorInstructions::nop;
  vInst0.vOp0Src1 = VectorInstructions::readInterface;
  vInst0.vOp0 = VectorInstructions::nop;
  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3Src1 = VectorInstructions::nop;
  vInst0.vOp3 = VectorInstructions::nop;
  vInst0.vOp4 = VectorInstructions::nop;
  vInst0.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = size;

  vectorInstructionConfig->instLen = 1;
  vectorInstructionConfig->instLoopCount = 1;

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
  opMemoryMaps.push_back(acceleratorMemoryMap);
}
