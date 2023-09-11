#include "test/toolchain/operations/Operations.h"

// This operation performs:
// weight = weight - learningRate * gradient;
// it handles the case when the weight is 8b or 16b
void MapWeightUpdate(const SimplifiedParams &params, const MemoryMap &memoryMap,
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

  // std::cout << "FX: " << FX << " FY: " << FY << " C: " << C << "K: " << K
  //           << std::endl;

  int factor0, factor1;
  int totalSize = X * C / DIMENSION;
  if (totalSize > 512) {  // address generator is 10 bit, so we need to split
                          // it up if it's too big
    factor0 = 512;
    factor1 = totalSize / 512;

  } else {
    if (totalSize == 128) {
      factor0 = 1;
      factor1 = 128;
    } else {
      factor0 = totalSize;
      factor1 = 1;
    }
    // factor0 = totalSize/16;
    // factor1 = 16;
  }

  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;
  AcceleratorMemoryMap acceleratorMemoryMap;

  // this are gradients (called inputs in the gold model)
  acceleratorMemoryMap["vector0"] = memoryMap.inputs;
  vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen0Enable = true;
  vectorParams->addressGen0Broadcast = false;
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen0Loop[0][i] = 1;
  }
  vectorParams->addressGen0Loop[1][0] = 1;
  vectorParams->addressGen0Loop[1][1] = factor1;
  vectorParams->addressGen0Loop[1][2] = factor0;
  vectorParams->DP_VEC0 = params.ACC_T_INPUT;

  // address gen 1 (disabled)
  vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen1Mode = 0;  // disable

  // these are weights (called weights in the gold model)
  acceleratorMemoryMap["vector2"] = memoryMap.weights;
  vectorParams->ADDRESS_GEN2_OFFSET = params.WEIGHT_OFFSET;
  vectorParams->addressGen2Mode = 2;  // 2d tensor
  vectorParams->addressGen2Loops[0][0] = 1;
  vectorParams->addressGen2Loops[0][1] = factor1;
  vectorParams->addressGen2Loops[0][2] = factor0;
  vectorParams->DP_VEC2 = params.ACC_T_WEIGHT;

  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

  // vectorParams->scalarOutputCount = 0;
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
  vectorParams->outputLoops[1][1] = factor1;
  vectorParams->outputLoops[1][2] = factor0;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 0;
  vectorParams->outputXLoopIndex[1] = 1;
  vectorParams->DP_OUTPUT = params.ACC_T_WEIGHT;

  // inst 1- (-learning_rate) * gradients + weights
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vAccumulatePush = VectorInstructions::nop;
  vInst0.vOp0Src1 = VectorInstructions::op0immediate0;
  vInst0.immediate0 = (Posit<8, 1>(-1 * params.learningRate)).bits;
  vInst0.vOp0 = VectorInstructions::vmult;
  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3Src1 = VectorInstructions::readNormalInterface;
  vInst0.vOp3 = VectorInstructions::vadd;
  vInst0.vOp4 = VectorInstructions::nop;
  vInst0.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig->inst[0] = vInst0;
  vectorInstructionConfig->instCount[0] = totalSize;
  if (totalSize == 128) {
    // vectorInstructionConfig->instCount[0] = 1;
  }

  vectorInstructionConfig->instLen = 1;
  vectorInstructionConfig->instLoopCount = 1;
  if (totalSize == 128) {
    // vectorInstructionConfig->instLoopCount = 16;
  }

  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
  opMemoryMaps.push_back(acceleratorMemoryMap);
}