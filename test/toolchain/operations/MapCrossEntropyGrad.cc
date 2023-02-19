#include "test/toolchain/operations/Operations.h"

void MapCrossEntropyGrad(const SimplifiedParams &params,
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

  // softmax, but with an additional subtraction

  // softmax uses a different convention for X and Y
  // in softmax, X is the vertical dimension and Y is the horizontal
  // in cross entropy grad, X is the horizontal
  // so to make them match up, we create a copy of the params with the
  // softmax convention for mapping the softmax part.
  SimplifiedParams modifiedSoftmaxParams = params;
  modifiedSoftmaxParams.loops[1][modifiedSoftmaxParams.inputYLoopIndex[1]] = X;
  modifiedSoftmaxParams.loops[1][modifiedSoftmaxParams.inputXLoopIndex[1]] = 1;
  MapSoftmax(modifiedSoftmaxParams, memoryMap, mappedParams, opMemoryMaps);

  VectorParams *softmaxVectorParams =
      dynamic_cast<VectorParams *>(mappedParams.at(0));
  VectorInstructionConfig *softmaxVectorInstructionConfig =
      dynamic_cast<VectorInstructionConfig *>(mappedParams.at(1));

  // make softmax output in DP
  softmaxVectorParams->DP_OUTPUT = true;

  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;
  AcceleratorMemoryMap acceleratorMemoryMap;

  acceleratorMemoryMap["vector0"] = memoryMap.outputs;
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
  acceleratorMemoryMap["vector1"] = memoryMap.weights;
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
  vectorParams->outputLoops[1][1] = 1;
  vectorParams->outputLoops[1][2] = X / DIMENSION;
  vectorParams->outputWeightLoopIndex[1] = 2;
  vectorParams->outputYLoopIndex[1] = 1;
  vectorParams->outputXLoopIndex[1] = 0;
  vectorParams->DP_OUTPUT = false;

  // logits[i] - labels[i]
  // and scale output exponent
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromVectorFetch;
  vInst0.vOp0Src1 = VectorInstructions::readInterface;
  vInst0.vOp0 = VectorInstructions::vsub;
  vInst0.vOp1 = VectorInstructions::vscaleexp;
  vInst0.vOp1Src1 = VectorInstructions::op1immediate0;
  vInst0.vOp2 = VectorInstructions::nop;
  vInst0.vOp3 = VectorInstructions::nop;
  vInst0.immediate0 = params.outputExpBias;
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
  opMemoryMaps.push_back(acceleratorMemoryMap);
}