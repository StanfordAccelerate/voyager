#include "test/toolchain/operations/Operations.h"

// Performs a VectorUnit-based matrix multiplication along with a
// residual
void MapLoRACombination(const SimplifiedParams &params,
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

  // if the output is too large, then we need to tile the operation
  int xTileSize = 128;
  int numTiles = 1;
  if (X > 128) {
    numTiles = X / xTileSize;
  }

  for (int tile = 0; tile < numTiles; tile++) {
    VectorParams *vectorParams = new VectorParams;
    VectorInstructionConfig *vectorInstructionConfig =
        new VectorInstructionConfig;
    AcceleratorMemoryMap acceleratorMemoryMap;

    // input is a vector of size C
    acceleratorMemoryMap["vector0"] = memoryMap.inputs;
    vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET + tile * xTileSize * C;
    vectorParams->addressGen0Enable = true;
    for (int i = 0; i < 3; i++) {
      vectorParams->addressGen0Loop[0][i] = 1;
    }
    vectorParams->addressGen0Loop[0][1] = xTileSize;
    vectorParams->addressGen0Loop[1][0] = K;
    vectorParams->addressGen0Loop[1][1] = 1;
    vectorParams->addressGen0Loop[1][2] = C / DIMENSION;
    vectorParams->addressGen0Broadcast = false;
    vectorParams->addressGen0BroadcastCount = K;
    vectorParams->DP_VEC0 = true;

    // weights is a matrix of K x C
    acceleratorMemoryMap["vector1"] = memoryMap.weights;
    vectorParams->ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
    vectorParams->addressGen1Mode = 2;  // 2d tensor
    vectorParams->DP_VEC1 = true;

    vectorParams->addressGen1Loops[0][0] = xTileSize;
    vectorParams->addressGen1Loops[0][1] = 1;
    vectorParams->addressGen1Loops[0][2] = 1;
    vectorParams->addressGen1Loops[1][0] = 1;
    vectorParams->addressGen1Loops[1][1] = K;
    vectorParams->addressGen1Loops[1][2] = C / DIMENSION;

    // bias (used as residual here)
    acceleratorMemoryMap["vector2"] = memoryMap.residual;
    vectorParams->ADDRESS_GEN2_OFFSET =
        params.RESIDUAL_OFFSET + tile * xTileSize * K;
    vectorParams->addressGen2Mode = 2;  // 2d tensor
    for (int i = 0; i < 3; i++) {
      vectorParams->addressGen2Loops[0][i] = 1;
    }
    vectorParams->addressGen2InputXLoopIndex[0] = 0;
    vectorParams->addressGen2InputYLoopIndex[0] = 1;
    vectorParams->addressGen2WeightLoopIndex[0] = 2;

    vectorParams->addressGen2Loops[0][0] = 1;
    vectorParams->addressGen2Loops[0][1] = xTileSize;
    vectorParams->addressGen2Loops[0][2] = K / DIMENSION;
    vectorParams->addressGen2Loops[1][0] = 1;
    vectorParams->addressGen2Loops[1][1] = 1;
    vectorParams->addressGen2Loops[1][2] = 1;
    vectorParams->addressGen2WeightLoopIndex[1] = 0;
    vectorParams->addressGen2InputYLoopIndex[1] = 1;
    vectorParams->addressGen2InputXLoopIndex[1] = 2;

    vectorParams->VECTOR_OUTPUT_OFFSET =
        params.OUTPUT_OFFSET + tile * xTileSize * K;
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
    vectorParams->outputLoops[1][1] = xTileSize;
    vectorParams->outputLoops[1][2] = K / DIMENSION;
    vectorParams->outputXLoopIndex[1] = 0;
    vectorParams->outputYLoopIndex[1] = 1;
    vectorParams->outputWeightLoopIndex[1] = 2;
    vectorParams->SPLIT_OUTPUT = false;
    vectorParams->DP_OUTPUT = false;

    // sendSerializedParams<VectorParams, 32>(vectorParams,
    // &serialVectorParamsIn);

    // create instruction stream
    //    VectorInstructionConfig vectorInstructionConfig;

    // inst0- start reduction engine
    VectorInstructions vInst0;
    vInst0.instType = VectorInstructions::reduction;
    vInst0.rCount = C / DIMENSION;
    vInst0.rOp = VectorInstructions::radd;
    vInst0.rDuplicate = 0;
    vInst0.rDest = VectorInstructions::toVectorOp0Src0;
    vectorInstructionConfig->inst[0] = vInst0;
    vectorInstructionConfig->instCount[0] = 1;

    // inst 1- inputs x weights, send to reduce
    VectorInstructions vInst1;
    vInst1.instType = VectorInstructions::vector;
    vInst1.vInput = VectorInstructions::readFromVectorFetch;
    vInst1.vAccumulatePush = VectorInstructions::nop;
    vInst1.vOp0Src1 = VectorInstructions::readInterface;
    vInst1.vOp0 = VectorInstructions::vmult;
    vInst1.vOp1 = VectorInstructions::nop;
    vInst1.vOp2 = VectorInstructions::toReduce;
    vInst1.vOp3Src1 = VectorInstructions::nop;
    vInst1.vOp3 = VectorInstructions::nop;
    vInst1.vOp4 = VectorInstructions::nop;
    vInst1.vDest = VectorInstructions::nop;
    vectorInstructionConfig->inst[1] = vInst1;

    // C/DIMENSION to do the complete reduction
    // DIMENSION to fill up the entire vector (this is now K dimension)
    vectorInstructionConfig->instCount[1] = DIMENSION * C / DIMENSION;

    // inst2- add bias, write out
    VectorInstructions vInst2;
    vInst2.instType = VectorInstructions::vector;
    vInst2.vInput = VectorInstructions::readFromReduce;
    vInst2.vAccumulatePush = VectorInstructions::nop;
    vInst2.vOp0Src1 = VectorInstructions::nop;
    vInst2.vOp0 = VectorInstructions::nop;
    vInst2.vOp1 = VectorInstructions::nop;
    vInst2.vOp2 = VectorInstructions::nop;
    vInst2.vOp3Src1 = VectorInstructions::readNormalInterface;
    vInst2.vOp3 = VectorInstructions::vadd;
    vInst2.vOp4 = VectorInstructions::nop;
    vInst2.vDest = VectorInstructions::vWriteOut;
    vectorInstructionConfig->inst[2] = vInst2;
    vectorInstructionConfig->instCount[2] = 1;

    // unroll to reduce loop count
    vectorInstructionConfig->inst[3] = vInst0;
    vectorInstructionConfig->instCount[3] =
        vectorInstructionConfig->instCount[0];
    vectorInstructionConfig->inst[4] = vInst1;
    vectorInstructionConfig->instCount[4] =
        vectorInstructionConfig->instCount[1];
    vectorInstructionConfig->inst[5] = vInst2;
    vectorInstructionConfig->instCount[5] =
        vectorInstructionConfig->instCount[2];

    vectorInstructionConfig->instLen = 6;
    vectorInstructionConfig->instLoopCount = xTileSize * K / DIMENSION / 2;

    mappedParams.push_back(vectorParams);
    mappedParams.push_back(vectorInstructionConfig);
    opMemoryMaps.push_back(acceleratorMemoryMap);
  }
}
