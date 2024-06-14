#include "test/toolchain/operations/Operations.h"

void MapMatrixOp(const SimplifiedParams &originalParams,
                 const MemoryMap &memoryMap,
                 std::deque<BaseParams *> &mappedParams,
                 std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  SimplifiedParams params = originalParams;
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

  if (IC_DIMENSION < 16) {
    params.loops[1][params.reductionLoopIndex[1]] *= (16 / IC_DIMENSION);
  } else if (IC_DIMENSION > 16) {
    if (!params.REPLICATION) {
      params.loops[1][params.reductionLoopIndex[1]] /= (IC_DIMENSION / 16);
    }
  }

  if (OC_DIMENSION < 16) {
    params.loops[0][params.weightLoopIndex[0]] *= (16 / OC_DIMENSION);
  } else if (OC_DIMENSION > 16) {
    if ((params.loops[1][params.weightLoopIndex[1]] >= 4 &&
         params.loops[1][params.fxIndex] > 1 &&
         params.loops[1][params.fyIndex] > 1)) {
      params.loops[1][params.weightLoopIndex[1]] /= (OC_DIMENSION / 16);
    } else if (params.loops[0][params.weightLoopIndex[0]] <
                   (OC_DIMENSION / 16) &&
               params.loops[0][params.weightLoopIndex[0]] != 1) {
      int reductionFactor = OC_DIMENSION / 16;
      reductionFactor =
          reductionFactor / params.loops[0][params.weightLoopIndex[0]];
      params.loops[0][params.weightLoopIndex[0]] = 1;
      params.loops[1][params.weightLoopIndex[1]] /= reductionFactor;
    } else if (params.loops[0][params.weightLoopIndex[0]] == 1) {
      params.loops[1][params.weightLoopIndex[1]] /= (OC_DIMENSION / 16);
    } else {
      params.loops[0][params.weightLoopIndex[0]] /= (OC_DIMENSION / 16);
    }
  }

  MatrixParams *matrixParams = new MatrixParams;
  VectorParams *vectorParams = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig =
      new VectorInstructionConfig;
  AcceleratorMemoryMap acceleratorMemoryMap;

  // matrix params
  acceleratorMemoryMap["inputs"] = memoryMap.inputs;
  matrixParams->INPUT_OFFSET = params.INPUT_OFFSET;
  acceleratorMemoryMap["weights"] = memoryMap.weights;
  matrixParams->WEIGHT_OFFSET = params.WEIGHT_OFFSET;
  matrixParams->WEIGHT_TRANSPOSE = params.WEIGHT_TRANSPOSE;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      matrixParams->loops[i][j] = params.loops[i][j];
    }
    matrixParams->inputXLoopIndex[i] = params.inputXLoopIndex[i];
    matrixParams->inputYLoopIndex[i] = params.inputYLoopIndex[i];
    matrixParams->reductionLoopIndex[i] = params.reductionLoopIndex[i];
    matrixParams->weightLoopIndex[i] = params.weightLoopIndex[i];
    matrixParams->weightReuseIndex[i] = params.weightReuseIndex[i];
  }
  matrixParams->fxIndex = params.fxIndex;
  matrixParams->fyIndex = params.fyIndex;

  // set outer loop values
  for (int j = 0; j < 5; j++) {
    matrixParams->weightAddressGenLoops[0][j] = params.loops[0][j];
  }
  // matrixParams->weightAddressGenInputXLoopIndex = params.inputXLoopIndex[0];
  // matrixParams->weightAddressGenInputYLoopIndex = params.inputYLoopIndex[0];
  matrixParams->weightAddressGenWeightLoopIndex[0] = params.weightLoopIndex[0];

  // set inner loop values
  if (params.WEIGHT_TRANSPOSE) {
    // for tranpose, we need to enforce that the innermost loop is the
    // unrolled reduction loop
    // we can just use the following loop nest:
    // C1, K, FY, FX, C0
    matrixParams->weightAddressGenLoops[1][4] = OC_DIMENSION;
    matrixParams->weightAddressGenReductionLoopIndex[1] = 4;
    matrixParams->weightAddressGenLoops[1][3] = params.loops[1][params.fxIndex];
    matrixParams->weightAddressGenFxIndex = 3;
    matrixParams->weightAddressGenLoops[1][2] = params.loops[1][params.fyIndex];
    matrixParams->weightAddressGenFyIndex = 2;
    matrixParams->weightAddressGenLoops[1][1] =
        params.loops[1][params.weightLoopIndex[1]];

    if (OC_DIMENSION > IC_DIMENSION) {
      // matrixParams->weightAddressGenLoops[1][1] =
      //     params.loops[1][params.weightLoopIndex[1]] /
      //     (OC_DIMENSION / IC_DIMENSION);
    }

    matrixParams->weightAddressGenWeightLoopIndex[1] = 1;
    matrixParams->weightAddressGenLoops[1][0] =
        params.loops[1][params.reductionLoopIndex[1]];

    if (OC_DIMENSION > IC_DIMENSION) {
      // we can reduce the number of iterations, since we have already fetched
      // the values
      matrixParams->weightAddressGenLoops[1][0] =
          params.loops[1][params.reductionLoopIndex[1]] /
          (OC_DIMENSION / IC_DIMENSION);
    }
    matrixParams->weightAddressGenReductionLoopIndex[0] = 0;
  } else {  // if not tranpose, then we have freedom to pick any loop order
    // for efficient memory accesses, addresses should be consecutive
    // or least, not multiples of 4, due to interleaving.
    // given that weights are arranged as: FY,FX,C,K
    // the following loop nest should work:
    // C1, C0, FX, FY, K
    // int index = 0;
    // for (int j = 0; j < 6; j++) {
    //   if (j == matrixParams->inputXLoopIndex[1] ||
    //       j == matrixParams->inputYLoopIndex[1]) {
    //     continue;
    //   }
    //   matrixParams->weightAddressGenLoops[1][index] = params.loops[1][j];

    //   if (j == matrixParams->reductionLoopIndex[1]) {
    //     matrixParams->weightAddressGenReductionLoopIndex[0] = index;
    //   }
    //   if (j == matrixParams->fxIndex) {
    //     matrixParams->weightAddressGenFxIndex = index;
    //   }
    //   if (j == matrixParams->fyIndex) {
    //     matrixParams->weightAddressGenFyIndex = index;
    //   }
    //   if (j == matrixParams->weightLoopIndex[1]) {
    //     matrixParams->weightAddressGenWeightLoopIndex[1] = index;
    //   }

    //   index++;
    // }
    // matrixParams->weightAddressGenLoops[1][4] = DIMENSION;
    // matrixParams->weightAddressGenReductionLoopIndex[1] = 4;

    matrixParams->weightAddressGenLoops[1][4] =
        params.loops[1][params.weightLoopIndex[1]];
    matrixParams->weightAddressGenWeightLoopIndex[1] = 4;

    matrixParams->weightAddressGenLoops[1][3] = params.loops[1][params.fyIndex];
    matrixParams->weightAddressGenFyIndex = 3;

    matrixParams->weightAddressGenLoops[1][2] = params.loops[1][params.fxIndex];
    if (params.REPLICATION) {
      matrixParams->weightAddressGenLoops[1][2] = 7;
    }
    matrixParams->weightAddressGenFxIndex = 2;

    if (params.REPLICATION) {
      matrixParams->weightAddressGenLoops[1][1] = 3;
      matrixParams->weightAddressGenReductionLoopIndex[1] = 1;
    } else {
      matrixParams->weightAddressGenLoops[1][1] = IC_DIMENSION;
      matrixParams->weightAddressGenReductionLoopIndex[1] = 1;
    }
    matrixParams->weightAddressGenLoops[1][0] =
        params.loops[1][params.reductionLoopIndex[1]];
    matrixParams->weightAddressGenReductionLoopIndex[0] = 0;
  }

  matrixParams->STRIDE = params.STRIDE;
  // matrixParams->HEAD_SIZE_LG2 = 0;
  matrixParams->REPLICATION = params.REPLICATION;
  matrixParams->STORE_IN_ACC = params.STORE_IN_ACC;
  matrixParams->ACC_FROM_ACC = params.ACC_FROM_ACC;
  matrixParams->CONCAT_INPUT = params.CONCAT_INPUT;
  matrixParams->CONCAT_HEAD_WEIGHTS = params.CONCAT_WEIGHT;
  matrixParams->TRANPOSE_INPUTS = params.INPUT_TRANSPOSE;

  // bias
  matrixParams->BIAS_OFFSET = params.BIAS_OFFSET;
  matrixParams->BIAS = params.BIAS;
  acceleratorMemoryMap["bias"] = memoryMap.bias;

  // P8 learningRate = static_cast<P8>(params.learningRate);
  // matrixParams->learningRate = learningRate.bits;

  // sendSerializedParams<MatrixParams, 32>(matrixParams,
  // &serialMatrixParamsIn);

  // memset(vectorParams, 0, sizeof(*vectorParams));

  vectorParams->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams->addressGen0Mode = false;  // use matrix unit outputs

  // residual
  acceleratorMemoryMap["vector1"] = memoryMap.residual;
  vectorParams->ADDRESS_GEN1_OFFSET = params.RESIDUAL_OFFSET;
  vectorParams->addressGen1Mode = params.RESIDUAL || params.RELU_GRAD;
  vectorParams->DP_VEC1 = false;

  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen1Loops[0][i] = params.loops[0][i];
  }
  vectorParams->addressGen1InputXLoopIndex[0] = params.inputXLoopIndex[0];
  vectorParams->addressGen1InputYLoopIndex[0] = params.inputYLoopIndex[0];
  vectorParams->addressGen1WeightLoopIndex[0] = params.weightLoopIndex[0];

  int residualLoopIndex = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == params.weightLoopIndex[1] || i == params.inputXLoopIndex[1] ||
        i == params.inputYLoopIndex[1]) {
      vectorParams->addressGen1Loops[1][residualLoopIndex] = params.loops[1][i];
      if (i == params.inputXLoopIndex[1]) {
        vectorParams->addressGen1InputXLoopIndex[1] = residualLoopIndex;
      }
      if (i == params.inputYLoopIndex[1]) {
        vectorParams->addressGen1InputYLoopIndex[1] = residualLoopIndex;
      }
      if (i == params.weightLoopIndex[1]) {
        vectorParams->addressGen1WeightLoopIndex[1] = residualLoopIndex;
      }
      residualLoopIndex++;
    }
  }

  // bias
  acceleratorMemoryMap["vector2"] = memoryMap.bias;
  vectorParams->ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
  vectorParams->addressGen2Mode = 0;
  for (int i = 0; i < 3; i++) {
    vectorParams->addressGen2Loops[0][i] = params.loops[0][i];
  }

  vectorParams->addressGen2InputXLoopIndex[0] = params.inputXLoopIndex[0];
  vectorParams->addressGen2InputYLoopIndex[0] = params.inputYLoopIndex[0];
  vectorParams->addressGen2WeightLoopIndex[0] = params.weightLoopIndex[0];

  int biasLoopIndex = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == params.weightLoopIndex[1] || i == params.inputXLoopIndex[1] ||
        i == params.inputYLoopIndex[1]) {
      vectorParams->addressGen2Loops[1][biasLoopIndex] = params.loops[1][i];
      if (i == params.inputXLoopIndex[1]) {
        vectorParams->addressGen2InputXLoopIndex[1] = biasLoopIndex;
      }
      if (i == params.inputYLoopIndex[1]) {
        vectorParams->addressGen2InputYLoopIndex[1] = biasLoopIndex;
      }
      if (i == params.weightLoopIndex[1]) {
        vectorParams->addressGen2WeightLoopIndex[1] = biasLoopIndex;
      }
      biasLoopIndex++;
    }
  }

  // vectorParams->FULL_HEAD_SIZE = 0;
  vectorParams->SPLIT_OUTPUT = params.SPLIT_OUTPUT;
  vectorParams->DP_OUTPUT = params.ACC_T_OUTPUT;
  vectorParams->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams->SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  // vectorParams->scalarOutputCount = 0;
  vectorParams->MAXPOOL = params.MAXPOOL;
  vectorParams->AVGPOOL = params.AVGPOOL;

  // output
  acceleratorMemoryMap["outputs"] = memoryMap.outputs;
  for (int i = 0; i < 3; i++) {
    vectorParams->outputLoops[0][i] = params.loops[0][i];
  }
  vectorParams->outputXLoopIndex[0] = params.inputXLoopIndex[0];
  vectorParams->outputYLoopIndex[0] = params.inputYLoopIndex[0];
  vectorParams->outputWeightLoopIndex[0] = params.weightLoopIndex[0];

  int outputLoopIndex = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == params.weightLoopIndex[1] || i == params.inputXLoopIndex[1] ||
        i == params.inputYLoopIndex[1]) {
      vectorParams->outputLoops[1][outputLoopIndex] = params.loops[1][i];
      if (i == params.inputXLoopIndex[1]) {
        vectorParams->outputXLoopIndex[1] = outputLoopIndex;
      }
      if (i == params.inputYLoopIndex[1]) {
        vectorParams->outputYLoopIndex[1] = outputLoopIndex;
      }
      if (i == params.weightLoopIndex[1]) {
        vectorParams->outputWeightLoopIndex[1] = outputLoopIndex;
      }
      outputLoopIndex++;
    }
  }

  vectorParams->SPLIT_OUTPUT = params.SPLIT_OUTPUT;

  // sendSerializedParams<VectorParams, 32>(vectorParams,
  // &serialVectorParamsIn);

  // create instruction stream
  //    VectorInstructionConfig vectorInstructionConfig;

  // memset(vectorInstructionConfig, 0, sizeof(*vectorInstructionConfig));

  VectorInstructions vInst0;
  memset(&vInst0, 0, sizeof(vInst0));
  vInst0.instType = VectorInstructions::vector;
  vInst0.vInput = VectorInstructions::readFromSystolicArray;
  vInst0.vAccumulatePush = 0;

  if (params.RESIDUAL) {
    vInst0.vOp0Src1 = VectorInstructions::readInterface;
    vInst0.vOp0 = VectorInstructions::vadd;
  } else if (params.ATTENTION_SCALING) {
    vInst0.vOp0Src1 = VectorInstructions::op0immediate0;
    float fpscale = (1.0 / sqrt(32));
    INPUT_DATATYPE scale(fpscale);
    vInst0.immediate0 = scale.bits_rep();
    vInst0.vOp0 = VectorInstructions::vmult;
  } else {
    vInst0.vOp0Src1 = VectorInstructions::nop;
    vInst0.vOp0 = VectorInstructions::nop;
  }

  vInst0.vOp1 = VectorInstructions::nop;
  vInst0.vOp2 = VectorInstructions::nop;

  // if (params.BIAS) {
  //   vInst0.vOp3Src1 = VectorInstructions::readNormalInterface;
  //   vInst0.vOp3 = VectorInstructions::vadd;
  // } else {
  vInst0.vOp3Src1 = VectorInstructions::nop;
  vInst0.vOp3 = VectorInstructions::nop;
  // }

  if (params.RELU) {
    vInst0.vOp4 = VectorInstructions::vrelu;
  } else if (params.RELU_GRAD) {
    vInst0.vOp0Src1 = VectorInstructions::readInterface;
    vInst0.vOp4 = VectorInstructions::vrelumask;
  } else {
    vInst0.vOp4 = VectorInstructions::nop;
  }

  if (params.AVGPOOL) {
    // accumulate over X*Y
    VectorInstructions accumulatorInst;
    accumulatorInst.instType = VectorInstructions::accumulation;
    accumulatorInst.rOp = VectorInstructions::radd;
    accumulatorInst.rCount = X * Y;
    vectorInstructionConfig->instCount[0] = 1;
    vectorInstructionConfig->inst[0] = accumulatorInst;

    // build on top of existing inst0
    // send to accumulator
    vInst0.vAccumulatePush = 1;
    vectorInstructionConfig->inst[1] = vInst0;
    vectorInstructionConfig->instCount[1] = X * Y;

    // pull from accumulator and divide by X*Y
    VectorInstructions vInst2;
    vInst2.instType = VectorInstructions::vector;
    vInst2.vInput = VectorInstructions::readFromAccumulation;
    vInst2.vAccumulatePush = 0;
    vInst2.vOp0Src1 = VectorInstructions::nop;
    vInst2.vOp0 = VectorInstructions::nop;
    vInst2.vOp1 = VectorInstructions::nop;
    vInst2.vOp1 = VectorInstructions::nop;
    vInst2.vOp3Src1 = VectorInstructions::op3immediate0;
    vInst2.vOp3 = VectorInstructions::vmult;
    vInst2.vOp4 = VectorInstructions::nop;
    vInst2.vDest = VectorInstructions::vWriteOut;
    float fpscale = (1.0 / (X * Y));
    INPUT_DATATYPE scale(fpscale);
    vInst2.immediate0 = scale.bits_rep();
    vectorInstructionConfig->inst[2] = vInst2;
    vectorInstructionConfig->instCount[2] = 1;

    vectorInstructionConfig->instLen = 3;
    vectorInstructionConfig->instLoopCount = K / OC_DIMENSION;
  } else {
    vInst0.vDest = VectorInstructions::vWriteOut;
    vectorInstructionConfig->inst[0] = vInst0;

    // total output count
    vectorInstructionConfig->instCount[0] =
        params.loops[0][params.inputXLoopIndex[0]] *
        params.loops[1][params.inputXLoopIndex[1]] *
        params.loops[0][params.inputYLoopIndex[0]] *
        params.loops[1][params.inputYLoopIndex[1]] *
        params.loops[0][params.weightLoopIndex[0]] *
        params.loops[1][params.weightLoopIndex[1]];
    vectorInstructionConfig->instLen = 1;
    vectorInstructionConfig->instLoopCount = 1;
  }

  mappedParams.push_back(matrixParams);
  mappedParams.push_back(vectorParams);
  mappedParams.push_back(vectorInstructionConfig);
  opMemoryMaps.push_back(acceleratorMemoryMap);
}
