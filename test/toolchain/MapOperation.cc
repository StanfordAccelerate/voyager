#include "test/toolchain/MapOperation.h"

// create Matrix and Vector Params from SimplifiedParams
void map_operation(const SimplifiedParams &params, MatrixParams &matrixParams,
                   bool &matrixParamsValid, VectorParams &vectorParams,
                   VectorInstructionConfig &vectorInstructionConfig,
                   bool &vectorParamsValid) {
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

  if (params.SOFTMAX) {
    matrixParamsValid = false;
    vectorParamsValid = true;

    vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
    vectorParams.addressGen0Enable = true;
    vectorParams.addressGen0Broadcast = false;
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen0Loop[0][i] = 1;
    }
    vectorParams.addressGen0Loop[1][0] = 1;
    vectorParams.addressGen0Loop[1][1] = Y;
    vectorParams.addressGen0Loop[1][2] = C / DIMENSION;

    // address gen 1 (weights)
    vectorParams.ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
    vectorParams.addressGen1Mode = 2;  // 2d tensor
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen1Loops[0][i] = 1;
    }
    vectorParams.addressGen1Loops[1][0] = Y;
    vectorParams.addressGen1Loops[1][1] = 1;
    vectorParams.addressGen1Loops[1][2] = C / DIMENSION;

    vectorParams.ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
    vectorParams.addressGen2Mode = 2;  // 2d tensor
    vectorParams.addressGen2Loops[0][0] = Y;
    vectorParams.addressGen2Loops[0][1] = 1;
    vectorParams.addressGen2Loops[0][2] = C / DIMENSION;

    vectorParams.VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
    vectorParams.SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

    vectorParams.scalarOutputCount = 0;
    vectorParams.MAXPOOL = params.MAXPOOL;
    vectorParams.AVGPOOL = params.AVGPOOL;

    // output
    for (int i = 0; i < 3; i++) {
      vectorParams.outputLoops[0][i] = params.loops[0][i];
    }
    vectorParams.outputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.outputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.outputWeightLoopIndex[0] = params.weightLoopIndex[0];

    vectorParams.outputLoops[1][0] = 1;
    vectorParams.outputLoops[1][1] = Y;
    vectorParams.outputLoops[1][2] = C / DIMENSION;
    vectorParams.outputWeightLoopIndex[1] = 2;
    vectorParams.outputYLoopIndex[1] = 1;
    vectorParams.outputXLoopIndex[1] = 0;

    // sendSerializedParams<VectorParams, 32>(vectorParams,
    // &serialVectorParamsIn);

    // create instruction stream
    //    VectorInstructionConfig vectorInstructionConfig;

    // inst 0- calculate max value and save
    VectorInstructions vInst0;
    vInst0.instType = VectorInstructions::vector;
    vInst0.vInput = VectorInstructions::readFromVectorFetch;
    vInst0.vAccumulatePush = VectorInstructions::nop;
    vInst0.vOp0Src1 = VectorInstructions::readInterface;
    vInst0.vOp0 = VectorInstructions::vmult;
    vInst0.vOp1 = VectorInstructions::nop;
    vInst0.vOp2 = VectorInstructions::nop;
    vInst0.vOp3Src0 = VectorInstructions::nop;
    vInst0.vOp3Src1 = VectorInstructions::readNormalInterface;
    vInst0.vOp3 = VectorInstructions::vadd;
    vInst0.vOp4 = params.RELU;
    vInst0.vDest = VectorInstructions::vWriteOut;

    vectorInstructionConfig.inst[0] = vInst0;
    // C/DIMENSION to do the complete reduction
    // DIMENSION to fill up the entire vector
    vectorInstructionConfig.instCount[0] = Y * C / DIMENSION;

    // inst 1- subtract max and exp, and reduce

    // inst 2- subtract max and exp, and divide by reduced value

    vectorInstructionConfig.instLen = 1;
    vectorInstructionConfig.instLoopCount = 1;

    // sendSerializedParams<VectorInstructionConfig,
    // 32>(vectorInstructionConfig,
    //   &serialVectorParamsIn);

    //     vectorUnitStartSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Started.");
    //     vectorUnitDoneSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Finished.");
  } else if (params.SOFTMAX_GRAD) {
    matrixParamsValid = false;
    vectorParamsValid = true;

    vectorParams.VECTOR_OFFSET = params.RESIDUAL_OFFSET;
    vectorParams.addressGen0Enable = true;
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen0Loop[0][i] = 1;
    }
    vectorParams.addressGen0Loop[1][0] = 1;
    vectorParams.addressGen0Loop[1][1] = X;
    vectorParams.addressGen0Loop[1][2] = Y;
    vectorParams.addressGen0Broadcast = true;
    vectorParams.addressGen0BroadcastCount = Y;
    vectorParams.SOFTMAX_GRAD_NEGATE = true;

    vectorParams.ADDRESS_GEN1_OFFSET = params.RESIDUAL_OFFSET;
    vectorParams.addressGen1Mode = 2;  // 2d tensor
    vectorParams.addressGen1Loops[0][0] = 1;
    vectorParams.addressGen1Loops[0][1] = X;
    vectorParams.addressGen1Loops[0][2] = 1;
    vectorParams.addressGen1Loops[1][0] = Y;  // repeat
    vectorParams.addressGen1Loops[1][1] = 1;
    vectorParams.addressGen1Loops[1][2] = Y;

    vectorParams.ADDRESS_GEN2_OFFSET = params.INPUT_OFFSET;
    vectorParams.addressGen2Mode = 2;  // 2d tensor
    vectorParams.addressGen2Loops[0][0] = 1;
    vectorParams.addressGen2Loops[0][1] = X;
    vectorParams.addressGen2Loops[0][2] = 1;
    vectorParams.addressGen2Loops[1][0] = Y;  // repeat
    vectorParams.addressGen2Loops[1][1] = 1;
    vectorParams.addressGen2Loops[1][2] = Y;

    vectorParams.VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
    vectorParams.SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

    vectorParams.scalarOutputCount = 0;
    vectorParams.MAXPOOL = params.MAXPOOL;
    vectorParams.AVGPOOL = params.AVGPOOL;

    // output
    for (int i = 0; i < 3; i++) {
      vectorParams.outputLoops[0][i] = params.loops[0][i];
    }
    vectorParams.outputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.outputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.outputWeightLoopIndex[0] = params.weightLoopIndex[0];

    vectorParams.outputLoops[1][0] = 1;
    vectorParams.outputLoops[1][1] = X;
    vectorParams.outputLoops[1][2] = Y;
    vectorParams.outputWeightLoopIndex[1] = 2;
    vectorParams.outputYLoopIndex[1] = 1;
    vectorParams.outputXLoopIndex[1] = 0;

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
    vInst0.rDest = VectorInstructions::toVectorSrc0;
    vectorInstructionConfig.inst[0] = vInst0;
    vectorInstructionConfig.instCount[0] = 1;

    // perform multiplication (-a_i * a_j) or (a_i * (1-a_i))
    VectorInstructions vInst1;
    vInst1.instType = VectorInstructions::vector;
    vInst1.vInput = VectorInstructions::readFromVectorFetch;
    vInst1.vAccumulatePush = VectorInstructions::nop;
    vInst1.vOp0Src1 = VectorInstructions::readInterface;
    vInst1.vOp0 = VectorInstructions::vmult;
    vInst1.vOp1 = VectorInstructions::nop;
    vInst1.vOp2 = VectorInstructions::toReduce;
    vInst1.vOp3Src0 = VectorInstructions::nop;
    vInst1.vOp3Src1 = VectorInstructions::readNormalInterface;
    vInst1.vOp3 = VectorInstructions::vmult;
    vInst1.vOp4 = VectorInstructions::nop;
    vInst1.vDest = VectorInstructions::nop;
    vectorInstructionConfig.inst[1] = vInst1;
    vectorInstructionConfig.instCount[1] = Y * DIMENSION;

    // pull out from reduce and multiply by 1/sqrt(32)
    VectorInstructions vInst2;
    vInst2.instType = VectorInstructions::vector;
    vInst2.vInput = VectorInstructions::nop;
    vInst2.vAccumulatePush = VectorInstructions::nop;
    vInst2.vOp0Src1 = VectorInstructions::nop;
    vInst2.vOp0 = VectorInstructions::vmult;
    vInst2.vOp1 = VectorInstructions::nop;
    vInst2.vOp2 = VectorInstructions::toReduce;
    vInst2.vOp3Src0 = VectorInstructions::readReduceInterface;
    vInst2.vOp3Src1 = VectorInstructions::op3immediate0;

    vInst2.vOp3 = VectorInstructions::vmult;
    vInst2.vOp4 = VectorInstructions::nop;
    vInst2.vDest = VectorInstructions::vWriteOut;
    vectorInstructionConfig.inst[2] = vInst2;
    vectorInstructionConfig.instCount[2] = 1;
    Posit<16, 1> scale = (1.0 / sqrt(32));
    vInst2.immediate0 = scale.bits;

    vectorInstructionConfig.instLen = 3;
    vectorInstructionConfig.instLoopCount = X * Y / DIMENSION;

    // sendSerializedParams<VectorInstructionConfig,
    // 32>(vectorInstructionConfig,
    //   &serialVectorParamsIn);

    //     vectorUnitStartSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Started.");
    //     vectorUnitDoneSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Finished.");
  } else if (params.FC) {
    matrixParamsValid = false;
    vectorParamsValid = true;

    vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
    vectorParams.addressGen0Enable = true;
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen0Loop[0][i] = 1;
    }
    vectorParams.addressGen0Loop[1][0] = K;
    vectorParams.addressGen0Loop[1][1] = 1;
    vectorParams.addressGen0Loop[1][2] = C / DIMENSION;
    vectorParams.addressGen0Broadcast = false;

    // address gen 1 (weights)
    vectorParams.ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
    vectorParams.addressGen1Mode = 2;  // 2d tensor
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen1Loops[0][i] = 1;
    }
    vectorParams.addressGen1Loops[0][0] = 1;
    vectorParams.addressGen1Loops[0][1] = K;
    vectorParams.addressGen1Loops[0][2] = C / DIMENSION;

    // TODO: deal with bias
    // bias
    vectorParams.ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
    vectorParams.addressGen2Mode = 0;

    vectorParams.VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
    vectorParams.SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

    vectorParams.scalarOutputCount = 0;
    vectorParams.MAXPOOL = params.MAXPOOL;
    vectorParams.AVGPOOL = params.AVGPOOL;

    // output
    for (int i = 0; i < 3; i++) {
      vectorParams.outputLoops[0][i] = params.loops[0][i];
    }
    vectorParams.outputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.outputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.outputWeightLoopIndex[0] = params.weightLoopIndex[0];

    int outputLoopIndex = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == params.weightLoopIndex[1] || i == params.inputXLoopIndex[1] ||
          i == params.inputYLoopIndex[1]) {
        vectorParams.outputLoops[1][outputLoopIndex] = params.loops[1][i];
        if (i == params.inputXLoopIndex[1]) {
          vectorParams.outputXLoopIndex[1] = outputLoopIndex;
        }
        if (i == params.inputYLoopIndex[1]) {
          vectorParams.outputYLoopIndex[1] = outputLoopIndex;
        }
        if (i == params.weightLoopIndex[1]) {
          vectorParams.outputWeightLoopIndex[1] = outputLoopIndex;
        }
        outputLoopIndex++;
      }
    }

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
    vInst0.rDest = VectorInstructions::toVectorSrc0;
    vectorInstructionConfig.inst[0] = vInst0;
    vectorInstructionConfig.instCount[0] = 1;

    // inst 1- inputs x weights, send to reduce
    VectorInstructions vInst1;
    vInst1.instType = VectorInstructions::vector;
    vInst1.vInput = VectorInstructions::readFromVectorFetch;
    vInst1.vAccumulatePush = VectorInstructions::nop;
    vInst1.vOp0Src1 = VectorInstructions::readInterface;
    vInst1.vOp0 = VectorInstructions::vmult;
    vInst1.vOp1 = VectorInstructions::nop;
    vInst1.vOp2 = VectorInstructions::toReduce;
    vInst1.vOp3Src0 = VectorInstructions::nop;
    vInst1.vOp3Src1 = VectorInstructions::nop;
    vInst1.vOp3 = VectorInstructions::nop;
    vInst1.vOp4 = VectorInstructions::nop;
    vInst1.vDest = VectorInstructions::nop;
    vectorInstructionConfig.inst[1] = vInst1;

    // C/DIMENSION to do the complete reduction
    // DIMENSION to fill up the entire vector
    vectorInstructionConfig.instCount[1] = DIMENSION * C / DIMENSION;

    // inst2- add bias, write out
    VectorInstructions vInst2;
    vInst2.instType = VectorInstructions::vector;
    vInst2.vInput = VectorInstructions::nop;
    vInst2.vAccumulatePush = VectorInstructions::nop;
    vInst2.vOp0Src1 = VectorInstructions::nop;
    vInst2.vOp0 = VectorInstructions::vmult;
    vInst2.vOp1 = VectorInstructions::nop;
    vInst2.vOp2 = VectorInstructions::nop;
    vInst2.vOp3Src0 = VectorInstructions::readReduceInterface;
    vInst2.vOp3Src1 = VectorInstructions::nop;  // TODO: change to add for bias
    vInst2.vOp3 = VectorInstructions::nop;
    vInst2.vOp4 = params.RELU;
    vInst2.vDest = VectorInstructions::vWriteOut;
    vectorInstructionConfig.inst[2] = vInst2;
    vectorInstructionConfig.instCount[2] = 1;

    vectorInstructionConfig.instLen = 3;
    vectorInstructionConfig.instLoopCount = K / DIMENSION;

    // sendSerializedParams<VectorInstructionConfig,
    // 32>(vectorInstructionConfig,
    //   &serialVectorParamsIn);

    //     vectorUnitStartSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Started.");
    //     vectorUnitDoneSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Finished.");
  } else if (params.NO_NORM) {
    matrixParamsValid = false;
    vectorParamsValid = true;

    vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
    vectorParams.addressGen0Enable = true;
    vectorParams.addressGen0Broadcast = false;
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen0Loop[0][i] = 1;
    }
    vectorParams.addressGen0Loop[1][0] = 1;
    vectorParams.addressGen0Loop[1][1] = X;
    vectorParams.addressGen0Loop[1][2] = C / DIMENSION;

    // address gen 1 (weights)
    vectorParams.ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
    vectorParams.addressGen1Mode = 2;  // 2d tensor
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen1Loops[0][i] = 1;
    }
    vectorParams.addressGen1Loops[1][0] = X;
    vectorParams.addressGen1Loops[1][1] = 1;
    vectorParams.addressGen1Loops[1][2] = C / DIMENSION;

    vectorParams.ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
    vectorParams.addressGen2Mode = 1;  // use bias mode
    vectorParams.addressGen2Loops[0][0] = X;
    vectorParams.addressGen2Loops[0][1] = 1;
    vectorParams.addressGen2Loops[0][2] = 1;
    vectorParams.addressGen2Loops[1][0] = C / DIMENSION;
    vectorParams.addressGen2Loops[1][1] = 1;
    vectorParams.addressGen2Loops[1][2] = 1;
    vectorParams.addressGen2InputXLoopIndex[1] = 2;
    vectorParams.addressGen2InputYLoopIndex[1] = 1;
    vectorParams.addressGen2WeightLoopIndex[1] = 0;
    vectorParams.addressGen2WeightLoopIndex[0] = 2;

    vectorParams.VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
    vectorParams.SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

    vectorParams.scalarOutputCount = 0;
    vectorParams.MAXPOOL = params.MAXPOOL;
    vectorParams.AVGPOOL = params.AVGPOOL;
    vectorParams.SPLIT_OUTPUT = params.SPLIT_OUTPUT;

    // output
    for (int i = 0; i < 3; i++) {
      vectorParams.outputLoops[0][i] = params.loops[0][i];
    }
    vectorParams.outputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.outputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.outputWeightLoopIndex[0] = params.weightLoopIndex[0];

    vectorParams.outputLoops[1][0] = 1;
    vectorParams.outputLoops[1][1] = X;
    vectorParams.outputLoops[1][2] = C / DIMENSION;
    vectorParams.outputWeightLoopIndex[1] = 2;
    vectorParams.outputYLoopIndex[1] = 0;
    vectorParams.outputXLoopIndex[1] = 1;

    // sendSerializedParams<VectorParams, 32>(vectorParams,
    // &serialVectorParamsIn);

    // create instruction stream
    //    VectorInstructionConfig vectorInstructionConfig;

    // inst 1- (inputs x weights) + bias
    VectorInstructions vInst0;
    vInst0.instType = VectorInstructions::vector;
    vInst0.vInput = VectorInstructions::readFromVectorFetch;
    vInst0.vAccumulatePush = VectorInstructions::nop;
    vInst0.vOp0Src1 = VectorInstructions::readInterface;
    vInst0.vOp0 = VectorInstructions::vmult;
    vInst0.vOp1 = VectorInstructions::nop;
    vInst0.vOp2 = VectorInstructions::nop;
    vInst0.vOp3Src0 = VectorInstructions::nop;
    vInst0.vOp3Src1 = VectorInstructions::readNormalInterface;
    vInst0.vOp3 = VectorInstructions::vadd;
    vInst0.vOp4 = params.RELU;
    vInst0.vDest = VectorInstructions::vWriteOut;
    vectorInstructionConfig.inst[0] = vInst0;

    // C/DIMENSION to do the complete reduction
    // DIMENSION to fill up the entire vector
    vectorInstructionConfig.instCount[0] = X * C / DIMENSION;

    vectorInstructionConfig.instLen = 1;
    vectorInstructionConfig.instLoopCount = 1;

    // sendSerializedParams<VectorInstructionConfig,
    // 32>(vectorInstructionConfig,
    //   &serialVectorParamsIn);

    //     vectorUnitStartSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Started.");
    //     vectorUnitDoneSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Finished.");
  } else if (params.NO_NORM_GRAD) {
    matrixParamsValid = false;
    vectorParamsValid = true;

    vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
    vectorParams.addressGen0Enable = true;
    vectorParams.addressGen0Broadcast = false;

    vectorParams.addressGen0Loop[0][0] = 1;
    vectorParams.addressGen0Loop[0][1] = 1;
    vectorParams.addressGen0Loop[0][2] = K / DIMENSION;
    vectorParams.addressGen0Loop[1][0] = 1;
    vectorParams.addressGen0Loop[1][1] = X;
    vectorParams.addressGen0Loop[1][2] = 1;

    // address gen 1 (weights)
    vectorParams.ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
    vectorParams.addressGen1Mode = 2;  // 2d tensor
    vectorParams.addressGen1Loops[0][0] = 1;
    vectorParams.addressGen1Loops[0][1] = 1;
    vectorParams.addressGen1Loops[0][2] = K / DIMENSION;
    vectorParams.addressGen1Loops[1][0] = 1;
    vectorParams.addressGen1Loops[1][1] = X;
    vectorParams.addressGen1Loops[1][2] = 1;

    vectorParams.ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
    vectorParams.addressGen2Mode = 0;  // no bias

    vectorParams.VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
    vectorParams.SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;

    vectorParams.scalarOutputCount = 0;
    vectorParams.MAXPOOL = false;
    vectorParams.AVGPOOL = false;
    vectorParams.SPLIT_OUTPUT = false;

    // output
    for (int i = 0; i < 3; i++) {
      vectorParams.outputLoops[0][i] = 1;
    }
    vectorParams.outputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.outputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.outputWeightLoopIndex[0] = params.weightLoopIndex[0];

    vectorParams.outputLoops[1][0] = 1;
    vectorParams.outputLoops[1][1] = 1;
    vectorParams.outputLoops[1][2] = K / DIMENSION;
    vectorParams.outputWeightLoopIndex[1] = 2;
    vectorParams.outputYLoopIndex[1] = 0;
    vectorParams.outputXLoopIndex[1] = 1;

    // sendSerializedParams<VectorParams, 32>(vectorParams,
    // &serialVectorParamsIn);

    // create instruction stream
    //    VectorInstructionConfig vectorInstructionConfig;

    // inst 0- accumulate
    VectorInstructions vInst0;
    vInst0.instType = VectorInstructions::accumulation;
    vInst0.rCount = X;
    vectorInstructionConfig.instCount[0] = 1;
    vectorInstructionConfig.inst[0] = vInst0;

    // inst 1- (inputs x weights), send to accumulator
    VectorInstructions vInst1;
    vInst1.instType = VectorInstructions::vector;
    vInst1.vInput = VectorInstructions::readFromVectorFetch;
    vInst1.vOp0Src1 = VectorInstructions::readInterface;
    vInst1.vOp0 = VectorInstructions::vmult;
    vInst1.vOp1 = VectorInstructions::nop;
    vInst1.vOp2 = VectorInstructions::nop;
    vInst1.vOp3Src0 = VectorInstructions::nop;
    vInst1.vOp3Src1 = VectorInstructions::nop;
    vInst1.vOp3 = VectorInstructions::nop;
    vInst1.vOp4 = VectorInstructions::nop;
    vInst1.vAccumulatePush = 1;
    vInst1.vDest = VectorInstructions::nop;
    vectorInstructionConfig.inst[1] = vInst1;

    // C/DIMENSION to do the complete reduction
    // DIMENSION to fill up the entire vector
    vectorInstructionConfig.instCount[1] = X;

    // inst 2- pull from accumulator
    VectorInstructions vInst2;
    vInst2.instType = VectorInstructions::vector;
    vInst2.vInput = VectorInstructions::readFromAccumulation;
    vInst2.vOp0Src1 = VectorInstructions::nop;
    vInst2.vOp0 = VectorInstructions::nop;
    vInst2.vOp1 = VectorInstructions::nop;
    vInst2.vOp2 = VectorInstructions::nop;
    vInst2.vOp3Src0 = VectorInstructions::nop;
    vInst2.vOp3Src1 = VectorInstructions::nop;
    vInst2.vOp3 = VectorInstructions::nop;
    vInst2.vOp4 = VectorInstructions::nop;
    vInst2.vAccumulatePush = 0;
    vInst2.vDest = VectorInstructions::vWriteOut;
    vectorInstructionConfig.inst[2] = vInst2;
    vectorInstructionConfig.instCount[2] = 1;

    vectorInstructionConfig.instLen = 3;
    vectorInstructionConfig.instLoopCount = C / DIMENSION;

    // sendSerializedParams<VectorInstructionConfig,
    // 32>(vectorInstructionConfig,
    //   &serialVectorParamsIn);

    //     vectorUnitStartSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Started.");
    //     vectorUnitDoneSignal.SyncPop();
    //     CCS_LOG("Accelerator
    //     Layer Finished.");
  } else {
    matrixParamsValid = true;
    vectorParamsValid = true;

    // matrix params
    matrixParams.INPUT_OFFSET = params.INPUT_OFFSET;
    matrixParams.WEIGHT_OFFSET = params.WEIGHT_OFFSET;
    matrixParams.WEIGHT_TRANSPOSE = params.WEIGHT_TRANSPOSE;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        matrixParams.loops[i][j] = params.loops[i][j];
      }
      matrixParams.inputXLoopIndex[i] = params.inputXLoopIndex[i];
      matrixParams.inputYLoopIndex[i] = params.inputYLoopIndex[i];
      matrixParams.reductionLoopIndex[i] = params.reductionLoopIndex[i];
      matrixParams.weightLoopIndex[i] = params.weightLoopIndex[i];
      matrixParams.weightReuseIndex[i] = params.weightReuseIndex[i];
    }
    matrixParams.fxIndex = params.fxIndex;
    matrixParams.fyIndex = params.fyIndex;

    // set outer loop values
    for (int j = 0; j < 5; j++) {
      matrixParams.weightAddressGenLoops[0][j] = params.loops[0][j];
    }
    matrixParams.weightAddressGenInputXLoopIndex = params.inputXLoopIndex[0];
    matrixParams.weightAddressGenInputYLoopIndex = params.inputYLoopIndex[0];
    matrixParams.weightAddressGenWeightLoopIndex[0] = params.weightLoopIndex[0];

    // set inner loop values
    if (params.WEIGHT_TRANSPOSE) {
      // for tranpose, we need to enforce that the innermost loop is the
      // unrolled reduction loop
      // we can just use the following loop nest:
      // C1, K, FY, FX, C0
      matrixParams.weightAddressGenLoops[1][4] = DIMENSION;
      matrixParams.weightAddressGenReductionLoopIndex[1] = 4;
      matrixParams.weightAddressGenLoops[1][3] =
          params.loops[1][params.fxIndex];
      matrixParams.weightAddressGenFxIndex = 3;
      matrixParams.weightAddressGenLoops[1][2] =
          params.loops[1][params.fyIndex];
      matrixParams.weightAddressGenFyIndex = 2;
      matrixParams.weightAddressGenLoops[1][1] =
          params.loops[1][params.weightLoopIndex[1]];
      matrixParams.weightAddressGenWeightLoopIndex[1] = 1;
      matrixParams.weightAddressGenLoops[1][0] =
          params.loops[1][params.reductionLoopIndex[1]];
      matrixParams.weightAddressGenReductionLoopIndex[0] = 0;
    } else {  // if not tranpose, then we have freedom to pick any loop order
      // for efficient memory accesses, addresses should be consecutive
      // or least, not multiples of 4, due to interleaving.
      // given that weights are arranged as: FY,FX,C,K
      // the following loop nest should work:
      // C1, C0, FX, FY, K
      // int index = 0;
      // for (int j = 0; j < 6; j++) {
      //   if (j == matrixParams.inputXLoopIndex[1] ||
      //       j == matrixParams.inputYLoopIndex[1]) {
      //     continue;
      //   }
      //   matrixParams.weightAddressGenLoops[1][index] = params.loops[1][j];

      //   if (j == matrixParams.reductionLoopIndex[1]) {
      //     matrixParams.weightAddressGenReductionLoopIndex[0] = index;
      //   }
      //   if (j == matrixParams.fxIndex) {
      //     matrixParams.weightAddressGenFxIndex = index;
      //   }
      //   if (j == matrixParams.fyIndex) {
      //     matrixParams.weightAddressGenFyIndex = index;
      //   }
      //   if (j == matrixParams.weightLoopIndex[1]) {
      //     matrixParams.weightAddressGenWeightLoopIndex[1] = index;
      //   }

      //   index++;
      // }
      // matrixParams.weightAddressGenLoops[1][4] = DIMENSION;
      // matrixParams.weightAddressGenReductionLoopIndex[1] = 4;

      matrixParams.weightAddressGenLoops[1][4] =
          params.loops[1][params.weightLoopIndex[1]];
      matrixParams.weightAddressGenWeightLoopIndex[1] = 4;

      matrixParams.weightAddressGenLoops[1][3] =
          params.loops[1][params.fyIndex];
      matrixParams.weightAddressGenFyIndex = 3;

      matrixParams.weightAddressGenLoops[1][2] =
          params.loops[1][params.fxIndex];
      if (params.REPLICATION) {
        matrixParams.weightAddressGenLoops[1][2] = 7;
      }
      matrixParams.weightAddressGenFxIndex = 2;

      if (params.REPLICATION) {
        matrixParams.weightAddressGenLoops[1][1] = 3;
        matrixParams.weightAddressGenReductionLoopIndex[1] = 1;
      } else {
        matrixParams.weightAddressGenLoops[1][1] = DIMENSION;
        matrixParams.weightAddressGenReductionLoopIndex[1] = 1;
      }
      matrixParams.weightAddressGenLoops[1][0] =
          params.loops[1][params.reductionLoopIndex[1]];
      matrixParams.weightAddressGenReductionLoopIndex[0] = 0;
    }

    matrixParams.STRIDE = params.STRIDE;
    matrixParams.HEAD_SIZE_LG2 = 0;
    matrixParams.REPLICATION = params.REPLICATION;
    matrixParams.STORE_IN_ACC = params.STORE_IN_ACC;
    matrixParams.ACC_FROM_ACC = params.ACC_FROM_ACC;
    matrixParams.CONCAT_INPUT = params.CONCAT_INPUT;
    matrixParams.CONCAT_HEAD_WEIGHTS = params.CONCAT_WEIGHT;
    matrixParams.TRANPOSE_INPUTS = params.INPUT_TRANSPOSE;

    // sendSerializedParams<MatrixParams, 32>(matrixParams,
    // &serialMatrixParamsIn);

    memset(&vectorParams, 0, sizeof(vectorParams));

    vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
    vectorParams.addressGen0Enable = false;  // use matrix unit outputs

    // residual
    vectorParams.ADDRESS_GEN1_OFFSET = params.RESIDUAL_OFFSET;
    vectorParams.addressGen1Mode = params.RESIDUAL;

    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen1Loops[0][i] = params.loops[0][i];
    }
    vectorParams.addressGen1InputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.addressGen1InputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.addressGen1WeightLoopIndex[0] = params.weightLoopIndex[0];

    int residualLoopIndex = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == params.weightLoopIndex[1] || i == params.inputXLoopIndex[1] ||
          i == params.inputYLoopIndex[1]) {
        vectorParams.addressGen1Loops[1][residualLoopIndex] =
            params.loops[1][i];
        if (i == params.inputXLoopIndex[1]) {
          vectorParams.addressGen1InputXLoopIndex[1] = residualLoopIndex;
        }
        if (i == params.inputYLoopIndex[1]) {
          vectorParams.addressGen1InputYLoopIndex[1] = residualLoopIndex;
        }
        if (i == params.weightLoopIndex[1]) {
          vectorParams.addressGen1WeightLoopIndex[1] = residualLoopIndex;
        }
        residualLoopIndex++;
      }
    }

    // bias
    vectorParams.ADDRESS_GEN2_OFFSET = params.BIAS_OFFSET;
    vectorParams.addressGen2Mode = params.BIAS;
    for (int i = 0; i < 3; i++) {
      vectorParams.addressGen2Loops[0][i] = params.loops[0][i];
    }

    vectorParams.addressGen2InputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.addressGen2InputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.addressGen2WeightLoopIndex[0] = params.weightLoopIndex[0];

    int biasLoopIndex = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == params.weightLoopIndex[1] || i == params.inputXLoopIndex[1] ||
          i == params.inputYLoopIndex[1]) {
        vectorParams.addressGen2Loops[1][biasLoopIndex] = params.loops[1][i];
        if (i == params.inputXLoopIndex[1]) {
          vectorParams.addressGen2InputXLoopIndex[1] = biasLoopIndex;
        }
        if (i == params.inputYLoopIndex[1]) {
          vectorParams.addressGen2InputYLoopIndex[1] = biasLoopIndex;
        }
        if (i == params.weightLoopIndex[1]) {
          vectorParams.addressGen2WeightLoopIndex[1] = biasLoopIndex;
        }
        biasLoopIndex++;
      }
    }

    vectorParams.FULL_HEAD_SIZE = 0;
    vectorParams.SPLIT_OUTPUT = params.SPLIT_OUTPUT;
    vectorParams.DP_OUTPUT = false;
    vectorParams.VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
    vectorParams.SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
    vectorParams.scalarOutputCount = 0;
    vectorParams.MAXPOOL = params.MAXPOOL;
    vectorParams.AVGPOOL = params.AVGPOOL;

    // output
    for (int i = 0; i < 3; i++) {
      vectorParams.outputLoops[0][i] = params.loops[0][i];
    }
    vectorParams.outputXLoopIndex[0] = params.inputXLoopIndex[0];
    vectorParams.outputYLoopIndex[0] = params.inputYLoopIndex[0];
    vectorParams.outputWeightLoopIndex[0] = params.weightLoopIndex[0];

    int outputLoopIndex = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == params.weightLoopIndex[1] || i == params.inputXLoopIndex[1] ||
          i == params.inputYLoopIndex[1]) {
        vectorParams.outputLoops[1][outputLoopIndex] = params.loops[1][i];
        if (i == params.inputXLoopIndex[1]) {
          vectorParams.outputXLoopIndex[1] = outputLoopIndex;
        }
        if (i == params.inputYLoopIndex[1]) {
          vectorParams.outputYLoopIndex[1] = outputLoopIndex;
        }
        if (i == params.weightLoopIndex[1]) {
          vectorParams.outputWeightLoopIndex[1] = outputLoopIndex;
        }
        outputLoopIndex++;
      }
    }

    vectorParams.SPLIT_OUTPUT = params.SPLIT_OUTPUT;

    // sendSerializedParams<VectorParams, 32>(vectorParams,
    // &serialVectorParamsIn);

    // create instruction stream
    //    VectorInstructionConfig vectorInstructionConfig;

    memset(&vectorInstructionConfig, 0, sizeof(vectorInstructionConfig));

    if (params.AVGPOOL) {
      // accumulate over X*Y
      VectorInstructions vInst0;
      vInst0.instType = VectorInstructions::accumulation;
      vInst0.rCount = X * Y;
      vectorInstructionConfig.instCount[0] = 1;
      vectorInstructionConfig.inst[0] = vInst0;

      // send to accumulator
      VectorInstructions vInst1;
      vInst1.instType = VectorInstructions::vector;
      vInst1.vInput = VectorInstructions::readFromSystolicArray;
      vInst1.vAccumulatePush = 1;
      vInst1.vOp0Src1 = VectorInstructions::nop;
      vInst1.vOp0 = VectorInstructions::nop;
      vInst1.vOp1 = VectorInstructions::nop;
      vInst1.vOp1 = VectorInstructions::nop;
      vInst1.vOp3Src0 = VectorInstructions::nop;  // use existing
      vInst1.vOp3Src1 = VectorInstructions::nop;
      vInst1.vOp3 = VectorInstructions::nop;
      vInst1.vOp4 = VectorInstructions::nop;
      vInst1.vDest = VectorInstructions::nop;
      vectorInstructionConfig.inst[1] = vInst1;
      vectorInstructionConfig.instCount[1] = X * Y;

      // pull from accumulator and divide by X*Y
      VectorInstructions vInst2;
      vInst2.instType = VectorInstructions::vector;
      vInst2.vInput = VectorInstructions::readFromAccumulation;
      vInst2.vAccumulatePush = 0;
      vInst2.vOp0Src1 = VectorInstructions::nop;
      vInst2.vOp0 = VectorInstructions::nop;
      vInst2.vOp1 = VectorInstructions::nop;
      vInst2.vOp1 = VectorInstructions::nop;
      vInst2.vOp3Src0 = VectorInstructions::nop;  // use existing
      vInst2.vOp3Src1 = VectorInstructions::op3immediate0;
      vInst2.vOp3 = VectorInstructions::vmult;
      vInst2.vOp4 = VectorInstructions::nop;
      vInst2.vDest = VectorInstructions::vWriteOut;
      float fpscale = (1.0 / (X * Y));
      Posit<8, 1> scale = static_cast<Posit<8, 1> >(fpscale);
      vInst2.immediate0 = scale.bits;
      vectorInstructionConfig.inst[2] = vInst2;
      vectorInstructionConfig.instCount[2] = 1;

      vectorInstructionConfig.instLen = 3;
      vectorInstructionConfig.instLoopCount = K / DIMENSION;
    } else {
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
        Posit<8, 1> scale = static_cast<Posit<8, 1> >(fpscale);
        vInst0.immediate0 = scale.bits;
        vInst0.vOp0 = VectorInstructions::vmult;
      } else {
        vInst0.vOp0Src1 = VectorInstructions::nop;
        vInst0.vOp0 = VectorInstructions::nop;
      }

      vInst0.vOp1 = VectorInstructions::nop;
      vInst0.vOp2 = VectorInstructions::nop;
      vInst0.vOp3Src0 = VectorInstructions::nop;  // use existing

      if (params.BIAS) {
        vInst0.vOp3Src1 = VectorInstructions::readNormalInterface;
        vInst0.vOp3 = VectorInstructions::vadd;
      } else {
        vInst0.vOp3Src1 = VectorInstructions::nop;
        vInst0.vOp3 = VectorInstructions::nop;
      }

      if (params.RELU) {
        vInst0.vOp4 = VectorInstructions::vrelu;
      } else {
        vInst0.vOp4 = VectorInstructions::nop;
      }

      vInst0.vDest = VectorInstructions::vWriteOut;
      vectorInstructionConfig.inst[0] = vInst0;

      // total output count
      vectorInstructionConfig.instCount[0] =
          params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]] *
          params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]] *
          params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]];
      std::cout << "count: " << vectorInstructionConfig.instCount[0]
                << std::endl;
      vectorInstructionConfig.instLen = 1;
      vectorInstructionConfig.instLoopCount = 1;
    }
  }
}