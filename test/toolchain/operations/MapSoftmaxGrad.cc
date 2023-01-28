#include "test/toolchain/operations/Operations.h"

/*
 * softmax_gradient
 * for x:
 *  sum = 0
 *  for y:
 *    tmp0[x,y] = activation[x,y] * attention_prob[x,y]
 *    sum += activation[x,y] * attention_prob[x,y]
 *  for y:
 *    tmp1[x,y] = sum * attention_prob[x,y]
 *
 * for x:
 *   for y:
 *     output[x,y] = tmp0[x,y] - tmp1[x,y]
 *
 */
void MapSoftmaxGrad(const SimplifiedParams &params,
                    std::deque<BaseParams *> &mappedParams) {
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

  // softmax gradient is performed with two operations:
  // 1. calculate tmp0 and tmp1
  // 2. tmp0 - tmp1

  VectorParams *vectorParams_0 = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig_0 =
      new VectorInstructionConfig;

  vectorParams_0->VECTOR_OFFSET = params.INPUT_OFFSET;
  vectorParams_0->addressGen0Enable = true;
  for (int i = 0; i < 3; i++) {
    vectorParams_0->addressGen0Loop[0][i] = 1;
  }
  vectorParams_0->addressGen0Loop[1][0] = 1;
  vectorParams_0->addressGen0Loop[1][1] = X;
  vectorParams_0->addressGen0Loop[1][2] = Y / DIMENSION;
  vectorParams_0->addressGen0Broadcast = false;
  vectorParams_0->SOFTMAX_GRAD_NEGATE = false;
  vectorParams_0->DP_VEC0 = params.ACC_T_INPUT;

  vectorParams_0->ADDRESS_GEN1_OFFSET = params.RESIDUAL_OFFSET;
  vectorParams_0->addressGen1Mode = 2;  // 2d tensor
  vectorParams_0->addressGen1Loops[0][0] = 1;
  vectorParams_0->addressGen1Loops[0][1] = X;
  vectorParams_0->addressGen1Loops[0][2] = 1;
  vectorParams_0->addressGen1Loops[1][0] = 2;  // requires two passes
  vectorParams_0->addressGen1Loops[1][1] = 1;
  vectorParams_0->addressGen1Loops[1][2] = Y / DIMENSION;
  vectorParams_0->DP_VEC1 = params.ACC_T_OUTPUT;

  vectorParams_0->addressGen2Mode = 0;  // disable address gen 2

  vectorParams_0->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams_0->SCALAR_OUTPUT_OFFSET =
      params.OUTPUT_OFFSET + X * Y / DIMENSION;

  vectorParams_0->scalarOutputCount = 0;
  vectorParams_0->MAXPOOL = false;
  vectorParams_0->AVGPOOL = false;

  // output
  // we are producting two output tensors (tmp0, tmp1)
  // in order to get them placed in memory consecutively,
  // we need to mess with the output loop bounds
  // we can basically treat it as if we have a single tensor
  // of dimension 2 * X * Y
  vectorParams_0->outputLoops[0][0] = 1;
  vectorParams_0->outputYLoopIndex[0] = 0;

  vectorParams_0->outputLoops[0][1] = X;
  vectorParams_0->outputXLoopIndex[0] = 1;

  vectorParams_0->outputLoops[0][2] = 1;
  vectorParams_0->outputWeightLoopIndex[0] = 2;

  vectorParams_0->outputLoops[1][0] = 2;
  vectorParams_0->outputYLoopIndex[1] = 0;

  vectorParams_0->outputLoops[1][1] = 1;
  vectorParams_0->outputXLoopIndex[1] = 1;

  vectorParams_0->outputLoops[1][2] = Y / DIMENSION;
  vectorParams_0->outputWeightLoopIndex[1] = 2;
  vectorParams_0->DP_OUTPUT = true;  // intermediate needs to be in dp output
  vectorParams_0->SPLIT_OUTPUT = false;

  // inst0- start reduction engine
  VectorInstructions vInst0;
  vInst0.instType = VectorInstructions::reduction;
  vInst0.rCount = Y / DIMENSION;
  vInst0.rOp = VectorInstructions::radd;
  vInst0.rDuplicate = 1;
  vInst0.rDest = VectorInstructions::toVectorOp0Src0;
  vInst0.rBroadcast = true;
  vInst0.immediate0 = Y / DIMENSION;
  vInst0.immediate1 = 0;
  vectorInstructionConfig_0->inst[0] = vInst0;
  vectorInstructionConfig_0->instCount[0] = 1;

  // inst 1- multiply activation * attention_prob
  // and send to reduction unit as well as write out
  VectorInstructions vInst1;
  vInst1.instType = VectorInstructions::vector;
  vInst1.vInput = VectorInstructions::readFromVectorFetch;
  vInst1.vOp0Src1 = VectorInstructions::readInterface;
  vInst1.vOp0 = VectorInstructions::vmult;
  vInst1.vOp1 = VectorInstructions::nop;
  vInst1.vOp2 = VectorInstructions::toReduce;
  vInst1.vOp3 = VectorInstructions::nop;
  vInst1.vOp3Src1 = VectorInstructions::nop;
  vInst1.vOp4 = VectorInstructions::nop;
  vInst1.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig_0->inst[1] = vInst1;
  vectorInstructionConfig_0->instCount[1] = Y / DIMENSION;

  // inst 2- pull from reduction and multiply with attention_prob
  VectorInstructions vInst2;
  vInst2.instType = VectorInstructions::vector;
  vInst2.vInput = VectorInstructions::readFromReduce;
  vInst2.vOp0Src1 = VectorInstructions::readInterface;
  vInst2.vOp0 = VectorInstructions::vmult;
  vInst2.vOp1 = VectorInstructions::nop;
  vInst2.vOp2 = VectorInstructions::nop;
  vInst2.vOp3 = VectorInstructions::nop;
  vInst2.vOp3Src1 = VectorInstructions::nop;
  vInst2.vOp4 = VectorInstructions::nop;
  vInst2.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig_0->inst[2] = vInst2;
  vectorInstructionConfig_0->instCount[2] = Y / DIMENSION;

  vectorInstructionConfig_0->instLen = 3;
  vectorInstructionConfig_0->instLoopCount = X;

  mappedParams.push_back(vectorParams_0);
  mappedParams.push_back(vectorInstructionConfig_0);

  //================================================
  VectorParams *vectorParams_1 = new VectorParams;
  VectorInstructionConfig *vectorInstructionConfig_1 =
      new VectorInstructionConfig;

  vectorParams_1->VECTOR_OFFSET = params.OUTPUT_OFFSET;
  vectorParams_1->addressGen0Enable = true;
  for (int i = 0; i < 3; i++) {
    vectorParams_1->addressGen0Loop[0][i] = 1;
  }
  vectorParams_1->addressGen0Loop[1][0] = 1;
  vectorParams_1->addressGen0Loop[1][1] = X;
  vectorParams_1->addressGen0Loop[1][2] = Y / DIMENSION;
  vectorParams_1->addressGen0Broadcast = false;
  vectorParams_1->SOFTMAX_GRAD_NEGATE = false;
  vectorParams_1->DP_VEC0 = true;

  vectorParams_1->ADDRESS_GEN1_OFFSET = params.OUTPUT_OFFSET + 2 * (X * Y);
  std::cout << "Second op tmp1: " << vectorParams_1->ADDRESS_GEN1_OFFSET
            << std::endl;
  vectorParams_1->addressGen1Mode = 2;  // 2d tensor
  vectorParams_1->addressGen1Loops[0][0] = 1;
  vectorParams_1->addressGen1Loops[0][1] = 1;
  vectorParams_1->addressGen1Loops[0][2] = 1;
  vectorParams_1->addressGen1Loops[1][0] = 1;
  vectorParams_1->addressGen1Loops[1][1] = X;
  vectorParams_1->addressGen1Loops[1][2] = Y / DIMENSION;
  vectorParams_1->DP_VEC1 = true;

  vectorParams_1->addressGen2Mode = 0;  // disable address gen 2

  vectorParams_1->VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  vectorParams_1->SCALAR_OUTPUT_OFFSET =
      params.OUTPUT_OFFSET + X * Y / DIMENSION;

  vectorParams_1->scalarOutputCount = 0;
  vectorParams_1->MAXPOOL = false;
  vectorParams_1->AVGPOOL = false;

  // output
  vectorParams_1->outputLoops[0][0] = 1;
  vectorParams_1->outputYLoopIndex[0] = 0;

  vectorParams_1->outputLoops[0][1] = 1;
  vectorParams_1->outputXLoopIndex[0] = 1;

  vectorParams_1->outputLoops[0][2] = 1;
  vectorParams_1->outputWeightLoopIndex[0] = 2;

  vectorParams_1->outputLoops[1][0] = 1;
  vectorParams_1->outputYLoopIndex[1] = 0;

  vectorParams_1->outputLoops[1][1] = X;
  vectorParams_1->outputXLoopIndex[1] = 1;

  vectorParams_1->outputLoops[1][2] = Y / DIMENSION;
  vectorParams_1->outputWeightLoopIndex[1] = 2;
  vectorParams_1->DP_OUTPUT = params.ACC_T_OUTPUT;
  vectorParams_1->SPLIT_OUTPUT = false;

  // inst- subtract tmp0 - tmp1
  VectorInstructions op2SubInst;
  op2SubInst.instType = VectorInstructions::vector;
  op2SubInst.vInput = VectorInstructions::readFromVectorFetch;
  op2SubInst.vOp0Src1 = VectorInstructions::readInterface;
  op2SubInst.vOp0 = VectorInstructions::vsub;
  op2SubInst.vOp1 = VectorInstructions::nop;
  op2SubInst.vOp2 = VectorInstructions::nop;
  op2SubInst.vOp3 = VectorInstructions::nop;
  op2SubInst.vOp3Src1 = VectorInstructions::nop;
  op2SubInst.vOp4 = VectorInstructions::nop;
  op2SubInst.vDest = VectorInstructions::vWriteOut;
  vectorInstructionConfig_1->inst[0] = op2SubInst;
  vectorInstructionConfig_1->instCount[0] = X * Y / DIMENSION;

  vectorInstructionConfig_1->instLen = 1;
  vectorInstructionConfig_1->instLoopCount = 1;

  mappedParams.push_back(vectorParams_1);
  mappedParams.push_back(vectorInstructionConfig_1);
}
