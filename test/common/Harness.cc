#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

Harness::Harness(sc_module_name name, std::vector<SimplifiedParams> params_list,
                 INPUT_DATATYPE *sram, INPUT_DATATYPE *rram,
                 MemoryMap memoryMap)
    : sc_module(name),
      clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
      params_list(params_list),
      sramMemory(sram),
      rramMemory(rram),
      memoryMap(memoryMap) {
  accelerator.clk(clk);
  accelerator.rstn(rstn);
  accelerator.serialParamsIn(serialParamsIn);

  accelerator.inputAddressRequest(inputAddressRequest);
  accelerator.inputDataResponse(inputDataResponse);
  accelerator.weightAddressRequest(weightAddressRequest);
  accelerator.weightDataResponse(weightDataResponse);
  accelerator.vectorFetch0AddressRequest(vectorFetch0AddressRequest);
  accelerator.vectorFetch0DataResponse(vectorFetch0DataResponse);
  accelerator.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
  accelerator.vectorFetch1DataResponse(vectorFetch1DataResponse);
  accelerator.vectorFetch2AddressRequest(vectorFetch2AddressRequest);
  accelerator.vectorFetch2DataResponse(vectorFetch2DataResponse);
  accelerator.vectorOutput(vectorOutput);
  accelerator.vectorOutputAddress(vectorOutputAddress);
  accelerator.scalarUnitOutput(scalarUnitOutput);
  accelerator.scalarOutputAddress(scalarOutputAddress);
  accelerator.startSignal(start);
  accelerator.doneSignal(done);

  SC_CTHREAD(reset, clk);

  SC_THREAD(memAccessInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessVector0);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessVector1);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessVector2);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(storeVectorOutputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(storeScalarOutputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendParams);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(waitForStart);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(waitForDone);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
}

void Harness::reset() {
  rstn.write(0);
  wait(5);
  rstn.write(1);
  wait();
}

void Harness::memAccessBurst(
    Connections::Combinational<MemoryRequest> *addressRequest,
    Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
        *dataResponse,
    MemorySource memSource) {
  INPUT_DATATYPE *memory = memSource == SRAM ? sramMemory : rramMemory;

  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    MemoryRequest memRequest = addressRequest->Pop();

    for (int b = 0; b < memRequest.burstSize / DIMENSION; b++) {
      Pack1D<INPUT_DATATYPE, DIMENSION> data;
      for (int i = 0; i < DIMENSION; i++) {
        data[i] = memory[memRequest.address + b * DIMENSION + i];
      }
      dataResponse->Push(data);
    }
  }
}

/* Special variation of memAccessBurst for FC layers
 */
void Harness::memAccessBurstVariable(
    Connections::Combinational<MemoryRequest> *addressRequest,
    Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
        *dataResponse) {
  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    MemoryRequest memRequest = addressRequest->Pop();
    INPUT_DATATYPE *memory;
    if (currentParams.FC || currentParams.NO_NORM) {
      memory = rramMemory;
    } else {
      memory = sramMemory;
    }

    for (int b = 0; b < memRequest.burstSize / DIMENSION; b++) {
      Pack1D<INPUT_DATATYPE, DIMENSION> data;
      for (int i = 0; i < DIMENSION; i++) {
        data[i] = memory[memRequest.address + b * DIMENSION + i];
      }
      dataResponse->Push(data);
    }
  }
}

void Harness::memAccessPack(
    Connections::Combinational<int> *addressRequest,
    Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
        *dataResponse,
    MemorySource memSource) {
  INPUT_DATATYPE *memory = memSource == SRAM ? sramMemory : rramMemory;

  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    Pack1D<INPUT_DATATYPE, DIMENSION> data;
    int count = 0;
    while (count < DIMENSION) {
      int address = addressRequest->Pop();
      data[count] = memory[address];
      count++;
    }
    dataResponse->Push(data);
  }
}

void Harness::memAccess(
    Connections::Combinational<int> *addressRequest,
    Connections::Combinational<INPUT_DATATYPE> *dataResponse,
    MemorySource memSource) {
  INPUT_DATATYPE *memory = memSource == SRAM ? sramMemory : rramMemory;

  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    int address = addressRequest->Pop();
    dataResponse->Push(memory[address]);
  }
}

void Harness::memAccessInputs() {
  memAccessBurst(&inputAddressRequest, &inputDataResponse, memoryMap.inputs);
}

void Harness::memAccessWeights() {
  memAccessBurst(&weightAddressRequest, &weightDataResponse, memoryMap.weights);
}

void Harness::memAccessVector0() {
  memAccessBurst(&vectorFetch0AddressRequest, &vectorFetch0DataResponse,
                 memoryMap.inputs);
}

void Harness::memAccessVector1() {
  memAccessBurstVariable(&vectorFetch1AddressRequest,
                         &vectorFetch1DataResponse);
}

void Harness::memAccessVector2() {
  memAccessBurst(&vectorFetch2AddressRequest, &vectorFetch2DataResponse,
                 memoryMap.bias);
}

template <typename T, unsigned int interfaceWidth>
void sendSerializedParams(T params,
                          Connections::Combinational<int> *serialParamsIn) {
  ac_int<T::width, false> serializedParam;
  vector_to_type(TypeToBits<T>(params), false, &serializedParam);

  // round up to the nearest multiple of interfaceWidth
  ac_int<((T::width + interfaceWidth - 1) / interfaceWidth) * interfaceWidth,
         false>
      serializedParamsPadded = serializedParam;

  for (int i = 0; i < serializedParamsPadded.width / interfaceWidth; i++) {
    serialParamsIn->Push(serializedParamsPadded.template slc<interfaceWidth>(
        i * interfaceWidth));
  }
}

void Harness::sendParams() {
  done.ResetRead();
  serialParamsIn.ResetWrite();

  wait();

  for (SimplifiedParams params : params_list) {
    currentParams = params;

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

    // create Matrix and Vector Params from SimplifiedParams
    if (params.SOFTMAX) {
      // TODO
    } else if (params.FC) {
      // no matrix parameters
      serialParamsIn.Push(0);

      serialParamsIn.Push(1);
      VectorParams vectorParams;
      vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
      vectorParams.addressGen0Enable = true;
      vectorParams.addressGen0Loop[0] = K;
      vectorParams.addressGen0Loop[1] = 1;
      vectorParams.addressGen0Loop[2] = C / DIMENSION;

      // address gen 1 (weights)
      vectorParams.ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
      vectorParams.addressGen1Mode = 2;  // 2d tensor
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

      sendSerializedParams<VectorParams, 32>(vectorParams, &serialParamsIn);

      // create instruction stream
      VectorInstructionConfig vectorInstructionConfig;

      // inst0- start reduction engine
      vectorInstructionConfig.inst[0].instType = VectorInstructions::reduction;
      vectorInstructionConfig.inst[0].rCount = C / DIMENSION;
      vectorInstructionConfig.inst[0].rOp = VectorInstructions::radd;
      vectorInstructionConfig.inst[0].rDuplicate = 0;
      vectorInstructionConfig.inst[0].rDest = VectorInstructions::toVectorSrc0;
      vectorInstructionConfig.instCount[0] = 1;

      // inst 1- inputs x weights, send to reduce
      vectorInstructionConfig.inst[1].instType = VectorInstructions::vector;
      vectorInstructionConfig.inst[1].vInput =
          VectorInstructions::readFromVectorFetch;
      vectorInstructionConfig.inst[1].vAccumulatePush = VectorInstructions::nop;
      vectorInstructionConfig.inst[1].vOp0Src1 =
          VectorInstructions::readInterface;
      vectorInstructionConfig.inst[1].vOp0 = VectorInstructions::vmult;
      vectorInstructionConfig.inst[1].vOp1 = VectorInstructions::nop;
      vectorInstructionConfig.inst[1].vOp2 = VectorInstructions::toReduce;
      vectorInstructionConfig.inst[1].vOp3Src0 = VectorInstructions::nop;
      vectorInstructionConfig.inst[1].vOp3Src1 = VectorInstructions::nop;
      vectorInstructionConfig.inst[1].vOp3 = VectorInstructions::nop;
      vectorInstructionConfig.inst[1].vOp4 = VectorInstructions::nop;
      vectorInstructionConfig.inst[1].vDest = VectorInstructions::nop;
      // C/DIMENSION to do the complete reduction
      // DIMENSION to fill up the entire vector
      vectorInstructionConfig.instCount[1] = DIMENSION * C / DIMENSION;

      // inst2- add bias, write out
      vectorInstructionConfig.inst[2].instType = VectorInstructions::vector;
      vectorInstructionConfig.inst[2].vInput = VectorInstructions::nop;
      vectorInstructionConfig.inst[2].vAccumulatePush = VectorInstructions::nop;
      vectorInstructionConfig.inst[2].vOp0Src1 = VectorInstructions::nop;
      vectorInstructionConfig.inst[2].vOp0 = VectorInstructions::vmult;
      vectorInstructionConfig.inst[2].vOp1 = VectorInstructions::nop;
      vectorInstructionConfig.inst[2].vOp2 = VectorInstructions::nop;
      vectorInstructionConfig.inst[2].vOp3Src0 =
          VectorInstructions::readReduceInterface;
      vectorInstructionConfig.inst[2].vOp3Src1 =
          VectorInstructions::nop;  // TODO: change to add for bias
      vectorInstructionConfig.inst[2].vOp3 = VectorInstructions::nop;
      vectorInstructionConfig.inst[2].vOp4 = params.RELU;
      vectorInstructionConfig.inst[2].vDest = VectorInstructions::vWriteOut;
      vectorInstructionConfig.instCount[2] = 1;

      vectorInstructionConfig.instLen = 3;
      vectorInstructionConfig.instLoopCount = K / DIMENSION;

      sendSerializedParams<VectorInstructionConfig, 32>(vectorInstructionConfig,
                                                        &serialParamsIn);
      serialParamsIn.Push(0);

      done.SyncPop();
      CCS_LOG("Accelerator Layer Finished.");
    } else if (params.NO_NORM) {
      // no matrix parameters
      serialParamsIn.Push(0);

      serialParamsIn.Push(1);
      VectorParams vectorParams;
      vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
      vectorParams.addressGen0Enable = true;
      vectorParams.addressGen0Loop[0] = 1;
      vectorParams.addressGen0Loop[1] = Y;
      vectorParams.addressGen0Loop[2] = C / DIMENSION;

      // address gen 1 (weights)
      vectorParams.ADDRESS_GEN1_OFFSET = params.WEIGHT_OFFSET;
      vectorParams.addressGen1Mode = 2;  // 2d tensor
      vectorParams.addressGen1Loops[0][0] = Y;
      vectorParams.addressGen1Loops[0][1] = 1;
      vectorParams.addressGen1Loops[0][2] = C / DIMENSION;

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

      sendSerializedParams<VectorParams, 32>(vectorParams, &serialParamsIn);

      // create instruction stream
      VectorInstructionConfig vectorInstructionConfig;

      // inst 1- inputs x weights, send to reduce
      vectorInstructionConfig.inst[0].instType = VectorInstructions::vector;
      vectorInstructionConfig.inst[0].vInput =
          VectorInstructions::readFromVectorFetch;
      vectorInstructionConfig.inst[0].vAccumulatePush = VectorInstructions::nop;
      vectorInstructionConfig.inst[0].vOp0Src1 =
          VectorInstructions::readInterface;
      vectorInstructionConfig.inst[0].vOp0 = VectorInstructions::vmult;
      vectorInstructionConfig.inst[0].vOp1 = VectorInstructions::nop;
      vectorInstructionConfig.inst[0].vOp2 = VectorInstructions::nop;
      vectorInstructionConfig.inst[0].vOp3Src0 = VectorInstructions::nop;
      vectorInstructionConfig.inst[0].vOp3Src1 =
          VectorInstructions::readNormalInterface;
      vectorInstructionConfig.inst[0].vOp3 = VectorInstructions::vadd;
      vectorInstructionConfig.inst[0].vOp4 = params.RELU;
      vectorInstructionConfig.inst[0].vDest = VectorInstructions::vWriteOut;
      // C/DIMENSION to do the complete reduction
      // DIMENSION to fill up the entire vector
      vectorInstructionConfig.instCount[0] = Y * C / DIMENSION;

      vectorInstructionConfig.instLen = 1;
      vectorInstructionConfig.instLoopCount = 1;

      sendSerializedParams<VectorInstructionConfig, 32>(vectorInstructionConfig,
                                                        &serialParamsIn);
      serialParamsIn.Push(0);

      done.SyncPop();
      CCS_LOG("Accelerator Layer Finished.");
    } else {
      // matrix params
      serialParamsIn.Push(1);

      MatrixParams matrixParams;
      matrixParams.INPUT_OFFSET = params.INPUT_OFFSET;
      matrixParams.WEIGHT_OFFSET = params.WEIGHT_OFFSET;
      matrixParams.OUTPUT_OFFSET = params.OUTPUT_OFFSET;
      matrixParams.SOFTMAX = params.SOFTMAX;
      matrixParams.SCALE = 0;  // unused
      matrixParams.TRANSPOSE = params.TRANSPOSE;
      matrixParams.VECTOR_OFFSET = 0;     // unused
      matrixParams.VEC_OP = 0;            // unused
      matrixParams.VEC_SUB = 0;           // unused
      matrixParams.VEC_SQUARE = 0;        // unused
      matrixParams.VEC_REDUCE = 0;        // unused
      matrixParams.CONST_SCALE = 0;       // unused
      matrixParams.VEC_SCALE_OFFSET = 0;  // unused
      matrixParams.VEC_SUB_OFFSET = 0;    // unused
      matrixParams.RELU = params.RELU;
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
      matrixParams.matMul = false;  // unused
      matrixParams.STRIDE = params.STRIDE;
      matrixParams.REPLICATION = params.REPLICATION;
      matrixParams.MAXPOOL = params.MAXPOOL;
      matrixParams.BIAS = params.BIAS;
      matrixParams.BIAS_OFFSET = params.BIAS_OFFSET;
      matrixParams.RESIDUAL = params.RESIDUAL;
      matrixParams.RESIDUAL_OFFSET = params.RESIDUAL_OFFSET;
      matrixParams.AVGPOOL = params.AVGPOOL;
      sendSerializedParams<MatrixParams, 32>(matrixParams, &serialParamsIn);

      serialParamsIn.Push(0);

      serialParamsIn.Push(1);
      VectorParams vectorParams;

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

      sendSerializedParams<VectorParams, 32>(vectorParams, &serialParamsIn);

      // create instruction stream
      VectorInstructionConfig vectorInstructionConfig;

      if (params.AVGPOOL) {
        vectorInstructionConfig.inst[0].instType =
            VectorInstructions::accumulation;
        vectorInstructionConfig.inst[0].rCount = X * Y;
        vectorInstructionConfig.instCount[0] = 1;

        vectorInstructionConfig.inst[1].instType = VectorInstructions::vector;
        vectorInstructionConfig.inst[1].vInput =
            VectorInstructions::readFromSystolicArray;
        vectorInstructionConfig.inst[1].vAccumulatePush = 1;

        vectorInstructionConfig.inst[1].vOp0Src1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[1].vOp0 = VectorInstructions::nop;

        vectorInstructionConfig.inst[1].vOp1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[1].vOp1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[1].vOp3Src0 =
            VectorInstructions::nop;  // use existing
        vectorInstructionConfig.inst[1].vOp3Src1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[1].vOp3 = VectorInstructions::nop;

        vectorInstructionConfig.inst[1].vOp4 = VectorInstructions::nop;
        vectorInstructionConfig.inst[1].vDest = VectorInstructions::nop;
        vectorInstructionConfig.instCount[1] = X * Y;

        vectorInstructionConfig.inst[2].instType = VectorInstructions::vector;
        vectorInstructionConfig.inst[2].vInput =
            VectorInstructions::readFromAccumulation;
        vectorInstructionConfig.inst[2].vAccumulatePush = 0;

        vectorInstructionConfig.inst[2].vOp0Src1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[2].vOp0 = VectorInstructions::nop;

        vectorInstructionConfig.inst[2].vOp1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[2].vOp1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[2].vOp3Src0 =
            VectorInstructions::nop;  // use existing
        vectorInstructionConfig.inst[2].vOp3Src1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[2].vOp3 = VectorInstructions::nop;

        vectorInstructionConfig.inst[2].vOp4 = VectorInstructions::nop;
        vectorInstructionConfig.inst[2].vDest = VectorInstructions::vWriteOut;
        vectorInstructionConfig.instCount[2] = 1;

        vectorInstructionConfig.instLen = 3;
        vectorInstructionConfig.instLoopCount = K / DIMENSION;
      } else {
        vectorInstructionConfig.inst[0].instType = VectorInstructions::vector;
        vectorInstructionConfig.inst[0].vInput =
            VectorInstructions::readFromSystolicArray;

        vectorInstructionConfig.inst[0].vAccumulatePush = 0;

        if (params.RESIDUAL) {
          vectorInstructionConfig.inst[0].vOp0Src1 =
              VectorInstructions::readInterface;
          vectorInstructionConfig.inst[0].vOp0 = VectorInstructions::vadd;
        } else {
          vectorInstructionConfig.inst[0].vOp0Src1 = VectorInstructions::nop;
          vectorInstructionConfig.inst[0].vOp0 = VectorInstructions::nop;
        }

        vectorInstructionConfig.inst[0].vOp1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[0].vOp1 = VectorInstructions::nop;
        vectorInstructionConfig.inst[0].vOp3Src0 =
            VectorInstructions::nop;  // use existing

        if (params.BIAS) {
          vectorInstructionConfig.inst[0].vOp3Src1 =
              VectorInstructions::readNormalInterface;
          vectorInstructionConfig.inst[0].vOp3 = VectorInstructions::vadd;
        } else {
          vectorInstructionConfig.inst[0].vOp3Src1 = VectorInstructions::nop;
          vectorInstructionConfig.inst[0].vOp3 = VectorInstructions::nop;
        }

        if (params.RELU) {
          vectorInstructionConfig.inst[0].vOp4 = VectorInstructions::vrelu;
        } else {
          vectorInstructionConfig.inst[0].vOp4 = VectorInstructions::nop;
        }

        vectorInstructionConfig.inst[0].vDest = VectorInstructions::vWriteOut;

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
      sendSerializedParams<VectorInstructionConfig, 32>(vectorInstructionConfig,
                                                        &serialParamsIn);

      serialParamsIn.Push(0);

      done.SyncPop();
      CCS_LOG("Accelerator Layer Finished.");
    }

    sc_stop();
  }
}

void Harness::storeVectorOutputs() {
  vectorOutput.ResetRead();
  vectorOutputAddress.ResetRead();

  wait();

  while (true) {
    Pack1D<OUTPUT_DATATYPE, DIMENSION> data = vectorOutput.Pop();
    int address = vectorOutputAddress.Pop();
    DLOG("address: " << address << " data: " << data);
    for (int i = 0; i < DIMENSION; i++) {
      sramMemory[address + i] = data[i];
    }
  }
}

void Harness::storeScalarOutputs() {
  scalarUnitOutput.ResetRead();
  scalarOutputAddress.ResetRead();

  wait();

  while (true) {
    Pack1D<OUTPUT_DATATYPE, DIMENSION> data = scalarUnitOutput.Pop();
    int address = scalarOutputAddress.Pop();
  }
}

void Harness::waitForStart() {
  start.ResetRead();

  wait();

  int i = 0;
  for (SimplifiedParams params : params_list) {
    start.SyncPop();
    CCS_LOG("Accelerator Layer " + std::to_string(i) + " Started.");
    i++;
  }
}
void Harness::waitForDone() {
  // done.ResetRead();
  // wait();

  // for (SimplifiedParams params : params_list) {

  // }

  // CCS_LOG("Accelerator Finished.");
  // sc_stop();
}

void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE *sramMemory, INPUT_DATATYPE *rramMemory,
            MemoryMap memoryMap) {
  Harness harness("harness", params_list, sramMemory, rramMemory, memoryMap);
  sc_start();
}
