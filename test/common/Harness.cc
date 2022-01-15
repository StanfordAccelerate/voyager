#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

Harness::Harness(sc_module_name name, Params params, INPUT_DATATYPE *sram, INPUT_DATATYPE *rram, bool weightFromRRAM)
    : sc_module(name),
      clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
      params(params),
      sramMemory(sram),
      rramMemory(rram),
      weightFromRRAM(weightFromRRAM) {
  accelerator.clk(clk);
  accelerator.rstn(rstn);
  accelerator.serialParamsIn(serialParamsIn);

  accelerator.inputAddressRequest(inputAddressRequest);
  accelerator.inputDataResponse(inputDataResponse);
  accelerator.weightAddressRequest(weightAddressRequest);
  accelerator.weightDataResponse(weightDataResponse);
  accelerator.vectorFetchAddressRequest(vectorAddressRequest);
  accelerator.vectorFetchDataResponse(vectorDataResponse);
  accelerator.scalarAddressRequest(scalarAddressRequest);
  accelerator.scalarDataResponse(scalarDataResponse);
  accelerator.varianceAddressRequest(varianceAddressRequest);
  accelerator.varianceDataResponse(varianceDataResponse);
  accelerator.biasAddressRequest(biasAddressRequest);
  accelerator.biasDataResponse(biasDataResponse);
  accelerator.residualAddressRequest(residualAddressRequest);
  accelerator.residualDataResponse(residualDataResponse);
  accelerator.vectorUnitOutput(vectorOutput);
  accelerator.outputAddress(vectorOutputAddress);
  accelerator.startSignal(start);
  accelerator.doneSignal(done);

  SC_CTHREAD(reset, clk);

  SC_THREAD(memAccessInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessVector);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessScalar);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessVariance);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessResidual);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(storeOutputs);
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
        *dataResponse, bool useRRAM) {
  INPUT_DATATYPE* memory = useRRAM ? rramMemory : sramMemory;
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

void Harness::memAccessPack(
    Connections::Combinational<int> *addressRequest,
    Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
        *dataResponse, bool useRRAM) {
  INPUT_DATATYPE* memory = useRRAM ? rramMemory : sramMemory;
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
    Connections::Combinational<INPUT_DATATYPE> *dataResponse, bool useRRAM) {
  INPUT_DATATYPE* memory = useRRAM ? rramMemory : sramMemory;
  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    int address = addressRequest->Pop();
    dataResponse->Push(memory[address]);
  }
}

void Harness::memAccessInputs() {
  memAccessBurst(&inputAddressRequest, &inputDataResponse, USE_SRAM);
}

void Harness::memAccessWeights() {
  memAccessBurst(&weightAddressRequest, &weightDataResponse, weightFromRRAM);
}

void Harness::memAccessVector() {
  memAccessPack(&vectorAddressRequest, &vectorDataResponse, USE_SRAM);
}

void Harness::memAccessScalar() {
  memAccess(&scalarAddressRequest, &scalarDataResponse, USE_SRAM);
}

void Harness::memAccessVariance() {
  memAccess(&varianceAddressRequest, &varianceDataResponse, USE_SRAM);
}

void Harness::memAccessBias() {
  memAccessBurst(&biasAddressRequest, &biasDataResponse, USE_RRAM);
}

void Harness::memAccessResidual() {
  memAccessBurst(&residualAddressRequest, &residualDataResponse, USE_SRAM);
}

void Harness::sendParams() {
  serialParamsIn.ResetWrite();

  wait();

  serialParamsIn.Push(params.INPUT_OFFSET);
  serialParamsIn.Push(params.WEIGHT_OFFSET);
  serialParamsIn.Push(params.OUTPUT_OFFSET);
  serialParamsIn.Push(params.SOFTMAX);
  serialParamsIn.Push(params.SCALE);
  serialParamsIn.Push(params.TRANSPOSE);
  serialParamsIn.Push(params.VECTOR_OFFSET);
  serialParamsIn.Push(params.VEC_OP);
  serialParamsIn.Push(params.VEC_SUB);
  serialParamsIn.Push(params.VEC_SQUARE);
  serialParamsIn.Push(params.VEC_REDUCE);
  serialParamsIn.Push(params.CONST_SCALE);
  serialParamsIn.Push(params.VEC_SCALE_OFFSET);
  serialParamsIn.Push(params.VEC_SUB_OFFSET);
  serialParamsIn.Push(params.RELU);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      serialParamsIn.Push(params.loops[i][j]);
    }
  }
  for (int i = 0; i < 2; i++) {
    serialParamsIn.Push(params.inputXLoopIndex[i]);
  }
  for (int i = 0; i < 2; i++) {
    serialParamsIn.Push(params.inputYLoopIndex[i]);
  }
  for (int i = 0; i < 2; i++) {
    serialParamsIn.Push(params.reductionLoopIndex[i]);
  }
  for (int i = 0; i < 2; i++) {
    serialParamsIn.Push(params.weightLoopIndex[i]);
  }
  serialParamsIn.Push(params.fxIndex);
  serialParamsIn.Push(params.fyIndex);
  for (int i = 0; i < 2; i++) {
    serialParamsIn.Push(params.weightReuseIndex[i]);
  }
  serialParamsIn.Push(params.matMul);
  serialParamsIn.Push(params.STRIDE);
  serialParamsIn.Push(params.REPLICATION);
  serialParamsIn.Push(params.MAXPOOL);

  serialParamsIn.Push(params.BIAS);
  serialParamsIn.Push(params.BIAS_OFFSET);

  serialParamsIn.Push(params.RESIDUAL);
  serialParamsIn.Push(params.RESIDUAL_OFFSET);

  serialParamsIn.Push(params.AVGPOOL);

  wait();
}

void Harness::storeOutputs() {
  vectorOutput.ResetRead();
  vectorOutputAddress.ResetRead();

  wait();

  while (true) {
    Pack1D<OUTPUT_DATATYPE, DIMENSION> data = vectorOutput.Pop();
    int address = vectorOutputAddress.Pop();
    for (int i = 0; i < DIMENSION; i++) {
      sramMemory[address + i] = data[i];
    }
  }
}

void Harness::waitForStart() {
  start.ResetRead();

  wait();

  start.SyncPop();

  CCS_LOG("Accelerator Started.");
}

void Harness::waitForDone() {
  done.ResetRead();
  wait();

  done.SyncPop();

  CCS_LOG("Accelerator Finished.");
  sc_stop();
}

void run_op(const Params params, INPUT_DATATYPE *sramMemory, INPUT_DATATYPE *rramMemory, bool weightFromRRAM) {
  Harness harness("harness", params, sramMemory, rramMemory, weightFromRRAM);
  sc_start();
}
