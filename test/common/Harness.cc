#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "test/toolchain/MapOperation.h"

#ifdef SOC_COSIM
extern bool syscDone;
void register_interface(
    std::deque<sc_lv<Wrapped<int>::width> > *serialParamsIn,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width> > *inputAddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION> >::width> >
        *inputAddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width> > *weightAddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION> >::width> >
        *weightAddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width> >
        *vectorFetch0AddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION> >::width> >
        *vectorFetch0AddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width> >
        *vectorFetch1AddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION> >::width> >
        *vectorFetch1AddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width> >
        *vectorFetch2AddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION> >::width> >
        *vectorFetch2AddressResponse,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION> >::width> >
        *vectorOutput,
    std::deque<sc_lv<Wrapped<int>::width> > *vectorOutputAddress,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION> >::width> >
        *scalarUnitOutput,
    std::deque<sc_lv<Wrapped<int>::width> > *scalarOutputAddress);
void copy_output(void *sram, int size, int data_size);
#endif

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
  accelerator.serialMatrixParamsIn(serialMatrixParamsIn);
  accelerator.serialVectorParamsIn(serialVectorParamsIn);
  accelerator.inputAddressRequest(inputAddressRequest);
  accelerator.inputDataResponse(inputDataResponse);
  accelerator.weightAddressRequest(weightAddressRequest);
  accelerator.weightDataResponse(weightDataResponse);
  accelerator.gradAddressRequest(gradAddressRequest);
  accelerator.gradDataResponse(gradDataResponse);
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

  accelerator.matrixUnitStartSignal(matrixUnitStartSignal);
  accelerator.matrixUnitDoneSignal(matrixUnitDoneSignal);
  accelerator.vectorUnitStartSignal(vectorUnitStartSignal);
  accelerator.vectorUnitDoneSignal(vectorUnitDoneSignal);

#ifdef SOC_COSIM
  register_interface(
      serialParamsIn.getDataQueue(), inputAddressRequest.getDataQueue(),
      inputDataResponse.getDataQueue(), weightAddressRequest.getDataQueue(),
      weightDataResponse.getDataQueue(),
      vectorFetch0AddressRequest.getDataQueue(),
      vectorFetch0DataResponse.getDataQueue(),
      vectorFetch1AddressRequest.getDataQueue(),
      vectorFetch1DataResponse.getDataQueue(),
      vectorFetch2AddressRequest.getDataQueue(),
      vectorFetch2DataResponse.getDataQueue(), vectorOutput.getDataQueue(),
      vectorOutputAddress.getDataQueue(), scalarUnitOutput.getDataQueue(),
      scalarOutputAddress.getDataQueue());
#endif

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

  SC_THREAD(memAccessGrad);
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
}

void Harness::reset() {
  rstn.write(0);
  wait(5);
  rstn.write(1);
  wait();
}

void Harness::memAccessBurst(
    CombinationalInterface<MemoryRequest> *addressRequest,
    CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > *dataResponse,
    MemorySource memSource) {
  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    MemoryRequest memRequest = addressRequest->Pop();
    INPUT_DATATYPE *memory;
    if (currentParams.WEIGHT && memSource == RRAM) {
      memory = rramMemory;
    } else {
      memory = sramMemory;
    }

    for (int b = 0; b < memRequest.burstSize / DIMENSION; b++) {
      Pack1D<INPUT_DATATYPE, DIMENSION> data;
      for (int i = 0; i < DIMENSION; i++) {
        data[i] = memory[memRequest.address + b * DIMENSION + i];
      }
      DLOG(memSource << " access at addr: " << memRequest.address
                     << " data: " << data << std::endl);
      dataResponse->Push(data);
    }
  }
}

/* Special variation of memAccessBurst for FC layers
 */
void Harness::memAccessBurstVariable(
    CombinationalInterface<MemoryRequest> *addressRequest,
    CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > *dataResponse) {
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
    CombinationalInterface<int> *addressRequest,
    CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > *dataResponse,
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

void Harness::memAccess(CombinationalInterface<int> *addressRequest,
                        CombinationalInterface<INPUT_DATATYPE> *dataResponse,
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
  memAccessBurst(&inputAddressRequest, &inputDataResponse, SRAM);
}

void Harness::memAccessWeights() {
  memAccessBurst(&weightAddressRequest, &weightDataResponse, RRAM);
}

void Harness::memAccessGrad() {
  memAccessBurst(&gradAddressRequest, &gradDataResponse, SRAM);
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
                          CombinationalInterface<int> *serialParamsIn) {
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
  matrixUnitStartSignal.ResetRead();
  matrixUnitDoneSignal.ResetRead();
  vectorUnitStartSignal.ResetRead();
  vectorUnitDoneSignal.ResetRead();

  serialMatrixParamsIn.ResetWrite();
  serialVectorParamsIn.ResetWrite();

  wait();

  for (SimplifiedParams params : params_list) {
    currentParams = params;

    MatrixParams matrixParams;
    VectorParams vectorParams;
    VectorInstructionConfig vectorInstructionConfig;
    bool matrixParamsValid, vectorParamsValid;

    map_operation(params, matrixParams, matrixParamsValid, vectorParams,
                  vectorInstructionConfig, vectorParamsValid);

    if (matrixParamsValid) {
      sendSerializedParams<MatrixParams, 32>(matrixParams,
                                             &serialMatrixParamsIn);
      matrixUnitStartSignal.SyncPop();
    }
    if (vectorParamsValid) {
      sendSerializedParams<VectorParams, 32>(vectorParams,
                                             &serialVectorParamsIn);
      sendSerializedParams<VectorInstructionConfig, 32>(vectorInstructionConfig,
                                                        &serialVectorParamsIn);
      vectorUnitStartSignal.SyncPop();
    }
    CCS_LOG("Accelerator Layer Started.");

    if (matrixParamsValid) {
      matrixUnitDoneSignal.SyncPop();
    }
    if (vectorParamsValid) {
      vectorUnitDoneSignal.SyncPop();
    }
    CCS_LOG("Accelerator Layer Finshed.");

#ifdef SOC_COSIM
    copy_output(sramMemory, sizeof(INPUT_DATATYPE) * 2 * 1024 * 1024,
                sizeof(INPUT_DATATYPE));
    syscDone = true;
#endif
  }
  sc_stop();
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

void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE *sramMemory, INPUT_DATATYPE *rramMemory,
            MemoryMap memoryMap) {
  Harness harness("harness", params_list, sramMemory, rramMemory, memoryMap);
  sc_start();
}
