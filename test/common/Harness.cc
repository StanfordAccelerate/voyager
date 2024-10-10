#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include <cassert>

#include "AccelTypes.h"
#include "sysc/kernel/sc_time.h"
#include "test/toolchain/MapOperation.h"

#ifdef SOC_COSIM
extern bool syscDone;
void init_checkers();
void register_interface(
    std::deque<sc_lv<Wrapped<int>::width>> *serialMatrixParamsIn,
    std::deque<sc_lv<Wrapped<int>::width>> *serialVectorParamsIn,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width>> *inputAddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION>>::width>>
        *inputAddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width>> *weightAddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION>>::width>>
        *weightAddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width>>
        *vectorFetch0AddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION>>::width>>
        *vectorFetch0AddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width>>
        *vectorFetch1AddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION>>::width>>
        *vectorFetch1AddressResponse,
    std::deque<sc_lv<Wrapped<MemoryRequest>::width>>
        *vectorFetch2AddressRequest,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION>>::width>>
        *vectorFetch2AddressResponse,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION>>::width>>
        *vectorOutput,
    std::deque<sc_lv<Wrapped<int>::width>> *vectorOutputAddress);
// void copy_output(void *sram, int size, int data_size);
#endif

Harness::Harness(sc_module_name name,
                 std::vector<codegen::AcceleratorParam> params, char *sram,
                 char *rram)
    : sc_module(name),
      clk("clk", std::stod(std::getenv("CLOCK_PERIOD")), SC_NS, 0.5, 0, SC_NS,
          true),
      params(params),
      sramMemory(sram),
      rramMemory(rram),
      inputDataResponse_fifo("inputDataResponse_fifo", 1024),
      weightDataResponse_fifo("weightDataResponse_fifo", 1024),
      biasDataResponse_fifo("biasDataResponse_fifo", 1024),
      vectorFetch0DataResponse_fifo("vectorFetch0DataResponse_fifo", 1024),
      vectorFetch1DataResponse_fifo("vectorFetch1DataResponse_fifo", 1024),
      vectorFetch2DataResponse_fifo("vectorFetch2DataResponse_fifo", 1024) {
  accelerator.clk(clk);
  accelerator.rstn(rstn);
  accelerator.serialMatrixParamsIn(serialMatrixParamsIn);
  accelerator.serialVectorParamsIn(serialVectorParamsIn);
  accelerator.inputAddressRequest(inputAddressRequest);
  accelerator.inputDataResponse(inputDataResponse);
  accelerator.weightAddressRequest(weightAddressRequest);
  accelerator.weightDataResponse(weightDataResponse);
  accelerator.biasAddressRequest(biasAddressRequest);
  accelerator.biasDataResponse(biasDataResponse);
  accelerator.vectorFetch0AddressRequest(vectorFetch0AddressRequest);
  accelerator.vectorFetch0DataResponse(vectorFetch0DataResponse);
  accelerator.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
  accelerator.vectorFetch1DataResponse(vectorFetch1DataResponse);
  accelerator.vectorFetch2AddressRequest(vectorFetch2AddressRequest);
  accelerator.vectorFetch2DataResponse(vectorFetch2DataResponse);
  accelerator.vectorOutput(vectorOutput);
  accelerator.vectorOutputAddress(vectorOutputAddress);

  accelerator.matrixUnitStartSignal(matrixUnitStartSignal);
  accelerator.matrixUnitDoneSignal(matrixUnitDoneSignal);
  accelerator.vectorUnitStartSignal(vectorUnitStartSignal);
  accelerator.vectorUnitDoneSignal(vectorUnitDoneSignal);

#ifdef SOC_COSIM
  init_checkers();
  register_interface(
      serialMatrixParamsIn.getDataQueue(), serialVectorParamsIn.getDataQueue(),
      inputAddressRequest.getDataQueue(), inputDataResponse.getDataQueue(),
      weightAddressRequest.getDataQueue(), weightDataResponse.getDataQueue(),
      vectorFetch0AddressRequest.getDataQueue(),
      vectorFetch0DataResponse.getDataQueue(),
      vectorFetch1AddressRequest.getDataQueue(),
      vectorFetch1DataResponse.getDataQueue(),
      vectorFetch2AddressRequest.getDataQueue(),
      vectorFetch2DataResponse.getDataQueue(), vectorOutput.getDataQueue(),
      vectorOutputAddress.getDataQueue());
#endif

  SC_CTHREAD(reset, clk);

  SC_THREAD(readRequestInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestVector0);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector0);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestVector1);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector1);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestVector2);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector2);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(storeVectorOutputs);
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

template <long unsigned int DIMENSION>
void Harness::readMemoryRequest(
    CombinationalInterface<MemoryRequest> *addressRequest,
    sc_fifo<Pack1D<INPUT_DATATYPE, DIMENSION>> *dataResponse_fifo,
    std::string memSourceType) {
  addressRequest->ResetRead();

  wait();

  while (true) {
    MemoryRequest memRequest = addressRequest->Pop();
    MemorySource memSource;

    auto it = currentMemoryMap.find(memSourceType);
    if (it != currentMemoryMap.end()) {
      memSource = it->second;
    } else {
      std::cerr << "Memory interface " << memSourceType
                << " has not been specified for layer " << currentParams.name()
                << " but received memory requests. Fix the operation mapping."
                << std::endl;
      std::abort();
    }

    char *memory;
    if (memSource == RRAM) {
      memory = rramMemory;
    } else {
      memory = sramMemory;
    }

    for (int b = 0; b < memRequest.burstSize / DIMENSION; b++) {
      Pack1D<INPUT_DATATYPE, DIMENSION> data;
      for (int i = 0; i < DIMENSION; i++) {
        ac_int<INPUT_DATATYPE::width, false> bits;
        int num_bytes = INPUT_DATATYPE::width / 8;
        for (int byte = 0; byte < num_bytes; byte++) {
          bits.set_slc(
              byte * 8,
              static_cast<ac_int<8, false>>(
                  memory[memRequest.address + b * DIMENSION * num_bytes +
                         i * num_bytes + byte]));
        }
        data[i].setbits(bits);
      }
      DLOG(memSource << " access at addr: " << memRequest.address
                     << " data: " << data << std::endl);
      dataResponse_fifo->write(data);
    }
  }
}

template <long unsigned int DIMENSION>
void Harness::sendMemoryResponse(
    sc_fifo<Pack1D<INPUT_DATATYPE, DIMENSION>> *dataResponse_fifo,
    CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION>> *dataResponse) {
  dataResponse->ResetWrite();

  wait();

  while (true) {
    dataResponse->Push(dataResponse_fifo->read());
  }
}

void Harness::readRequestInputs() {
  readMemoryRequest(&inputAddressRequest, &inputDataResponse_fifo, "inputs");
}
void Harness::sendResponseInputs() {
  sendMemoryResponse(&inputDataResponse_fifo, &inputDataResponse);
}

void Harness::readRequestWeights() {
  readMemoryRequest(&weightAddressRequest, &weightDataResponse_fifo, "weights");
}
void Harness::sendResponseWeights() {
  sendMemoryResponse(&weightDataResponse_fifo, &weightDataResponse);
}

void Harness::readRequestVector0() {
  readMemoryRequest(&vectorFetch0AddressRequest, &vectorFetch0DataResponse_fifo,
                    "vector0");
}
void Harness::sendResponseVector0() {
  sendMemoryResponse(&vectorFetch0DataResponse_fifo, &vectorFetch0DataResponse);
}

void Harness::readRequestVector1() {
  readMemoryRequest(&vectorFetch1AddressRequest, &vectorFetch1DataResponse_fifo,
                    "vector1");
}
void Harness::sendResponseVector1() {
  sendMemoryResponse(&vectorFetch1DataResponse_fifo, &vectorFetch1DataResponse);
}

void Harness::readRequestVector2() {
  readMemoryRequest(&vectorFetch2AddressRequest, &vectorFetch2DataResponse_fifo,
                    "vector2");
}
void Harness::sendResponseVector2() {
  sendMemoryResponse(&vectorFetch2DataResponse_fifo, &vectorFetch2DataResponse);
}

void Harness::readRequestBias() {
  readMemoryRequest(&biasAddressRequest, &biasDataResponse_fifo, "bias");
}
void Harness::sendResponseBias() {
  sendMemoryResponse(&biasDataResponse_fifo, &biasDataResponse);
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

  // Iterate through all params, ie all layers
  for (int i = 0; i < params.size(); i++) {
    currentParams = params.at(i);

    std::deque<AcceleratorMemoryMap> accelerator_memory_maps;
    std::deque<BaseParams *> accelerator_params;
    MapOperation(currentParams, accelerator_params, accelerator_memory_maps);

    while (accelerator_params.size() > 0) {
      bool matrixParamsValid, vectorParamsValid;

      BaseParams *baseParam = accelerator_params.front();
      currentMemoryMap = accelerator_memory_maps.front();

      MatrixParams *matrixParams = dynamic_cast<MatrixParams *>(baseParam);
      matrixParamsValid = matrixParams != NULL;

      if (matrixParamsValid) {
        accelerator_params.pop_front();
        baseParam = accelerator_params.front();
      }

      VectorParams *vectorParams = dynamic_cast<VectorParams *>(baseParam);
      VectorInstructionConfig *vectorInstructionConfig;
      vectorParamsValid = vectorParams != NULL;

      if (vectorParamsValid) {
        accelerator_params.pop_front();
        baseParam = accelerator_params.front();

        vectorInstructionConfig =
            dynamic_cast<VectorInstructionConfig *>(baseParam);
        accelerator_params.pop_front();
      }

      if (matrixParamsValid) {
        sendSerializedParams<MatrixParams, 32>(*matrixParams,
                                               &serialMatrixParamsIn);
        matrixUnitStartSignal.SyncPop();
      }

      sc_time start = sc_time_stamp();
      CCS_LOG("----- Accelerator Layer '" << currentParams.name()
                                          << "' Started. -----");

      if (vectorParamsValid) {
        sendSerializedParams<VectorParams, 32>(*vectorParams,
                                               &serialVectorParamsIn);
        sendSerializedParams<VectorInstructionConfig, 32>(
            *vectorInstructionConfig, &serialVectorParamsIn);
        vectorUnitStartSignal.SyncPop();
      }

      if (matrixParamsValid) {
        matrixUnitDoneSignal.SyncPop();
      }
      if (vectorParamsValid) {
        vectorUnitDoneSignal.SyncPop();
      }
      CCS_LOG("----- Accelerator Layer '" << currentParams.name()
                                          << "' Finished. -----");
      sc_time end = sc_time_stamp();

      std::cout << "Default time unit: " << sc_get_default_time_unit()
                << std::endl;
      std::cout << "Runtime: "
                << int(end.to_default_time_units() -
                       start.to_default_time_units())
                << " ns" << std::endl;

      accelerator_memory_maps.pop_front();
    }

#ifdef SOC_COSIM
    // copy_output(sramMemory, sizeof(INPUT_DATATYPE) * 2 * 1024 * 1024,
    //             sizeof(INPUT_DATATYPE));
    syscDone = true;
#endif
  }

#ifndef SOC_COSIM
  sc_stop();
#endif
}

void Harness::storeVectorOutputs() {
  vectorOutput.ResetRead();
  vectorOutputAddress.ResetRead();

  wait();

  while (true) {
    Pack1D<OUTPUT_DATATYPE, OC_DIMENSION> data = vectorOutput.Pop();
    int address = vectorOutputAddress.Pop();
    DLOG("address: " << address << " data: " << data);
    for (int i = 0; i < OC_DIMENSION; i++) {
      char *memory =
          currentMemoryMap.at("outputs") == SRAM ? sramMemory : rramMemory;

      ac_int<OUTPUT_DATATYPE::width, false> bits = data[i].bits_rep();
      for (int byte = 0; byte < OUTPUT_DATATYPE::width / 8; byte++) {
        memory[address + i * OUTPUT_DATATYPE::width / 8 + byte] =
            bits.slc<8>(byte * 8);
      }
    }
  }
}

void run_accelerator(std::vector<codegen::AcceleratorParam> params,
                     char *sramMemory, char *rramMemory) {
  Harness harness("harness", params, sramMemory, rramMemory);
  sc_start();
}
