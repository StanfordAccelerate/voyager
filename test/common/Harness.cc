#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include <cassert>

#include "AccelTypes.h"
#include "sysc/kernel/sc_time.h"

#ifndef CFLOAT
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
    std::deque<sc_lv<Wrapped<MemoryRequest>::width>>
        *vectorFetch3AddressRequest,
    std::deque<sc_lv<
        Wrapped<Pack1D<INPUT_DATATYPE, 16 / INPUT_DATATYPE::width>>::width>>
        *vectorFetch3AddressResponse,
    std::deque<sc_lv<Wrapped<Pack1D<INPUT_DATATYPE, DIMENSION>>::width>>
        *vector_output,
    std::deque<sc_lv<Wrapped<uint64_t>::width>> *vector_output_address,
    std::deque<sc_lv<Wrapped<Pack1D<DataTypes::int8, 1>>::width>>
        *scalar_output,
    std::deque<sc_lv<Wrapped<uint64_t>::width>> *scalar_output_address);
// void copy_output(void *sram, int size, int data_size);
#endif

Harness::Harness(sc_module_name name, std::vector<Operation> operations,
                 char *memory)
    : sc_module(name),
      clk("clk", std::stod(std::getenv("CLOCK_PERIOD")), SC_NS, 0.5, 0, SC_NS,
          true),
      operations(operations),
      memory(memory),
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
  accelerator.vectorFetch3AddressRequest(vectorFetch3AddressRequest);
  accelerator.vectorFetch3DataResponse(vectorFetch3DataResponse);
  accelerator.vector_output(vector_output);
  accelerator.vector_output_address(vector_output_address);
  accelerator.scalar_output(scalar_output);
  accelerator.scalar_output_address(scalar_output_address);

  accelerator.matrixUnitStartSignal(matrixUnitStartSignal);
  accelerator.matrixUnitDoneSignal(matrixUnitDoneSignal);
  accelerator.vectorUnitStartSignal(vectorUnitStartSignal);
  accelerator.vectorUnitDoneSignal(vectorUnitDoneSignal);

#if SUPPORT_MX
  accelerator.inputScaleAddressRequest(inputScaleAddressRequest);
  accelerator.inputScaleDataResponse(inputScaleDataResponse);
  accelerator.weightScaleAddressRequest(weightScaleAddressRequest);
  accelerator.weightScaleDataResponse(weightScaleDataResponse);
#endif

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
      vectorFetch2DataResponse.getDataQueue(),
      vectorFetch3AddressRequest.getDataQueue(),
      vectorFetch3DataResponse.getDataQueue(), vector_output.getDataQueue(),
      vector_output_address.getDataQueue(), scalar_output.getDataQueue(),
      scalar_output_address.getDataQueue());
#endif

  SC_CTHREAD(reset, clk);

  SC_THREAD(readRequestInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MX
  SC_THREAD(readRequestInputScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseInputScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
#endif

  SC_THREAD(readRequestWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MX
  SC_THREAD(readRequestWeightScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseWeightScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
#endif

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

  SC_THREAD(readRequestVector3);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector3);
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

  SC_THREAD(storeScalarOutputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendParams);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  accessCounter = new AccessCounter();
// do not set access counters for an RTL simulation
#ifndef CCS_DUT_RTL
  accelerator.matrixUnit.inputBuffer.accessCounter = accessCounter;
  accelerator.matrixUnit.weightBuffer.accessCounter = accessCounter;
#endif
}

template <typename T, long unsigned int Dim>
void Harness::readMemoryRequest(
    CombinationalInterface<MemoryRequest> *request_out,
    sc_fifo<Pack1D<T, Dim>> *data_fifo) {
  request_out->ResetRead();

  wait();

  while (true) {
    MemoryRequest request = request_out->Pop();

    uint64_t base_address = request.address;
    int num_bytes = request.burst_size;
    int num_groups = num_bytes / (T::width / 8.0) / Dim;

    accessCounter->increment(std::string(name()), num_bytes);

    constexpr int buffer_size = (T::width / 8 + 2) * 8;
    ac_int<buffer_size, false> bits;

    for (int i = 0; i < num_groups; i++) {
      Pack1D<T, Dim> data;
      for (int j = 0; j < Dim; j++) {
        int start = (i * Dim + j) * T::width;
        int end = start + T::width;
        int offset = start % 8;

        int start_byte = start / 8;
        int end_byte = (end + 7) / 8;

        for (int byte = start_byte; byte < end_byte; byte++) {
          uint64_t address = base_address + byte;
          bits.set_slc((byte - start_byte) * 8,
                       static_cast<ac_int<8, false>>(memory[address]));
        }

        data[j].set_bits(bits >> offset);
      }

      DLOG("read addr: " << request.address << " data: " << data << std::endl);
      data_fifo->write(data);
    }
  }
}

template <typename T, long unsigned int Dim>
void Harness::sendMemoryResponse(
    sc_fifo<Pack1D<T, Dim>> *data_fifo,
    CombinationalInterface<Pack1D<T, Dim>> *response) {
  response->ResetWrite();

  wait();

  while (true) {
    response->Push(data_fifo->read());
  }
}

template <typename T, long unsigned int Dim>
void Harness::storeMemoryResponse(
    CombinationalInterface<Pack1D<T, Dim>> *data_out,
    CombinationalInterface<ac_int<64, false>> *address_out) {
  data_out->ResetRead();
  address_out->ResetRead();

  wait();

  while (true) {
    Pack1D<T, Dim> data = data_out->Pop();
    uint64_t address = address_out->Pop();
    DLOG("write address: " << address << " data: " << data);
    accessCounter->increment(std::string(name()) + "_" + "outputs",
                             OC_DIMENSION);

    constexpr int width = T::width;
    constexpr int buf_width = (width / 8 + 2) * 8;

    for (int i = 0; i < Dim; i++) {
      int start = i * width / 8;
      int end = (i + 1) * width / 8;
      int offset = (i * width) % 8;
      int num_bytes = (end - start + 1) * 8;

      int bits_remaining = num_bytes * 8 - width - offset;
      num_bytes = num_bytes - bits_remaining / 8;

      ac_int<buf_width, false> bytes = data[i].bits_rep();
      ac_int<buf_width, false> masks = ((1 << width) - 1);

      bytes = bytes << offset;
      masks = masks << offset;

      for (int i = 0; i < num_bytes; i++) {
        char mask = masks.template slc<8>(i * 8);
        char new_data = bytes.template slc<8>(i * 8) & mask;
        char orig_data = memory[address + start + i] & ~mask;
        memory[address + start + i] = new_data | orig_data;
      }
    }
  }
}

void Harness::readRequestInputs() {
  readMemoryRequest(&inputAddressRequest, &inputDataResponse_fifo);
}
void Harness::sendResponseInputs() {
  sendMemoryResponse(&inputDataResponse_fifo, &inputDataResponse);
}

void Harness::readRequestInputScale() {
#if SUPPORT_MX
  readMemoryRequest(&inputScaleAddressRequest, &inputScaleDataResponse_fifo);
#endif
}
void Harness::sendResponseInputScale() {
#if SUPPORT_MX
  sendMemoryResponse(&inputScaleDataResponse_fifo, &inputScaleDataResponse);
#endif
}

void Harness::readRequestWeights() {
  readMemoryRequest(&weightAddressRequest, &weightDataResponse_fifo);
}
void Harness::sendResponseWeights() {
  sendMemoryResponse(&weightDataResponse_fifo, &weightDataResponse);
}

void Harness::readRequestWeightScale() {
#if SUPPORT_MX
  readMemoryRequest(&weightScaleAddressRequest, &weightScaleDataResponse_fifo);
#endif
}
void Harness::sendResponseWeightScale() {
#if SUPPORT_MX
  sendMemoryResponse(&weightScaleDataResponse_fifo, &weightScaleDataResponse);
#endif
}

void Harness::readRequestVector0() {
  readMemoryRequest(&vectorFetch0AddressRequest,
                    &vectorFetch0DataResponse_fifo);
}
void Harness::sendResponseVector0() {
  sendMemoryResponse(&vectorFetch0DataResponse_fifo, &vectorFetch0DataResponse);
}

void Harness::readRequestVector1() {
  readMemoryRequest(&vectorFetch1AddressRequest,
                    &vectorFetch1DataResponse_fifo);
}
void Harness::sendResponseVector1() {
  sendMemoryResponse(&vectorFetch1DataResponse_fifo, &vectorFetch1DataResponse);
}

void Harness::readRequestVector2() {
  readMemoryRequest(&vectorFetch2AddressRequest,
                    &vectorFetch2DataResponse_fifo);
}
void Harness::sendResponseVector2() {
  sendMemoryResponse(&vectorFetch2DataResponse_fifo, &vectorFetch2DataResponse);
}

void Harness::readRequestVector3() {
  readMemoryRequest(&vectorFetch3AddressRequest,
                    &vectorFetch3DataResponse_fifo);
}
void Harness::sendResponseVector3() {
  sendMemoryResponse(&vectorFetch3DataResponse_fifo, &vectorFetch3DataResponse);
}

void Harness::readRequestBias() {
  readMemoryRequest(&biasAddressRequest, &biasDataResponse_fifo);
}
void Harness::sendResponseBias() {
  sendMemoryResponse(&biasDataResponse_fifo, &biasDataResponse);
}

void Harness::storeVectorOutputs() {
  storeMemoryResponse(&vector_output, &vector_output_address);
}

void Harness::storeScalarOutputs() {
  storeMemoryResponse(&scalar_output, &scalar_output_address);
}

void Harness::reset() {
  rstn.write(0);
  wait(5);
  rstn.write(1);
  wait();
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
  for (int i = 0; i < operations.size(); i++) {
    currentOperation = operations.at(i);

    std::deque<AcceleratorMemoryMap> accelerator_memory_maps;
    std::deque<BaseParams *> accelerator_params;
    MapOperation(currentOperation, accelerator_params, accelerator_memory_maps);

    while (accelerator_params.size() > 0) {
      bool matrixParamsValid, vectorParamsValid;

      BaseParams *baseParam = accelerator_params.front();

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
      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Started. -----");

      if (vectorParamsValid) {
        sendSerializedParams<VectorParams, 32>(*vectorParams,
                                               &serialVectorParamsIn);
        sendSerializedParams<VectorInstructionConfig, 32>(
            *vectorInstructionConfig, &serialVectorParamsIn);
        vectorUnitStartSignal.SyncPop();
      }

      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Started. -----");

      if (matrixParamsValid) {
        matrixUnitDoneSignal.SyncPop();
      }
      if (vectorParamsValid) {
        vectorUnitDoneSignal.SyncPop();
      }
      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Finished. -----");
      sc_time end = sc_time_stamp();

      std::cout << "Default time unit: " << sc_get_default_time_unit()
                << std::endl;
      std::cout << "Runtime: "
                << int(end.to_default_time_units() -
                       start.to_default_time_units())
                << " ns" << std::endl;

      accessCounter->print_summary(currentOperation.tiling,
                                   currentOperation.has_valid_tiling);

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

#endif

void run_accelerator(std::vector<Operation> operations, char *memory) {
#ifdef CFLOAT
  spdlog::error(
      "The SystemC model does not support the CFloat datatype. Only the gold "
      "model should be used for CFloat.\n");
  std::abort();
#else
  Harness harness("harness", operations, memory);
  sc_start();
#endif
}
