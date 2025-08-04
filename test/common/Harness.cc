#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include <cassert>

#include "AccelTypes.h"
#include "sysc/kernel/sc_time.h"

#ifndef CFLOAT
#include "test/toolchain/MapOperation.h"

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
      vectorFetch2DataResponse_fifo("vectorFetch2DataResponse_fifo", 1024)
#if SUPPORT_MVM
      ,
      matrix_vector_input_resp_fifo("matrix_vector_input_resp_fifo", 1024),
      matrix_vector_weight_resp_fifo("matrix_vector_weight_resp_fifo", 1024),
      matrix_vector_bias_resp_fifo("matrix_vector_bias_resp_fifo", 1024)
#endif
{
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
#if SUPPORT_MVM
  accelerator.serial_matrix_vector_params_in(serial_matrix_vector_params_in);
  accelerator.matrix_vector_input_req(matrix_vector_input_req);
  accelerator.matrix_vector_input_resp(matrix_vector_input_resp);
  accelerator.matrix_vector_weight_req(matrix_vector_weight_req);
  accelerator.matrix_vector_weight_resp(matrix_vector_weight_resp);
  accelerator.matrix_vector_bias_req(matrix_vector_bias_req);
  accelerator.matrix_vector_bias_resp(matrix_vector_bias_resp);
#if SUPPORT_MX
  accelerator.matrix_vector_input_scale_req(matrix_vector_input_scale_req);
  accelerator.matrix_vector_input_scale_resp(matrix_vector_input_scale_resp);
  accelerator.matrix_vector_weight_scale_req(matrix_vector_weight_scale_req);
  accelerator.matrix_vector_weight_scale_resp(matrix_vector_weight_scale_resp);
#endif
  accelerator.matrix_vector_unit_start_signal(matrix_vector_unit_start_signal);
  accelerator.matrix_vector_unit_done_signal(matrix_vector_unit_done_signal);
#endif
  accelerator.vector_fetch_0_req(vector_fetch_0_req);
  accelerator.vector_fetch_0_resp(vector_fetch_0_resp);
  accelerator.vector_fetch_1_req(vector_fetch_1_req);
  accelerator.vector_fetch_1_resp(vector_fetch_1_resp);
  accelerator.vector_fetch_2_req(vector_fetch_2_req);
  accelerator.vector_fetch_2_resp(vector_fetch_2_resp);
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

  SC_THREAD(readRequestBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MVM
  SC_THREAD(readRequestMatrixVectorInput);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseMatrixVectorInput);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestMatrixVectorWeight);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseMatrixVectorWeight);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestMatrixVectorBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseMatrixVectorBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MX
  SC_THREAD(readRequestMatrixVectorInputScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseMatrixVectorInputScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestMatrixVectorWeightScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseMatrixVectorWeightScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
#endif
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

template <int Width>
void Harness::readMemoryRequest(
    Connections::Combinational<MemoryRequest> *request_out,
    sc_fifo<ac_int<Width, false>> *data_fifo) {
  request_out->ResetRead();

  constexpr int num_bytes = Width / 8;

  wait();

  while (true) {
    MemoryRequest request = request_out->Pop();

    uint64_t base_address = request.address;
    int total_bytes = request.burst_size;
    int num_words = (total_bytes + num_bytes - 1) / num_bytes;

    accessCounter->increment(std::string(name()), total_bytes);

    ac_int<Width, false> bits;

    for (int i = 0; i < num_words; i++) {
      for (int j = 0; j < num_bytes; j++) {
        uint64_t address = base_address + i * num_bytes + j;
        DLOG("read addr: " << address << " data: " << memory[address]
                           << std::endl);
        bits.set_slc(j * 8, static_cast<ac_int<8, false>>(memory[address]));
      }

      data_fifo->write(bits);
    }
  }
}

template <int Width>
void Harness::sendMemoryResponse(
    sc_fifo<ac_int<Width, false>> *data_fifo,
    Connections::Combinational<ac_int<Width, false>> *response) {
  response->ResetWrite();

  wait();

  while (true) {
    response->Push(data_fifo->read());
  }
}

template <int Width>
void Harness::storeMemoryResponse(
    Connections::Combinational<ac_int<Width, false>> *data_out,
    Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> *address_out) {
  data_out->ResetRead();
  address_out->ResetRead();

  constexpr int num_bytes = Width / 8;

  wait();

  while (true) {
    uint64_t address = address_out->Pop();
    auto data = data_out->Pop();
    DLOG("write address: " << address << " data: " << data);

    accessCounter->increment(std::string(name()) + "_" + "outputs", num_bytes);

    for (int i = 0; i < num_bytes; i++) {
      memory[address + i] = data.template slc<8>(i * 8);
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

void Harness::readRequestBias() {
  readMemoryRequest(&biasAddressRequest, &biasDataResponse_fifo);
}

void Harness::sendResponseBias() {
  sendMemoryResponse(&biasDataResponse_fifo, &biasDataResponse);
}

#if SUPPORT_MVM
void Harness::readRequestMatrixVectorInput() {
  readMemoryRequest(&matrix_vector_input_req, &matrix_vector_input_resp_fifo);
}

void Harness::sendResponseMatrixVectorInput() {
  sendMemoryResponse(&matrix_vector_input_resp_fifo, &matrix_vector_input_resp);
}

void Harness::readRequestMatrixVectorWeight() {
  readMemoryRequest(&matrix_vector_weight_req, &matrix_vector_weight_resp_fifo);
}

void Harness::sendResponseMatrixVectorWeight() {
  sendMemoryResponse(&matrix_vector_weight_resp_fifo,
                     &matrix_vector_weight_resp);
}

void Harness::readRequestMatrixVectorBias() {
  readMemoryRequest(&matrix_vector_bias_req, &matrix_vector_bias_resp_fifo);
}

void Harness::sendResponseMatrixVectorBias() {
  sendMemoryResponse(&matrix_vector_bias_resp_fifo, &matrix_vector_bias_resp);
}

#if SUPPORT_MX
void Harness::readRequestMatrixVectorInputScale() {
  readMemoryRequest(&matrix_vector_input_scale_req,
                    &matrix_vector_input_scale_resp_fifo);
}

void Harness::sendResponseMatrixVectorInputScale() {
  sendMemoryResponse(&matrix_vector_input_scale_resp_fifo,
                     &matrix_vector_input_scale_resp);
}

void Harness::readRequestMatrixVectorWeightScale() {
  readMemoryRequest(&matrix_vector_weight_scale_req,
                    &matrix_vector_weight_scale_resp_fifo);
}

void Harness::sendResponseMatrixVectorWeightScale() {
  sendMemoryResponse(&matrix_vector_weight_scale_resp_fifo,
                     &matrix_vector_weight_scale_resp);
}
#endif
#endif

void Harness::readRequestVector0() {
  readMemoryRequest(&vector_fetch_0_req, &vectorFetch0DataResponse_fifo);
}

void Harness::sendResponseVector0() {
  sendMemoryResponse(&vectorFetch0DataResponse_fifo, &vector_fetch_0_resp);
}

void Harness::readRequestVector1() {
  readMemoryRequest(&vector_fetch_1_req, &vectorFetch1DataResponse_fifo);
}

void Harness::sendResponseVector1() {
  sendMemoryResponse(&vectorFetch1DataResponse_fifo, &vector_fetch_1_resp);
}

void Harness::readRequestVector2() {
  readMemoryRequest(&vector_fetch_2_req, &vectorFetch2DataResponse_fifo);
}

void Harness::sendResponseVector2() {
  sendMemoryResponse(&vectorFetch2DataResponse_fifo, &vector_fetch_2_resp);
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
void sendSerializedParams(
    T params,
    Connections::Combinational<ac_int<interfaceWidth, false>> *serialParamsIn) {
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

#if SUPPORT_MVM
  matrix_vector_unit_start_signal.ResetRead();
  matrix_vector_unit_done_signal.ResetRead();
  serial_matrix_vector_params_in.ResetWrite();
#endif

  wait();

  // Iterate through all params, ie all layers
  for (int i = 0; i < operations.size(); i++) {
    currentOperation = operations.at(i);

    std::deque<AcceleratorMemoryMap> accelerator_memory_maps;
    std::deque<BaseParams *> accelerator_params;
    MapOperation(currentOperation, accelerator_params, accelerator_memory_maps);

    int runtime_scale_factor = 1;
    std::cout << "Operation: " << currentOperation.name << std::endl;
    if (currentOperation.has_shrunk_tiling) {
      runtime_scale_factor = currentOperation.shrink_factor;
      std::cout << "Scaling operation by " << runtime_scale_factor << std::endl;
    }

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
#if SUPPORT_MVM
        if (matrixParams->is_fc) {
          sendSerializedParams<MatrixParams, 64>(
              *matrixParams, &serial_matrix_vector_params_in);
          matrix_vector_unit_start_signal.SyncPop();
        } else
#endif
        {
          sendSerializedParams<MatrixParams, 64>(*matrixParams,
                                                 &serialMatrixParamsIn);
          matrixUnitStartSignal.SyncPop();
        }
      }

      sc_time start = sc_time_stamp();
      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Started. -----");

      if (vectorParamsValid) {
        sendSerializedParams<VectorParams, 64>(*vectorParams,
                                               &serialVectorParamsIn);
        sendSerializedParams<VectorInstructionConfig, 64>(
            *vectorInstructionConfig, &serialVectorParamsIn);
        vectorUnitStartSignal.SyncPop();
      }

      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Started. -----");

      if (matrixParamsValid) {
#if SUPPORT_MVM
        if (matrixParams->is_fc) {
          matrix_vector_unit_done_signal.SyncPop();
        } else
#endif
        {
          matrixUnitDoneSignal.SyncPop();
        }
      }
      if (vectorParamsValid) {
        vectorUnitDoneSignal.SyncPop();
      }
      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Finished. -----");
      sc_time end = sc_time_stamp();

      std::cout << "Default time unit: " << sc_get_default_time_unit()
                << std::endl;

      int runtime = runtime_scale_factor * int(end.to_default_time_units() -
                                               start.to_default_time_units());
      std::cout << "Runtime: " << runtime << " ns" << std::endl;

      accessCounter->print_summary(currentOperation.tiling,
                                   currentOperation.has_valid_tiling);

      accelerator_memory_maps.pop_front();
    }
  }

  sc_stop();
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
