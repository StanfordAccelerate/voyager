#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>
#include <systemc.h>

#include <deque>
#include <vector>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "test/common/AccessCounter.h"
#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

#ifndef CFLOAT
#include "Accelerator.h"

template <typename T>
class LoggingCombinational : public Connections::Combinational<T> {
 public:
  typedef Wrapped<T> WT;
  static const unsigned int lv_width = T::width;
  typedef sc_lv<WT::width> lv_bits;

  std::deque<lv_bits> *dataQueue;

  LoggingCombinational()
      : Connections::Combinational<T>(), dataQueue(new std::deque<lv_bits>) {}
  explicit LoggingCombinational(const char *name)
      : Connections::Combinational<T>(name),
        dataQueue(new std::deque<lv_bits>) {}

  void Push(const T &m) {
    Marshaller<WT::width> marshaller;
    WT wt(m);
    wt.Marshall(marshaller);
    lv_bits bits = marshaller.GetResult();

    dataQueue->push_back(bits);
    Connections::Combinational<T>::Push(m);
  }

  T Pop() {
    T m = Connections::Combinational<T>::Pop();
    Marshaller<WT::width> marshaller;
    WT wt(m);
    wt.Marshall(marshaller);
    lv_bits bits = marshaller.GetResult();
    dataQueue->push_back(bits);
    return m;
  }

  std::deque<lv_bits> *getDataQueue() { return dataQueue; }
};

SC_MODULE(Harness) {
  sc_clock CCS_INIT_S1(clk);
  sc_signal<bool> CCS_INIT_S1(rstn);

  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      serialMatrixParamsIn);
  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      serialVectorParamsIn);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  sc_fifo<IC_PORT_TYPE> inputDataResponse_fifo;
  Connections::Combinational<IC_PORT_TYPE> CCS_INIT_S1(inputDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      inputScaleAddressRequest);
  sc_fifo<ac_int<SCALE_DATATYPE::width, false>> inputScaleDataResponse_fifo;
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleDataResponse);
#endif

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> weightDataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      weightScaleAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> weightScaleDataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleDataResponse);
#endif

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> biasDataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      biasDataResponse);

#if SUPPORT_MVM
  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      serial_matrix_vector_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_input_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_input_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_input_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_weight_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_weight_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_weight_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(matrix_vector_bias_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_bias_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(matrix_vector_bias_resp);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_input_scale_req);
  sc_fifo<ac_int<MVU_SCALE_PORT_WIDTH, false>>
      matrix_vector_input_scale_resp_fifo;
  Connections::Combinational<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_input_scale_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_weight_scale_req);
  sc_fifo<ac_int<MVU_SCALE_PORT_WIDTH, false>>
      matrix_vector_weight_scale_resp_fifo;
  Connections::Combinational<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_weight_scale_resp);
#endif

  Connections::SyncChannel CCS_INIT_S1(matrix_vector_unit_start_signal);
  Connections::SyncChannel CCS_INIT_S1(matrix_vector_unit_done_signal);
#endif

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch0DataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch1DataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch2DataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);

  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_output);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_address);
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      scalar_output);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      scalar_output_address);

  Connections::SyncChannel CCS_INIT_S1(matrixUnitStartSignal);
  Connections::SyncChannel CCS_INIT_S1(matrixUnitDoneSignal);
  Connections::SyncChannel CCS_INIT_S1(vectorUnitStartSignal);
  Connections::SyncChannel CCS_INIT_S1(vectorUnitDoneSignal);

  Harness(sc_module_name, std::vector<Operation>, char *);
  SC_HAS_PROCESS(Harness);

 private:
  std::vector<Operation> operations;
  Operation currentOperation;
  char *memory;
  AcceleratorMemoryMap currentMemoryMap;
  AccessCounter *accessCounter;

#ifdef SIM_Accelerator
  CCS_DESIGN(Accelerator) CCS_INIT_S1(accelerator);
#else
  Accelerator CCS_INIT_S1(accelerator);
#endif

  template <int Width>
  void readMemoryRequest(
      Connections::Combinational<MemoryRequest> * request_out,
      sc_fifo<ac_int<Width, false>> * data_fifo);

  template <int Width>
  void sendMemoryResponse(
      sc_fifo<ac_int<Width, false>> * data_fifo,
      Connections::Combinational<ac_int<Width, false>> * response);

  template <int Width>
  void storeMemoryResponse(
      Connections::Combinational<ac_int<Width, false>> * data_out,
      Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> * address_out);

  void readRequestInputs();
  void sendResponseInputs();

  void readRequestInputScale();
  void sendResponseInputScale();

  void readRequestWeights();
  void sendResponseWeights();

  void readRequestWeightScale();
  void sendResponseWeightScale();

  void readRequestBias();
  void sendResponseBias();

  void readRequestMatrixVectorInput();
  void sendResponseMatrixVectorInput();
  void readRequestMatrixVectorWeight();
  void sendResponseMatrixVectorWeight();
  void readRequestMatrixVectorBias();
  void sendResponseMatrixVectorBias();

#if SUPPORT_MX
  void readRequestMatrixVectorInputScale();
  void sendResponseMatrixVectorInputScale();
  void readRequestMatrixVectorWeightScale();
  void sendResponseMatrixVectorWeightScale();
#endif

  void readRequestVector0();
  void sendResponseVector0();

  void readRequestVector1();
  void sendResponseVector1();

  void readRequestVector2();
  void sendResponseVector2();

  void storeVectorOutputs();
  void storeScalarOutputs();

  void reset();
  void sendParams();
};
#endif
