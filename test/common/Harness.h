#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>
#include <systemc.h>

#include <deque>
#include <string>
#include <vector>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "spdlog/spdlog.h"
#include "test/common/AccessCounter.h"
#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

#ifndef CFLOAT
#include "Accelerator.h"

#ifdef SOC_COSIM
#define CombinationalInterface LoggingCombinational
#else
#define CombinationalInterface Connections::Combinational
#endif

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

  CombinationalInterface<int> CCS_INIT_S1(serialMatrixParamsIn);
  CombinationalInterface<int> CCS_INIT_S1(serialVectorParamsIn);

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  sc_fifo<ac_int<IC_PORT_WIDTH, false>> inputDataResponse_fifo;
  CombinationalInterface<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(
      inputDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      inputScaleAddressRequest);
  sc_fifo<ac_int<SCALE_DATATYPE::width, false>> inputScaleDataResponse_fifo;
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleDataResponse);
#endif

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> weightDataResponse_fifo;
  CombinationalInterface<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      weightScaleAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> weightScaleDataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleDataResponse);
#endif

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> biasDataResponse_fifo;
  CombinationalInterface<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      biasDataResponse);

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vector_fetch_0_request_out);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch0DataResponse_fifo;
  CombinationalInterface<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp_in);
  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vector_fetch_1_request_out);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch1DataResponse_fifo;
  CombinationalInterface<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp_in);
  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vector_fetch_2_request_out);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch2DataResponse_fifo;
  CombinationalInterface<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp_in);

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vector_fetch_3_request_out);
  sc_fifo<ac_int<16, false>> vectorFetch3DataResponse_fifo;
  CombinationalInterface<ac_int<16, false>> CCS_INIT_S1(vector_fetch_3_resp_in);

  CombinationalInterface<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_output);
  CombinationalInterface<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_address);
  CombinationalInterface<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      scalar_output);
  CombinationalInterface<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
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
  void readMemoryRequest(CombinationalInterface<MemoryRequest> * request_out,
                         sc_fifo<ac_int<Width, false>> * data_fifo);

  template <int Width>
  void sendMemoryResponse(
      sc_fifo<ac_int<Width, false>> * data_fifo,
      CombinationalInterface<ac_int<Width, false>> * response);

  template <int Width>
  void storeMemoryResponse(
      CombinationalInterface<ac_int<Width, false>> * data_out,
      CombinationalInterface<ac_int<ADDRESS_WIDTH, false>> * address_out);

  void readRequestInputs();
  void sendResponseInputs();

  void readRequestInputScale();
  void sendResponseInputScale();

  void readRequestWeights();
  void sendResponseWeights();

  void readRequestWeightScale();
  void sendResponseWeightScale();

  void readRequestVector0();
  void sendResponseVector0();

  void readRequestVector1();
  void sendResponseVector1();

  void readRequestVector2();
  void sendResponseVector2();

  void readRequestVector3();
  void sendResponseVector3();

  void readRequestBias();
  void sendResponseBias();

  void storeVectorOutputs();
  void storeScalarOutputs();

  void reset();
  void sendParams();
};
#endif
