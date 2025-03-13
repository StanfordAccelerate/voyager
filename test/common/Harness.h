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
  sc_fifo<Pack1D<INPUT_DATATYPE, IC_DIMENSION>> inputDataResponse_fifo;
  CombinationalInterface<Pack1D<INPUT_DATATYPE, IC_DIMENSION>> CCS_INIT_S1(
      inputDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      inputScaleAddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, 1>> inputScaleDataResponse_fifo;
  Connections::Combinational<Pack1D<INPUT_DATATYPE, 1>> CCS_INIT_S1(
      inputScaleDataResponse);
#endif

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> weightDataResponse_fifo;
  CombinationalInterface<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      weightDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      weightScaleAddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> weightScaleDataResponse_fifo;
  Connections::Combinational<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      weightScaleDataResponse);
#endif

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> biasDataResponse_fifo;
  CombinationalInterface<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      biasDataResponse);

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> vectorFetch0DataResponse_fifo;
  CombinationalInterface<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vectorFetch0DataResponse);
  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> vectorFetch1DataResponse_fifo;
  CombinationalInterface<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vectorFetch1DataResponse);
  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> vectorFetch2DataResponse_fifo;
  CombinationalInterface<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vectorFetch2DataResponse);

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  sc_fifo<Pack1D<INPUT_DATATYPE, 16 / INPUT_DATATYPE::width>>
      vectorFetch3DataResponse_fifo;
  CombinationalInterface<Pack1D<INPUT_DATATYPE, 16 / INPUT_DATATYPE::width>>
      CCS_INIT_S1(vectorFetch3DataResponse);

  CombinationalInterface<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vector_output);
  CombinationalInterface<ac_int<64, false>> CCS_INIT_S1(vector_output_address);
  CombinationalInterface<Pack1D<DataTypes::int8, 1>> CCS_INIT_S1(scalar_output);
  CombinationalInterface<ac_int<64, false>> CCS_INIT_S1(scalar_output_address);

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

  template <typename T, long unsigned int Dim>
  void readMemoryRequest(CombinationalInterface<MemoryRequest> * request_out,
                         sc_fifo<Pack1D<T, Dim>> * data_fifo);
  template <typename T, long unsigned int Dim>
  void sendMemoryResponse(sc_fifo<Pack1D<T, Dim>> * data_fifo,
                          CombinationalInterface<Pack1D<T, Dim>> * response);
  template <typename T, long unsigned int Dim>
  void storeMemoryResponse(
      CombinationalInterface<Pack1D<T, Dim>> * data_out,
      CombinationalInterface<ac_int<64, false>> * address_out);

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
