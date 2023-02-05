#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>
#include <systemc.h>

#include <deque>
#include <vector>
#include <string>

#include "AccelTypes.h"
#include "Accelerator.h"
#include "ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

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
  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputDataResponse);

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      weightDataResponse);
  CombinationalInterface<MemoryRequest> CCS_INIT_S1(gradAddressRequest);
  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      gradDataResponse);

  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch0DataResponse);
  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch1DataResponse);
  CombinationalInterface<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch2DataResponse);

  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorOutput);
  CombinationalInterface<int> CCS_INIT_S1(vectorOutputAddress);

  CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      scalarUnitOutput);
  CombinationalInterface<int> CCS_INIT_S1(scalarOutputAddress);

  Connections::SyncChannel CCS_INIT_S1(matrixUnitStartSignal);
  Connections::SyncChannel CCS_INIT_S1(matrixUnitDoneSignal);
  Connections::SyncChannel CCS_INIT_S1(vectorUnitStartSignal);
  Connections::SyncChannel CCS_INIT_S1(vectorUnitDoneSignal);

  Harness(sc_module_name, std::vector<SimplifiedParams>, INPUT_DATATYPE *,
          INPUT_DATATYPE *, std::vector<MemoryMap>);
  SC_HAS_PROCESS(Harness);

 private:
  std::vector<SimplifiedParams> params_list;
  SimplifiedParams currentParams;
  INPUT_DATATYPE *sramMemory, *rramMemory;
  std::vector<MemoryMap> memoryMap;
  MemoryMap currentMemoryMap;

#ifdef SIM_Accelerator
  CCS_DESIGN(Accelerator) CCS_INIT_S1(accelerator);
#else
  Accelerator CCS_INIT_S1(accelerator);
#endif

  void memAccessBurst(
      CombinationalInterface<MemoryRequest> * addressRequest,
      CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > * dataResponse,
      std::string memSourceType);
  void memAccessBurstVariable(
      CombinationalInterface<MemoryRequest> * addressRequest,
      CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > *
          dataResponse);
  // void memAccessPack(
  //     CombinationalInterface<int> * addressRequest,
  //     CombinationalInterface<Pack1D<INPUT_DATATYPE, DIMENSION> > * dataResponse,
  //     MemorySource memSource);
  // void memAccess(CombinationalInterface<int> * addressRequest,
  //                CombinationalInterface<INPUT_DATATYPE> * dataResponse,
  //                MemorySource memSource);

  void memAccessInputs();
  void memAccessWeights();
  void memAccessVector0();
  void memAccessVector1();
  void memAccessVector2();
  void memAccessGrad();

  void reset();
  void storeVectorOutputs();
  void storeScalarOutputs();
  void sendParams();
};
