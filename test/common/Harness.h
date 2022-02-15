#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "Accelerator.h"
#include "ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

#include <vector>

SC_MODULE(Harness) {
  sc_clock CCS_INIT_S1(clk);
  sc_signal<bool> CCS_INIT_S1(rstn);

  Connections::Combinational<int> CCS_INIT_S1(serialParamsIn);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputDataResponse);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      weightDataResponse);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      vectorFetch0AddressRequest);
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch0DataResponse);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      vectorFetch1AddressRequest);
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch1DataResponse);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      vectorFetch2AddressRequest);
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch2DataResponse);

  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorOutput);
  Connections::Combinational<int> CCS_INIT_S1(vectorOutputAddress);

  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      scalarUnitOutput);
  Connections::Combinational<int> CCS_INIT_S1(scalarOutputAddress);

  Connections::SyncChannel CCS_INIT_S1(start);
  Connections::SyncChannel CCS_INIT_S1(done);

  Harness(sc_module_name, std::vector<SimplifiedParams>, INPUT_DATATYPE *,
          INPUT_DATATYPE *, MemoryMap);
  SC_HAS_PROCESS(Harness);

 private:
  std::vector<SimplifiedParams> params_list;
  SimplifiedParams currentParams;
  INPUT_DATATYPE *sramMemory, *rramMemory;
  MemoryMap memoryMap;
  CCS_DESIGN(Accelerator) CCS_INIT_S1(accelerator);

  void memAccessBurst(
      Connections::Combinational<MemoryRequest> * addressRequest,
      Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > *
          dataResponse,
      MemorySource memSource);
  void memAccessPack(
      Connections::Combinational<int> * addressRequest,
      Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > *
          dataResponse,
      MemorySource memSource);
  void memAccess(Connections::Combinational<int> * addressRequest,
                 Connections::Combinational<INPUT_DATATYPE> * dataResponse,
                 MemorySource memSource);

  void memAccessInputs();
  void memAccessWeights();
  void memAccessVector0();
  void memAccessVector1();
  void memAccessVector2();

  void reset();
  void storeVectorOutputs();
  void storeScalarOutputs();
  void sendParams();
  void waitForStart();
  void waitForDone();
};
