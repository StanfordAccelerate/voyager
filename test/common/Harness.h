#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "Accelerator.h"
#include "ArchitectureParams.h"

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

  Connections::Combinational<int> CCS_INIT_S1(vectorAddressRequest);
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorDataResponse);

  Connections::Combinational<int> CCS_INIT_S1(scalarAddressRequest);
  Connections::Combinational<OUTPUT_DATATYPE> CCS_INIT_S1(scalarDataResponse);

  Connections::Combinational<int> CCS_INIT_S1(varianceAddressRequest);
  Connections::Combinational<OUTPUT_DATATYPE> CCS_INIT_S1(varianceDataResponse);

  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorOutput);
  Connections::Combinational<int> CCS_INIT_S1(vectorOutputAddress);

  Connections::SyncChannel done;

  Harness(sc_module_name, Params, INPUT_DATATYPE *);
  SC_HAS_PROCESS(Harness);

 private:
  Params params;
  INPUT_DATATYPE *mainMemory;
  CCS_DESIGN(Accelerator) CCS_INIT_S1(accelerator);

  void memAccessBurst(
      Connections::Combinational<MemoryRequest> * addressRequest,
      Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > *
          dataResponse);
  void memAccessPack(
      Connections::Combinational<int> * addressRequest,
      Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > *
          dataResponse);
  void memAccess(Connections::Combinational<int> * addressRequest,
                 Connections::Combinational<INPUT_DATATYPE> * dataResponse);

  void memAccessInputs();
  void memAccessWeights();
  void memAccessVector();
  void memAccessScalar();
  void memAccessVariance();

  void reset();
  void storeOutputs();
  void sendParams();
  void waitForDone();
};

void run_op(const Params params, INPUT_DATATYPE *mainMemory);