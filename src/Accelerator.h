#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "MatrixUnit.h"
#include "ParamsDeserializer.h"
#include "VectorUnit.h"
#include "mc_scverify.h"

SC_MODULE(Accelerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixUnit CCS_INIT_S1(matrixUnit);
  Connections::In<int> CCS_INIT_S1(serialMatrixParamsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, IC_DIMENSION> > CCS_INIT_S1(
      inputDataResponse);
#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputScaleAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, 1> > CCS_INIT_S1(
      inputScaleDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightScaleAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      weightScaleDataResponse);
#endif
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      weightDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      biasDataResponse);
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION> >
      CCS_INIT_S1(outputsFromSystolicArray);
  Connections::SyncOut CCS_INIT_S1(matrixUnitStartSignal);
  Connections::SyncOut CCS_INIT_S1(matrixUnitDoneSignal);

#ifdef SIM_VectorUnit
  // clang-format off
  CCS_DESIGN((VectorUnit<VECTOR_DATATYPE, VECTOR_ACCUM_DATATYPE, ACCUM_DATATYPE, OC_DIMENSION>)) CCS_INIT_S1(vectorUnit);
  // clang-format on
#else
  VectorUnit<INPUT_DATATYPE, VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE,
             MX_DATATYPE, OC_DIMENSION>
      CCS_INIT_S1(vectorUnit);
#endif
  Connections::In<int> CCS_INIT_S1(serialVectorParamsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::In<Pack1D<OUTPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      vectorFetch0DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::In<Pack1D<OUTPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      vectorFetch1DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::In<Pack1D<OUTPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      vectorFetch2DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  Connections::In<OUTPUT_DATATYPE> CCS_INIT_S1(vectorFetch3DataResponse);
  Connections::Out<Pack1D<OUTPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      vectorOutput);
  Connections::Out<ac_int<64, false> > CCS_INIT_S1(vectorOutputAddress);

  Connections::SyncOut CCS_INIT_S1(vectorUnitStartSignal);
  Connections::SyncOut CCS_INIT_S1(vectorUnitDoneSignal);

  SC_CTOR(Accelerator) {
    matrixUnit.clk(clk);
    matrixUnit.rstn(rstn);
    matrixUnit.serialMatrixParamsIn(serialMatrixParamsIn);
    matrixUnit.inputAddressRequest(inputAddressRequest);
    matrixUnit.inputDataResponse(inputDataResponse);
    matrixUnit.weightAddressRequest(weightAddressRequest);
    matrixUnit.weightDataResponse(weightDataResponse);
    matrixUnit.biasAddressRequest(biasAddressRequest);
    matrixUnit.biasDataResponse(biasDataResponse);
    matrixUnit.outputsFromSystolicArray(outputsFromSystolicArray);
    matrixUnit.startSignal(matrixUnitStartSignal);
    matrixUnit.doneSignal(matrixUnitDoneSignal);

#if SUPPORT_MX
    matrixUnit.inputScaleAddressRequest(inputScaleAddressRequest);
    matrixUnit.inputScaleDataResponse(inputScaleDataResponse);
    matrixUnit.weightScaleAddressRequest(weightScaleAddressRequest);
    matrixUnit.weightScaleDataResponse(weightScaleDataResponse);
#endif

    vectorUnit.clk(clk);
    vectorUnit.rstn(rstn);
    vectorUnit.serialParamsIn(serialVectorParamsIn);
    vectorUnit.systolicArrayOutput(outputsFromSystolicArray);
    vectorUnit.vectorFetch0AddressRequest(vectorFetch0AddressRequest);
    vectorUnit.vectorFetch0DataResponse(vectorFetch0DataResponse);
    vectorUnit.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
    vectorUnit.vectorFetch1DataResponse(vectorFetch1DataResponse);
    vectorUnit.vectorFetch2AddressRequest(vectorFetch2AddressRequest);
    vectorUnit.vectorFetch2DataResponse(vectorFetch2DataResponse);
    vectorUnit.vectorFetch3AddressRequest(vectorFetch3AddressRequest);
    vectorUnit.vectorFetch3DataResponse(vectorFetch3DataResponse);
    vectorUnit.vectorOutputAddress(vectorOutputAddress);
    vectorUnit.finalVectorOutput(vectorOutput);
    vectorUnit.start(vectorUnitStartSignal);
    vectorUnit.done(vectorUnitDoneSignal);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    wait();

    while (true) {
      wait();
    }
  }
};
