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
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);

  Connections::In<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(inputDataResponse);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weightDataResponse);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(biasDataResponse);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputScaleAddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightScaleAddressRequest);

  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleDataResponse);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleDataResponse);
#endif

  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(outputsFromSystolicArray);
  Connections::SyncOut CCS_INIT_S1(matrixUnitStartSignal);
  Connections::SyncOut CCS_INIT_S1(matrixUnitDoneSignal);

#ifdef SIM_VectorUnit
  // clang-format off
  CCS_DESIGN((VectorUnit<VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE, OC_DIMENSION>)) CCS_INIT_S1(vectorUnit);
  // clang-format on
#else
  VectorUnit<VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE,
             OC_DIMENSION>
      CCS_INIT_S1(vectorUnit);
#endif

  Connections::In<int> CCS_INIT_S1(serialVectorParamsIn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vectorFetch0DataResponse);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vectorFetch1DataResponse);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vectorFetch2DataResponse);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(vectorFetch3DataResponse);

  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_output);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_address);
  Connections::Out<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      scalar_output);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      scalar_output_address);

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
    vectorUnit.vector_output(vector_output);
    vectorUnit.vector_output_address(vector_output_address);
    vectorUnit.scalar_output(scalar_output);
    vectorUnit.scalar_output_address(scalar_output_address);
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
