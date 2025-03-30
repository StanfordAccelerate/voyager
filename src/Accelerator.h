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
  Connections::In<Pack1D<INPUT_DATATYPE, IC_DIMENSION>> CCS_INIT_S1(
      inputDataResponse);
#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputScaleAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, 1>> CCS_INIT_S1(
      inputScaleDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightScaleAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      weightScaleDataResponse);
#endif
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      weightDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      biasDataResponse);
  Connections::SyncOut CCS_INIT_S1(matrixUnitStartSignal);
  Connections::SyncOut CCS_INIT_S1(matrixUnitDoneSignal);

  Connections::Combinational<ac_int<16, false>>
      accumulation_buffer_vu_read_address[2];
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_vu_read_data[2];
  Connections::Combinational<
      BufferWriteRequest<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_vu_write_request[2];
  Connections::SyncChannel accumulation_buffer_vu_done[2];

#ifdef SIM_VectorUnit
  // clang-format off
  CCS_DESIGN((VectorUnit<VECTOR_DATATYPE, VECTOR_ACCUM_DATATYPE, ACCUM_DATATYPE, OC_DIMENSION>)) CCS_INIT_S1(vectorUnit);
  // clang-format on
#else
  VectorUnit<INPUT_DATATYPE, VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE,
             SCALE_DATATYPE, OC_DIMENSION>
      CCS_INIT_S1(vectorUnit);
#endif
  Connections::In<int> CCS_INIT_S1(serialVectorParamsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vectorFetch0DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vectorFetch1DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vectorFetch2DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, 16 / INPUT_DATATYPE::width>>
      CCS_INIT_S1(vectorFetch3DataResponse);
  Connections::Out<Pack1D<INPUT_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      vector_output);
  Connections::Out<ac_int<64, false>> CCS_INIT_S1(vector_output_address);
  Connections::Out<Pack1D<DataTypes::int8, 1>> CCS_INIT_S1(scalar_output);
  Connections::Out<ac_int<64, false>> CCS_INIT_S1(scalar_output_address);

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
    matrixUnit.startSignal(matrixUnitStartSignal);
    matrixUnit.doneSignal(matrixUnitDoneSignal);

    for (int i = 0; i < 2; i++) {
      matrixUnit.accumulation_buffer_vu_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      matrixUnit.accumulation_buffer_vu_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      matrixUnit.accumulation_buffer_vu_write_request[i](
          accumulation_buffer_vu_write_request[i]);
      matrixUnit.accumulation_buffer_vu_done[i](accumulation_buffer_vu_done[i]);
    }

#if SUPPORT_MX
    matrixUnit.inputScaleAddressRequest(inputScaleAddressRequest);
    matrixUnit.inputScaleDataResponse(inputScaleDataResponse);
    matrixUnit.weightScaleAddressRequest(weightScaleAddressRequest);
    matrixUnit.weightScaleDataResponse(weightScaleDataResponse);
#endif

    vectorUnit.clk(clk);
    vectorUnit.rstn(rstn);
    vectorUnit.serialParamsIn(serialVectorParamsIn);
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

    for (int i = 0; i < 2; i++) {
      vectorUnit.accumulation_buffer_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      vectorUnit.accumulation_buffer_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      vectorUnit.accumulation_buffer_write_request[i](
          accumulation_buffer_vu_write_request[i]);
      vectorUnit.accumulation_buffer_done[i](accumulation_buffer_vu_done[i]);
    }

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
