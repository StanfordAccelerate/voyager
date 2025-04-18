#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "MatrixUnit.h"
#include "VectorUnit.h"
#include "mc_scverify.h"

SC_MODULE(Accelerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixUnit CCS_INIT_S1(matrixUnit);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialMatrixParamsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);

  Connections::In<IC_PORT_TYPE> CCS_INIT_S1(inputDataResponse);
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

  Connections::SyncOut CCS_INIT_S1(matrixUnitStartSignal);
  Connections::SyncOut CCS_INIT_S1(matrixUnitDoneSignal);

  Connections::Combinational<ac_int<16, false>>
      accumulation_buffer_vu_read_address[2];
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_vu_read_data[2];
  Connections::Combinational<
      BufferWriteRequest<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>>
      accumulation_buffer_vu_write_request[2];
  Connections::SyncChannel accumulation_buffer_vu_done[2];

#ifdef SIM_VectorUnit
  // clang-format off
  CCS_DESIGN((VectorUnit<VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE, OC_DIMENSION>)) CCS_INIT_S1(vector_unit);
  // clang-format on
#else
  VectorUnit<VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE,
             OC_DIMENSION>
      CCS_INIT_S1(vector_unit);
#endif

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialVectorParamsIn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_request_out);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_request_out);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_request_out);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp_in);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp_in);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp_in);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_3_request_out);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(vector_fetch_3_resp_in);

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

    vector_unit.clk(clk);
    vector_unit.rstn(rstn);
    vector_unit.serial_params_in(serialVectorParamsIn);
    vector_unit.vector_fetch_0_request_out(vector_fetch_0_request_out);
    vector_unit.vector_fetch_0_response_in(vector_fetch_0_resp_in);
    vector_unit.vector_fetch_1_request_out(vector_fetch_1_request_out);
    vector_unit.vector_fetch_1_response_in(vector_fetch_1_resp_in);
    vector_unit.vector_fetch_2_request_out(vector_fetch_2_request_out);
    vector_unit.vector_fetch_2_response_in(vector_fetch_2_resp_in);
    vector_unit.vector_fetch_3_request_out(vector_fetch_3_request_out);
    vector_unit.vector_fetch_3_response_in(vector_fetch_3_resp_in);
    vector_unit.vector_out(vector_output);
    vector_unit.vector_address_out(vector_output_address);
    vector_unit.scale_out(scalar_output);
    vector_unit.scale_address_out(scalar_output_address);
    vector_unit.start(vectorUnitStartSignal);
    vector_unit.done(vectorUnitDoneSignal);

    for (int i = 0; i < 2; i++) {
      vector_unit.accumulation_buffer_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      vector_unit.accumulation_buffer_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      vector_unit.accumulation_buffer_write_request[i](
          accumulation_buffer_vu_write_request[i]);
      vector_unit.accumulation_buffer_done[i](accumulation_buffer_vu_done[i]);
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
