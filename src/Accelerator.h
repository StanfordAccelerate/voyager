#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "MatrixUnit.h"
#include "MatrixVectorUnit2.h"
#include "mc_scverify.h"
#include "vector_unit/main.h"

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

  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(matrixUnitOutput);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::Combinational<ac_int<16, false>>
      accumulation_buffer_vu_read_address[2];
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_vu_read_data[2];
  Connections::Combinational<
      BufferWriteRequest<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>>
      accumulation_buffer_vu_write_request[2];
  Connections::SyncChannel accumulation_buffer_vu_done[2];
#endif

#if SUPPORT_MVM
  MatrixVectorUnit<InputTypeList, WeightTypeList, SA_INPUT_TYPE, SA_WEIGHT_TYPE,
                   ACCUM_DATATYPE, VECTOR_DATATYPE, SCALE_DATATYPE,
                   OC_PORT_WIDTH, MV_UNIT_WIDTH, IC_DIMENSION,
                   VECTOR_UNIT_WIDTH>
      CCS_INIT_S1(matrix_vector_unit);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(
      serial_matrix_vector_params_in);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_input_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_weight_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_bias_req);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_input_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_weight_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_bias_resp);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_input_scale_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_weight_scale_req);

  Connections::In<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_input_scale_resp);
  Connections::In<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_weight_scale_resp);
#endif

  Connections::Combinational<Pack1D<VECTOR_DATATYPE, VECTOR_UNIT_WIDTH>>
      matrix_vector_unit_data;
  Connections::SyncOut CCS_INIT_S1(matrix_vector_unit_start_signal);
  Connections::SyncOut CCS_INIT_S1(matrix_vector_unit_done_signal);
#endif

  VectorUnit<VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE,
             VECTOR_UNIT_WIDTH, OC_DIMENSION>
      CCS_INIT_S1(vector_unit);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialVectorParamsIn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_3_req);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(vector_fetch_3_resp);

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
    matrixUnit.matrixUnitOutputChannel(matrixUnitOutput);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    for (int i = 0; i < 2; i++) {
      matrixUnit.accumulation_buffer_vu_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      matrixUnit.accumulation_buffer_vu_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      matrixUnit.accumulation_buffer_vu_write_request[i](
          accumulation_buffer_vu_write_request[i]);
      matrixUnit.accumulation_buffer_vu_done[i](accumulation_buffer_vu_done[i]);
    }
#endif

#if SUPPORT_MX
    matrixUnit.inputScaleAddressRequest(inputScaleAddressRequest);
    matrixUnit.inputScaleDataResponse(inputScaleDataResponse);
    matrixUnit.weightScaleAddressRequest(weightScaleAddressRequest);
    matrixUnit.weightScaleDataResponse(weightScaleDataResponse);
#endif

#if SUPPORT_MVM
    matrix_vector_unit.clk(clk);
    matrix_vector_unit.rstn(rstn);
    matrix_vector_unit.serial_params_in(serial_matrix_vector_params_in);
    matrix_vector_unit.input_req(matrix_vector_input_req);
    matrix_vector_unit.input_resp(matrix_vector_input_resp);
    matrix_vector_unit.weight_req(matrix_vector_weight_req);
    matrix_vector_unit.weight_resp(matrix_vector_weight_resp);
    matrix_vector_unit.bias_req(matrix_vector_bias_req);
    matrix_vector_unit.bias_resp(matrix_vector_bias_resp);
#if SUPPORT_MX
    matrix_vector_unit.input_scale_req(matrix_vector_input_scale_req);
    matrix_vector_unit.input_scale_resp(matrix_vector_input_scale_resp);
    matrix_vector_unit.weight_scale_req(matrix_vector_weight_scale_req);
    matrix_vector_unit.weight_scale_resp(matrix_vector_weight_scale_resp);
#endif
    matrix_vector_unit.matrix_out(matrix_vector_unit_data);
    matrix_vector_unit.start_signal(matrix_vector_unit_start_signal);
    matrix_vector_unit.done_signal(matrix_vector_unit_done_signal);

    vector_unit.matrix_vector_unit_data(matrix_vector_unit_data);
#endif

    vector_unit.clk(clk);
    vector_unit.rstn(rstn);
    vector_unit.serial_params_in(serialVectorParamsIn);
    vector_unit.vector_fetch_0_req(vector_fetch_0_req);
    vector_unit.vector_fetch_0_resp(vector_fetch_0_resp);
    vector_unit.vector_fetch_1_req(vector_fetch_1_req);
    vector_unit.vector_fetch_1_resp(vector_fetch_1_resp);
    vector_unit.vector_fetch_2_req(vector_fetch_2_req);
    vector_unit.vector_fetch_2_resp(vector_fetch_2_resp);
    vector_unit.vector_fetch_3_req(vector_fetch_3_req);
    vector_unit.vector_fetch_3_resp(vector_fetch_3_resp);
    vector_unit.vector_out(vector_output);
    vector_unit.vector_address_out(vector_output_address);
    vector_unit.scale_out(scalar_output);
    vector_unit.scale_address_out(scalar_output_address);
    vector_unit.start(vectorUnitStartSignal);
    vector_unit.done(vectorUnitDoneSignal);
    vector_unit.matrix_unit_output(matrixUnitOutput);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    for (int i = 0; i < 2; i++) {
      vector_unit.accumulation_buffer_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      vector_unit.accumulation_buffer_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      vector_unit.accumulation_buffer_write_request[i](
          accumulation_buffer_vu_write_request[i]);
      vector_unit.accumulation_buffer_done[i](accumulation_buffer_vu_done[i]);
    }
#endif

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
