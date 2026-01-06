#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "DwCUnit.h"
#include "MatrixUnit.h"
#if SUPPORT_SPMM
#include "SpMMUnit.h"
#endif
#include "matrix_vector_unit/main.h"
#include "mc_scverify.h"
#include "vector_unit/main.h"

SC_MODULE(Accelerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixUnit CCS_INIT_S1(matrix_unit);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(matrix_unit_params_in);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_unit_input_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_unit_weight_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_unit_bias_req);

  Connections::In<IC_PORT_TYPE> CCS_INIT_S1(matrix_unit_input_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_weight_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_bias_resp);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_unit_input_scale_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_unit_weight_scale_req);

  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      matrix_unit_input_scale_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_weight_scale_resp);
#endif

  Connections::SyncOut CCS_INIT_S1(matrix_unit_start);
  Connections::SyncOut CCS_INIT_S1(matrix_unit_done);

  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(matrix_unit_output);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::Combinational<ac_int<16, false>>
      accumulation_buffer_vu_read_address[2];
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_vu_read_data[2];
  Connections::SyncChannel accumulation_buffer_vu_done[2];
#endif

#if SUPPORT_MVM
  MatrixVectorUnit<InputTypeList, WeightTypeList, SA_INPUT_TYPE, SA_WEIGHT_TYPE,
                   ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE,
                   OC_PORT_WIDTH, MV_UNIT_WIDTH, IC_DIMENSION,
                   VECTOR_UNIT_WIDTH>
      CCS_INIT_S1(matrix_vector_unit);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(matrix_vector_unit_params_in);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_unit_input_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_unit_weight_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(matrix_vector_unit_bias_req);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_input_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_weight_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_bias_resp);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_input_scale_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_weight_scale_req);

  Connections::In<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_input_scale_resp);
  Connections::In<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_weight_scale_resp);
#endif

  Connections::Out<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_scale_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_zp_req);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_scale_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_zp_resp);

  Connections::Combinational<Pack1D<VECTOR_DATATYPE, VECTOR_UNIT_WIDTH>>
      matrix_vector_unit_data;
  Connections::SyncOut CCS_INIT_S1(matrix_vector_unit_start);
  Connections::SyncOut CCS_INIT_S1(matrix_vector_unit_done);
#endif

#if SUPPORT_SPMM
  SpMMUnit<WeightTypeList, VECTOR_DATATYPE, SA_WEIGHT_TYPE, SPMM_META_DATATYPE,
           VECTOR_DATATYPE, SCALE_DATATYPE, OC_PORT_WIDTH, SPMM_UNIT_WIDTH,
           IC_DIMENSION, VECTOR_UNIT_WIDTH>
      CCS_INIT_S1(spmm_unit);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(spmm_unit_params_in);
  Connections::Out<MemoryRequest> CCS_INIT_S1(spmm_unit_input_indptr_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(spmm_unit_input_indices_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(spmm_unit_input_data_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(spmm_unit_weight_req);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_input_indptr_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_input_indices_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_input_data_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_weight_resp);
#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(spmm_unit_weight_scale_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_weight_scale_resp);
#endif
  Connections::Combinational<Pack1D<VECTOR_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      spmm_unit_output);
  Connections::SyncOut CCS_INIT_S1(spmm_unit_start);
  Connections::SyncOut CCS_INIT_S1(spmm_unit_done);
#endif

  VectorUnit<VECTOR_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE,
             VECTOR_UNIT_WIDTH, OC_DIMENSION>
      CCS_INIT_S1(vector_unit);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(vector_unit_params_in);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);

  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_addr);
  Connections::Out<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      mx_scale_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      mx_scale_output_addr);
  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_addr);

  Connections::SyncOut CCS_INIT_S1(vector_unit_start);
  Connections::SyncOut CCS_INIT_S1(vector_unit_done);

#if SUPPORT_DWC
  DwCUnit<DWC_DATATYPE, DWC_DATATYPE, DWC_PSUM, ACCUM_BUFFER_DATATYPE,
          OC_DIMENSION, DWC_DATATYPE>
      CCS_INIT_S1(dwc_unit);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(dwc_unit_params_in);
  Connections::In<ac_int<UNROLLFACTOR * DWC_DATATYPE::width, false>>
      CCS_INIT_S1(dwc_unit_input_resp);
  Connections::In<ac_int<DWC_KERNEL_SIZE * DWC_DATATYPE::width, false>>
      CCS_INIT_S1(dwc_unit_weight_resp);
  Connections::In<ac_int<ACCUM_BUFFER_DATATYPE::width, false>> CCS_INIT_S1(
      dwc_unit_bias_resp);
  Connections::Out<MemoryRequest> CCS_INIT_S1(dwc_unit_input_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(dwc_unit_weight_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(dwc_unit_bias_req);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      dwc_output_address);
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(dwc_output);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(dwc_unit_input_scale_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(dwc_unit_weight_scale_req);

  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      dwc_unit_input_scale_resp);
  Connections::In<ac_int<DWC_KERNEL_SIZE * SCALE_DATATYPE::width, false>>
      CCS_INIT_S1(dwc_unit_weight_scale_resp);
#endif

  Connections::SyncOut CCS_INIT_S1(dwc_unit_start);
  Connections::SyncOut CCS_INIT_S1(dwc_unit_done);
#endif

  SC_CTOR(Accelerator) {
    matrix_unit.clk(clk);
    matrix_unit.rstn(rstn);
    matrix_unit.serial_params_in(matrix_unit_params_in);
    matrix_unit.input_req(matrix_unit_input_req);
    matrix_unit.input_resp(matrix_unit_input_resp);
    matrix_unit.weight_req(matrix_unit_weight_req);
    matrix_unit.weight_resp(matrix_unit_weight_resp);
    matrix_unit.bias_req(matrix_unit_bias_req);
    matrix_unit.bias_resp(matrix_unit_bias_resp);
#if SUPPORT_MX
    matrix_unit.input_scale_req(matrix_unit_input_scale_req);
    matrix_unit.input_scale_resp(matrix_unit_input_scale_resp);
    matrix_unit.weight_scale_req(matrix_unit_weight_scale_req);
    matrix_unit.weight_scale_resp(matrix_unit_weight_scale_resp);
#endif
    matrix_unit.start(matrix_unit_start);
    matrix_unit.done(matrix_unit_done);
    matrix_unit.output_channel(matrix_unit_output);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    for (int i = 0; i < 2; i++) {
      matrix_unit.accumulation_buffer_vu_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      matrix_unit.accumulation_buffer_vu_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      matrix_unit.accumulation_buffer_vu_done[i](
          accumulation_buffer_vu_done[i]);
    }
#endif

#if SUPPORT_MVM
    matrix_vector_unit.clk(clk);
    matrix_vector_unit.rstn(rstn);
    matrix_vector_unit.serial_params_in(matrix_vector_unit_params_in);
    matrix_vector_unit.input_req(matrix_vector_unit_input_req);
    matrix_vector_unit.input_resp(matrix_vector_unit_input_resp);
    matrix_vector_unit.weight_req(matrix_vector_unit_weight_req);
    matrix_vector_unit.weight_resp(matrix_vector_unit_weight_resp);
    matrix_vector_unit.bias_req(matrix_vector_unit_bias_req);
    matrix_vector_unit.bias_resp(matrix_vector_unit_bias_resp);
#if SUPPORT_MX
    matrix_vector_unit.input_scale_req(matrix_vector_unit_input_scale_req);
    matrix_vector_unit.input_scale_resp(matrix_vector_unit_input_scale_resp);
    matrix_vector_unit.weight_scale_req(matrix_vector_unit_weight_scale_req);
    matrix_vector_unit.weight_scale_resp(matrix_vector_unit_weight_scale_resp);
#endif
    matrix_vector_unit.weight_dq_scale_req(
        matrix_vector_unit_weight_dq_scale_req);
    matrix_vector_unit.weight_dq_scale_resp(
        matrix_vector_unit_weight_dq_scale_resp);
    matrix_vector_unit.weight_dq_zero_point_req(
        matrix_vector_unit_weight_dq_zp_req);
    matrix_vector_unit.weight_dq_zero_point_resp(
        matrix_vector_unit_weight_dq_zp_resp);
    matrix_vector_unit.matrix_out(matrix_vector_unit_data);
    matrix_vector_unit.start(matrix_vector_unit_start);
    matrix_vector_unit.done(matrix_vector_unit_done);

    vector_unit.matrix_vector_unit_output(matrix_vector_unit_data);
#endif

#if SUPPORT_SPMM
    spmm_unit.clk(clk);
    spmm_unit.rstn(rstn);
    spmm_unit.serial_params_in(spmm_unit_params_in);
    spmm_unit.input_indptr_req(spmm_unit_input_indptr_req);
    spmm_unit.input_indptr_resp(spmm_unit_input_indptr_resp);
    spmm_unit.input_indices_req(spmm_unit_input_indices_req);
    spmm_unit.input_indices_resp(spmm_unit_input_indices_resp);
    spmm_unit.input_data_req(spmm_unit_input_data_req);
    spmm_unit.input_data_resp(spmm_unit_input_data_resp);
    spmm_unit.weight_req(spmm_unit_weight_req);
    spmm_unit.weight_resp(spmm_unit_weight_resp);
#if SUPPORT_MX
    spmm_unit.weight_scale_req(spmm_unit_weight_scale_req);
    spmm_unit.weight_scale_resp(spmm_unit_weight_scale_resp);
#endif
    spmm_unit.spmm_unit_output(spmm_unit_output);
    spmm_unit.start(spmm_unit_start);
    spmm_unit.done(spmm_unit_done);

    vector_unit.spmm_unit_output(spmm_unit_output);
#endif

    vector_unit.clk(clk);
    vector_unit.rstn(rstn);
    vector_unit.serial_params_in(vector_unit_params_in);
    vector_unit.vector_fetch_0_req(vector_fetch_0_req);
    vector_unit.vector_fetch_0_resp(vector_fetch_0_resp);
    vector_unit.vector_fetch_1_req(vector_fetch_1_req);
    vector_unit.vector_fetch_1_resp(vector_fetch_1_resp);
    vector_unit.vector_fetch_2_req(vector_fetch_2_req);
    vector_unit.vector_fetch_2_resp(vector_fetch_2_resp);
    vector_unit.vector_output_data(vector_output_data);
    vector_unit.vector_output_addr(vector_output_addr);
    vector_unit.mx_scale_output_data(mx_scale_output_data);
    vector_unit.mx_scale_output_addr(mx_scale_output_addr);
    vector_unit.sparse_tensor_output_data(sparse_tensor_output_data);
    vector_unit.sparse_tensor_output_addr(sparse_tensor_output_addr);
    vector_unit.start(vector_unit_start);
    vector_unit.done(vector_unit_done);
    vector_unit.matrix_unit_output(matrix_unit_output);
#if SUPPORT_DWC
    vector_unit.dwc_unit_in(dwc_output);
    vector_unit.dwc_address_in(dwc_output_address);
#endif

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    for (int i = 0; i < 2; i++) {
      vector_unit.accumulation_buffer_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      vector_unit.accumulation_buffer_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      vector_unit.accumulation_buffer_done[i](accumulation_buffer_vu_done[i]);
    }
#endif

#if SUPPORT_DWC
    dwc_unit.clk(clk);
    dwc_unit.rstn(rstn);
    dwc_unit.serial_params_in(dwc_unit_params_in);
    dwc_unit.input_req(dwc_unit_input_req);
    dwc_unit.input_resp(dwc_unit_input_resp);
    dwc_unit.weight_req(dwc_unit_weight_req);
    dwc_unit.weight_resp(dwc_unit_weight_resp);
    dwc_unit.bias_req(dwc_unit_bias_req);
    dwc_unit.bias_resp(dwc_unit_bias_resp);
    dwc_unit.dwc_output_address(dwc_output_address);
    dwc_unit.dwc_output(dwc_output);
    dwc_unit.start(dwc_unit_start);
    dwc_unit.done(dwc_unit_done);
#if SUPPORT_MX
    dwc_unit.input_scale_req(dwc_unit_input_scale_req);
    dwc_unit.input_scale_resp(dwc_unit_input_scale_resp);
    dwc_unit.weight_scale_req(dwc_unit_weight_scale_req);
    dwc_unit.weight_scale_resp(dwc_unit_weight_scale_resp);
#endif
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
