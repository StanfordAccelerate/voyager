#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>
#include <systemc.h>

#include <deque>
#include <vector>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "test/common/AccessCounter.h"
#include "test/common/DataLoader.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"

#ifndef CFLOAT
#include "Accelerator.h"

SC_MODULE(Harness) {
  sc_clock CCS_INIT_S1(clk);
  sc_signal<bool> CCS_INIT_S1(rstn);

  //----------------------------------------------------------
  // MATRIX UNIT CONNECTIONS
  //----------------------------------------------------------

  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      matrix_unit_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(matrix_unit_input_req);
  sc_fifo<IC_PORT_TYPE> matrix_unit_input_resp_fifo;
  Connections::Combinational<IC_PORT_TYPE> CCS_INIT_S1(matrix_unit_input_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(matrix_unit_weight_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> matrix_unit_weight_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_weight_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(matrix_unit_bias_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> matrix_unit_bias_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_bias_resp);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_unit_input_scale_req);
  sc_fifo<ac_int<SCALE_DATATYPE::width, false>>
      matrix_unit_input_scale_resp_fifo;
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      matrix_unit_input_scale_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_unit_weight_scale_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> matrix_unit_weight_scale_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_weight_scale_resp);
#endif

  Connections::SyncChannel CCS_INIT_S1(matrix_unit_start);
  Connections::SyncChannel CCS_INIT_S1(matrix_unit_done);

  //----------------------------------------------------------
  // MATRIX VECTOR UNIT CONNECTIONS
  //----------------------------------------------------------

#if SUPPORT_MVM
  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      matrix_vector_unit_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_input_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_unit_input_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_unit_input_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_weight_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_unit_weight_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_unit_weight_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_bias_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_unit_bias_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_unit_bias_resp);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_input_scale_req);
  sc_fifo<ac_int<MVU_SCALE_PORT_WIDTH, false>>
      matrix_vector_unit_input_scale_resp_fifo;
  Connections::Combinational<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_input_scale_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_weight_scale_req);
  sc_fifo<ac_int<MVU_SCALE_PORT_WIDTH, false>>
      matrix_vector_unit_weight_scale_resp_fifo;
  Connections::Combinational<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_unit_weight_scale_resp);
#endif

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_scale_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_unit_weight_dq_scale_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_scale_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_zp_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_unit_weight_dq_zp_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_unit_weight_dq_zp_resp);

  Connections::SyncChannel CCS_INIT_S1(matrix_vector_unit_start);
  Connections::SyncChannel CCS_INIT_S1(matrix_vector_unit_done);
#endif

  //----------------------------------------------------------
  // SPMM UNIT CONNECTIONS
  //----------------------------------------------------------

#if SUPPORT_SPMM
  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      spmm_unit_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      spmm_unit_input_indptr_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> spmm_unit_input_indptr_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_input_indptr_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      spmm_unit_input_indices_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> spmm_unit_input_indices_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_input_indices_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      spmm_unit_input_data_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> spmm_unit_input_data_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_input_data_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(spmm_unit_weight_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> spmm_unit_weight_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_weight_resp);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      spmm_unit_weight_scale_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> spmm_unit_weight_scale_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      spmm_unit_weight_scale_resp);
#endif

  Connections::SyncChannel CCS_INIT_S1(spmm_unit_start);
  Connections::SyncChannel CCS_INIT_S1(spmm_unit_done);
#endif

  //----------------------------------------------------------
  // DWC UNIT CONNECTIONS
  //----------------------------------------------------------

#if SUPPORT_DWC
  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(dwc_unit_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(dwc_unit_input_req);
  sc_fifo<ac_int<UNROLLFACTOR * DWC_DATATYPE::width, false>>
      dwc_unit_input_resp_fifo;
  Connections::Combinational<ac_int<UNROLLFACTOR * DWC_DATATYPE::width, false>>
      CCS_INIT_S1(dwc_unit_input_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(dwc_unit_weight_req);
  sc_fifo<ac_int<DWC_KERNEL_SIZE * DWC_DATATYPE::width, false>>
      dwc_unit_weight_resp_fifo;
  Connections::Combinational<
      ac_int<DWC_KERNEL_SIZE * DWC_DATATYPE::width, false>>
      CCS_INIT_S1(dwc_unit_weight_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(dwc_unit_bias_req);
  sc_fifo<ac_int<ACCUM_BUFFER_DATATYPE::width, false>> dwc_unit_bias_resp_fifo;
  Connections::Combinational<ac_int<ACCUM_BUFFER_DATATYPE::width, false>>
      CCS_INIT_S1(dwc_unit_bias_resp);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      dwc_unit_input_scale_req);
  sc_fifo<ac_int<SCALE_DATATYPE::width, false>> dwc_unit_input_scale_resp_fifo;
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      dwc_unit_input_scale_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      dwc_unit_weight_scale_req);
  sc_fifo<ac_int<DWC_KERNEL_SIZE * SCALE_DATATYPE::width, false>>
      dwc_unit_weight_scale_resp_fifo;
  Connections::Combinational<
      ac_int<DWC_KERNEL_SIZE * SCALE_DATATYPE::width, false>>
      CCS_INIT_S1(dwc_unit_weight_scale_resp);
#endif

  Connections::SyncChannel CCS_INIT_S1(dwc_unit_start);
  Connections::SyncChannel CCS_INIT_S1(dwc_unit_done);
#endif

  //----------------------------------------------------------
  // VECTOR UNIT CONNECTIONS
  //----------------------------------------------------------

  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      vector_unit_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vector_fetch_0_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vector_fetch_1_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vector_fetch_2_resp_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);

  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_output_data);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_addr);
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      mx_scale_output_data);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      mx_scale_output_addr);
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_data);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_addr);

  Connections::SyncChannel CCS_INIT_S1(vector_unit_start);
  Connections::SyncChannel CCS_INIT_S1(vector_unit_done);

  Connections::SyncChannel CCS_INIT_S1(tile_done);
  Connections::SyncChannel CCS_INIT_S1(operation_done);

  std::deque<sc_time> start_times;
  std::deque<sc_time> operation_start_times;

  Harness(sc_module_name, std::vector<Operation>, DataLoader*);
  SC_HAS_PROCESS(Harness);

 private:
  std::vector<Operation> operations;
  DataLoader* dataloader;
  AccessCounter* access_counter;

#ifdef SIM_Accelerator
  CCS_DESIGN(Accelerator) CCS_INIT_S1(accelerator);
#else
  Accelerator CCS_INIT_S1(accelerator);
#endif

  template <int width>
  void process_read_request(
      Connections::Combinational<MemoryRequest> * request_out,
      sc_fifo<ac_int<width, false>> * data_fifo);

  template <int width>
  void send_data_response(
      sc_fifo<ac_int<width, false>> * data_fifo,
      Connections::Combinational<ac_int<width, false>> * response);

  template <int width>
  void process_write_request(
      Connections::Combinational<ac_int<width, false>> * data_out,
      Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> * address_out);

  void read_matrix_unit_input_request();
  void send_matrix_unit_input_response();
  void read_matrix_unit_weight_request();
  void send_matrix_unit_weight_response();
  void read_matrix_unit_bias_request();
  void send_matrix_unit_bias_response();
#if SUPPORT_MX
  void read_matrix_unit_input_scale_request();
  void send_matrix_unit_input_scale_response();
  void read_matrix_unit_weight_scale_request();
  void send_matrix_unit_weight_scale_response();
#endif

  void read_matrix_vector_unit_input_request();
  void send_matrix_vector_unit_input_response();
  void read_matrix_vector_unit_weight_request();
  void send_matrix_vector_unit_weight_response();
  void read_matrix_vector_unit_bias_request();
  void send_matrix_vector_unit_bias_response();
#if SUPPORT_MX
  void read_matrix_vector_unit_input_scale_request();
  void send_matrix_vector_unit_input_scale_response();
  void read_matrix_vector_unit_weight_scale_request();
  void send_matrix_vector_unit_weight_scale_response();
#endif
  void read_matrix_vector_unit_weight_dq_scale_request();
  void send_matrix_vector_unit_weight_dq_scale_response();
  void read_matrix_vector_unit_weight_dq_zp_request();
  void send_matrix_vector_unit_weight_dq_zp_response();

#if SUPPORT_SPMM
  void read_spmm_unit_input_indptr_request();
  void send_spmm_unit_input_indptr_response();
  void read_spmm_unit_input_indices_request();
  void send_spmm_unit_input_indices_response();
  void read_spmm_unit_input_data_request();
  void send_spmm_unit_input_data_response();
  void read_spmm_unit_weight_request();
  void send_spmm_unit_weight_response();
#if SUPPORT_MX
  void read_spmm_unit_weight_scale_request();
  void send_spmm_unit_weight_scale_response();
#endif
#endif

#if SUPPORT_DWC
  void read_dwc_unit_input_request();
  void send_dwc_unit_input_response();
  void read_dwc_unit_weight_request();
  void send_dwc_unit_weight_response();
  void read_dwc_unit_bias_request();
  void send_dwc_unit_bias_response();
#if SUPPORT_MX
  void read_dwc_unit_input_scale_request();
  void send_dwc_unit_input_scale_response();
  void read_dwc_unit_weight_scale_request();
  void send_dwc_unit_weight_scale_response();
#endif
#endif

  void read_vector_fetch_0_request();
  void send_vector_fetch_0_response();
  void read_vector_fetch_1_request();
  void send_vector_fetch_1_response();
  void read_vector_fetch_2_request();
  void send_vector_fetch_2_response();

  void store_vector_output();
  void store_mx_scale_output();
  void store_sparse_tensor_output();

  void reset();
  void param_sender();
  void start_monitor();
  void done_monitor();

  void send_params(const std::deque<BaseParams*>& params);
  void record_start(const std::deque<BaseParams*>& params,
                    const Operation& operation, bool is_first);
  void record_done(const std::deque<BaseParams*>& params,
                   const Operation& operation, bool is_last);
};
#endif
