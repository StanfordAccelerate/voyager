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
#include "test/common/VerificationTypes.h"

#ifndef CFLOAT
#include "Accelerator.h"

SC_MODULE(Harness) {
  sc_clock CCS_INIT_S1(clk);
  sc_signal<bool> CCS_INIT_S1(rstn);

  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      serialMatrixParamsIn);
  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      serial_vector_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  sc_fifo<IC_PORT_TYPE> inputDataResponse_fifo;
  Connections::Combinational<IC_PORT_TYPE> CCS_INIT_S1(inputDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      inputScaleAddressRequest);
  sc_fifo<ac_int<SCALE_DATATYPE::width, false>> inputScaleDataResponse_fifo;
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleDataResponse);
#endif

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> weightDataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightDataResponse);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      weightScaleAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> weightScaleDataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleDataResponse);
#endif

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> biasDataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      biasDataResponse);

#if SUPPORT_MVM
  Connections::Combinational<ac_int<64, false>> CCS_INIT_S1(
      serial_matrix_vector_params_in);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_input_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_input_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_input_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_weight_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_weight_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(
      matrix_vector_weight_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(matrix_vector_bias_req);
  sc_fifo<OC_PORT_TYPE> matrix_vector_bias_resp_fifo;
  Connections::Combinational<OC_PORT_TYPE> CCS_INIT_S1(matrix_vector_bias_resp);

#if SUPPORT_MX
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_input_scale_req);
  sc_fifo<ac_int<MVU_SCALE_PORT_WIDTH, false>>
      matrix_vector_input_scale_resp_fifo;
  Connections::Combinational<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_input_scale_resp);

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(
      matrix_vector_weight_scale_req);
  sc_fifo<ac_int<MVU_SCALE_PORT_WIDTH, false>>
      matrix_vector_weight_scale_resp_fifo;
  Connections::Combinational<ac_int<MVU_SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_vector_weight_scale_resp);
#endif

  Connections::SyncChannel CCS_INIT_S1(matrix_vector_unit_start_signal);
  Connections::SyncChannel CCS_INIT_S1(matrix_vector_unit_done_signal);
#endif

  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch0DataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch1DataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::Combinational<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);
  sc_fifo<ac_int<OC_PORT_WIDTH, false>> vectorFetch2DataResponse_fifo;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);

  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_output);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_address);
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      scalar_output);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      scalar_output_address);

  Connections::SyncChannel CCS_INIT_S1(matrixUnitStartSignal);
  Connections::SyncChannel CCS_INIT_S1(matrixUnitDoneSignal);
  Connections::SyncChannel CCS_INIT_S1(vector_unit_start_signal);
  Connections::SyncChannel CCS_INIT_S1(vector_unit_done_signal);

  Connections::SyncChannel CCS_INIT_S1(tile_done);
  Connections::SyncChannel CCS_INIT_S1(operation_done);

  std::deque<sc_time> start_times;
  std::deque<sc_time> operation_start_times;

  Harness(sc_module_name, std::vector<Operation>, DataLoader *);
  SC_HAS_PROCESS(Harness);

 private:
  std::vector<Operation> operations;
  DataLoader *dataloader;
  AccessCounter *accessCounter;

#ifdef SIM_Accelerator
  CCS_DESIGN(Accelerator) CCS_INIT_S1(accelerator);
#else
  Accelerator CCS_INIT_S1(accelerator);
#endif

  template <int Width>
  void process_read_request(
      Connections::Combinational<MemoryRequest> * request_out,
      sc_fifo<ac_int<Width, false>> * data_fifo);

  template <int Width>
  void send_data_response(
      sc_fifo<ac_int<Width, false>> * data_fifo,
      Connections::Combinational<ac_int<Width, false>> * response);

  template <int Width>
  void process_write_request(
      Connections::Combinational<ac_int<Width, false>> * data_out,
      Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> * address_out);

  void read_input_request();
  void send_input_response();
  void read_weight_request();
  void send_weight_response();
  void read_bias_request();
  void send_bias_response();

#if SUPPORT_MX
  void read_input_scale_request();
  void send_input_scale_response();
  void read_weight_scale_request();
  void send_weight_scale_response();
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

  void read_vector_fetch_0_request();
  void send_vector_fetch_0_response();
  void read_vector_fetch_1_request();
  void send_vector_fetch_1_response();
  void read_vector_fetch_2_request();
  void send_vector_fetch_2_response();

  void store_vector_outputs();
  void store_scale_outputs();

  void reset();
  void param_sender();
  void start_monitor();
  void done_monitor();

  void send_params(const std::deque<BaseParams *> &params);
  void record_start(const std::deque<BaseParams *> &params,
                    const Operation &operation, bool is_first);
  void record_done(const std::deque<BaseParams *> &params,
                   const Operation &operation, int runtime_scale, bool is_last);
};
#endif
