#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>
#include <systemc.h>

#include <deque>
#include <vector>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "test/common/AccessCounter.h"
#include "test/common/ArrayMemory.h"
#include "test/common/Backend.h"
#include "test/common/Interpreter.h"
#include "test/common/Model.h"
#include "test/common/Utils.h"

#ifndef CFLOAT
#include "Accelerator.h"

struct Harness : public sc_module, public Backend {
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

  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_output_data);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_output_addr);

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

  Harness(sc_module_name name, const Model& model,
          const Model::Selection& selection, MemoryInterface* memory);
  SC_HAS_PROCESS(Harness);

  // Backend: run one accelerator operation on the DUT.
  void execute(const voyager::Operation& op, const ScalarEnv& env) override;

  // Backend: asynchronous commits. The walk pushes a commit's params and moves
  // on -- the units' double-buffered controllers start fetching the next
  // tile's operands while the current tile computes -- and two threads pace
  // the start/done handshakes. wait_semaphore blocks the walk until the
  // completion thread posts, which is the back-pressure that keeps the walk at
  // most one ping-pong slot ahead of the hardware.
  bool is_committed_async() const override { return true; }
  void begin_commit() override;
  void end_commit(bool has_post, const std::string& post_node,
                  int64_t post_slot, int64_t post_amount) override;
  void init_semaphore(const std::string& node, int64_t slot,
                      int64_t value) override;
  void post_semaphore(const std::string& node, int64_t slot,
                      int64_t amount) override;
  void wait_semaphore(const std::string& node, int64_t slot,
                      int64_t amount) override;

 private:
  const Model& model;
  Model::Selection selection;
  MemoryInterface* memory;
  AccessCounter* access_counter;

#ifdef SIM_Accelerator
  CCS_DESIGN(Accelerator) CCS_INIT_S1(accelerator);
#else
  Accelerator CCS_INIT_S1(accelerator);
#endif

  template <int width>
  void process_read_request(
      Connections::Combinational<MemoryRequest>* request_out,
      sc_fifo<ac_int<width, false>>* data_fifo);

  template <int width>
  void send_data_response(
      sc_fifo<ac_int<width, false>>* data_fifo,
      Connections::Combinational<ac_int<width, false>>* response);

  template <int width>
  void process_write_request(
      Connections::Combinational<ac_int<width, false>>* data_out,
      Connections::Combinational<ac_int<ADDRESS_WIDTH, false>>* address_out);

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

  void store_matrix_unit_output();
  void store_vector_output();
  void store_mx_scale_output();
  void store_sparse_tensor_output();

  void reset();

  // One params group in flight: which units it started, and what to do when
  // it retires. A group with no units is a commit's post token -- it rides
  // the queues behind the commit's groups so the post fires only after every
  // group ahead of it has retired.
  struct InvocationGroup {
    bool matrix = false;
    bool matrix_vector = false;
    bool spmm = false;
    bool dwc = false;
    bool vector = false;
    // Set on the first / last group of an operation, for the per-op logging
    // and access summary. start_time is filled in by release_starts on the
    // op_end group.
    const voyager::Operation* op_begin = nullptr;
    const voyager::Operation* op_end = nullptr;
    sc_time start_time;
    bool has_post = false;
    std::string post_node;
    int64_t post_slot = 0;
    int64_t post_amount = 0;
  };

  // A counting semaphore the walk can block on while another thread posts.
  struct SyncSemaphore {
    int64_t count = 0;
    sc_event posted;
  };

  std::map<std::pair<std::string, int64_t>, SyncSemaphore> sync_semaphores;
  std::deque<InvocationGroup> start_queue;
  std::deque<InvocationGroup> done_queue;
  sc_event start_pushed;
  sc_event done_pushed;
  sc_event group_retired;
  sc_time op_release_time;
  long pending_groups = 0;
  bool in_commit = false;

  // Serialize one operation's params to the units and queue its invocation
  // groups; does not wait for them to execute.
  void dispatch_params(const voyager::Operation& op,
                       const std::deque<BaseParams*>& params);

  // Block the walk until every queued group has retired.
  void drain();

  // The thread that drives the DUT: it walks the graph exactly as the gold
  // model does -- same loops, same conditionals, same DMAs -- and dispatches
  // each accelerator operation through execute() above. Two companion threads
  // pace the handshakes so consecutive tiles overlap in the datapath.
  void run_walker();
  void release_starts();
  void retire_dones();
};

#endif

// Declared outside the CFLOAT guard to mirror the definition: for CFloat it
// compiles (the harness itself cannot) and aborts if actually called.
void run_accelerator(const Model& model, const Model::Selection& selection,
                     MemoryInterface* memory);
