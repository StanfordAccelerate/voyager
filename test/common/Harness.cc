#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include <cassert>

#include "AccelTypes.h"
#include "sysc/kernel/sc_time.h"

#ifndef CFLOAT
#include "test/toolchain/MapOperation.h"

Harness::Harness(sc_module_name name, std::vector<Operation> operations,
                 DataLoader* dataloader)
    : sc_module(name),
      clk("clk", std::stod(std::getenv("CLOCK_PERIOD")), SC_NS, 0.5, 0, SC_NS,
          true),
      operations(operations),
      dataloader(dataloader) {
  accelerator.clk(clk);
  accelerator.rstn(rstn);
  accelerator.matrix_unit_params_in(matrix_unit_params_in);
  accelerator.matrix_unit_input_req(matrix_unit_input_req);
  accelerator.matrix_unit_input_resp(matrix_unit_input_resp);
  accelerator.matrix_unit_weight_req(matrix_unit_weight_req);
  accelerator.matrix_unit_weight_resp(matrix_unit_weight_resp);
  accelerator.matrix_unit_bias_req(matrix_unit_bias_req);
  accelerator.matrix_unit_bias_resp(matrix_unit_bias_resp);
#if SUPPORT_MX
  accelerator.matrix_unit_input_scale_req(matrix_unit_input_scale_req);
  accelerator.matrix_unit_input_scale_resp(matrix_unit_input_scale_resp);
  accelerator.matrix_unit_weight_scale_req(matrix_unit_weight_scale_req);
  accelerator.matrix_unit_weight_scale_resp(matrix_unit_weight_scale_resp);
#endif
  accelerator.matrix_unit_start(matrix_unit_start);
  accelerator.matrix_unit_done(matrix_unit_done);
#if SUPPORT_MVM
  accelerator.matrix_vector_unit_params_in(matrix_vector_unit_params_in);
  accelerator.matrix_vector_unit_input_req(matrix_vector_unit_input_req);
  accelerator.matrix_vector_unit_input_resp(matrix_vector_unit_input_resp);
  accelerator.matrix_vector_unit_weight_req(matrix_vector_unit_weight_req);
  accelerator.matrix_vector_unit_weight_resp(matrix_vector_unit_weight_resp);
  accelerator.matrix_vector_unit_bias_req(matrix_vector_unit_bias_req);
  accelerator.matrix_vector_unit_bias_resp(matrix_vector_unit_bias_resp);
#if SUPPORT_MX
  accelerator.matrix_vector_unit_input_scale_req(
      matrix_vector_unit_input_scale_req);
  accelerator.matrix_vector_unit_input_scale_resp(
      matrix_vector_unit_input_scale_resp);
  accelerator.matrix_vector_unit_weight_scale_req(
      matrix_vector_unit_weight_scale_req);
  accelerator.matrix_vector_unit_weight_scale_resp(
      matrix_vector_unit_weight_scale_resp);
#endif
  accelerator.matrix_vector_unit_weight_dq_scale_req(
      matrix_vector_unit_weight_dq_scale_req);
  accelerator.matrix_vector_unit_weight_dq_scale_resp(
      matrix_vector_unit_weight_dq_scale_resp);
  accelerator.matrix_vector_unit_weight_dq_zp_req(
      matrix_vector_unit_weight_dq_zp_req);
  accelerator.matrix_vector_unit_weight_dq_zp_resp(
      matrix_vector_unit_weight_dq_zp_resp);
  accelerator.matrix_vector_unit_start(matrix_vector_unit_start);
  accelerator.matrix_vector_unit_done(matrix_vector_unit_done);
#endif
#if SUPPORT_SPMM
  accelerator.spmm_unit_params_in(spmm_unit_params_in);
  accelerator.spmm_unit_input_indptr_req(spmm_unit_input_indptr_req);
  accelerator.spmm_unit_input_indptr_resp(spmm_unit_input_indptr_resp);
  accelerator.spmm_unit_input_indices_req(spmm_unit_input_indices_req);
  accelerator.spmm_unit_input_indices_resp(spmm_unit_input_indices_resp);
  accelerator.spmm_unit_input_data_req(spmm_unit_input_data_req);
  accelerator.spmm_unit_input_data_resp(spmm_unit_input_data_resp);
  accelerator.spmm_unit_weight_req(spmm_unit_weight_req);
  accelerator.spmm_unit_weight_resp(spmm_unit_weight_resp);
#if SUPPORT_MX
  accelerator.spmm_unit_weight_scale_req(spmm_unit_weight_scale_req);
  accelerator.spmm_unit_weight_scale_resp(spmm_unit_weight_scale_resp);
#endif
  accelerator.spmm_unit_start(spmm_unit_start);
  accelerator.spmm_unit_done(spmm_unit_done);
#endif
#if SUPPORT_DWC
  accelerator.dwc_unit_params_in(dwc_unit_params_in);
  accelerator.dwc_unit_input_req(dwc_unit_input_req);
  accelerator.dwc_unit_input_resp(dwc_unit_input_resp);
  accelerator.dwc_unit_weight_req(dwc_unit_weight_req);
  accelerator.dwc_unit_weight_resp(dwc_unit_weight_resp);
  accelerator.dwc_unit_bias_req(dwc_unit_bias_req);
  accelerator.dwc_unit_bias_resp(dwc_unit_bias_resp);
#if SUPPORT_MX
  accelerator.dwc_unit_input_scale_req(dwc_unit_input_scale_req);
  accelerator.dwc_unit_input_scale_resp(dwc_unit_input_scale_resp);
  accelerator.dwc_unit_weight_scale_req(dwc_unit_weight_scale_req);
  accelerator.dwc_unit_weight_scale_resp(dwc_unit_weight_scale_resp);
#endif
  accelerator.dwc_unit_start(dwc_unit_start);
  accelerator.dwc_unit_done(dwc_unit_done);
#endif
  accelerator.vector_unit_params_in(vector_unit_params_in);
  accelerator.vector_fetch_0_req(vector_fetch_0_req);
  accelerator.vector_fetch_0_resp(vector_fetch_0_resp);
  accelerator.vector_fetch_1_req(vector_fetch_1_req);
  accelerator.vector_fetch_1_resp(vector_fetch_1_resp);
  accelerator.vector_fetch_2_req(vector_fetch_2_req);
  accelerator.vector_fetch_2_resp(vector_fetch_2_resp);
  accelerator.vector_output_data(vector_output_data);
  accelerator.vector_output_addr(vector_output_addr);
  accelerator.mx_scale_output_data(mx_scale_output_data);
  accelerator.mx_scale_output_addr(mx_scale_output_addr);
  accelerator.sparse_tensor_output_data(sparse_tensor_output_data);
  accelerator.sparse_tensor_output_addr(sparse_tensor_output_addr);
  accelerator.vector_unit_start(vector_unit_start);
  accelerator.vector_unit_done(vector_unit_done);

  SC_CTHREAD(reset, clk);

#define REGISTER_FN(NAME)           \
  SC_THREAD(NAME);                  \
  sensitive << clk.posedge_event(); \
  async_reset_signal_is(rstn, false);

#define REGISTER_IO_FN(NAME)          \
  REGISTER_FN(read_##NAME##_request); \
  REGISTER_FN(send_##NAME##_response);

  REGISTER_IO_FN(matrix_unit_input)
  REGISTER_IO_FN(matrix_unit_weight)
  REGISTER_IO_FN(matrix_unit_bias)
#if SUPPORT_MX
  REGISTER_IO_FN(matrix_unit_input_scale)
  REGISTER_IO_FN(matrix_unit_weight_scale)
#endif

#if SUPPORT_MVM
  REGISTER_IO_FN(matrix_vector_unit_input)
  REGISTER_IO_FN(matrix_vector_unit_weight)
  REGISTER_IO_FN(matrix_vector_unit_bias)
#if SUPPORT_MX
  REGISTER_IO_FN(matrix_vector_unit_input_scale)
  REGISTER_IO_FN(matrix_vector_unit_weight_scale)
#endif
  REGISTER_IO_FN(matrix_vector_unit_weight_dq_scale)
  REGISTER_IO_FN(matrix_vector_unit_weight_dq_zp)
#endif

#if SUPPORT_SPMM
  REGISTER_IO_FN(spmm_unit_input_indptr)
  REGISTER_IO_FN(spmm_unit_input_indices)
  REGISTER_IO_FN(spmm_unit_input_data)
  REGISTER_IO_FN(spmm_unit_weight)
#if SUPPORT_MX
  REGISTER_IO_FN(spmm_unit_weight_scale)
#endif
#endif

#if SUPPORT_DWC
  REGISTER_IO_FN(dwc_unit_input)
  REGISTER_IO_FN(dwc_unit_weight)
  REGISTER_IO_FN(dwc_unit_bias)
#if SUPPORT_MX
  REGISTER_IO_FN(dwc_unit_input_scale)
  REGISTER_IO_FN(dwc_unit_weight_scale)
#endif
#endif

  REGISTER_IO_FN(vector_fetch_0)
  REGISTER_IO_FN(vector_fetch_1)
  REGISTER_IO_FN(vector_fetch_2)

  REGISTER_FN(store_vector_output)
  REGISTER_FN(store_mx_scale_output)
  REGISTER_FN(store_sparse_tensor_output)

  REGISTER_FN(param_sender)
  REGISTER_FN(start_monitor)
  REGISTER_FN(done_monitor)

  access_counter = new AccessCounter();
// do not set access counters for an RTL simulation
#ifndef CCS_DUT_RTL
  accelerator.matrix_unit.input_buffer.access_counter = access_counter;
  accelerator.matrix_unit.weight_buffer.access_counter = access_counter;
#endif
}

template <int width>
void Harness::process_read_request(
    Connections::Combinational<MemoryRequest>* request_out,
    sc_fifo<ac_int<width, false>>* data_fifo) {
  request_out->ResetRead();

  constexpr int num_bytes = width / 8;

  const auto array_memory = (ArrayMemory*)(dataloader->memory);
  const int mem_idx = is_soc_sim() ? 1 : 0;
  char* memory = array_memory->memories[mem_idx];

  wait();

  while (true) {
    MemoryRequest request = request_out->Pop();

    uint64_t base_address = request.address;
    int total_bytes = request.burst_size;
    int num_words = (total_bytes + num_bytes - 1) / num_bytes;

    access_counter->increment(std::string(name()), total_bytes);

    ac_int<width, false> bits;

    for (int i = 0; i < num_words; i++) {
      for (int j = 0; j < num_bytes; j++) {
        uint64_t address = base_address + i * num_bytes + j;
        bits.set_slc(j * 8, static_cast<ac_int<8, false>>(memory[address]));
      }

      data_fifo->write(bits);
    }
  }
}

template <int width>
void Harness::send_data_response(
    sc_fifo<ac_int<width, false>>* data_fifo,
    Connections::Combinational<ac_int<width, false>>* response) {
  response->ResetWrite();

  wait();

  while (true) {
    response->Push(data_fifo->read());
  }
}

template <int width>
void Harness::process_write_request(
    Connections::Combinational<ac_int<width, false>>* data_out,
    Connections::Combinational<ac_int<ADDRESS_WIDTH, false>>* address_out) {
  data_out->ResetRead();
  address_out->ResetRead();

  constexpr int num_bytes = width / 8;

  const auto array_memory = (ArrayMemory*)(dataloader->memory);
  const int mem_idx = is_soc_sim() ? 1 : 0;
  char* memory = array_memory->memories[mem_idx];

  wait();

  while (true) {
    uint64_t address = address_out->Pop();
    auto data = data_out->Pop();

    access_counter->increment(std::string(name()) + "_" + "outputs", num_bytes);

    for (int i = 0; i < num_bytes; i++) {
      memory[address + i] = data.template slc<8>(i * 8);
    }
  }
}

#define DEFINE_IO_FN(NAME)                                \
  void Harness::read_##NAME##_request() {                 \
    process_read_request(&NAME##_req, &NAME##_resp_fifo); \
  }                                                       \
                                                          \
  void Harness::send_##NAME##_response() {                \
    send_data_response(&NAME##_resp_fifo, &NAME##_resp);  \
  }

DEFINE_IO_FN(matrix_unit_input)
DEFINE_IO_FN(matrix_unit_weight)
DEFINE_IO_FN(matrix_unit_bias)
#if SUPPORT_MX
DEFINE_IO_FN(matrix_unit_input_scale)
DEFINE_IO_FN(matrix_unit_weight_scale)
#endif

#if SUPPORT_MVM
DEFINE_IO_FN(matrix_vector_unit_input)
DEFINE_IO_FN(matrix_vector_unit_weight)
DEFINE_IO_FN(matrix_vector_unit_bias)
#if SUPPORT_MX
DEFINE_IO_FN(matrix_vector_unit_input_scale)
DEFINE_IO_FN(matrix_vector_unit_weight_scale)
#endif
DEFINE_IO_FN(matrix_vector_unit_weight_dq_scale)
DEFINE_IO_FN(matrix_vector_unit_weight_dq_zp)
#endif

#if SUPPORT_SPMM
DEFINE_IO_FN(spmm_unit_input_indptr)
DEFINE_IO_FN(spmm_unit_input_indices)
DEFINE_IO_FN(spmm_unit_input_data)
DEFINE_IO_FN(spmm_unit_weight)
#if SUPPORT_MX
DEFINE_IO_FN(spmm_unit_weight_scale)
#endif
#endif

#if SUPPORT_DWC
DEFINE_IO_FN(dwc_unit_input)
DEFINE_IO_FN(dwc_unit_weight)
DEFINE_IO_FN(dwc_unit_bias)
#if SUPPORT_MX
DEFINE_IO_FN(dwc_unit_input_scale)
DEFINE_IO_FN(dwc_unit_weight_scale)
#endif
#endif

DEFINE_IO_FN(vector_fetch_0)
DEFINE_IO_FN(vector_fetch_1)
DEFINE_IO_FN(vector_fetch_2)

void Harness::store_vector_output() {
  process_write_request(&vector_output_data, &vector_output_addr);
}

void Harness::store_mx_scale_output() {
  process_write_request(&mx_scale_output_data, &mx_scale_output_addr);
}

void Harness::store_sparse_tensor_output() {
  process_write_request(&sparse_tensor_output_data, &sparse_tensor_output_addr);
}

void Harness::reset() {
  rstn.write(0);
  wait(5);
  rstn.write(1);
  wait();
}

template <typename T, unsigned int width>
void send_serialized_params(
    T params,
    Connections::Combinational<ac_int<width, false>>* serial_params_in) {
  ac_int<T::width, false> serialized_params;
  vector_to_type(TypeToBits<T>(params), false, &serialized_params);

  // round up to the nearest multiple of width
  ac_int<((T::width + width - 1) / width) * width, false> padded_params =
      serialized_params;

  for (int i = 0; i < padded_params.width / width; i++) {
    serial_params_in->Push(padded_params.template slc<width>(i * width));
  }
}

void Harness::send_params(const std::deque<BaseParams*>& params) {
  for (size_t idx = 0; idx < params.size(); idx++) {
    if (auto* matrix_params = dynamic_cast<MatrixParams*>(params[idx])) {
#if SUPPORT_MVM
      if (matrix_params->is_fc) {
        send_serialized_params<MatrixParams, 64>(*matrix_params,
                                                 &matrix_vector_unit_params_in);
      } else
#endif
#if SUPPORT_SPMM
          if (matrix_params->is_spmm) {
        send_serialized_params<MatrixParams, 64>(*matrix_params,
                                                 &spmm_unit_params_in);
      } else
#endif
      {
        send_serialized_params<MatrixParams, 64>(*matrix_params,
                                                 &matrix_unit_params_in);
      }

      idx++;
    }

#if SUPPORT_DWC
    if (auto* dwc_params = dynamic_cast<DwCParams*>(params[idx])) {
      send_serialized_params<DwCParams, 64>(*dwc_params, &dwc_unit_params_in);
      idx++;
    }
#endif

    if (auto* vector_params = dynamic_cast<VectorParams*>(params[idx])) {
      VectorInstructionConfig* vector_config =
          dynamic_cast<VectorInstructionConfig*>(params[++idx]);

      send_serialized_params<VectorParams, 64>(*vector_params,
                                               &vector_unit_params_in);
      send_serialized_params<VectorInstructionConfig, 64>(
          *vector_config, &vector_unit_params_in);
    }

    // Wait for last operation to finish
    if (idx != params.size() - 1) {
      operation_done.SyncPop();
    }
  }
}

void Harness::record_start(const std::deque<BaseParams*>& params,
                           const Operation& operation, bool is_first) {
  for (size_t idx = 0; idx < params.size(); idx++) {
    BaseParams* base_param = params[idx];
    MatrixParams* matrix_params = dynamic_cast<MatrixParams*>(base_param);
    bool has_matrix_params = matrix_params != nullptr;

    if (has_matrix_params) {
      base_param = params[++idx];
    }

#if SUPPORT_DWC
    DwCParams* dwc_params = dynamic_cast<DwCParams*>(base_param);
    bool has_dwc_params = dwc_params != nullptr;

    if (has_dwc_params) {
      base_param = params[++idx];
    }
#endif

    VectorParams* vector_params = dynamic_cast<VectorParams*>(base_param);
    VectorInstructionConfig* vector_config = nullptr;
    bool has_vector_params = vector_params != nullptr;

    if (has_vector_params) {
      base_param = params[++idx];
      vector_config = dynamic_cast<VectorInstructionConfig*>(base_param);
    }

    // Wait for start signals
    if (has_matrix_params) {
#if SUPPORT_MVM
      if (matrix_params->is_fc) {
        matrix_vector_unit_start.SyncPop();
      } else
#endif
#if SUPPORT_SPMM
          if (matrix_params->is_spmm) {
        spmm_unit_start.SyncPop();
      } else
#endif
      {
        matrix_unit_start.SyncPop();
      }

      auto start = sc_time_stamp();
      start_times.push_back(start);

      if (is_first) {
        operation_start_times.push_back(start);
        is_first = false;
      }

      CCS_LOG("----- Accelerator Layer '" << operation.name
                                          << "' Started. -----");

      if (has_vector_params) {
        vector_unit_start.SyncPop();
      }
    }
#if SUPPORT_DWC
    else if (has_dwc_params) {
      dwc_unit_start.SyncPop();

      auto start = sc_time_stamp();
      start_times.push_back(start);

      if (is_first) {
        operation_start_times.push_back(start);
        is_first = false;
      }

      CCS_LOG("----- Accelerator Layer '" << operation.name
                                          << "' Started. -----");

      if (has_vector_params) {
        vector_unit_start.SyncPop();
      }
    }
#endif
    else if (has_vector_params) {
      vector_unit_start.SyncPop();

      auto start = sc_time_stamp();
      start_times.push_back(start);

      if (is_first) {
        operation_start_times.push_back(start);
        is_first = false;
      }

      CCS_LOG("----- Accelerator Layer '" << operation.name
                                          << "' Started. -----");
    }
  }
}

void Harness::record_done(const std::deque<BaseParams*>& params,
                          const Operation& operation, bool is_last) {
  for (size_t idx = 0; idx < params.size(); idx++) {
    BaseParams* base_param = params[idx];
    MatrixParams* matrix_params = dynamic_cast<MatrixParams*>(base_param);
    bool has_matrix_params = matrix_params != nullptr;

    if (has_matrix_params) {
      base_param = params[++idx];
    }

#if SUPPORT_DWC
    DwCParams* dwc_params = dynamic_cast<DwCParams*>(base_param);
    bool has_dwc_params = dwc_params != nullptr;

    if (has_dwc_params) {
      base_param = params[++idx];
    }
#endif

    VectorParams* vector_params = dynamic_cast<VectorParams*>(base_param);
    VectorInstructionConfig* vector_config = nullptr;
    bool has_vector_params = vector_params != nullptr;

    if (has_vector_params) {
      base_param = params[++idx];
      vector_config = dynamic_cast<VectorInstructionConfig*>(base_param);
    }

    // Wait for done signals
    if (has_matrix_params) {
#if SUPPORT_MVM
      if (matrix_params->is_fc) {
        matrix_vector_unit_done.SyncPop();
      } else
#endif
#if SUPPORT_SPMM
          if (matrix_params->is_spmm) {
        spmm_unit_done.SyncPop();
      } else
#endif
      {
        matrix_unit_done.SyncPop();
      }
    }
#if SUPPORT_DWC
    else if (has_dwc_params) {
      dwc_unit_done.SyncPop();
    }
#endif

    if (has_vector_params) {
      vector_unit_done.SyncPop();
    }

    sc_time end = sc_time_stamp();
    CCS_LOG("----- Accelerator Layer '" << operation.name
                                        << "' Finished. -----");

    sc_time start = start_times.front();
    start_times.pop_front();

    int runtime = end.to_default_time_units() - start.to_default_time_units();
    std::cout << "Runtime: " << runtime << " ns" << std::endl;
    access_counter->print_summary(operation.tiling, operation.has_valid_tiling);

    if (idx != params.size() - 1) {
      operation_done.SyncPush();
    }

    if (is_last && idx == params.size() - 1) {
      auto start = operation_start_times.front();
      operation_start_times.pop_front();

      int total_runtime =
          end.to_default_time_units() - start.to_default_time_units();
      std::cout << "Total Runtime: " << total_runtime << " ns" << std::endl;
    }
  }
}

std::deque<BaseParams*> offset_param_addresses(std::deque<BaseParams*> params,
                                               int offset) {
  std::deque<BaseParams*> new_params;
  for (const auto base_param : params) {
    if (auto* mp = dynamic_cast<MatrixParams*>(base_param)) {
      auto* param = new MatrixParams(*mp);
      param->input_offset += offset;
      param->weight_offset += offset;
      param->bias_offset += offset;
      param->input_scale_offset += offset;
      param->weight_scale_offset += offset;
      param->dq_scale_offset += offset;
      param->dq_zero_point_offset += offset;
#if SUPPORT_SPMM
      param->spmm_indices_offset += offset;
      param->spmm_indptr_offset += offset;
      param->spmm_data_offset += offset;
#endif
      new_params.push_back(param);
    } else if (auto* vp = dynamic_cast<VectorParams*>(base_param)) {
      auto* param = new VectorParams(*vp);
      param->vector_fetch_0_offset += offset;
      param->vector_fetch_1_offset += offset;
      param->vector_fetch_2_offset += offset;
      param->vector_output_offset += offset;
      param->mx_scale_offset += offset;
      param->csr_data_offset += offset;
      param->csr_indices_offset += offset;
      param->csr_indptr_offset += offset;
      new_params.push_back(param);
    } else {
      new_params.push_back(base_param);
    }
  }
  return new_params;
}

void Harness::param_sender() {
  matrix_unit_params_in.ResetWrite();
  vector_unit_params_in.ResetWrite();
#if SUPPORT_MVM
  matrix_vector_unit_params_in.ResetWrite();
#endif
#if SUPPORT_SPMM
  spmm_unit_params_in.ResetWrite();
#endif
#if SUPPORT_DWC
  dwc_unit_params_in.ResetWrite();
#endif
  operation_done.ResetRead();
  tile_done.ResetRead();

  wait();

  for (int i = 0; i < operations.size(); i++) {
    const auto operation = operations.at(i);
    const auto param = operation.param;

    std::deque<BaseParams*> accelerator_params;
    map_operation(operation, accelerator_params);

    if (is_soc_sim()) {
      const int num_tiles = get_tile_count(param);
      const int bank_size = getenv_int("CACHE_SIZE", 8 * 1024 * 1024);

      auto params_offseted =
          offset_param_addresses(accelerator_params, bank_size);

      for (int j = 0; j < num_tiles; j++) {
        if (accelerator_params.size() < 4 && j > 1) {
          tile_done.SyncPop();
        }

        spdlog::debug("Sending tile {} params\n", j);
        send_params(j % 2 == 0 ? accelerator_params : params_offseted);
      }

      // drain out remaining done signals
      if (accelerator_params.size() < 4) {
        tile_done.SyncPop();

        if (num_tiles > 1) {
          tile_done.SyncPop();
        }
      }
    } else {
      send_params(accelerator_params);
    }
  }
}

void Harness::start_monitor() {
  matrix_unit_start.ResetRead();
  vector_unit_start.ResetRead();
#if SUPPORT_MVM
  matrix_vector_unit_start.ResetRead();
#endif
#if SUPPORT_SPMM
  spmm_unit_start.ResetRead();
#endif
#if SUPPORT_DWC
  dwc_unit_start.ResetRead();
#endif

  wait();

  for (int i = 0; i < operations.size(); i++) {
    const auto operation = operations.at(i);
    const auto param = operation.param;

    std::deque<BaseParams*> accelerator_params;
    map_operation(operation, accelerator_params);

    if (is_soc_sim()) {
      const int num_tiles = get_tile_count(param);
      for (int j = 0; j < num_tiles; j++) {
        spdlog::debug("Waiting for tile {} to start\n", j);
        bool is_first = j == 0;
        record_start(accelerator_params, operation, is_first);
      }
    } else {
      record_start(accelerator_params, operation, true);
    }
  }
}

void Harness::done_monitor() {
  matrix_unit_done.ResetRead();
  vector_unit_done.ResetRead();
#if SUPPORT_MVM
  matrix_vector_unit_done.ResetRead();
#endif
#if SUPPORT_SPMM
  spmm_unit_done.ResetRead();
#endif
#if SUPPORT_DWC
  dwc_unit_done.ResetRead();
#endif

  operation_done.ResetWrite();
  tile_done.ResetWrite();

  wait();

  for (int i = 0; i < operations.size(); i++) {
    const auto operation = operations.at(i);
    const auto param = operation.param;

    std::deque<BaseParams*> accelerator_params;
    map_operation(operation, accelerator_params);

    if (is_soc_sim()) {
      const int num_tiles = get_tile_count(param);
      const int bank_size = getenv_int("CACHE_SIZE", 8 * 1024 * 1024);

      dataloader->load_scratchpad(param, 0, 0);

      if (num_tiles > 1) {
        dataloader->load_scratchpad(param, 1, bank_size);
      }

      for (int j = 0; j < num_tiles; j++) {
        spdlog::debug("Waiting for tile {} to finish\n", j);
        bool is_last = j == num_tiles - 1;
        record_done(accelerator_params, operation, is_last);

        int offset = j % 2 == 0 ? 0 : bank_size;
        dataloader->store_scratchpad(param, j, offset);

        if (j + 2 < num_tiles) {
          dataloader->load_scratchpad(param, j + 2, offset);
        }

        if (accelerator_params.size() < 4) {
          tile_done.SyncPush();
        }
      }
    } else {
      record_done(accelerator_params, operation, true);
    }
  }

  sc_stop();
}

#endif

void run_accelerator(std::vector<Operation> operations,
                     DataLoader* dataloader) {
#ifdef CFLOAT
  spdlog::error(
      "The SystemC model does not support the CFloat datatype. Only the gold "
      "model should be used for CFloat.\n");
  std::abort();
#else
  Harness harness("harness", operations, dataloader);
  sc_start();
#endif
}
