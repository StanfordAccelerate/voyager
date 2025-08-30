#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include <cassert>

#include "AccelTypes.h"
#include "sysc/kernel/sc_time.h"

#ifndef CFLOAT
#include "test/toolchain/MapOperation.h"

Harness::Harness(sc_module_name name, std::vector<Operation> operations,
                 DataLoader *dataloader)
    : sc_module(name),
      clk("clk", std::stod(std::getenv("CLOCK_PERIOD")), SC_NS, 0.5, 0, SC_NS,
          true),
      operations(operations),
      dataloader(dataloader),
      inputDataResponse_fifo("inputDataResponse_fifo", 1024),
      weightDataResponse_fifo("weightDataResponse_fifo", 1024),
      biasDataResponse_fifo("biasDataResponse_fifo", 1024),
      vectorFetch0DataResponse_fifo("vectorFetch0DataResponse_fifo", 1024),
      vectorFetch1DataResponse_fifo("vectorFetch1DataResponse_fifo", 1024),
      vectorFetch2DataResponse_fifo("vectorFetch2DataResponse_fifo", 1024)
#if SUPPORT_MVM
      ,
      matrix_vector_input_resp_fifo("matrix_vector_input_resp_fifo", 1024),
      matrix_vector_weight_resp_fifo("matrix_vector_weight_resp_fifo", 1024),
      matrix_vector_bias_resp_fifo("matrix_vector_bias_resp_fifo", 1024)
#endif
{
  accelerator.clk(clk);
  accelerator.rstn(rstn);
  accelerator.serialMatrixParamsIn(serialMatrixParamsIn);
  accelerator.inputAddressRequest(inputAddressRequest);
  accelerator.inputDataResponse(inputDataResponse);
  accelerator.weightAddressRequest(weightAddressRequest);
  accelerator.weightDataResponse(weightDataResponse);
  accelerator.biasAddressRequest(biasAddressRequest);
  accelerator.biasDataResponse(biasDataResponse);
#if SUPPORT_MVM
  accelerator.serial_matrix_vector_params_in(serial_matrix_vector_params_in);
  accelerator.matrix_vector_input_req(matrix_vector_input_req);
  accelerator.matrix_vector_input_resp(matrix_vector_input_resp);
  accelerator.matrix_vector_weight_req(matrix_vector_weight_req);
  accelerator.matrix_vector_weight_resp(matrix_vector_weight_resp);
  accelerator.matrix_vector_bias_req(matrix_vector_bias_req);
  accelerator.matrix_vector_bias_resp(matrix_vector_bias_resp);
#if SUPPORT_MX
  accelerator.matrix_vector_input_scale_req(matrix_vector_input_scale_req);
  accelerator.matrix_vector_input_scale_resp(matrix_vector_input_scale_resp);
  accelerator.matrix_vector_weight_scale_req(matrix_vector_weight_scale_req);
  accelerator.matrix_vector_weight_scale_resp(matrix_vector_weight_scale_resp);
#endif
  accelerator.matrix_vector_unit_start_signal(matrix_vector_unit_start_signal);
  accelerator.matrix_vector_unit_done_signal(matrix_vector_unit_done_signal);
#endif
  accelerator.serial_vector_params_in(serial_vector_params_in);
  accelerator.vector_fetch_0_req(vector_fetch_0_req);
  accelerator.vector_fetch_0_resp(vector_fetch_0_resp);
  accelerator.vector_fetch_1_req(vector_fetch_1_req);
  accelerator.vector_fetch_1_resp(vector_fetch_1_resp);
  accelerator.vector_fetch_2_req(vector_fetch_2_req);
  accelerator.vector_fetch_2_resp(vector_fetch_2_resp);
  accelerator.vector_output(vector_output);
  accelerator.vector_output_address(vector_output_address);
  accelerator.scalar_output(scalar_output);
  accelerator.scalar_output_address(scalar_output_address);

  accelerator.matrixUnitStartSignal(matrixUnitStartSignal);
  accelerator.matrixUnitDoneSignal(matrixUnitDoneSignal);
  accelerator.vector_unit_start_signal(vector_unit_start_signal);
  accelerator.vector_unit_done_signal(vector_unit_done_signal);

#if SUPPORT_MX
  accelerator.inputScaleAddressRequest(inputScaleAddressRequest);
  accelerator.inputScaleDataResponse(inputScaleDataResponse);
  accelerator.weightScaleAddressRequest(weightScaleAddressRequest);
  accelerator.weightScaleDataResponse(weightScaleDataResponse);
#endif

  SC_CTHREAD(reset, clk);

  SC_THREAD(read_input_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_input_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_weight_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_weight_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_bias_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_bias_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MX
  SC_THREAD(read_input_scale_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_input_scale_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_weight_scale_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_weight_scale_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
#endif

#if SUPPORT_MVM
  SC_THREAD(read_matrix_vector_unit_input_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_matrix_vector_unit_input_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_matrix_vector_unit_weight_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_matrix_vector_unit_weight_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_matrix_vector_unit_bias_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_matrix_vector_unit_bias_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MX
  SC_THREAD(read_matrix_vector_unit_input_scale_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_matrix_vector_unit_input_scale_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_matrix_vector_unit_weight_scale_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_matrix_vector_unit_weight_scale_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
#endif
#endif

  SC_THREAD(read_vector_fetch_0_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_vector_fetch_0_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_vector_fetch_1_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_vector_fetch_1_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(read_vector_fetch_2_request);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(send_vector_fetch_2_response);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(store_vector_outputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(store_scale_outputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(param_sender);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(start_monitor);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(done_monitor);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  accessCounter = new AccessCounter();
// do not set access counters for an RTL simulation
#ifndef CCS_DUT_RTL
  accelerator.matrixUnit.inputBuffer.accessCounter = accessCounter;
  accelerator.matrixUnit.weightBuffer.accessCounter = accessCounter;
#endif
}

template <int Width>
void Harness::process_read_request(
    Connections::Combinational<MemoryRequest> *request_out,
    sc_fifo<ac_int<Width, false>> *data_fifo) {
  request_out->ResetRead();

  constexpr int num_bytes = Width / 8;

  const auto array_memory = (ArrayMemory *)(dataloader->memory_interface);
  const int mem_idx = is_soc_sim() ? 1 : 0;
  char *memory = array_memory->memories[mem_idx];

  wait();

  while (true) {
    MemoryRequest request = request_out->Pop();

    uint64_t base_address = request.address;
    int total_bytes = request.burst_size;
    int num_words = (total_bytes + num_bytes - 1) / num_bytes;

    accessCounter->increment(std::string(name()), total_bytes);

    ac_int<Width, false> bits;

    for (int i = 0; i < num_words; i++) {
      for (int j = 0; j < num_bytes; j++) {
        uint64_t address = base_address + i * num_bytes + j;
        DLOG("read addr: " << address << " data: " << memory[address]
                           << std::endl);
        bits.set_slc(j * 8, static_cast<ac_int<8, false>>(memory[address]));
      }

      data_fifo->write(bits);
    }
  }
}

template <int Width>
void Harness::send_data_response(
    sc_fifo<ac_int<Width, false>> *data_fifo,
    Connections::Combinational<ac_int<Width, false>> *response) {
  response->ResetWrite();

  wait();

  while (true) {
    response->Push(data_fifo->read());
  }
}

template <int Width>
void Harness::process_write_request(
    Connections::Combinational<ac_int<Width, false>> *data_out,
    Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> *address_out) {
  data_out->ResetRead();
  address_out->ResetRead();

  constexpr int num_bytes = Width / 8;

  const auto array_memory = (ArrayMemory *)(dataloader->memory_interface);
  const int mem_idx = is_soc_sim() ? 1 : 0;
  char *memory = array_memory->memories[mem_idx];

  wait();

  while (true) {
    uint64_t address = address_out->Pop();
    auto data = data_out->Pop();
    DLOG("write address: " << address << " data: " << data);

    accessCounter->increment(std::string(name()) + "_" + "outputs", num_bytes);

    for (int i = 0; i < num_bytes; i++) {
      memory[address + i] = data.template slc<8>(i * 8);
    }
  }
}

void Harness::read_input_request() {
  process_read_request(&inputAddressRequest, &inputDataResponse_fifo);
}

void Harness::send_input_response() {
  send_data_response(&inputDataResponse_fifo, &inputDataResponse);
}

void Harness::read_weight_request() {
  process_read_request(&weightAddressRequest, &weightDataResponse_fifo);
}

void Harness::send_weight_response() {
  send_data_response(&weightDataResponse_fifo, &weightDataResponse);
}

void Harness::read_bias_request() {
  process_read_request(&biasAddressRequest, &biasDataResponse_fifo);
}

void Harness::send_bias_response() {
  send_data_response(&biasDataResponse_fifo, &biasDataResponse);
}

#if SUPPORT_MX
void Harness::read_input_scale_request() {
  process_read_request(&inputScaleAddressRequest, &inputScaleDataResponse_fifo);
}

void Harness::send_input_scale_response() {
  send_data_response(&inputScaleDataResponse_fifo, &inputScaleDataResponse);
}

void Harness::read_weight_scale_request() {
  process_read_request(&weightScaleAddressRequest,
                       &weightScaleDataResponse_fifo);
}

void Harness::send_weight_scale_response() {
  send_data_response(&weightScaleDataResponse_fifo, &weightScaleDataResponse);
}
#endif

#if SUPPORT_MVM
void Harness::read_matrix_vector_unit_input_request() {
  process_read_request(&matrix_vector_input_req,
                       &matrix_vector_input_resp_fifo);
}

void Harness::send_matrix_vector_unit_input_response() {
  send_data_response(&matrix_vector_input_resp_fifo, &matrix_vector_input_resp);
}

void Harness::read_matrix_vector_unit_weight_request() {
  process_read_request(&matrix_vector_weight_req,
                       &matrix_vector_weight_resp_fifo);
}

void Harness::send_matrix_vector_unit_weight_response() {
  send_data_response(&matrix_vector_weight_resp_fifo,
                     &matrix_vector_weight_resp);
}

void Harness::read_matrix_vector_unit_bias_request() {
  process_read_request(&matrix_vector_bias_req, &matrix_vector_bias_resp_fifo);
}

void Harness::send_matrix_vector_unit_bias_response() {
  send_data_response(&matrix_vector_bias_resp_fifo, &matrix_vector_bias_resp);
}

#if SUPPORT_MX
void Harness::read_matrix_vector_unit_input_scale_request() {
  process_read_request(&matrix_vector_input_scale_req,
                       &matrix_vector_input_scale_resp_fifo);
}

void Harness::send_matrix_vector_unit_input_scale_response() {
  send_data_response(&matrix_vector_input_scale_resp_fifo,
                     &matrix_vector_input_scale_resp);
}

void Harness::read_matrix_vector_unit_weight_scale_request() {
  process_read_request(&matrix_vector_weight_scale_req,
                       &matrix_vector_weight_scale_resp_fifo);
}

void Harness::send_matrix_vector_unit_weight_scale_response() {
  send_data_response(&matrix_vector_weight_scale_resp_fifo,
                     &matrix_vector_weight_scale_resp);
}
#endif
#endif

void Harness::read_vector_fetch_0_request() {
  process_read_request(&vector_fetch_0_req, &vectorFetch0DataResponse_fifo);
}

void Harness::send_vector_fetch_0_response() {
  send_data_response(&vectorFetch0DataResponse_fifo, &vector_fetch_0_resp);
}

void Harness::read_vector_fetch_1_request() {
  process_read_request(&vector_fetch_1_req, &vectorFetch1DataResponse_fifo);
}

void Harness::send_vector_fetch_1_response() {
  send_data_response(&vectorFetch1DataResponse_fifo, &vector_fetch_1_resp);
}

void Harness::read_vector_fetch_2_request() {
  process_read_request(&vector_fetch_2_req, &vectorFetch2DataResponse_fifo);
}

void Harness::send_vector_fetch_2_response() {
  send_data_response(&vectorFetch2DataResponse_fifo, &vector_fetch_2_resp);
}

void Harness::store_vector_outputs() {
  process_write_request(&vector_output, &vector_output_address);
}

void Harness::store_scale_outputs() {
  process_write_request(&scalar_output, &scalar_output_address);
}

void Harness::reset() {
  rstn.write(0);
  wait(5);
  rstn.write(1);
  wait();
}

template <typename T, unsigned int interfaceWidth>
void send_serialized_params(
    T params,
    Connections::Combinational<ac_int<interfaceWidth, false>> *serialParamsIn) {
  ac_int<T::width, false> serializedParam;
  vector_to_type(TypeToBits<T>(params), false, &serializedParam);

  // round up to the nearest multiple of interfaceWidth
  ac_int<((T::width + interfaceWidth - 1) / interfaceWidth) * interfaceWidth,
         false>
      serializedParamsPadded = serializedParam;

  for (int i = 0; i < serializedParamsPadded.width / interfaceWidth; i++) {
    serialParamsIn->Push(serializedParamsPadded.template slc<interfaceWidth>(
        i * interfaceWidth));
  }
}

void Harness::send_params(const std::deque<BaseParams *> &params) {
  for (size_t idx = 0; idx < params.size(); idx++) {
    if (auto *matrix_params = dynamic_cast<MatrixParams *>(params[idx])) {
#if SUPPORT_MVM
      if (matrix_params->is_fc) {
        send_serialized_params<MatrixParams, 64>(
            *matrix_params, &serial_matrix_vector_params_in);
      } else
#endif
      {
        send_serialized_params<MatrixParams, 64>(*matrix_params,
                                                 &serialMatrixParamsIn);
      }

      idx++;
    }

    if (auto *vector_params = dynamic_cast<VectorParams *>(params[idx])) {
      VectorInstructionConfig *vector_config =
          dynamic_cast<VectorInstructionConfig *>(params[++idx]);

      send_serialized_params<VectorParams, 64>(*vector_params,
                                               &serial_vector_params_in);
      send_serialized_params<VectorInstructionConfig, 64>(
          *vector_config, &serial_vector_params_in);
    }
  }
}

void Harness::record_start(const std::deque<BaseParams *> &params,
                           const Operation &operation, bool is_first) {
  for (size_t idx = 0; idx < params.size(); idx++) {
    BaseParams *base_param = params[idx];
    MatrixParams *matrix_params = dynamic_cast<MatrixParams *>(base_param);
    bool has_matrix_params = matrix_params != nullptr;

    if (has_matrix_params) {
      base_param = params[++idx];
    }

    VectorParams *vector_params = dynamic_cast<VectorParams *>(base_param);
    VectorInstructionConfig *vector_config = nullptr;
    bool has_vector_params = vector_params != nullptr;

    if (has_vector_params) {
      base_param = params[++idx];
      vector_config = dynamic_cast<VectorInstructionConfig *>(base_param);
    }

    // Wait for start signals
    if (has_matrix_params) {
#if SUPPORT_MVM
      if (matrix_params->is_fc) {
        matrix_vector_unit_start_signal.SyncPop();
      } else
#endif
      {
        matrixUnitStartSignal.SyncPop();
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
        vector_unit_start_signal.SyncPop();
      }
    } else if (has_vector_params) {
      vector_unit_start_signal.SyncPop();

      auto start = sc_time_stamp();
      start_times.push_back(start);

      if (is_first) {
        operation_start_times.push_back(start);
        is_first = false;
      }

      CCS_LOG("----- Accelerator Layer '" << operation.name
                                          << "' Started. -----");
    }

    // Wait for last operation to finish
    if (idx != params.size() - 1) {
      operation_done.SyncPop();
    }
  }
}

void Harness::record_done(const std::deque<BaseParams *> &params,
                          const Operation &operation, int runtime_scale,
                          bool is_last) {
  for (size_t idx = 0; idx < params.size(); idx++) {
    BaseParams *base_param = params[idx];
    MatrixParams *matrix_params = dynamic_cast<MatrixParams *>(base_param);
    bool has_matrix_params = matrix_params != nullptr;

    if (has_matrix_params) {
      base_param = params[++idx];
    }

    VectorParams *vector_params = dynamic_cast<VectorParams *>(base_param);
    VectorInstructionConfig *vector_config = nullptr;
    bool has_vector_params = vector_params != nullptr;

    if (has_vector_params) {
      base_param = params[++idx];
      vector_config = dynamic_cast<VectorInstructionConfig *>(base_param);
    }

    // Wait for done signals
    if (has_matrix_params) {
#if SUPPORT_MVM
      if (matrix_params->is_fc) {
        matrix_vector_unit_done_signal.SyncPop();
      } else
#endif
      {
        matrixUnitDoneSignal.SyncPop();
      }
    }

    if (has_vector_params) {
      vector_unit_done_signal.SyncPop();
    }

    sc_time end = sc_time_stamp();
    CCS_LOG("----- Accelerator Layer '" << operation.name
                                        << "' Finished. -----");

    sc_time start = start_times.front();
    start_times.pop_front();

    int runtime = runtime_scale * int(end.to_default_time_units() -
                                      start.to_default_time_units());

    std::cout << "Runtime: " << runtime << " ns" << std::endl;
    accessCounter->print_summary(operation.tiling, operation.has_valid_tiling);

    if (idx != params.size() - 1) {
      operation_done.SyncPush();
    }

    if (is_last && idx == params.size() - 1) {
      auto start = operation_start_times.front();
      operation_start_times.pop_front();

      int total_runtime = runtime_scale * int(end.to_default_time_units() -
                                              start.to_default_time_units());
      std::cout << "Total Runtime: " << total_runtime << " ns" << std::endl;
    }
  }
}

std::deque<BaseParams *> offset_param_addresses(std::deque<BaseParams *> params,
                                                int offset) {
  std::deque<BaseParams *> new_params;
  for (const auto base_param : params) {
    if (auto *mp = dynamic_cast<MatrixParams *>(base_param)) {
      auto *param = new MatrixParams(*mp);
      param->input_offset += offset;
      param->weight_offset += offset;
      param->bias_offset += offset;
      param->input_scale_offset += offset;
      param->weight_scale_offset += offset;
      new_params.push_back(param);
    } else if (auto *vp = dynamic_cast<VectorParams *>(base_param)) {
      auto *param = new VectorParams(*vp);
      param->vector_fetch_0_offset += offset;
      param->vector_fetch_1_offset += offset;
      param->vector_fetch_2_offset += offset;
      param->vector_output_offset += offset;
      param->mx_scale_offset += offset;
      new_params.push_back(param);
    } else {
      new_params.push_back(base_param);
    }
  }
  return new_params;
}

void Harness::param_sender() {
  serialMatrixParamsIn.ResetWrite();
  serial_vector_params_in.ResetWrite();
#if SUPPORT_MVM
  serial_matrix_vector_params_in.ResetWrite();
#endif
  tile_done.ResetRead();

  wait();

  for (int i = 0; i < operations.size(); i++) {
    const auto operation = operations.at(i);
    const auto param = operation.param;

    std::deque<AcceleratorMemoryMap> memory_maps;
    std::deque<BaseParams *> accelerator_params;
    MapOperation(operation, accelerator_params, memory_maps);

    if (is_soc_sim()) {
      const auto l2_tiling = get_l2_tiling(param);
      const int num_tiles = get_num_tiles(l2_tiling);
      std::cerr << "Number of accelerator tiles to run: " << num_tiles
                << std::endl;

      const int cache_size = getenv_int("CACHE_SIZE", 8 * 1024 * 1024);
      auto adjusted_params =
          offset_param_addresses(accelerator_params, cache_size);

      bool bank_index = 0;

      for (int j = 0; j < num_tiles; j++) {
        if (accelerator_params.size() < 4 && j > 1) {
          tile_done.SyncPop();
        }

        std::cerr << "Sending tile " << j << " params" << std::endl;
        dataloader->load_scratchpad(param, j, bank_index ? cache_size : 0);
        send_params(bank_index ? adjusted_params : accelerator_params);
        bank_index = !bank_index;
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
  matrixUnitStartSignal.ResetRead();
  vector_unit_start_signal.ResetRead();
#if SUPPORT_MVM
  matrix_vector_unit_start_signal.ResetRead();
#endif
  operation_done.ResetRead();

  wait();

  for (int i = 0; i < operations.size(); i++) {
    const auto operation = operations.at(i);
    const auto param = operation.param;

    std::deque<AcceleratorMemoryMap> memory_maps;
    std::deque<BaseParams *> accelerator_params;
    MapOperation(operation, accelerator_params, memory_maps);

    if (is_soc_sim()) {
      const auto l2_tiling = get_l2_tiling(param);
      const int num_tiles = get_num_tiles(l2_tiling);

      for (int j = 0; j < num_tiles; j++) {
        std::cerr << "Waiting for tile " << j << " start signal" << std::endl;
        bool is_first = j == 0;
        record_start(accelerator_params, operation, is_first);
        std::cerr << "Tile " << j << " started" << std::endl;
      }
    } else {
      record_start(accelerator_params, operation, true);
    }
  }
}

void Harness::done_monitor() {
  matrixUnitDoneSignal.ResetRead();
  vector_unit_done_signal.ResetRead();
#if SUPPORT_MVM
  matrix_vector_unit_done_signal.ResetRead();
#endif
  tile_done.ResetWrite();
  operation_done.ResetWrite();

  wait();

  for (int i = 0; i < operations.size(); i++) {
    const auto operation = operations.at(i);
    const auto param = operation.param;

    std::deque<AcceleratorMemoryMap> memory_maps;
    std::deque<BaseParams *> accelerator_params;
    MapOperation(operation, accelerator_params, memory_maps);

    int runtime_scale = 1;
    if (operation.has_shrunk_tiling) {
      runtime_scale = operation.shrink_factor;
      std::cout << "Scale operation" << operation.name << " by "
                << runtime_scale << std::endl;
    }

    if (is_soc_sim()) {
      const auto l2_tiling = get_l2_tiling(param);
      const int num_tiles = get_num_tiles(l2_tiling);

      const int cache_size = getenv_int("CACHE_SIZE", 8 * 1024 * 1024);
      bool bank_index = 0;

      for (int j = 0; j < num_tiles; j++) {
        std::cerr << "Waiting for tile " << j << " to complete" << std::endl;

        bool is_last = j == num_tiles - 1;
        record_done(accelerator_params, operation, runtime_scale, is_last);
        dataloader->store_scratchpad(param, j, bank_index ? cache_size : 0);
        bank_index = !bank_index;

        std::cerr << "Tile " << j << " finished" << std::endl;

        if (accelerator_params.size() < 4) {
          tile_done.SyncPush();
        }
      }
    } else {
      record_done(accelerator_params, operation, runtime_scale, true);
    }
  }

  sc_stop();
}

#endif

void run_accelerator(std::vector<Operation> operations,
                     DataLoader *dataloader) {
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
