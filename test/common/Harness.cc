#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include <cassert>
#include <cstdlib>
#include <memory>

#include "AccelTypes.h"
#include "sysc/kernel/sc_time.h"

#ifndef CFLOAT
#include "test/common/GoldModel.h"
#include "test/toolchain/MapOperation.h"

Harness::Harness(sc_module_name name, const Model& model,
                 const Model::Selection& selection, MemoryInterface* memory)
    : sc_module(name),
      clk("clk", std::stod(getenv("CLOCK_PERIOD", "1")), SC_NS, 0.5, 0, SC_NS,
          true),
      model(model),
      selection(selection),
      memory(memory) {
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
  accelerator.matrix_unit_output_data(matrix_unit_output_data);
  accelerator.matrix_unit_output_addr(matrix_unit_output_addr);
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

  REGISTER_FN(store_matrix_unit_output)
  REGISTER_FN(store_vector_output)
  REGISTER_FN(store_mx_scale_output)
  REGISTER_FN(store_sparse_tensor_output)

  REGISTER_FN(run_walker)
  REGISTER_FN(release_starts)
  REGISTER_FN(retire_dones)

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

  // Every datapath operand lives on chip: the graph's async_copies stage it
  // there before the operation runs, and stage the results back out.
  const auto array_memory = (ArrayMemory*)(this->memory);
  char* memory = array_memory->memories[SRAM_PARTITION];

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

  wait();

  while (true) {
    uint64_t address = address_out->Pop();
    auto data = data_out->Pop();

    access_counter->increment(std::string(name()) + "_" + "outputs", num_bytes);

    char bytes[num_bytes];
    for (int i = 0; i < num_bytes; i++) {
      bytes[i] = data.template slc<8>(i * 8);
    }
    this->memory->write_bytes_to_memory(address, SRAM_PARTITION, num_bytes,
                                        bytes);
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

void Harness::store_matrix_unit_output() {
  process_write_request(&matrix_unit_output_data, &matrix_unit_output_addr);
}

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

template <typename T>
T* get_param(const std::deque<BaseParams*>& params, int& index) {
  if (index < params.size()) {
    if (auto* casted = dynamic_cast<T*>(params[index])) {
      index++;  // Advance index only if the cast succeeds
      return casted;
    }
  }
  return nullptr;
}

// Serialize one operation's params and queue its invocation groups.
//
// The old harness drove each group to completion before sending the next:
// params, start, done, one group at a time, which serialized every tile's
// buffer loads against the previous tile's drain. Here the walk only pushes
// params -- the units' deserializers hand them to the double-buffered
// controllers, so the next tile's input and weight fetches begin while the
// current tile computes -- and queues a descriptor per group; the
// release_starts and retire_dones threads pace the handshakes.
//
// Only consecutive operations overlap. Groups within one operation are the
// passes of a multi-pass op -- softmax and layer_norm expand into three
// vector passes, a strided quantize_mx into a scale pass and a main pass --
// and each pass reads what the previous one wrote. So every group after the
// first drains the pipeline before its params go out, exactly as the old
// harness gated each pass on the previous one's done.
void Harness::dispatch_params(const voyager::Operation& op,
                              const std::deque<BaseParams*>& params) {
  int idx = 0;
  bool first = true;
  while (idx < params.size()) {
    if (!first) drain();

    InvocationGroup group;

    if (auto* matrix_params = get_param<MatrixParams>(params, idx)) {
#if SUPPORT_MVM
      if (matrix_params->is_fc) {
        send_serialized_params<MatrixParams, 64>(*matrix_params,
                                                 &matrix_vector_unit_params_in);
        group.matrix_vector = true;
      } else
#endif
#if SUPPORT_SPMM
          if (matrix_params->is_spmm) {
        // SpMM operation could have a fused dense matrix op
        if (auto* dense_params = get_param<MatrixParams>(params, idx)) {
          send_serialized_params<MatrixParams, 64>(*dense_params,
                                                   &matrix_unit_params_in);
          group.matrix = true;
        }
        send_serialized_params<MatrixParams, 64>(*matrix_params,
                                                 &spmm_unit_params_in);
        group.spmm = true;
      } else
#endif
      {
        send_serialized_params<MatrixParams, 64>(*matrix_params,
                                                 &matrix_unit_params_in);
        group.matrix = true;
      }
    } else if (auto* dwc_params = get_param<DwCParams>(params, idx)) {
#if SUPPORT_DWC
      send_serialized_params<DwCParams, 64>(*dwc_params, &dwc_unit_params_in);
      group.dwc = true;
#else
      throw std::runtime_error("DWC support is not enabled.");
#endif
    }

    if (auto* vector_params = get_param<VectorParams>(params, idx)) {
      send_serialized_params<VectorParams, 64>(*vector_params,
                                               &vector_unit_params_in);

      auto* vector_config = get_param<VectorInstructionConfig>(params, idx);
      send_serialized_params<VectorInstructionConfig, 64>(
          *vector_config, &vector_unit_params_in);
      group.vector = true;
    }

    group.op_begin = first ? &op : nullptr;
    first = false;
    if (idx >= params.size()) group.op_end = &op;

    pending_groups++;
    start_queue.push_back(group);
    start_pushed.notify(SC_ZERO_TIME);
  }
}

// Release each queued group's units in program order. SyncPop completes when
// the unit offers its start handshake -- the matrix unit raises it from the
// processor at the front of its pipeline, so a release can land while the
// previous group still drains through the accumulation buffer and output
// controller behind it.
void Harness::release_starts() {
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

  while (true) {
    while (start_queue.empty()) wait(start_pushed);
    InvocationGroup group = start_queue.front();
    start_queue.pop_front();

    if (group.op_begin != nullptr) {
      op_release_time = sc_time_stamp();
      CCS_LOG("----- Accelerator Layer '" << group.op_begin->name()
                                          << "' Started. -----");
    }

    // Popping every start before any done (in retire_dones) preserves
    // concurrency within a group -- a fused SpMM runs its dense matrix and
    // sparse passes at once.
    if (group.matrix) matrix_unit_start.SyncPop();
#if SUPPORT_MVM
    if (group.matrix_vector) matrix_vector_unit_start.SyncPop();
#endif
#if SUPPORT_SPMM
    if (group.spmm) spmm_unit_start.SyncPop();
#endif
#if SUPPORT_DWC
    if (group.dwc) dwc_unit_start.SyncPop();
#endif
    if (group.vector) vector_unit_start.SyncPop();

    // An operation's groups are contiguous, so the release time captured at
    // its op_begin group is still current at its op_end group.
    if (group.op_end != nullptr) group.start_time = op_release_time;

    done_queue.push_back(group);
    done_pushed.notify(SC_ZERO_TIME);
  }
}

// Retire groups in the order they started: pop the done handshakes, then fire
// whatever the group carried -- a commit's post, and the per-op logging. A
// post token is safe to fire here because everything queued ahead of it has
// already retired.
void Harness::retire_dones() {
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

  wait();

  while (true) {
    while (done_queue.empty()) wait(done_pushed);
    InvocationGroup group = done_queue.front();
    done_queue.pop_front();

    if (group.matrix) matrix_unit_done.SyncPop();
#if SUPPORT_MVM
    if (group.matrix_vector) matrix_vector_unit_done.SyncPop();
#endif
#if SUPPORT_SPMM
    if (group.spmm) spmm_unit_done.SyncPop();
#endif
#if SUPPORT_DWC
    if (group.dwc) dwc_unit_done.SyncPop();
#endif
    if (group.vector) vector_unit_done.SyncPop();

    if (group.has_post) {
      post_semaphore(group.post_node, group.post_slot, group.post_amount);
    }

    if (group.op_end != nullptr) {
      const sc_time end = sc_time_stamp();
      CCS_LOG("----- Accelerator Layer '" << group.op_end->name()
                                          << "' Finished. -----");
      // to_default_time_units returns a double; print whole nanoseconds so
      // large values do not switch to scientific notation.
      std::cout << "Runtime: "
                << static_cast<long long>(
                       end.to_default_time_units() -
                       group.start_time.to_default_time_units())
                << " ns" << std::endl;
      if (group.op_end->has_tiling()) {
        access_counter->print_summary(group.op_end->tiling(), true);
      }
    }

    pending_groups--;
    if (pending_groups == 0) group_retired.notify(SC_ZERO_TIME);
  }
}

void Harness::drain() {
  while (pending_groups > 0) wait(group_retired);
}

void Harness::begin_commit() { in_commit = true; }

// A commit's post must not fire until the whole region has retired. The walk
// reaches end_commit as soon as it has pushed the params, so the post rides
// the queues as a unit-less token behind the region's groups.
void Harness::end_commit(bool has_post, const std::string& post_node,
                         int64_t post_slot, int64_t post_amount) {
  in_commit = false;
  if (!has_post) return;

  InvocationGroup token;
  token.has_post = true;
  token.post_node = post_node;
  token.post_slot = post_slot;
  token.post_amount = post_amount;

  pending_groups++;
  start_queue.push_back(token);
  start_pushed.notify(SC_ZERO_TIME);
}

void Harness::init_semaphore(const std::string& node, int64_t slot,
                             int64_t value) {
  auto& semaphore = sync_semaphores[{node, slot}];
  semaphore.count = value;
}

void Harness::post_semaphore(const std::string& node, int64_t slot,
                             int64_t amount) {
  auto& semaphore = sync_semaphores[{node, slot}];
  semaphore.count += amount;
  semaphore.posted.notify(SC_ZERO_TIME);
}

// Called from the walk; blocks until retire_dones (or an async_copy on the
// walk itself) has posted enough credits. This is the back-pressure that
// keeps the walk at most one ping-pong slot ahead of the hardware.
void Harness::wait_semaphore(const std::string& node, int64_t slot,
                             int64_t amount) {
  auto& semaphore = sync_semaphores[{node, slot}];
  while (semaphore.count < amount) wait(semaphore.posted);
  semaphore.count -= amount;
}

// Drives the DUT. The bufferized graph already states the tile loops, the
// reduction conditionals and every DMA, so there is nothing left for the
// harness to synthesize: it walks the graph exactly as the gold model does and,
// when the walk reaches an accelerator operation, hands it to the hardware
// instead of to a kernel.
void Harness::run_walker() {
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

  wait();

  const sc_time start = sc_time_stamp();

  Interpreter interpreter(model, memory, this);
  interpreter.run(selection);

  // The walk returning only means every group was queued; wait for the
  // hardware to retire them all before stamping the runtime.
  drain();

  const sc_time end = sc_time_stamp();
  std::cout << "Total Runtime: "
            << static_cast<long long>(end.to_default_time_units() -
                                      start.to_default_time_units())
            << " ns" << std::endl;

  sc_stop();
}

void Harness::execute(const voyager::Operation& op, const ScalarEnv& env) {
  const std::vector<Tensor> outputs = resolve_outputs(op, env);

  // Anything the compiler tagged "cpu" runs on the control processor, not the
  // datapath -- a slice, a host-side pad. Those still need a kernel, so let the
  // gold model do them in place, exactly as it does in the gold flow. Host ops
  // are not sequenced by the semaphore protocol, so quiesce the datapath
  // first.
  if (!is_datapath(op)) {
    drain();
    run_host_operation(op, env, memory);
    return;
  }

  std::deque<BaseParams*> params;
  map_operation(op, env, params);

#ifndef __SYNTHESIS__
  if (std::getenv("PRINT_PARAMS")) {
    std::cout << "===== params for " << op.name() << " (" << params.size()
              << ") =====" << std::endl;
    for (auto* p : params) {
      if (auto* mp = dynamic_cast<MatrixParams*>(p)) std::cout << *mp;
      if (auto* vp = dynamic_cast<VectorParams*>(p)) std::cout << *vp;
    }
  }
#endif

  // A synchronous operation is a barrier on both sides: it waits on none of
  // the semaphores the in-flight commits will post, so only a drain orders its
  // fetches after their writes, and the interpreter posts Operation.semaphore
  // the moment execute returns.
  if (!in_commit) drain();
  dispatch_params(op, params);
  if (!in_commit) drain();

  for (auto* param : params) delete param;
}

#endif  // CFLOAT: the harness instantiates the SystemC Accelerator, which the
        // CFloat datatype cannot compile. run_accelerator below just aborts.

void run_accelerator(const Model& model, const Model::Selection& selection,
                     MemoryInterface* memory) {
#ifdef CFLOAT
  spdlog::error(
      "The SystemC model does not support the CFloat datatype. Only the gold "
      "model should be used for CFloat.\n");
  std::abort();
#else
  // CONNECTIONS_FAST_SIM makes Harness larger than the default process stack
  // (about 12 MiB for a 64x64 MX build). Construct it on the heap so entering
  // run_accelerator cannot overflow the stack before the constructor runs.
  auto harness = std::make_unique<Harness>("harness", model, selection, memory);
  sc_start();
#endif
}
