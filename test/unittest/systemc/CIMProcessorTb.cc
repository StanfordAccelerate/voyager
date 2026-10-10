// SystemC tests for CIMProcessor using INT8 operands

#include <ac_int.h>
#include <mc_connections.h>
#include <systemc.h>

#include <deque>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cim/CIMProcessor.h"

#ifndef CIM_PROCESSOR_TEST_MODE
#define CIM_PROCESSOR_TEST_MODE 0
#endif

static constexpr int MACRO_INPUT_LANES = 2;
static constexpr int MACRO_OUTPUT_LANES = 2;
static constexpr int WEIGHT_SETS = 3;
static constexpr int BASE_A_WIDTH = 4;
static constexpr int BASE_B_WIDTH = 4;
static constexpr int BASE_C_WIDTH = 12;
static constexpr int TILE_INPUT_AXIS_ELEMENTS = 2;
static constexpr int TILE_OUTPUT_AXIS_ELEMENTS = 1;
static constexpr int INPUT_AXIS_TILES = 2;
static constexpr int OUTPUT_AXIS_TILES = 2;
static constexpr int MACRO_WRITE_INPUT_LANES = 1;
static constexpr int MAC_LATENCY = 1;
static constexpr int MODE = CIM_PROCESSOR_TEST_MODE;
static constexpr int A_WIDTH = 8;
static constexpr int B_WIDTH = 8;
static constexpr bool SIGNED = true;
static constexpr int A_PORT_TILES = INPUT_AXIS_TILES;
// A narrow weight beat spans one output tile at a time.
static constexpr int B_PORT_TILES = 1;
static constexpr int C_PORT_TILES = OUTPUT_AXIS_TILES;
static constexpr int RESULT_SLOTS_PER_OUTPUT_TILE = INPUT_AXIS_TILES;
static constexpr int INPUT_LANES =
    MACRO_INPUT_LANES * TILE_INPUT_AXIS_ELEMENTS * INPUT_AXIS_TILES;
static constexpr int TILE_OUTPUT_LANES =
    (MACRO_OUTPUT_LANES / (B_WIDTH / BASE_B_WIDTH)) * TILE_OUTPUT_AXIS_ELEMENTS;
static constexpr int OUTPUT_LANES = TILE_OUTPUT_LANES * OUTPUT_AXIS_TILES;
static constexpr int BUFFER_DEPTH = 16;

using Processor =
    CIMProcessor<std::tuple<DataTypes::int8>, std::tuple<DataTypes::int8>,
                 DataTypes::int8, DataTypes::int8, DataTypes::int24,
                 DataTypes::int24, DataTypes::fp8_e8m0, INPUT_LANES,
                 OUTPUT_LANES, BUFFER_DEPTH, MACRO_INPUT_LANES,
                 MACRO_OUTPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH, BASE_B_WIDTH,
                 BASE_C_WIDTH, MACRO_WRITE_INPUT_LANES, MAC_LATENCY, MODE,
                 SIGNED, TILE_INPUT_AXIS_ELEMENTS, TILE_OUTPUT_AXIS_ELEMENTS,
                 INPUT_AXIS_TILES, OUTPUT_AXIS_TILES, A_PORT_TILES,
                 B_PORT_TILES, C_PORT_TILES, CIM_C_BEAT_OUTPUT_MAJOR,
                 RESULT_SLOTS_PER_OUTPUT_TILE, CIM_LOCAL_ACCUM_CONTEXTS>;

using Dut = Processor;
using Buffer = DataTypes::int24;
using BufferVector = Pack1D<Buffer, OUTPUT_LANES>;
using WriteRequest = typename Processor::AccumulationWriteRequest;

// CIMProcessorTb checks scheduling, resident-weight reuse, backpressure, and
// persistent accumulation
SC_MODULE(CIMProcessorTb) {
  // One queued result lets the consumer independently delay ready and
  // backpressure the processor
  struct ExpectedOutput {
    std::string label;
    BufferVector values;
    int stall_cycles;
  };

  Dut dut;
  sc_clock clk;
  sc_signal<bool> rstn;
#if ENABLE_PERF_COUNTERS
  sc_signal<MatrixPerformance::Counter>
      perf_counters[MatrixPerformance::PROCESSOR_COUNTER_COUNT];
#endif

  Connections::Combinational<ac_int<INPUT_BUFFER_WIDTH, false>> input_channel;
  Connections::Combinational<ac_int<Processor::WEIGHT_WRITE_WIDTH, false>>
      weight_channel;
  Connections::Combinational<cim::WeightDescriptor> weight_descriptor_channel;
  Connections::Combinational<BufferVector> bias_channel;
  Connections::Combinational<MatrixParams> params_channel;
  Connections::Combinational<BufferVector> output_channel;
  Connections::SyncChannel start_channel;

  Connections::Combinational<ac_int<16, false>> accumulation_read_address_0;
  Connections::Combinational<BufferVector> accumulation_read_data_0;
  Connections::Combinational<WriteRequest> accumulation_write_request_0;
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::Combinational<ac_int<16, false>> accumulation_read_address_1;
  Connections::Combinational<BufferVector> accumulation_read_data_1;
  Connections::Combinational<WriteRequest> accumulation_write_request_1;
  Connections::SyncChannel accumulation_done_0;
  Connections::SyncChannel accumulation_done_1;
#endif

  BufferVector accumulation_memory[Processor::ACCUM_BUFFER_BANKS][BUFFER_DEPTH];
  bool pending_read[Processor::ACCUM_BUFFER_BANKS];
  ac_int<16, false> pending_read_address[Processor::ACCUM_BUFFER_BANKS];
  unsigned long pending_read_ready_cycle[Processor::ACCUM_BUFFER_BANKS];
  int read_count[Processor::ACCUM_BUFFER_BANKS];
  int write_count[Processor::ACCUM_BUFFER_BANKS];
  int done_count[Processor::ACCUM_BUFFER_BANKS];
  unsigned long last_write_completion_cycle[Processor::ACCUM_BUFFER_BANKS]
                                           [BUFFER_DEPTH];
  unsigned long last_read_write_cycle[Processor::ACCUM_BUFFER_BANKS]
                                     [BUFFER_DEPTH];
  int ordered_dependent_read_count;
  std::vector<int> read_addresses[Processor::ACCUM_BUFFER_BANKS];
  std::vector<int> write_addresses[Processor::ACCUM_BUFFER_BANKS];
  unsigned long buffer_cycle;
  bool independent_read_probe = false;
  bool independent_read_seen = false;
  int independent_probe_writes_begin = 0;
  unsigned long independent_probe_hold_begin = 0;

  std::deque<ExpectedOutput> expected_outputs;
  std::deque<BufferVector> pending_biases;
  std::deque<cim::WeightDescriptor> pending_weight_descriptors;
  std::deque<int> pending_weight_sets;
  sc_event expected_output_event;
  sc_event bias_event;
  sc_event weight_descriptor_event;
  sc_event weight_event;
  sc_event weight_completion_event;
  int queued_weight_sets;
  int completed_weight_sets;
  int checked_outputs;
  int pushed_weight_beats = 0;
  bool test_failed;

  SC_HAS_PROCESS(CIMProcessorTb);

  // Construct the processor and independent ready/valid peers around it
  explicit CIMProcessorTb(sc_module_name name)
      : sc_module(name),
        dut("dut"),
        clk("clk", 10, SC_NS),
        start_channel("start_channel"),
#if DOUBLE_BUFFERED_ACCUM_BUFFER
        accumulation_done_0("accumulation_done_0"),
        accumulation_done_1("accumulation_done_1"),
#endif
        ordered_dependent_read_count(0),
        buffer_cycle(0),
        queued_weight_sets(0),
        completed_weight_sets(0),
        checked_outputs(0),
        test_failed(false) {
    dut.clk(clk);
    dut.rstn(rstn);
#if ENABLE_PERF_COUNTERS
    for (int i = 0; i < MatrixPerformance::PROCESSOR_COUNTER_COUNT; ++i)
      dut.perf_counters[i](perf_counters[i]);
#endif
    dut.input_channel(input_channel);
    dut.weight_channel(weight_channel);
    dut.weight_descriptor_channel(weight_descriptor_channel);
    dut.bias_channel(bias_channel);
    dut.params_in(params_channel);
    dut.output_channel(output_channel);
    dut.start(start_channel);
    dut.accumulation_buffer_read_address[0](accumulation_read_address_0);
    dut.accumulation_buffer_read_data[0](accumulation_read_data_0);
    dut.accumulation_buffer_write_request[0](accumulation_write_request_0);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    dut.accumulation_buffer_read_address[1](accumulation_read_address_1);
    dut.accumulation_buffer_read_data[1](accumulation_read_data_1);
    dut.accumulation_buffer_write_request[1](accumulation_write_request_1);
    dut.accumulation_buffer_done[0](accumulation_done_0);
    dut.accumulation_buffer_done[1](accumulation_done_1);
#endif

    for (int bank = 0; bank < Processor::ACCUM_BUFFER_BANKS; bank++) {
      pending_read[bank] = false;
      pending_read_address[bank] = 0;
      pending_read_ready_cycle[bank] = 0;
      read_count[bank] = 0;
      write_count[bank] = 0;
      done_count[bank] = 0;
      for (int address = 0; address < BUFFER_DEPTH; address++) {
        accumulation_memory[bank][address] = BufferVector::zero();
        last_write_completion_cycle[bank][address] = 0;
        last_read_write_cycle[bank][address] = 0;
      }
    }

    SC_THREAD(run);
    sensitive << clk.posedge_event();

    SC_THREAD(drive_bias);
    sensitive << clk.posedge_event();

    SC_THREAD(drive_weight_descriptors);
    sensitive << clk.posedge_event();

    SC_THREAD(drive_weights);
    sensitive << clk.posedge_event();

    SC_THREAD(check_outputs);
    sensitive << clk.posedge_event();

    SC_THREAD(run_accumulation_buffer);
    sensitive << clk.posedge_event();

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    SC_THREAD(consume_done_0);
    sensitive << clk.posedge_event();

    SC_THREAD(consume_done_1);
    sensitive << clk.posedge_event();
#endif

    SC_THREAD(watchdog);
  }

  // Advance one processor cycle
  void tick() {
    wait(clk.posedge_event());
    wait(SC_ZERO_TIME);
  }

  // Record a deterministic failure without abandoning channel cleanup
  void require(bool condition, const std::string& message) {
    if (condition) {
      return;
    }
    std::cerr << "[FAIL] " << message << std::endl;
    test_failed = true;
  }

  // Stop a ready/valid or scheduling deadlock with a bounded failure
  void watchdog() {
    wait(1, SC_MS);
    require(false, "timed out waiting for CIMProcessor completion");
    sc_stop();
  }

  // Initialize mapper loop indices shared by all scenarios
  MatrixParams make_base_params() const {
    MatrixParams params;
    for (int level = 0; level < 2; level++) {
      for (int loop = 0; loop < 6; loop++) {
        params.loops[level][loop] = 1;
      }
    }

    // Level 0 orders output Y, output X, weights, filter Y, then reduction
    params.y_loop_idx[0] = 0;
    params.x_loop_idx[0] = 1;
    params.weight_loop_idx[0] = 2;
    params.fy_loop_idx[0] = 3;
    params.reduction_loop_idx[0] = 4;

    // Level 1 orders filter Y/X, output Y, weights, output X, then reduction
    params.fy_loop_idx[1] = 0;
    params.fx_loop_idx = 1;
    params.y_loop_idx[1] = 2;
    params.weight_loop_idx[1] = 3;
    params.x_loop_idx[1] = 4;
    params.reduction_loop_idx[1] = 5;

    params.use_input_codebook = false;
    params.use_weight_codebook = false;
    return params;
  }

  // Create two output-X addresses with two temporal contributions each
  MatrixParams make_accumulation_params(bool has_bias) const {
    MatrixParams params = make_base_params();
    params.loops[1][params.x_loop_idx[1]] = 2;
    params.loops[1][params.reduction_loop_idx[1]] = 2;
    params.has_bias = has_bias;
    return params;
  }

  // Replay each outer-reduction set across two live outer-X contexts
  MatrixParams make_set_major_replay_params() const {
    MatrixParams params = make_base_params();
    params.reduction_loop_idx[0] = 0;
    params.x_loop_idx[0] = 1;
    params.y_loop_idx[0] = 2;
    params.weight_loop_idx[0] = 3;
    params.fy_loop_idx[0] = 4;
    params.loops[0][params.reduction_loop_idx[0]] = 2;
    params.loops[0][params.x_loop_idx[0]] = 2;
    params.has_bias = false;
    return params;
  }

  // Create output-X or output-Y traversal inside one resident-weight lifetime
  MatrixParams make_weight_reuse_params(bool traverse_x) const {
    MatrixParams params = make_base_params();
    params.weight_loop_idx[0] = 1;
    params.x_loop_idx[0] = traverse_x ? 2 : 0;
    params.y_loop_idx[0] = traverse_x ? 0 : 2;
    params.fy_loop_idx[0] = 3;
    params.reduction_loop_idx[0] = 4;
    params.loops[0][traverse_x ? params.x_loop_idx[0] : params.y_loop_idx[0]] =
        4;
    params.has_bias = false;
    return params;
  }

  // Create one direct output per independently loaded resident set
  MatrixParams make_weight_reload_params(int operations) const {
    MatrixParams params = make_base_params();
    params.loops[0][params.weight_loop_idx[0]] = operations;
    params.has_bias = false;
    return params;
  }

  // Create a resident set sequence replayed across two outer-X positions
  MatrixParams make_multiset_reuse_params(int set_count) const {
    MatrixParams params = make_base_params();
    params.weight_loop_idx[0] = 0;
    params.x_loop_idx[0] = 1;
    params.y_loop_idx[0] = 2;
    params.fy_loop_idx[0] = 3;
    params.reduction_loop_idx[0] = 4;
    params.loops[0][params.x_loop_idx[0]] = 2;
    params.loops[1][params.weight_loop_idx[1]] = set_count;
    params.has_bias = false;
    return params;
  }

  // Return one signed activation pattern so heterogeneous jobs cannot alias
  int input_value(int input_pattern, int input_index) const {
    return input_index == 0 ? -(input_pattern + 2) : input_pattern + 1;
  }

  // Return one deterministic signed weight from a logical B row
  int weight_value(int weight_pattern, int input_index, int output_index)
      const {
    const int output_tile_index = output_index / TILE_OUTPUT_LANES;
    const ac_int<B_WIDTH, true> value =
        weight_pattern + 1 + output_tile_index + input_index;
    return value.to_int();
  }

  // Pack one complete signed A vector
  ac_int<INPUT_BUFFER_WIDTH, false> make_inputs(int input_pattern) const {
    ac_int<INPUT_BUFFER_WIDTH, false> inputs = 0;
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
      const ac_int<A_WIDTH, true> value =
          input_value(input_pattern, input_index);
      inputs.set_slc(input_index * A_WIDTH, value.template slc<A_WIDTH>(0));
    }
    return inputs;
  }

  // Pack one output-axis span in the weight-channel lane order
  ac_int<Processor::WEIGHT_WRITE_WIDTH, false> make_weight_beat(
      int weight_pattern, int input_index, int span) const {
    ac_int<Processor::WEIGHT_WRITE_WIDTH, false> beat = 0;
    for (int port_tile = 0; port_tile < B_PORT_TILES; port_tile++) {
      for (int tile_output_index = 0; tile_output_index < TILE_OUTPUT_LANES;
           tile_output_index++) {
        const int output_index =
            (span * B_PORT_TILES + port_tile) * TILE_OUTPUT_LANES +
            tile_output_index;
        const int lane = port_tile * TILE_OUTPUT_LANES + tile_output_index;
        const ac_int<B_WIDTH, true> value =
            weight_value(weight_pattern, input_index, output_index);
        beat.set_slc(lane * B_WIDTH, value.template slc<B_WIDTH>(0));
      }
    }
    return beat;
  }

  // Push one complete row-major resident-set payload
  void write_weight_set(int weight_pattern) {
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
      for (int span = 0; span < Processor::WEIGHT_BEATS_PER_ROW; span++) {
        weight_channel.Push(
            make_weight_beat(weight_pattern, input_index, span));
        pushed_weight_beats++;
      }
    }
  }

  // Program every queued set from one channel-driving process
  void drive_weights() {
    weight_channel.ResetWrite();
    wait();
    while (!rstn.read()) {
      wait();
    }

    while (true) {
      if (pending_weight_sets.empty()) {
        wait(weight_event);
        continue;
      }
      const int weight_pattern = pending_weight_sets.front();
      pending_weight_sets.pop_front();
      write_weight_set(weight_pattern);
      completed_weight_sets++;
      weight_completion_event.notify(SC_ZERO_TIME);
    }
  }

  // Queue one set without coupling its weight load to input acceptance
  int queue_weight_set(int weight_pattern) {
    pending_weight_sets.push_back(weight_pattern);
    queued_weight_sets++;
    weight_event.notify(SC_ZERO_TIME);
    return queued_weight_sets;
  }

  // Wait until every queued set has reached the processor input
  void wait_for_weight_sets(int target) {
    while (completed_weight_sets < target) {
      wait(weight_completion_event);
    }
  }

  // Program one complete set before the caller continues
  void push_weight_set(int weight_pattern) {
    wait_for_weight_sets(queue_weight_set(weight_pattern));
  }

  // Compute one complete golden MAC result
  BufferVector expected_partial(int input_pattern, int weight_pattern) const {
    BufferVector expected = BufferVector::zero();
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
      int value = 0;
      for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
        value += input_value(input_pattern, input_index) *
                 weight_value(weight_pattern, input_index, output_index);
      }
      expected[output_index] = Buffer(value);
    }
    return expected;
  }

  // Add one vector into another with the processor's Buffer arithmetic
  void add_vector(BufferVector & destination, const BufferVector& source)
      const {
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
      destination[output_index] += source[output_index];
    }
  }

  // Queue one bias vector for the independent bias producer
  BufferVector queue_bias(int base) {
    BufferVector bias = BufferVector::zero();
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
      bias[output_index] = Buffer(base + output_index);
    }
    pending_biases.push_back(bias);
    bias_event.notify(SC_ZERO_TIME);
    return bias;
  }

  // Create one tile descriptor for consecutive physical sets
  cim::WeightDescriptor make_weight_descriptor(int set_count, int replay_count)
      const {
    cim::WeightDescriptor descriptor;
    descriptor.set_count = set_count;
    descriptor.replay_count = replay_count;
    return descriptor;
  }

  // Match WeightController's resident-capacity decision for a tile sequence
  std::vector<cim::WeightDescriptor> make_weight_descriptors(int set_count)
      const {
    std::vector<cim::WeightDescriptor> descriptors;
    if (set_count <= Processor::RESIDENT_SET_COUNT) {
      descriptors.push_back(make_weight_descriptor(set_count, 1));
    } else {
      for (int set = 0; set < set_count; set++) {
        descriptors.push_back(make_weight_descriptor(1, 1));
      }
    }
    return descriptors;
  }

  // Queue tile metadata without blocking the weight-data producer
  void queue_weight_descriptors(
      const std::vector<cim::WeightDescriptor>& descriptors) {
    for (const cim::WeightDescriptor& descriptor : descriptors) {
      pending_weight_descriptors.push_back(descriptor);
    }
    weight_descriptor_event.notify(SC_ZERO_TIME);
  }

  // Queue one expected output and its deliberate ready stall
  void expect_output(const std::string& label, const BufferVector& values,
                     int stall_cycles) {
    expected_outputs.push_back(ExpectedOutput{label, values, stall_cycles});
    expected_output_event.notify(SC_ZERO_TIME);
  }

  // Send one mapper job whose operations may use distinct input patterns
  void send_job(const MatrixParams& params,
                const std::vector<int>& weight_patterns,
                const std::vector<bool>& load_weights,
                const std::vector<cim::WeightDescriptor>& descriptors,
                const std::vector<int>& input_patterns) {
    require(weight_patterns.size() == load_weights.size() &&
                weight_patterns.size() == input_patterns.size(),
            "test job vectors must have equal lengths");
    queue_weight_descriptors(descriptors);
    params_channel.Push(params);
    start_channel.SyncPop();

    for (std::size_t operation = 0; operation < weight_patterns.size();
         operation++) {
      if (load_weights[operation]) {
        push_weight_set(weight_patterns[operation]);
      }
      input_channel.Push(make_inputs(input_patterns[operation]));
    }
  }

  // Send one mapper job with a shared input pattern
  void send_job(const MatrixParams& params,
                const std::vector<int>& weight_patterns,
                const std::vector<bool>& load_weights,
                const std::vector<cim::WeightDescriptor>& descriptors,
                int input_pattern) {
    send_job(params, weight_patterns, load_weights, descriptors,
             std::vector<int>(weight_patterns.size(), input_pattern));
  }

  // Check one job's exact accumulation-buffer address subsequence
  void require_address_sequence(
      const std::string& label, const std::vector<int>& addresses,
      std::size_t begin, const std::vector<int>& expected) {
    std::ostringstream count_message;
    count_message << label << " expected " << expected.size()
                  << " addresses got " << addresses.size() - begin;
    require(addresses.size() == begin + expected.size(), count_message.str());
    if (addresses.size() < begin + expected.size()) {
      return;
    }
    for (std::size_t index = 0; index < expected.size(); index++) {
      std::ostringstream address_message;
      address_message << label << " index " << index << " expected "
                      << expected[index] << " got " << addresses[begin + index];
      require(addresses[begin + index] == expected[index],
              address_message.str());
    }
  }

  // Check register reuse and excess-row spills with independent arithmetic and
  // address traces
  void check_context_rows(int rows, int contributions, bool interleaved,
                          bool banked, int bank, int reduction = 0) {
    MatrixParams params = make_base_params();
    const int reduction_position = interleaved ? 4 : 5;
    params.x_loop_idx[1] = interleaved ? 5 : 4;
    if (reduction == 1) {
      params.reduction_loop_idx[1] = 1;
      params.fx_loop_idx = reduction_position;
    } else if (reduction == 2) {
      params.reduction_loop_idx[1] = 0;
      params.fy_loop_idx[1] = reduction_position;
    } else {
      params.reduction_loop_idx[1] = reduction_position;
    }
    params.loops[1][params.x_loop_idx[1]] = rows;
    params.loops[1][reduction_position] = contributions;
    params.has_bias = true;
    params.write_output_to_accum_buffer = banked;
    const BufferVector bias = queue_bias(-300 - checked_outputs);
    const int outputs_before = checked_outputs;
    const int done_before = done_count[bank];
    const auto reads_begin = read_addresses[bank].size();
    const auto writes_begin = write_addresses[bank].size();
    std::vector<BufferVector> expected(rows, bias);
    std::vector<int> weights, inputs, expected_reads, expected_writes;
    std::vector<bool> loads;
    for (int operation = 0; operation < rows * contributions; operation++) {
      const int row =
          interleaved ? operation % rows : operation / contributions;
      const int term =
          interleaved ? operation / rows : operation % contributions;
      weights.push_back(220 + term);
      inputs.push_back(20 + row);
      loads.push_back((!interleaved && contributions > 1) || row == 0);
      add_vector(expected[row],
                 expected_partial(inputs.back(), weights.back()));
      const bool local = !interleaved || row < CIM_LOCAL_ACCUM_CONTEXTS;
      if (!local && term > 0) expected_reads.push_back(row);
      if ((!local && term + 1 < contributions) ||
          (banked && term + 1 == contributions))
        expected_writes.push_back(row);
    }
    if (!banked) {
      for (int row = 0; row < rows; row++) {
        expect_output("reusable context row " + std::to_string(row),
                      expected[row], 25);
      }
    }
    send_job(params, weights, loads,
             make_weight_descriptors(interleaved || contributions == 1
                                         ? contributions
                                         : rows * contributions),
             inputs);
    while (banked ? (done_count[bank] == done_before ||
                     write_addresses[bank].size() <
                         writes_begin + expected_writes.size())
                  : checked_outputs < outputs_before + rows) {
      tick();
    }
    require_address_sequence("context reads", read_addresses[bank], reads_begin,
                             expected_reads);
    require_address_sequence("context writes", write_addresses[bank],
                             writes_begin, expected_writes);
    if (banked) {
      for (int row = 0; row < rows; row++) {
        for (int lane = 0; lane < OUTPUT_LANES; lane++) {
          require(accumulation_memory[bank][row][lane] == expected[row][lane],
                  "buffered context result mismatch");
        }
      }
    }
    std::cout << "CIM_CONTEXT_REUSE contexts=" << CIM_LOCAL_ACCUM_CONTEXTS
              << " rows=" << rows << " contributions=" << contributions
              << " interleaved=" << interleaved << " reduction=" << reduction
              << " intermediate_reads=" << expected_reads.size()
              << " intermediate_writes="
              << expected_writes.size() - (banked ? rows : 0)
              << " final_writes=" << (banked ? rows : 0) << std::endl;
  }

  // Hold one spill write until a later spill reads its committed partial sum
  void check_independent_sram_read() {
    independent_read_probe = true;
    independent_read_seen = false;
    independent_probe_writes_begin = write_count[0];
    independent_probe_hold_begin = 0;
    check_context_rows(CIM_LOCAL_ACCUM_CONTEXTS + 3, 3, true, false, 0);
    independent_read_probe = false;
    require(independent_read_seen,
            "independent SRAM read waited for an unrelated write");
    if (independent_read_seen) {
      std::cout << "[PASS] cim_processor_independent_sram_read" << std::endl;
    }
  }

  // Release a deliberately stalled write only after the unrelated read
  // progresses
  bool hold_independent_probe_write() {
    if (!independent_read_probe || independent_read_seen ||
        write_count[0] < independent_probe_writes_begin + 3) {
      return false;
    }
    if (independent_probe_hold_begin == 0) {
      independent_probe_hold_begin = buffer_cycle;
    }
    if (buffer_cycle - independent_probe_hold_begin < 1000) {
      return true;
    }
    require(false,
            "unrelated SRAM write blocked read progress for 1000 cycles");
    independent_read_probe = false;
    return false;
  }

  // Interleave final uses from a full tile with the next tile's ring refills
  void send_progressive_release_job(int input_pattern, int weight_pattern) {
    queue_weight_descriptors(
        {make_weight_descriptor(WEIGHT_SETS, 1), make_weight_descriptor(2, 1)});
    params_channel.Push(make_weight_reload_params(WEIGHT_SETS + 2));
    start_channel.SyncPop();

    for (int set = 0; set < WEIGHT_SETS; set++) {
      push_weight_set(weight_pattern + set);
    }

    // The next push cannot start until the first old set is released
    input_channel.Push(make_inputs(input_pattern));
    push_weight_set(weight_pattern + WEIGHT_SETS);
    input_channel.Push(make_inputs(input_pattern));
    push_weight_set(weight_pattern + WEIGHT_SETS + 1);

    for (int set = 2; set < WEIGHT_SETS; set++) {
      input_channel.Push(make_inputs(input_pattern));
    }
    input_channel.Push(make_inputs(input_pattern));
    input_channel.Push(make_inputs(input_pattern));
  }

  // Refill the complete ring while the next descriptor is already runnable
  void send_consecutive_full_ring_replay_job(int input_pattern,
                                             int first_weight_pattern,
                                             int second_weight_pattern,
                                             int replays_per_descriptor = 2) {
    MatrixParams params = make_multiset_reuse_params(WEIGHT_SETS);
    params.loops[0][params.x_loop_idx[0]] = 2 * replays_per_descriptor;
    queue_weight_descriptors(
        {make_weight_descriptor(WEIGHT_SETS, replays_per_descriptor),
         make_weight_descriptor(WEIGHT_SETS, replays_per_descriptor)});
    params_channel.Push(params);
    start_channel.SyncPop();

    for (int set = 0; set < WEIGHT_SETS; set++) {
      push_weight_set(first_weight_pattern + set);
    }

    int completion_target = completed_weight_sets;
    for (int set = 0; set < WEIGHT_SETS; set++) {
      completion_target = queue_weight_set(second_weight_pattern + set);
    }

    for (int descriptor = 0; descriptor < 2; descriptor++) {
      for (int replay = 0; replay < replays_per_descriptor; replay++) {
        for (int set = 0; set < WEIGHT_SETS; set++) {
          input_channel.Push(make_inputs(input_pattern));
        }
      }
    }
    wait_for_weight_sets(completion_target);
  }

  // Drive queued biases only when the processor requests them
  void drive_bias() {
    bias_channel.ResetWrite();
    wait();
    while (!rstn.read()) {
      wait();
    }

    while (true) {
      if (pending_biases.empty()) {
        wait(bias_event);
        continue;
      }
      const BufferVector bias = pending_biases.front();
      pending_biases.pop_front();
      bias_channel.Push(bias);
    }
  }

  // Drive resident-tile metadata independently of the weight beat stream
  void drive_weight_descriptors() {
    weight_descriptor_channel.ResetWrite();
    wait();
    while (!rstn.read()) {
      wait();
    }

    while (true) {
      if (pending_weight_descriptors.empty()) {
        wait(weight_descriptor_event);
        continue;
      }
      const cim::WeightDescriptor descriptor =
          pending_weight_descriptors.front();
      pending_weight_descriptors.pop_front();
      weight_descriptor_channel.Push(descriptor);
    }
  }

#if ENABLE_PERF_COUNTERS
  unsigned counter(MatrixPerformance::CounterId id) {
    return perf_counters[MatrixPerformance::storage_index(id)].read().to_uint();
  }

  // Counters publish one cycle after the observed handshake. The held result
  // backpressure job must stall the array result channel, and the first
  // job's weights arrive after its parameters.
  void check_performance_counters() {
    using namespace MatrixPerformance;
    tick();
    tick();
    const unsigned issued = counter(ARRAY_ISSUE_CYCLES);
    require(issued > 0, "counter: no MAC vectors issued");
    require(counter(PROCESSOR_ACTIVE_CYCLES) >= issued,
            "counter: active cycles below issued vectors");
    require(counter(CIM_WEIGHT_LOAD_CYCLES) == unsigned(pushed_weight_beats),
            "counter: weight-write beats differ from pushed weight beats");
    require(counter(MAC_WAIT_WEIGHT_SET_LOAD_CYCLES) > 0,
            "counter: scheduler never waited for a resident set");
    require(counter(RESULT_BACKPRESSURE_CYCLES) > 0,
            "counter: delayed output ready never stalled the result channel");
    require(counter(INPUT_BACKPRESSURE_CYCLES) > 0,
            "counter: MAC issue never stalled");
    std::cout << "counters: active=" << counter(PROCESSOR_ACTIVE_CYCLES)
              << " issued=" << issued
              << " input_backpressure=" << counter(INPUT_BACKPRESSURE_CYCLES)
              << " result_backpressure=" << counter(RESULT_BACKPRESSURE_CYCLES)
              << " weight_wait=" << counter(MAC_WAIT_WEIGHT_SET_LOAD_CYCLES)
              << " weight_beats=" << counter(CIM_WEIGHT_LOAD_CYCLES)
              << std::endl;
  }
#endif

  // Delay output ready to prove result-channel backpressure is lossless
  void check_outputs() {
    output_channel.ResetRead();
    wait();
    while (!rstn.read()) {
      wait();
    }

    while (true) {
      if (expected_outputs.empty()) {
        wait(expected_output_event);
        continue;
      }

      const ExpectedOutput expected = expected_outputs.front();
      expected_outputs.pop_front();
      for (int cycle = 0; cycle < expected.stall_cycles; cycle++) {
        wait();
      }

      const BufferVector result = output_channel.Pop();
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
        std::ostringstream message;
        message << expected.label << " output_index " << output_index
                << " expected "
                << expected.values[output_index].int_val.to_int() << " got "
                << result[output_index].int_val.to_int();
        require(result[output_index].int_val.to_int() ==
                    expected.values[output_index].int_val.to_int(),
                message.str());
      }
      checked_outputs++;
    }
  }

  // Apply deterministic read-address and write-request backpressure to bank 0
  void service_bank_0() {
    if (pending_read[0] && buffer_cycle >= pending_read_ready_cycle[0]) {
      BufferVector value =
          accumulation_memory[0][pending_read_address[0].to_int()];

      accumulation_read_data_0.Push(value);
      pending_read[0] = false;
    }

    ac_int<16, false> address;
    if (!pending_read[0] && buffer_cycle % 3 == 0 &&
        accumulation_read_address_0.PopNB(address)) {
      require(write_count[0] > read_count[0] &&
                  last_write_completion_cycle[0][address.to_int()] >
                      last_read_write_cycle[0][address.to_int()],
              "bank 0 dependent read preceded its physical write completion");
      last_read_write_cycle[0][address.to_int()] =
          last_write_completion_cycle[0][address.to_int()];
      ordered_dependent_read_count++;
      pending_read[0] = true;
      pending_read_address[0] = address;
      pending_read_ready_cycle[0] = buffer_cycle + 2;
      read_addresses[0].push_back(address.to_int());
      read_count[0]++;
      if (independent_read_probe && address == CIM_LOCAL_ACCUM_CONTEXTS + 2) {
        independent_read_seen = true;
      }
    }

    WriteRequest write;
    if (!hold_independent_probe_write() && buffer_cycle % 4 == 0 &&
        accumulation_write_request_0.PopNB(write)) {
      accumulation_memory[0][write.address.to_int()] = write.data;
      last_write_completion_cycle[0][write.address.to_int()] = buffer_cycle + 1;
      write_addresses[0].push_back(write.address.to_int());
      write_count[0]++;
    }
  }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  // Apply a different deterministic backpressure phase to bank 1
  void service_bank_1() {
    if (pending_read[1] && buffer_cycle >= pending_read_ready_cycle[1]) {
      accumulation_read_data_1.Push(
          accumulation_memory[1][pending_read_address[1].to_int()]);
      pending_read[1] = false;
    }

    ac_int<16, false> address;
    if (!pending_read[1] && buffer_cycle % 3 == 1 &&
        accumulation_read_address_1.PopNB(address)) {
      require(write_count[1] > read_count[1] &&
                  last_write_completion_cycle[1][address.to_int()] >
                      last_read_write_cycle[1][address.to_int()],
              "bank 1 dependent read preceded its physical write completion");
      last_read_write_cycle[1][address.to_int()] =
          last_write_completion_cycle[1][address.to_int()];
      ordered_dependent_read_count++;
      pending_read[1] = true;
      pending_read_address[1] = address;
      pending_read_ready_cycle[1] = buffer_cycle + 2;
      read_addresses[1].push_back(address.to_int());
      read_count[1]++;
    }

    WriteRequest write;
    if (buffer_cycle % 4 == 1 && accumulation_write_request_1.PopNB(write)) {
      accumulation_memory[1][write.address.to_int()] = write.data;
      last_write_completion_cycle[1][write.address.to_int()] = buffer_cycle + 1;
      write_addresses[1].push_back(write.address.to_int());
      write_count[1]++;
    }
  }
#endif

  // Emulate MatrixUnit's persistent accumulation memory and ready/valid ports
  void run_accumulation_buffer() {
    accumulation_read_address_0.ResetRead();
    accumulation_read_data_0.ResetWrite();
    accumulation_write_request_0.ResetRead();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_read_address_1.ResetRead();
    accumulation_read_data_1.ResetWrite();
    accumulation_write_request_1.ResetRead();
#endif
    wait();

    while (true) {
      service_bank_0();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      service_bank_1();
#endif
      buffer_cycle++;
      wait();
    }
  }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  // Consume the bank-0 completion handshake
  void consume_done_0() {
    accumulation_done_0.ResetRead();
    wait();
    while (!rstn.read()) {
      wait();
    }
    while (true) {
      accumulation_done_0.SyncPop();
      done_count[0]++;
    }
  }

  // Consume the bank-1 completion handshake
  void consume_done_1() {
    accumulation_done_1.ResetRead();
    wait();
    while (!rstn.read()) {
      wait();
    }
    while (true) {
      accumulation_done_1.SyncPop();
      done_count[1]++;
    }
  }
#endif

  // Exercise complete jobs, changing weight patterns and schedules without
  // reset.
  void run() {
    params_channel.ResetWrite();
    input_channel.ResetWrite();
    start_channel.ResetRead();
    rstn.write(false);
    tick();
    tick();
    rstn.write(true);
    tick();
#if ENABLE_PERF_COUNTERS
    for (int i = 0; i < MatrixPerformance::PROCESSOR_COUNTER_COUNT; i++) {
      require(perf_counters[i].read() == 0, "counter: not zero after reset");
    }
#endif

    const BufferVector bias = queue_bias(10);
    for (int row = 0; row < 2; row++) {
      BufferVector expected = bias;
      add_vector(expected, expected_partial(row, 2 * row));
      add_vector(expected, expected_partial(row, 2 * row + 1));
      expect_output("biased reduction", expected, 8);
    }
    send_job(make_accumulation_params(true), {0, 1, 2, 3},
             {true, true, true, true}, make_weight_descriptors(4),
             {0, 0, 1, 1});

    for (bool traverse_x : {true, false}) {
      for (int output = 0; output < 4; output++) {
        expect_output("resident reuse", expected_partial(2, 20), 12);
      }
      send_job(make_weight_reuse_params(traverse_x), {20, 20, 20, 20},
               {true, false, false, false}, {make_weight_descriptor(1, 4)}, 2);
    }

    for (int row = 0; row < 2; row++) {
      BufferVector expected = expected_partial(3 + row, 30);
      add_vector(expected, expected_partial(3 + row, 31));
      expect_output("outer reduction replay", expected, 20);
    }
    send_job(make_set_major_replay_params(), {30, 30, 31, 31},
             {true, false, true, false},
             {make_weight_descriptor(1, 2), make_weight_descriptor(1, 2)},
             {3, 4, 3, 4});

    for (int descriptor = 0; descriptor < 2; descriptor++) {
      for (int replay = 0; replay < 2; replay++) {
        for (int set = 0; set < WEIGHT_SETS; set++) {
          expect_output("consecutive full-ring replay",
                        expected_partial(5, 40 + descriptor * 10 + set), 15);
        }
      }
    }
    send_consecutive_full_ring_replay_job(5, 40, 50);

    for (int set = 0; set < WEIGHT_SETS + 2; set++) {
      expect_output("progressive set release", expected_partial(6, 60 + set),
                    5);
    }
    send_progressive_release_job(6, 60);

    // Hold the first result of a long resident-set replay. The output FIFO
    // and result path absorb about twelve results; the remaining results
    // stall the array result channel, which check_performance_counters
    // requires.
    static constexpr int kBackpressureReplays = 8;
    for (int descriptor = 0; descriptor < 2; descriptor++) {
      for (int replay = 0; replay < kBackpressureReplays; replay++) {
        for (int set = 0; set < WEIGHT_SETS; set++) {
          const bool first = descriptor == 0 && replay == 0 && set == 0;
          expect_output("result backpressure",
                        expected_partial(2, 70 + descriptor * 10 + set),
                        first ? 800 : 0);
        }
      }
    }
    send_consecutive_full_ring_replay_job(2, 70, 80, kBackpressureReplays);
    const int expected_count = 12 + 4 * WEIGHT_SETS + WEIGHT_SETS + 2 +
                               2 * kBackpressureReplays * WEIGHT_SETS;
    while (checked_outputs < expected_count) tick();

    // Reuse a context across completed rows, then exceed local context
    // capacity.
    check_context_rows(6, 3, false, false, 0);
    check_context_rows(CIM_LOCAL_ACCUM_CONTEXTS + 1, 3, true, false, 0);
    check_context_rows(3, 3, false, false, 0, 1);
    check_context_rows(3, 3, false, false, 0, 2);
    check_independent_sram_read();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    check_context_rows(3, 3, false, true, 0);
    check_context_rows(3, 3, true, true, 1);
    check_context_rows(3, 1, false, false, 0);
#endif
    require(pending_weight_sets.empty() && pending_weight_descriptors.empty(),
            "unconsumed weight stream or descriptor");
#if ENABLE_PERF_COUNTERS
    check_performance_counters();
#endif
    if (!test_failed) {
      std::cout << "[PASS] cim_processor_scheduling_and_accumulation"
                << std::endl;
    }
    sc_stop();
  }
};

// Elaborate the selected CIMProcessor geometry and scenario set
int sc_main(int argc, char** argv) {
  CIMProcessorTb testbench("cim_processor_int8");
  sc_start();
  return testbench.test_failed ? 1 : 0;
}
