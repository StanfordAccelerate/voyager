// SystemC array contract tests: routing, reduction, resident weights, and backpressure


#include <ac_int.h>
#include <mc_connections.h>
#include <systemc.h>

#include <deque>
#include <iostream>
#include <sstream>
#include <string>

#include "cim/CIMArray.h"

static constexpr int CIM_MODE_BIT_PARALLEL_VALUE = 0;
static constexpr int CIM_MODE_BIT_SERIAL_VALUE = 1;

static int g_cases_remaining = 0;

// Return a mask covering the requested bit width
static constexpr long long mask_for_width(int width) {
  return (1LL << width) - 1;
}

// Exercise one parameterized CIMArray geometry using logical A/B/C coordinates
template <
    int MACRO_INPUT_LANES, int MACRO_OUTPUT_LANES, int WEIGHT_SETS, int BASE_A_WIDTH, int BASE_B_WIDTH,
    int BASE_C_WIDTH, int MACRO_WRITE_INPUT_LANES, int MAC_LATENCY, int MODE, int A_WIDTH,
    int B_WIDTH, bool IS_SIGNED, int TILE_INPUT_AXIS_ELEMENTS,
    int TILE_OUTPUT_AXIS_ELEMENTS, int INPUT_AXIS_TILES = 1,
    int OUTPUT_AXIS_TILES = 1, int A_PORT_TILES = INPUT_AXIS_TILES,
    int B_PORT_TILES = OUTPUT_AXIS_TILES, int C_PORT_TILES = INPUT_AXIS_TILES,
    int C_BEAT_LAYOUT = CIM_C_BEAT_INPUT_MAJOR,
    // Accumulator width; the processor takes this from ACCUM_DATATYPE, so the
    // standalone testbench over-provisions from the base width. The array's own
    // static_assert enforces that it holds one reduced result
    int C_WIDTH = BASE_C_WIDTH + 8,
    int RESULT_SLOTS_PER_OUTPUT_TILE = INPUT_AXIS_TILES,
    typename DutType = CIMArray<
        MACRO_INPUT_LANES, MACRO_OUTPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH, BASE_B_WIDTH, BASE_C_WIDTH,
        MACRO_WRITE_INPUT_LANES, MAC_LATENCY, MODE, A_WIDTH, B_WIDTH, C_WIDTH, IS_SIGNED,
        TILE_INPUT_AXIS_ELEMENTS, TILE_OUTPUT_AXIS_ELEMENTS, INPUT_AXIS_TILES,
        OUTPUT_AXIS_TILES, A_PORT_TILES, B_PORT_TILES, C_PORT_TILES,
        C_BEAT_LAYOUT, RESULT_SLOTS_PER_OUTPUT_TILE>>
struct CIMArrayTbCase : sc_module {
  using Dut = DutType;
  using ABeat = typename Dut::ABeat;
  using CBeat = typename Dut::CBeat;
  using MACRequest = typename Dut::MACRequest;
  using WriteRequest = typename Dut::WriteRequest;
  using Set = typename Dut::Set;

  static_assert(A_WIDTH < 31, "A_WIDTH must fit this unit test golden model");
  static_assert(B_WIDTH < 31, "B_WIDTH must fit this unit test golden model");
  static_assert(C_WIDTH < 62, "C_WIDTH must fit this unit test golden model");

  Dut dut;
  sc_clock clk;
  sc_signal<bool> rstn;
  Connections::Combinational<MACRequest> mac_request_channel;
  Connections::Combinational<WriteRequest> write_request_channel;
  Connections::Combinational<CBeat> result_channel;
  unsigned mac_accept_count = 0;
  unsigned total_mac_accept_count = 0;
  unsigned result_fire_count = 0;

  ac_int<B_WIDTH, false> expected_b[WEIGHT_SETS][Dut::INPUT_LANES][Dut::OUTPUT_LANES];

  // ExpectedBeat mirrors one C-port beat in issue order
  struct ExpectedBeat {
    long long value[C_PORT_TILES][Dut::TILE_OUTPUT_LANES];
  };
  std::deque<ExpectedBeat> expected_beats;

  SC_HAS_PROCESS(CIMArrayTbCase);

  // Construct one case and bind the standalone CIMArray ports
  explicit CIMArrayTbCase(sc_module_name name)
      : sc_module(name), dut("dut"), clk("clk", 10, SC_NS) {
    g_cases_remaining++;

    dut.clk(clk);
    dut.rstn(rstn);
    dut.mac_request_channel(mac_request_channel);
    dut.write_request_channel(write_request_channel);
    dut.result_channel(result_channel);

    clear_expected_state();

    SC_THREAD(run);
    sensitive << clk.posedge_event();

    SC_METHOD(observe_mac_acceptance);
    sensitive << clk.posedge_event();
    dont_initialize();

    SC_THREAD(watchdog);
  }

  // Fail this case with a contextual SystemC report
  void require(bool condition, const std::string& message) const {
    if (condition) {
      return;
    }

    std::ostringstream text;
    text << name() << ": " << message;
    const std::string report = text.str();
    SC_REPORT_FATAL("CIMArrayTb", report.c_str());
  }

  // Stop a lost request with a bounded simulation failure
  void watchdog() {
    wait(100, SC_US);
    require(false, "timed out waiting for CIMArray completion");
  }

  // Wait enough delta cycles for combinational methods to settle
  void settle() {
    for (int delta = 0; delta < 4; delta++) {
      wait(SC_ZERO_TIME);
    }
  }

  // Advance one CIMArray clock edge
  void tick() {
    wait(clk.posedge_event());
    settle();
  }

  // Count actual array and result transfers at their DUT-side interfaces
  void observe_mac_acceptance() {
    if (!rstn.read()) {
      mac_accept_count = 0;
      return;
    }
#ifdef CONNECTIONS_SIM_ONLY
    const bool mac_fired = mac_request_channel._VLDNAMEOUT_.read() &&
                           mac_request_channel._RDYNAMEOUT_.read();
    const bool result_fired =
        result_channel._VLDNAMEIN_.read() && result_channel._RDYNAMEIN_.read();
#else
    const bool mac_fired = mac_request_channel._VLDNAME_.read() &&
                           mac_request_channel._RDYNAME_.read();
    const bool result_fired =
        result_channel._VLDNAME_.read() && result_channel._RDYNAME_.read();
#endif
    if (mac_fired) {
      mac_accept_count++;
      total_mac_accept_count++;
    }
    if (result_fired) {
      result_fire_count++;
    }
  }

  // Reset the testbench-side Connections endpoints
  void reset_channels() {
    mac_request_channel.ResetWrite();
    write_request_channel.ResetWrite();
    result_channel.ResetRead();
  }

  // Encode one integer as an unsigned ac_int bit pattern
  template <int WIDTH>
  ac_int<WIDTH, false> encode_value(long long value) const {
    const long long mask = mask_for_width(WIDTH);
    return ac_int<WIDTH, false>(value & mask);
  }

  // Decode one unsigned ac_int bit pattern using the case signedness
  template <int WIDTH>
  long long decode_value(ac_int<WIDTH, false> value) const {
    const long long mask = mask_for_width(WIDTH);
    long long raw = static_cast<long long>(value.to_int()) & mask;
    if (!IS_SIGNED) {
      return raw;
    }

    const long long sign_bit = 1LL << (WIDTH - 1);
    return (raw & sign_bit) == 0 ? raw : raw | ~mask;
  }

  // Create deterministic A data for one array input channel
  ac_int<A_WIDTH, false> a_value(int input_index, int phase) const {
    if (IS_SIGNED) {
      return encode_value<A_WIDTH>(((input_index * 5 + phase) % 13) - 6);
    }
    return encode_value<A_WIDTH>((input_index + 2) * 3 + phase);
  }

  // Create deterministic B data for one logical matrix coordinate
  ac_int<B_WIDTH, false> b_value(int input_index, int output_index, int phase) const {
    if (IS_SIGNED) {
      return encode_value<B_WIDTH>(((input_index * 13 + output_index * 5 + phase) % 11) - 5);
    }
    return encode_value<B_WIDTH>((input_index + 1) * 2 + output_index * 3 + phase);
  }

  // Clear resident B state and pending expected C beats
  void clear_expected_state() {
    for (int set_idx = 0; set_idx < WEIGHT_SETS; set_idx++) {
      for (int input_index = 0; input_index < Dut::INPUT_LANES; input_index++) {
        for (int output_index = 0; output_index < Dut::OUTPUT_LANES; output_index++) {
          expected_b[set_idx][input_index][output_index] = 0;
        }
      }
    }
    expected_beats.clear();
  }

  // Apply reset and release into the B-loading phase
  void apply_reset() {
    rstn.write(false);
    reset_channels();
    settle();
    tick();
    rstn.write(true);
    tick();
  }

  // Drive one direct B-port span and mirror it into the logical B model
  void drive_write_direct(Set write_set, int input_tile_index,
                          int output_axis_tile_base, int tile_write_input_index, int phase) {
    WriteRequest request;
    request.write_set = write_set;
    request.input_tile_index = input_tile_index;
    request.output_axis_tile_base = output_axis_tile_base;
    request.write_input_index = tile_write_input_index;
    request.replicate = 0;

    for (int port_tile_idx = 0; port_tile_idx < B_PORT_TILES; port_tile_idx++) {
      for (int tile_output_index = 0; tile_output_index < Dut::TILE_OUTPUT_LANES; tile_output_index++) {
        for (int tile_write_input_offset = 0; tile_write_input_offset < Dut::TILE_WRITE_INPUT_LANES; tile_write_input_offset++) {
          const int input_index = input_tile_index * Dut::TILE_INPUT_LANES + tile_write_input_index + tile_write_input_offset;
          const int output_index =
              (output_axis_tile_base + port_tile_idx) * Dut::TILE_OUTPUT_LANES + tile_output_index;
          request.data[port_tile_idx][tile_write_input_offset][tile_output_index] = b_value(input_index, output_index, phase);
          expected_b[write_set.to_int()][input_index][output_index] =
              request.data[port_tile_idx][tile_write_input_offset][tile_output_index];
        }
      }
    }

    write_request_channel.Push(request);
    settle();
  }

  // Drive one replicated B tile and mirror beat tile zero across the output
  // axis
  void drive_write_replicate(Set write_set, int input_tile_index, int tile_write_input_index,
                             int phase) {
    WriteRequest request;
    request.write_set = write_set;
    request.input_tile_index = input_tile_index;
    request.output_axis_tile_base = 0;
    request.write_input_index = tile_write_input_index;
    request.replicate = 1;

    for (int port_tile_idx = 0; port_tile_idx < B_PORT_TILES; port_tile_idx++) {
      for (int tile_output_index = 0; tile_output_index < Dut::TILE_OUTPUT_LANES; tile_output_index++) {
        for (int tile_write_input_offset = 0; tile_write_input_offset < Dut::TILE_WRITE_INPUT_LANES; tile_write_input_offset++) {
          const int input_index = input_tile_index * Dut::TILE_INPUT_LANES + tile_write_input_index + tile_write_input_offset;
          const int source_output_index = port_tile_idx * Dut::TILE_OUTPUT_LANES + tile_output_index;
          request.data[port_tile_idx][tile_write_input_offset][tile_output_index] =
              b_value(input_index, source_output_index, phase);
        }
      }
    }

    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      for (int tile_output_index = 0; tile_output_index < Dut::TILE_OUTPUT_LANES; tile_output_index++) {
        for (int tile_write_input_offset = 0; tile_write_input_offset < Dut::TILE_WRITE_INPUT_LANES; tile_write_input_offset++) {
          const int input_index = input_tile_index * Dut::TILE_INPUT_LANES + tile_write_input_index + tile_write_input_offset;
          const int output_index = output_tile_index * Dut::TILE_OUTPUT_LANES + tile_output_index;
          expected_b[write_set.to_int()][input_index][output_index] = request.data[0][tile_write_input_offset][tile_output_index];
        }
      }
    }

    write_request_channel.Push(request);
    settle();
  }

  // Load every direct B-port span required by one resident weight set
  void load_weight_set(Set write_set, int phase, bool insert_idle = false) {
    for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
         input_tile_index++) {
      for (int tile_write_input_index = 0; tile_write_input_index < Dut::TILE_INPUT_LANES;
           tile_write_input_index += Dut::TILE_WRITE_INPUT_LANES) {
        for (int output_axis_tile_base = 0;
             output_axis_tile_base < OUTPUT_AXIS_TILES;
             output_axis_tile_base += B_PORT_TILES) {
          drive_write_direct(write_set, input_tile_index, output_axis_tile_base,
                             tile_write_input_index, phase);
          if (insert_idle) {
            tick();
            tick();
          }
        }
      }
    }
    tick();
  }

  // Load one weight set by replicating beat tile zero
  void load_weight_set_replicate(Set write_set, int phase,
                                 bool insert_idle = false) {
    for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
         input_tile_index++) {
      for (int tile_write_input_index = 0; tile_write_input_index < Dut::TILE_INPUT_LANES;
           tile_write_input_index += Dut::TILE_WRITE_INPUT_LANES) {
        drive_write_replicate(write_set, input_tile_index, tile_write_input_index, phase);
        if (insert_idle) {
          tick();
          tick();
        }
      }
    }
    tick();
  }

  // Compute one tile C vector for an expected beat
  void compute_expected_tile(ExpectedBeat& beat, int port_tile_idx, Set compute_set,
                             int input_tile_index, int output_tile_index,
                             int phase) const {
    for (int tile_output_index = 0; tile_output_index < Dut::TILE_OUTPUT_LANES; tile_output_index++) {
      const int output_index = output_tile_index * Dut::TILE_OUTPUT_LANES + tile_output_index;
      long long sum = 0;
      for (int tile_input_index = 0; tile_input_index < Dut::TILE_INPUT_LANES; tile_input_index++) {
        const int input_index = input_tile_index * Dut::TILE_INPUT_LANES + tile_input_index;
        const long long a = decode_value<A_WIDTH>(a_value(input_index, phase));
        const long long b =
            decode_value<B_WIDTH>(expected_b[compute_set.to_int()][input_index][output_index]);
        sum += a * b;
      }
      beat.value[port_tile_idx][tile_output_index] += sum;
    }
  }

  // Queue the C beats produced by one targeted or multicast MAC request
  void queue_expected_beats(Set compute_set, bool multicast,
                            int target_output_tile_index, bool reduce,
                            int phase) {
    const int selected_output_tiles = multicast ? OUTPUT_AXIS_TILES : 1;
    const int logical_results = reduce
                                    ? selected_output_tiles
                                    : selected_output_tiles * INPUT_AXIS_TILES;
    const int result_beats =
        (logical_results + C_PORT_TILES - 1) / C_PORT_TILES;
    for (int result_beat_idx = 0; result_beat_idx < result_beats;
         result_beat_idx++) {
      ExpectedBeat beat = {};
      for (int port_idx = 0; port_idx < C_PORT_TILES; port_idx++) {
        const int logical_idx = result_beat_idx * C_PORT_TILES + port_idx;
        if (logical_idx >= logical_results) {
          continue;
        }

        int output_axis_ordinal = 0;
        int input_tile_index = 0;
        if (reduce) {
          output_axis_ordinal = logical_idx;
        } else if constexpr (C_BEAT_LAYOUT == CIM_C_BEAT_INPUT_MAJOR) {
          output_axis_ordinal = logical_idx / INPUT_AXIS_TILES;
          input_tile_index = logical_idx % INPUT_AXIS_TILES;
        } else if (multicast) {
          input_tile_index = logical_idx / OUTPUT_AXIS_TILES;
          output_axis_ordinal = logical_idx % OUTPUT_AXIS_TILES;
        } else {
          input_tile_index = logical_idx;
        }
        const int output_tile_index =
            multicast ? output_axis_ordinal : target_output_tile_index;

        if (reduce) {
          for (int sum_input_tile_index = 0;
               sum_input_tile_index < INPUT_AXIS_TILES; sum_input_tile_index++) {
            compute_expected_tile(beat, port_idx, compute_set, sum_input_tile_index,
                                  output_tile_index, phase);
          }
        } else {
          compute_expected_tile(beat, port_idx, compute_set, input_tile_index,
                                output_tile_index, phase);
        }
      }
      expected_beats.push_back(beat);
    }
  }

  // Build one complete A beat for a deterministic phase
  ABeat build_a_beat(int phase) const {
    ABeat beat;
    for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
         input_tile_index++) {
      for (int tile_input_index = 0; tile_input_index < Dut::TILE_INPUT_LANES; tile_input_index++) {
        const int input_index = input_tile_index * Dut::TILE_INPUT_LANES + tile_input_index;
        beat[input_tile_index][tile_input_index] = a_value(input_index, phase);
      }
    }
    return beat;
  }

  // Build one atomic MAC request with its deterministic A beat
  MACRequest build_mac_request(Set compute_set, int phase) const {
    MACRequest request;
    request.compute_set = compute_set;
    request.output_tile_index = 0;
    request.multicast = 0;
    request.reduce = 0;
    request.a = build_a_beat(phase);
    return request;
  }

  // Wait until one result has transferred into the testbench endpoint
  void wait_for_result_fire(unsigned start_count) {
    for (int cycle = 0; cycle < 100 && result_fire_count == start_count;
         cycle++) {
      tick();
    }
    require(result_fire_count == start_count + 1,
            "timed out waiting for one DUT-side result transfer");
  }

  // Admit one MAC while independently draining older ready results
  void push_mac_request(const MACRequest& request) {
    const unsigned accepted_before = mac_accept_count;
    while (!mac_request_channel.PushNB(request)) {
      CBeat actual;
      if (result_channel.PopNB(actual)) {
        check_beat(actual);
      }
      tick();
    }
    settle();
    while (mac_accept_count == accepted_before) {
      CBeat actual;
      if (result_channel.PopNB(actual)) {
        check_beat(actual);
      }
      tick();
    }
  }

  // Drive one targeted MAC request and queue its expected C beats
  void drive_targeted_mac(Set compute_set, int output_tile_index, int phase,
                          bool reduce = false) {
    MACRequest request = build_mac_request(compute_set, phase);
    request.output_tile_index = output_tile_index;
    request.reduce = reduce;
    queue_expected_beats(compute_set, false, output_tile_index, reduce, phase);
    push_mac_request(request);
  }

  // Drive one multicast MAC request and queue its expected C beats
  void drive_multicast_mac(Set compute_set, int phase, bool reduce = false) {
    MACRequest request = build_mac_request(compute_set, phase);
    request.multicast = 1;
    request.reduce = reduce;
    queue_expected_beats(compute_set, true, 0, reduce, phase);
    push_mac_request(request);
  }

  // Admit one MAC while deliberately holding all older output beats
  void drive_held_mac(Set compute_set, int output_tile_index, bool multicast,
                      bool reduce, int phase) {
    MACRequest request = build_mac_request(compute_set, phase);
    request.output_tile_index = output_tile_index;
    request.multicast = multicast;
    request.reduce = reduce;
    queue_expected_beats(compute_set, multicast, output_tile_index, reduce, phase);
    mac_request_channel.Push(request);
    settle();
  }

  // Compare one received C beat against the oldest expected beat
  void check_beat(const CBeat& actual) {
    require(!expected_beats.empty(), "C beat popped with no expected beat");

    const ExpectedBeat expected = expected_beats.front();
    expected_beats.pop_front();

    for (int port_tile_idx = 0; port_tile_idx < C_PORT_TILES; port_tile_idx++) {
      for (int tile_output_index = 0; tile_output_index < Dut::TILE_OUTPUT_LANES; tile_output_index++) {
        const ac_int<C_WIDTH, false> expected_bits =
            encode_value<C_WIDTH>(expected.value[port_tile_idx][tile_output_index]);
        if (actual[port_tile_idx][tile_output_index] == expected_bits) {
          continue;
        }

        std::ostringstream text;
        text << "unexpected C beat[" << port_tile_idx << "][" << tile_output_index
             << "] got " << actual[port_tile_idx][tile_output_index].to_int()
             << " expected " << expected_bits.to_int();
        require(false, text.str());
      }
    }
  }

  // Pop one C beat and compare it against the oldest expected beat
  void pop_and_check() {
    check_beat(result_channel.Pop());
    settle();
  }

  // Pop and check every outstanding expected C beat
  void drain_expected_beats() {
    while (!expected_beats.empty()) {
      pop_and_check();
    }
  }

  // Return a valid weight set for one deterministic transaction phase
  Set transaction_write_set(int phase) const { return Set(phase % WEIGHT_SETS); }

  // Run one multicast transaction with optional idle and backpressure cycles
  void run_multicast_transaction(int phase, bool insert_idle_cycles,
                                 int c_backpressure_cycles) {
    const Set write_set = transaction_write_set(phase);
    load_weight_set(write_set, phase);
    if (insert_idle_cycles) {
      tick();
    }
    drive_multicast_mac(write_set, phase + 3);
    if (insert_idle_cycles) {
      tick();
    }
    for (int cycle = 0; cycle < c_backpressure_cycles; cycle++) {
      tick();
    }
    drain_expected_beats();
  }

  // Check direct B loading and C backpressure with and without idle spacing
  void run_basic_transaction_checks() {
    run_multicast_transaction(1, true, 3);
    run_multicast_transaction(13, false, 3);
  }

  // Check optional reduction for multicast and targeted requests
  void run_reduced_result_check() {
    const Set write_set = transaction_write_set(17);
    load_weight_set(write_set, 17);
    drive_multicast_mac(write_set, 19, true);
    drain_expected_beats();
    drive_targeted_mac(write_set, OUTPUT_AXIS_TILES - 1, 23, true);
    drain_expected_beats();
  }

  // Check direct and replicated B requests separated by empty cycles
  void run_gapped_write_check() {
    const Set direct_write_set = transaction_write_set(117);
    load_weight_set(direct_write_set, 117, true);
    drive_multicast_mac(direct_write_set, 121);
    drain_expected_beats();

    const Set replicate_write_set = transaction_write_set(127);
    load_weight_set_replicate(replicate_write_set, 127, true);
    drive_multicast_mac(replicate_write_set, 131);
    drain_expected_beats();
  }

  // Check that distinct weight sets remain independently addressable
  void run_set_retention_check() {
    const Set first_write_set = Set(0);
    const Set second_write_set = Set((WEIGHT_SETS > 1) ? 1 : 0);
    load_weight_set(first_write_set, 21);
    load_weight_set(second_write_set, 29);
    drive_multicast_mac(first_write_set, 35);
    drain_expected_beats();
    drive_multicast_mac(second_write_set, 39);
    drain_expected_beats();
    drive_multicast_mac(first_write_set, 43);
    drain_expected_beats();
  }

  // Check aligned B-port spans and preservation of an untouched span
  void run_b_port_span_retention_check() {
    if constexpr (OUTPUT_AXIS_TILES > B_PORT_TILES) {
      const Set write_set = transaction_write_set(137);
      load_weight_set(write_set, 137);
      for (int tile_write_input_index = 0; tile_write_input_index < Dut::TILE_INPUT_LANES;
           tile_write_input_index += Dut::TILE_WRITE_INPUT_LANES) {
        drive_write_direct(write_set, 0, 0, tile_write_input_index, 149);
      }
      tick();
      drive_multicast_mac(write_set, 151);
      drain_expected_beats();
    }
  }

  // Check targeted issue to every output-axis tile back to back
  void run_targeted_issue_check() {
    const Set compute_set = transaction_write_set(47);
    load_weight_set(compute_set, 47);
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      drive_targeted_mac(compute_set, output_tile_index, 49 + output_tile_index);
    }
    drain_expected_beats();
  }

  // Diverge output-tile cursors before reduced and raw multicasts join them
  void run_mixed_completion_cursor_check() {
    if constexpr (INPUT_AXIS_TILES != 2 || OUTPUT_AXIS_TILES < 3 ||
                  RESULT_SLOTS_PER_OUTPUT_TILE < 6) {
      return;
    } else {
      const Set compute_set = transaction_write_set(157);
      load_weight_set(compute_set, 157);
      const unsigned accepted_before = total_mac_accept_count;

      drive_held_mac(compute_set, 0, false, true, 163);
      drive_held_mac(compute_set, 1, false, false, 167);
      drive_held_mac(compute_set, 0, true, true, 173);
      drive_held_mac(compute_set, 0, false, false, 179);
      drive_held_mac(compute_set, 2, false, true, 181);
      drive_held_mac(compute_set, 0, true, false, 191);

      for (int cycle = 0; cycle < MAC_LATENCY + 6; cycle++) {
        tick();
      }
      require(total_mac_accept_count - accepted_before == 6,
              "mixed completion sequence did not accept every request");

      drain_expected_beats();
      require(expected_beats.empty(),
              "mixed completion sequence left expected results undrained");
    }
  }

  // Check sustained issue with the configured result slot count
  // Check that full lane-local result slots stop issue and retain every result
  // through output backpressure
  void run_result_slot_backpressure_check() {
    const Set compute_set = transaction_write_set(59);
    load_weight_set(compute_set, 59);

    for (int operation = 0; operation < RESULT_SLOTS_PER_OUTPUT_TILE;
         operation++) {
      const int phase = 300 + operation;
      MACRequest request = build_mac_request(compute_set, phase);
      request.output_tile_index = 0;
      request.reduce = 1;
      queue_expected_beats(compute_set, false, 0, true, phase);
      mac_request_channel.Push(request);
      settle();
    }

    MACRequest blocked_request = build_mac_request(compute_set, 400);
    blocked_request.output_tile_index = 0;
    blocked_request.reduce = 1;
    require(!mac_request_channel.PushNB(blocked_request),
            "completion storage accepted a request without a free result slot");

    drain_expected_beats();

    queue_expected_beats(compute_set, false, 0, true, 400);
    mac_request_channel.Push(blocked_request);
    pop_and_check();
  }

  // Replicate beat tile zero, then verify targeted and multicast requests
  void run_replicate_write_check() {
    const Set compute_set = transaction_write_set(107);
    load_weight_set_replicate(compute_set, 107);
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      drive_targeted_mac(compute_set, output_tile_index, 109 + output_tile_index);
    }
    drain_expected_beats();
    drive_multicast_mac(compute_set, 113);
    drain_expected_beats();
  }

  // Check that an old set can be overwritten after a new-set MAC is accepted
  void run_overwrite_after_set_switch_check() {
    if constexpr (WEIGHT_SETS < 2) {
      return;
    }

    const Set old_write_set = Set(0);
    const Set new_write_set = Set(1);
    load_weight_set(old_write_set, 79);
    load_weight_set(new_write_set, 89);
    drive_multicast_mac(old_write_set, 83);
    drive_multicast_mac(new_write_set, 97);
    load_weight_set(old_write_set, 101);
    drain_expected_beats();
    drive_multicast_mac(old_write_set, 103);
    drain_expected_beats();
  }

  // Reset a stalled final result and confirm the next operation completes
  void run_reset_recovery_check() {
    const Set compute_set = transaction_write_set(31);
    load_weight_set(compute_set, 31);

    const unsigned blocker_start = result_fire_count;
    drive_held_mac(compute_set, 0, false, true, 37);
    wait_for_result_fire(blocker_start);

    const unsigned staged_accept_start = total_mac_accept_count;
    drive_held_mac(compute_set, 0, false, true, 41);
    for (int cycle = 0;
         cycle < 100 && total_mac_accept_count != staged_accept_start + 1;
         cycle++) {
      tick();
    }
    require(total_mac_accept_count == staged_accept_start + 1,
            "reset test did not accept its staged completion");

    const unsigned stalled_final_count = result_fire_count;
    for (int cycle = 0; cycle < MAC_LATENCY + 6; cycle++) {
      tick();
    }
    require(result_fire_count == stalled_final_count,
            "reset test failed to hold its final result");

    rstn.write(false);
    reset_channels();
    expected_beats.clear();
    settle();
    tick();
    rstn.write(true);
    tick();

    run_multicast_transaction(103, true, 3);
  }

  // Run the full case sequence
  void run() {
    apply_reset();
    clear_expected_state();

    run_basic_transaction_checks();
    run_reduced_result_check();
    run_gapped_write_check();
    run_set_retention_check();
    run_b_port_span_retention_check();
    run_targeted_issue_check();
    run_mixed_completion_cursor_check();
    run_result_slot_backpressure_check();
    run_replicate_write_check();
    run_overwrite_after_set_switch_check();
    run_reset_recovery_check();

    std::cout << "[PASS] " << name() << std::endl;
    g_cases_remaining--;
    if (g_cases_remaining == 0) {
      sc_stop();
    }
  }
};

// Four fixtures cover distinct interfaces rather than sweeping geometry.
int sc_main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  CIMArrayTbCase<4, 2, 2, 8, 8, 20, 2, 1, CIM_MODE_BIT_PARALLEL_VALUE, 8, 8,
                 true, 2, 2, 1, 2>
      minimum_storage("minimum_storage");

  CIMArrayTbCase<4, 2, 2, 4, 4, 12, 2, 2, CIM_MODE_BIT_SERIAL_VALUE, 4, 4,
                 true, 2, 2, 2, 2>
      signed_serial("signed_serial");

  CIMArrayTbCase<4, 2, 2, 4, 4, 12, 2, 2, CIM_MODE_BIT_PARALLEL_VALUE, 4, 4,
                 true, 2, 2, 2, 3, 2, 3, 2, CIM_C_BEAT_INPUT_MAJOR, 20, 6>
      input_major("input_major");

  CIMArrayTbCase<4, 2, 2, 4, 4, 12, 2, 2, CIM_MODE_BIT_PARALLEL_VALUE, 4, 4,
                 true, 2, 2, 2, 4, 2, 2, 4, CIM_C_BEAT_OUTPUT_MAJOR, 20, 6>
      output_major_narrow_weights("output_major_narrow_weights");

  sc_start();
  return g_cases_remaining;
}
