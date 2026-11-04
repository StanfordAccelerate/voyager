// SystemC/RTL co-simulation tests for the PE-level CIMElement adapter
//
// These tests drive the behavioral SystemC CIMElement model and a Verilated
// CIMIntElementPacked RTL model through the Catapult blackbox ABI, then compare
// the observable issue/ready contract, retire pulses, and result payloads cycle by cycle

#include <ac_int.h>
#include <systemc.h>
#include <verilated.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "cim/CIMElement.h"

static constexpr int CIM_MODE_BIT_SERIAL_VALUE = 1;

static int g_cases_remaining = 0;

// One generated co-simulation case descriptor
struct CIMElementTbCaseDescriptor {
  const char* name;
  void (*instantiate)();
};

// Provide Verilator's non-SystemC $time hook from the active SystemC kernel
double sc_time_stamp() { return sc_core::sc_time_stamp().to_default_time_units(); }

// Return a mask covering the requested bit width
static constexpr std::uint64_t mask_for_width(int width) {
  return width >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << width) - 1);
}

// One parameterized CIMElement SystemC/RTL co-simulation case
template <typename RtlModel, int INPUT_LANES, int OUTPUT_LANES, int WEIGHT_SETS, int BASE_A_WIDTH, int BASE_B_WIDTH, int BASE_C_WIDTH, int WRITE_INPUT_LANES,
          int MAC_LATENCY, int MODE, int A_WIDTH, int B_WIDTH, bool IS_SIGNED>
struct CIMElementTbCase : sc_module {
  static constexpr int MACRO_INPUT_LANES = INPUT_LANES;
  static constexpr int MACRO_OUTPUT_LANES = OUTPUT_LANES * (B_WIDTH / BASE_B_WIDTH);
  static constexpr int MACRO_WRITE_INPUT_LANES = WRITE_INPUT_LANES;
  using Dut = CIMElement<MACRO_INPUT_LANES, MACRO_OUTPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH, BASE_B_WIDTH, BASE_C_WIDTH, MACRO_WRITE_INPUT_LANES, MAC_LATENCY,
                         MODE, A_WIDTH, B_WIDTH, IS_SIGNED>;

  static_assert(Dut::A_BUS_WIDTH <= 64, "Verilated packed a_bus must fit uint64_t");
  static_assert(Dut::B_BUS_WIDTH <= 64, "Verilated packed b_bus must fit uint64_t");
  static_assert(Dut::C_BUS_WIDTH <= 64, "Verilated packed c_bus must fit uint64_t");

  Dut dut;
  RtlModel rtl;
  sc_signal<bool> clk;
  sc_signal<bool> rstn;
  sc_signal<typename Dut::AData> a;
  sc_signal<typename Dut::BData> b;
  sc_signal<bool> wen;
  sc_signal<ac_int<Dut::INPUT_INDEX_WIDTH, false>> write_input_index;
  sc_signal<ac_int<Dut::SET_INDEX_WIDTH, false>> write_set;
  sc_signal<bool> mac_issue;
  sc_signal<ac_int<Dut::SET_INDEX_WIDTH, false>> compute_set;
  sc_signal<typename Dut::CData> c;
  sc_signal<bool> c_retire;
  sc_signal<bool> mac_ready;
  sc_signal<bool> mac_busy;

  SC_HAS_PROCESS(CIMElementTbCase);

  // Construct one case and bind the CIMElement ports
  explicit CIMElementTbCase(sc_module_name name) : sc_module(name), dut("dut"), rtl(name) {
    g_cases_remaining++;

    dut.wclk(clk);
    dut.mclk(clk);
    dut.rstn(rstn);
    dut.wen(wen);
    dut.write_input_index(write_input_index);
    dut.write_set(write_set);
    dut.mac_issue(mac_issue);
    dut.compute_set(compute_set);
    dut.c_retire(c_retire);
    dut.mac_ready(mac_ready);
    dut.mac_busy(mac_busy);
    dut.a(a);
    dut.b(b);
    dut.c(c);

    SC_THREAD(run);
  }

  // Fail this case with a contextual SystemC report
  void require(bool condition, const std::string& message) const {
    if (condition) {
      return;
    }

    std::ostringstream text;
    text << name() << ": " << message;
    const std::string report = text.str();
    SC_REPORT_FATAL("CIMElementTb", report.c_str());
  }

  // Return one unsigned SystemC signal value as a masked integer
  template <int WIDTH>
  static std::uint64_t signal_value(sc_signal<ac_int<WIDTH, false>>& signal) {
    return static_cast<std::uint64_t>(signal.read().to_uint64()) & mask_for_width(WIDTH);
  }

  // Encode one integer as an unsigned ac_int bit pattern
  template <int WIDTH>
  ac_int<WIDTH, false> encode_value(int value) const {
    const std::uint64_t raw = static_cast<std::uint64_t>(value) & mask_for_width(WIDTH);
    return ac_int<WIDTH, false>(raw);
  }

  // Create deterministic A data for one phase
  ac_int<A_WIDTH, false> a_value(int input_index, int phase) const {
    if (IS_SIGNED) {
      return encode_value<A_WIDTH>(((input_index * 3 + phase * 2) % 9) - 4);
    }
    return encode_value<A_WIDTH>(((input_index + 1) * (phase + 2)) & 0x1f);
  }

  // Create deterministic B data for one resident set and matrix coordinate
  ac_int<B_WIDTH, false> b_value(int set_idx, int output_index, int input_index) const {
    if (IS_SIGNED) {
      return encode_value<B_WIDTH>(((set_idx * 5 + output_index * 3 + input_index * 2) % 11) - 5);
    }
    return encode_value<B_WIDTH>(((set_idx + 1) * (output_index + 2) + input_index + 1) & 0x1f);
  }

  // Pack the INPUT_LANES vector into the blackbox ABI order
  std::uint64_t pack_a_bus() {
    std::uint64_t packed = 0;
    const typename Dut::AData a_input = a.read();
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
      packed |= static_cast<std::uint64_t>(a_input[input_index].to_uint64()) << (input_index * A_WIDTH);
    }
    return packed;
  }

  // Pack the WRITE_INPUT_LANES-by-OUTPUT_LANES B write block into the blackbox ABI order
  std::uint64_t pack_b_bus() {
    std::uint64_t packed = 0;
    const typename Dut::BData b_input = b.read();
    for (int write_input_offset = 0; write_input_offset < Dut::WRITE_INPUT_LANES; write_input_offset++) {
      for (int output_index = 0; output_index < Dut::OUTPUT_LANES; output_index++) {
        const int bit_offset = ((write_input_offset * Dut::OUTPUT_LANES) + output_index) * B_WIDTH;
        packed |= static_cast<std::uint64_t>(b_input[write_input_offset][output_index].to_uint64()) << bit_offset;
      }
    }
    return packed;
  }

  // Unpack one C matrix column from the blackbox ABI order
  std::uint64_t rtl_c_value(int output_index) const {
    const std::uint64_t packed = static_cast<std::uint64_t>(rtl.c_bus);
    return (packed >> (output_index * Dut::C_WIDTH)) & mask_for_width(Dut::C_WIDTH);
  }

  // Copy current SystemC inputs into the Verilated RTL model
  void drive_rtl_inputs() {
    rtl.wclk = clk.read();
    rtl.mclk = clk.read();
    rtl.rstn = rstn.read();
    rtl.wen = wen.read();
    rtl.write_input_index = signal_value(write_input_index);
    rtl.write_set = signal_value(write_set);
    rtl.mac_issue = mac_issue.read();
    rtl.compute_set = signal_value(compute_set);
    rtl.a_bus = pack_a_bus();
    rtl.b_bus = pack_b_bus();
  }

  // Evaluate RTL, settle SystemC deltas, and compare observable outputs
  void settle_and_compare(const char* label) {
    drive_rtl_inputs();
    rtl.eval();

    for (int delta = 0; delta < 4; delta++) {
      wait(SC_ZERO_TIME);
      drive_rtl_inputs();
      rtl.eval();
    }

    compare_outputs(label);
  }

  // Compare issue status, retirement, and the registered result every cycle
  void compare_outputs(const char* label) {
    if (mac_ready.read() != static_cast<bool>(rtl.mac_ready)) {
      std::ostringstream text;
      text << label << ": mac_ready mismatch, SystemC=" << mac_ready.read()
           << " RTL=" << static_cast<int>(rtl.mac_ready);
      require(false, text.str());
    }

    if (mac_busy.read() != static_cast<bool>(rtl.mac_busy)) {
      std::ostringstream text;
      text << label << ": mac_busy mismatch, SystemC=" << mac_busy.read()
           << " RTL=" << static_cast<int>(rtl.mac_busy);
      require(false, text.str());
    }

    if (c_retire.read() != static_cast<bool>(rtl.c_retire)) {
      std::ostringstream text;
      text << label << ": c_retire mismatch, SystemC=" << c_retire.read() << " RTL=" << static_cast<int>(rtl.c_retire);
      require(false, text.str());
    }

    const typename Dut::CData c_output = c.read();
    for (int output_index = 0; output_index < Dut::OUTPUT_LANES; output_index++) {
      const std::uint64_t sysc_value = static_cast<std::uint64_t>(c_output[output_index].to_uint64());
      const std::uint64_t rtl_value = rtl_c_value(output_index);
      if (sysc_value == rtl_value) {
        continue;
      }

      std::ostringstream text;
      text << label << ": c[" << output_index << "] mismatch, SystemC=" << sysc_value << " RTL=" << rtl_value;
      require(false, text.str());
    }
  }

  // Advance one shared wclk/mclk cycle and check both models after each phase
  void tick(const char* label) {
    clk.write(false);
    settle_and_compare(label);

    clk.write(true);
    settle_and_compare(label);

    clk.write(false);
    settle_and_compare(label);
  }

  // Drive inactive defaults onto all inputs
  void initialize_inputs() {
    clk.write(false);
    rstn.write(false);
    wen.write(false);
    write_input_index.write(0);
    write_set.write(0);
    mac_issue.write(false);
    compute_set.write(0);
    typename Dut::AData a_input;
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
      a_input[input_index] = 0;
    }
    a.write(a_input);
    typename Dut::BData b_input;
    for (int write_input_offset = 0; write_input_offset < Dut::WRITE_INPUT_LANES; write_input_offset++) {
      for (int output_index = 0; output_index < Dut::OUTPUT_LANES; output_index++) {
        b_input[write_input_offset][output_index] = 0;
      }
    }
    b.write(b_input);
    settle_and_compare("initialize");
  }

  // Apply reset and release into an idle ready state
  void apply_reset() {
    rstn.write(false);
    wen.write(false);
    mac_issue.write(false);
    settle_and_compare("reset asserted");
    tick("reset clock");

    rstn.write(true);
    settle_and_compare("reset release");
    require(mac_ready.read(), "mac_ready stayed low after reset release");
    require(!mac_busy.read(), "mac_busy stayed high after reset release");
    require(!c_retire.read(), "c_retire recovered high after reset release");
  }

  // Drive one A vector
  void drive_a(int phase) {
    typename Dut::AData a_input;
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
      a_input[input_index] = a_value(input_index, phase);
    }
    a.write(a_input);
  }

  // Write one WRITE_INPUT_LANES-wide B block into both models
  void write_b_block(int set_idx, int base_input_index) {
    wen.write(true);
    write_set.write(set_idx);
    write_input_index.write(base_input_index);

    typename Dut::BData b_input;
    for (int write_input_offset = 0; write_input_offset < Dut::WRITE_INPUT_LANES; write_input_offset++) {
      for (int output_index = 0; output_index < Dut::OUTPUT_LANES; output_index++) {
        const int input_index = base_input_index + write_input_offset;
        b_input[write_input_offset][output_index] = b_value(set_idx, output_index, input_index);
      }
    }
    b.write(b_input);

    tick("write B");
    wen.write(false);
    settle_and_compare("B write idle");
  }

  // Load every resident B set before any MAC reads it
  void load_b_sets() {
    for (int set_idx = 0; set_idx < WEIGHT_SETS; set_idx++) {
      for (int base_input_index = 0; base_input_index < INPUT_LANES; base_input_index += Dut::WRITE_INPUT_LANES) {
        write_b_block(set_idx, base_input_index);
      }
    }
  }

  // Wait until both models pulse c_retire or fail on timeout
  void wait_for_retire() {
    for (int cycle = 0; cycle < 128; cycle++) {
      if (c_retire.read()) {
        return;
      }
      tick("wait retire");
    }
    require(false, "timed out waiting for c_retire");
  }

  // Launch a MAC and check next-cycle readiness against current-cycle occupancy
  void run_mac_check(int set_idx, int phase, bool pulse_while_pending) {
    drive_a(phase);
    compute_set.write(set_idx);
    mac_issue.write(true);
    settle_and_compare("arm mac");
    require(mac_busy.read(), "mac_busy stayed low on the issue edge");
    require(mac_ready.read() == (Dut::issue_window() == 1),
            "mac_ready did not predict the post-issue window");
    tick("launch mac");
    mac_issue.write(false);
    settle_and_compare("mac launched");
    bool saw_ready_busy_overlap = mac_ready.read() && mac_busy.read();

    if (pulse_while_pending && Dut::issue_window() > 1) {
      mac_issue.write(true);
      settle_and_compare("arm ignored issue");
      require(mac_busy.read(), "mac_busy dropped before the issue window ended");
      tick("ignored issue pulse");
      mac_issue.write(false);
      settle_and_compare("ignored issue cleared");
      saw_ready_busy_overlap =
          saw_ready_busy_overlap || (mac_ready.read() && mac_busy.read());
    }

    if (Dut::issue_window() > 1) {
      for (int cycle = 0; cycle < Dut::issue_window(); cycle++) {
        if (!mac_busy.read() || saw_ready_busy_overlap) {
          break;
        }
        tick("advance issue window");
        saw_ready_busy_overlap =
            saw_ready_busy_overlap || (mac_ready.read() && mac_busy.read());
      }
      require(saw_ready_busy_overlap,
              "mac_ready did not overlap the final busy cycle");
      if (mac_busy.read()) {
        tick("finish issue window");
      }
      require(mac_ready.read(), "mac_ready dropped after the issue window");
      require(!mac_busy.read(), "mac_busy stayed high after operand use");
    } else {
      require(mac_ready.read(),
              "mac_ready dropped for a one-cycle issue window");
      require(!mac_busy.read(), "mac_busy stayed high after a one-cycle issue");
    }

    wait_for_retire();
    compare_outputs("completed mac");
    tick("hold result");
    compare_outputs("held result");
  }

  // Issue ops back to back at the ready cadence while comparing both models
  void run_pipelined_check(int set_idx, int base_phase) {
    constexpr int kOps = 3;
    int captured = 0;
    int retired = 0;
    int guard = 0;
    bool issue_pending = false;

    while (retired < kOps) {
      mac_issue.write(issue_pending);
      settle_and_compare("pipelined setup");
      const bool capture_next = captured < kOps && mac_ready.read();
      tick("pipelined");
      mac_issue.write(false);

      if (c_retire.read()) {
        retired++;
      }

      if (capture_next) {
        drive_a(base_phase + captured);
        compute_set.write(set_idx);
        captured++;
        issue_pending = true;
      } else {
        issue_pending = false;
      }
      settle_and_compare("pipelined capture");

      require(++guard < 512, "pipelined check stalled");
    }
  }

  // Reset an in-flight MAC and confirm resetless B storage serves later operations
  void run_reset_mid_operation_check() {
    drive_a(7);
    compute_set.write(0);
    mac_issue.write(true);
    tick("launch reset test");
    require(mac_busy.read(), "mac_busy did not cover the reset test operation");

    mac_issue.write(false);
    rstn.write(false);
    settle_and_compare("mid-operation reset");
    tick("mid-operation reset clock");
    rstn.write(true);
    settle_and_compare("mid-operation reset release");
    require(mac_ready.read(), "mac_ready stayed low after mid-operation reset");
    require(!mac_busy.read(), "mac_busy stayed high after mid-operation reset");
    require(!c_retire.read(), "c_retire recovered high after mid-operation reset");

    run_mac_check(WEIGHT_SETS - 1, 8, false);
  }

  // Reset while idle and confirm resetless B storage serves later operations
  void run_idle_reset_b_retention_check() {
    rstn.write(false);
    settle_and_compare("idle reset");
    tick("idle reset clock");
    rstn.write(true);
    settle_and_compare("idle reset release");
    require(mac_ready.read(), "mac_ready stayed low after idle reset");
    require(!mac_busy.read(), "mac_busy stayed high after idle reset");
    require(!c_retire.read(), "c_retire recovered high after idle reset");

    run_mac_check(WEIGHT_SETS - 1, 8, false);
  }

  // Run the full case sequence
  void run() {
    initialize_inputs();
    apply_reset();
    load_b_sets();
    run_mac_check(0, 1, true);
    run_mac_check(WEIGHT_SETS / 2, 3, false);
    run_pipelined_check(0, 20);
    if constexpr (MODE == CIM_MODE_BIT_SERIAL_VALUE) {
      run_idle_reset_b_retention_check();
    } else {
      run_reset_mid_operation_check();
    }

    rtl.final();
    std::cout << "[PASS] " << name() << std::endl;
    g_cases_remaining--;
    if (g_cases_remaining == 0) {
      sc_stop();
    }
  }
};

#include "CIMElementTbCases.inc"

// Elaborate all deterministic CIMElement co-simulation cases
int sc_main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  for (int index = 0; index < kCIMElementTbCaseCount; index++) {
    kCIMElementTbCases[index].instantiate();
  }

  sc_start();
  return g_cases_remaining;
}
