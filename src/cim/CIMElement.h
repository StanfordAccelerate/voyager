// SystemC HLS adapter for the integer CIM element RTL block
//
// CIMElement is a PE-level Catapult block boundary for the native CIM vector
// and matrix interface. The synthesized implementation blackboxes the existing
// SystemVerilog CIMIntElement, while the C++ body provides event-level
// simulation of the issue/retire protocol: mac_ready lets upstream commit a
// request one cycle before mac_issue, mac_busy covers current operand use, and
// results retire into c with a one-cycle c_retire pulse after a fixed latency
//
// See cim_element.sv for the element geometry and weight-slice diagrams.

#pragma once

#include <ac_blackbox.h>
#include <ac_int.h>
#include <systemc.h>

#include <sstream>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

// MACRO_INPUT_LANES, MACRO_OUTPUT_LANES, and MACRO_WRITE_INPUT_LANES describe the physical macro shape
// INPUT_LANES, OUTPUT_LANES, and WRITE_INPUT_LANES describe the logical tensor shape
// A_WIDTH, B_WIDTH, and SIGNED describe the logical arithmetic
// CIMElementPacked owns the Catapult blackbox ABI with packed A/B/C buses
template <int MACRO_INPUT_LANES, int MACRO_OUTPUT_LANES, int WEIGHT_SETS,
          int BASE_A_WIDTH, int BASE_B_WIDTH,
          int BASE_C_WIDTH, int MACRO_WRITE_INPUT_LANES, int MAC_LATENCY, int MODE,
          int A_WIDTH, int B_WIDTH, bool SIGNED>
SC_MODULE(CIMElementPacked) {
 private:
  // Return the ceil log2 used for static port widths
  static constexpr int log2_ceil(int value) {
    return (value <= 1) ? 0 : 1 + log2_ceil((value + 1) / 2);
  }

  // Return the ceiling division for static latency derivation
  static constexpr int ceil_div(int dividend, int divisor) {
    return (dividend + divisor - 1) / divisor;
  }

  // Return the smaller of two static integers
  static constexpr int min_int(int lhs, int rhs) {
    return (lhs < rhs) ? lhs : rhs;
  }

 public:
  static constexpr int CIM_MODE_BIT_PARALLEL_VALUE = 0;
  static constexpr int CIM_MODE_BIT_SERIAL_VALUE = 1;
  static constexpr int INPUT_LANES = MACRO_INPUT_LANES;
  static constexpr int WRITE_INPUT_LANES = MACRO_WRITE_INPUT_LANES;
  static constexpr int SUM_GUARD_WIDTH = (INPUT_LANES <= 1) ? 1 : log2_ceil(INPUT_LANES);
  static constexpr int WEIGHT_SLICES =
      (BASE_B_WIDTH > 0 && B_WIDTH >= BASE_B_WIDTH) ? (B_WIDTH / BASE_B_WIDTH)
                                                    : 0;
  static constexpr int OUTPUT_LANES = (WEIGHT_SLICES > 0) ? (MACRO_OUTPUT_LANES / WEIGHT_SLICES) : 0;
  static constexpr int C_WIDTH = A_WIDTH + B_WIDTH + SUM_GUARD_WIDTH;
  static constexpr int INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : log2_ceil(INPUT_LANES);
  static constexpr int SET_INDEX_WIDTH = (WEIGHT_SETS <= 1) ? 1 : log2_ceil(WEIGHT_SETS);
  static constexpr int A_BUS_WIDTH = INPUT_LANES * A_WIDTH;
  static constexpr int B_BUS_WIDTH = OUTPUT_LANES * WRITE_INPUT_LANES * B_WIDTH;
  static constexpr int C_BUS_WIDTH = OUTPUT_LANES * C_WIDTH;

  // B-set selector shared by B writes and MAC issues
  using WeightSet = ac_int<SET_INDEX_WIDTH, false>;

  static_assert(INPUT_LANES > 0, "INPUT_LANES must be positive");
  static_assert(MACRO_OUTPUT_LANES > 0, "MACRO_OUTPUT_LANES must be positive");
  static_assert(WEIGHT_SETS > 0, "WEIGHT_SETS must be positive");
  static_assert(BASE_A_WIDTH > 0, "BASE_A_WIDTH must be positive");
  static_assert(BASE_B_WIDTH > 0, "BASE_B_WIDTH must be positive");
  static_assert(BASE_C_WIDTH > 0, "BASE_C_WIDTH must be positive");
  static_assert(WRITE_INPUT_LANES > 0, "WRITE_INPUT_LANES must be positive");
  static_assert((INPUT_LANES % WRITE_INPUT_LANES) == 0, "INPUT_LANES must be divisible by WRITE_INPUT_LANES");
  static_assert(MAC_LATENCY > 0, "MAC_LATENCY must be positive");
  static_assert(A_WIDTH > 0, "A_WIDTH must be positive");
  static_assert(B_WIDTH > 0, "B_WIDTH must be positive");
  static_assert(B_WIDTH >= BASE_B_WIDTH,
                "B_WIDTH must be at least BASE_B_WIDTH");
  static_assert((BASE_B_WIDTH > 0) && ((B_WIDTH % BASE_B_WIDTH) == 0),
                "B_WIDTH must be a multiple of BASE_B_WIDTH");
  static_assert((WEIGHT_SLICES > 0) && ((MACRO_OUTPUT_LANES % WEIGHT_SLICES) == 0),
                "MACRO_OUTPUT_LANES must be divisible by WEIGHT_SLICES");
  static_assert(MODE == 0 || MODE == CIM_MODE_BIT_SERIAL_VALUE,
                "MODE must be bit-parallel (0) or bit-serial (1)");
  // A bit-serial macro shift-accumulates one A slice internally, so its own
  // accumulator has to hold that slice's partial sum: the slice itself, one B
  // operand, and the reduction guard over INPUT_LANES.
  static_assert(MODE == CIM_MODE_BIT_PARALLEL_VALUE ||
                    BASE_C_WIDTH > BASE_B_WIDTH + SUM_GUARD_WIDTH,
                "bit-serial requires BASE_C_WIDTH > BASE_B_WIDTH + "
                "SUM_GUARD_WIDTH so one A slice is at least 1 bit");

  // Return the minimum mclk edge spacing between accepted element issues
  static constexpr int issue_window() {
    constexpr int serial_max_slice_width =
        BASE_C_WIDTH - BASE_B_WIDTH - SUM_GUARD_WIDTH;
    constexpr int serial_slice_width = min_int(A_WIDTH, serial_max_slice_width);
    constexpr int slice_width =
        (MODE == CIM_MODE_BIT_SERIAL_VALUE) ? serial_slice_width : BASE_A_WIDTH;
    constexpr int num_slices = ceil_div(A_WIDTH, slice_width);
    constexpr int serial_slice_interval =
        ceil_div(serial_slice_width, BASE_A_WIDTH) * BASE_A_WIDTH;
    constexpr int slice_launch_interval =
        (MODE == CIM_MODE_BIT_SERIAL_VALUE) ? serial_slice_interval : 1;
    return num_slices * slice_launch_interval;
  }

  // Return the number of mclk cycles from an accepted issue to its retirement
  static constexpr int operation_latency() {
    return issue_window() + MAC_LATENCY - 1;
  }

  // CIMElement clock and reset interface
  sc_in<bool> CCS_INIT_S1(wclk);
  sc_in<bool> CCS_INIT_S1(mclk);
  sc_in<bool> CCS_INIT_S1(rstn);

  // Packed logical A vector; a_bus and compute_set remain stable throughout an
  // accepted issue window
  sc_in<ac_int<A_BUS_WIDTH, false>> CCS_INIT_S1(a_bus);

  // Packed weight block spanning WRITE_INPUT_LANES inputs and all output lanes
  sc_in<ac_int<B_BUS_WIDTH, false>> CCS_INIT_S1(b_bus);
  sc_in<bool> CCS_INIT_S1(wen);
  sc_in<ac_int<INPUT_INDEX_WIDTH, false>> CCS_INIT_S1(write_input_index);
  sc_in<WeightSet> CCS_INIT_S1(write_set);

  // A committed upstream request arrives as mac_issue one cycle later
  sc_in<bool> CCS_INIT_S1(mac_issue);
  sc_in<WeightSet> CCS_INIT_S1(compute_set);

  // mac_ready advertises upstream commit timing while mac_busy covers operand use
  sc_out<ac_int<C_BUS_WIDTH, false>> CCS_INIT_S1(c_bus);
  sc_out<bool> CCS_INIT_S1(c_retire);
  sc_out<bool> CCS_INIT_S1(mac_ready);
  sc_out<bool> CCS_INIT_S1(mac_busy);

 private:
  // Resetless weight storage; each write fills WRITE_INPUT_LANES input positions
  ac_int<B_WIDTH, false> b_mem[WEIGHT_SETS][INPUT_LANES][OUTPUT_LANES];

  // PendingResult carries one computed result through the fixed retire latency
  struct PendingResult {
    ac_int<C_WIDTH, false> value[OUTPUT_LANES];
    int cycles_remaining;
  };

  // Resettable C++ registers owned by the mclk process. The pending queue is
  // statically bounded so Catapult can analyze the behavioral model while the
  // RTL implementation remains an ac_blackbox.
  static constexpr int MAX_PENDING_RESULTS = operation_latency() + 1;
  PendingResult pending_results[MAX_PENDING_RESULTS];
  int pending_results_size;
  sc_signal<int> window_remaining;

 public:
  // Construct the packed CIMElement behavioral model and blackbox metadata
  SC_CTOR(CIMElementPacked) : pending_results_size(0) {
    initialize_model_state();

    SC_METHOD(write_b);
    sensitive << wclk.pos();
    dont_initialize();

    SC_METHOD(run_mclk);
    sensitive << mclk.pos() << rstn.neg();
    dont_initialize();

    SC_METHOD(drive_mac_status);
    sensitive << rstn << mac_issue << window_remaining;

#ifndef __SYNTHESIS__
    SC_METHOD(check_write_mac_collision);
    sensitive << wclk.pos();
    dont_initialize();
#endif

    ac_blackbox()
        .entity("CIMIntElementPacked")
        .verilog_files(
            "cim_macro_wrapper.sv cim_macro_model.sv "
            "cim_macro_1.sv cim_element.sv")
        .parameter("MACRO_INPUT_LANES", MACRO_INPUT_LANES)
        .parameter("MACRO_OUTPUT_LANES", MACRO_OUTPUT_LANES)
        .parameter("WEIGHT_SETS", WEIGHT_SETS)
        .parameter("BASE_A_WIDTH", BASE_A_WIDTH)
        .parameter("BASE_B_WIDTH", BASE_B_WIDTH)
        .parameter("BASE_C_WIDTH", BASE_C_WIDTH)
        .parameter("MACRO_WRITE_INPUT_LANES", MACRO_WRITE_INPUT_LANES)
        .parameter("MAC_LATENCY", MAC_LATENCY)
        .parameter("MODE", MODE)
        .parameter("A_WIDTH", A_WIDTH)
        .parameter("B_WIDTH", B_WIDTH)
        .parameter("SIGNED", static_cast<const int>(SIGNED))
        .inputs_registered(false)
        .end();
  }

 private:
  // Initialize observable model state while leaving resetless B storage
  // untouched
  void initialize_model_state() {
    pending_results_size = 0;
    window_remaining.write(0);
  }

  // Return the issue-window count after the upcoming edge
  static int next_window_remaining(int current_remaining, bool issue) {
    const bool issue_ready_now = current_remaining == 0;
    int next_remaining =
        current_remaining > 0 ? current_remaining - 1 : 0;
    if (issue && issue_ready_now) {
      next_remaining = issue_window() - 1;
    }
    return next_remaining;
  }

  // Decode a CIM operand with the configured signedness
  template <int WIDTH>
  static ac_int<WIDTH, SIGNED> decode_operand(ac_int<WIDTH, false> value) {
    ac_int<WIDTH, SIGNED> decoded;
    decoded.set_slc(0, value);
    return decoded;
  }

  // Return the packed B bus bit offset for one logical B value
  static constexpr int b_bus_offset(int write_input_offset, int output_index) {
    return ((write_input_offset * OUTPUT_LANES) + output_index) * B_WIDTH;
  }

  // Write one logical WRITE_INPUT_LANES-wide B block into the C++ model
  void write_b() {
    if (!wen.read()) {
      return;
    }

    const int set = write_set.read().to_int();
    const int base_input_index = write_input_index.read().to_int();

    if (set >= WEIGHT_SETS) {
      return;
    }

    const ac_int<B_BUS_WIDTH, false> b_value = b_bus.read();
    for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) {
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
        const int input_index = base_input_index + write_input_offset;
        if (input_index < INPUT_LANES) {
          b_mem[set][input_index][output_index] = b_value.template slc<B_WIDTH>(b_bus_offset(write_input_offset, output_index));
        }
      }
    }
  }

  // Compute one native CIM matrix-vector operation for the issued payload
  PendingResult compute_result() {
    PendingResult pending;
    pending.cycles_remaining = operation_latency();
    const int set = compute_set.read().to_int();
    const ac_int<A_BUS_WIDTH, false> a_value_bus = a_bus.read();

    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
      ac_int<C_WIDTH, SIGNED> acc = 0;

      if (set < WEIGHT_SETS) {
        for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
          const ac_int<A_WIDTH, SIGNED> a_value = decode_operand<A_WIDTH>(
              a_value_bus.template slc<A_WIDTH>(input_index * A_WIDTH));
          const ac_int<B_WIDTH, SIGNED> b_value =
              decode_operand<B_WIDTH>(b_mem[set][input_index][output_index]);
          acc += a_value * b_value;
        }
      }
      pending.value[output_index] = acc;
    }
    return pending;
  }

  // Clear resettable CIM element state while preserving resetless B storage
  void reset_element_state() {
    pending_results_size = 0;
    window_remaining.write(0);
    c_bus.write(0);
    c_retire.write(false);
  }

  // Advance the mclk-domain issue and retire state
  void run_mclk() {
    if (!rstn.read()) {
      reset_element_state();
      return;
    }

    // c_retire is a one-cycle pulse: default it low every edge and raise it
    // only on the edge a result retires
    c_retire.write(false);

    // Determine current-edge acceptance before advancing the window
    const int current_remaining = window_remaining.read();
    const bool issue_ready_now = current_remaining == 0;

    // Advance the retire pipeline; ops are spaced by at least the issue window,
    // so at most one result retires per edge
    if (pending_results_size > 0) {
      for (int pending_idx = 0; pending_idx < MAX_PENDING_RESULTS;
           pending_idx++) {
        if (pending_idx < pending_results_size) {
          pending_results[pending_idx].cycles_remaining--;
        }
      }
      if (pending_results[0].cycles_remaining == 0) {
        ac_int<C_BUS_WIDTH, false> packed_result = 0;
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
          packed_result.set_slc(output_index * C_WIDTH, pending_results[0].value[output_index]);
        }
        c_bus.write(packed_result);
        c_retire.write(true);
        for (int pending_idx = 1; pending_idx < MAX_PENDING_RESULTS;
             pending_idx++) {
          if (pending_idx < pending_results_size) {
            pending_results[pending_idx - 1] = pending_results[pending_idx];
          }
        }
        pending_results_size--;
      }
    }

    if (mac_issue.read() && issue_ready_now) {
      if (pending_results_size < MAX_PENDING_RESULTS) {
        pending_results[pending_results_size] = compute_result();
        pending_results_size++;
      }
    }

    window_remaining.write(
        next_window_remaining(current_remaining, mac_issue.read()));
  }

  // Drive next-cycle readiness and current-cycle operand occupancy
  void drive_mac_status() {
    const int current_remaining = window_remaining.read();
    const bool issue_ready_now = current_remaining == 0;
    const bool issue_accepts = mac_issue.read() && issue_ready_now;
    const int next_remaining =
        next_window_remaining(current_remaining, mac_issue.read());
    mac_ready.write(rstn.read() && next_remaining == 0);
    mac_busy.write(rstn.read() &&
                   (current_remaining > 0 || issue_accepts));
  }

#ifndef __SYNTHESIS__
  // Report an illegal write to the B set consumed by the active issue window
  void check_write_mac_collision() {
    const bool window_active = window_remaining.read() > 0;
    const bool issue_start = mac_issue.read() && !window_active;
    if (rstn.read() && wen.read() && (issue_start || window_active) &&
        write_set.read() == compute_set.read()) {
      std::ostringstream message;
      message << "write targets B set " << write_set.read().to_int()
              << " while its MAC issue window is active";
      SC_REPORT_ERROR("CIMElement B set protocol violation",
                      message.str().c_str());
    }
  }
#endif
};

// CIMElement keeps the native A/B/C interface and adapts it to the packed
// blackbox ABI
template <int MACRO_INPUT_LANES, int MACRO_OUTPUT_LANES, int WEIGHT_SETS,
          int BASE_A_WIDTH, int BASE_B_WIDTH,
          int BASE_C_WIDTH, int MACRO_WRITE_INPUT_LANES, int MAC_LATENCY, int MODE,
          int A_WIDTH, int B_WIDTH, bool SIGNED>
SC_MODULE(CIMElement) {
 private:
  using PackedElement =
      CIMElementPacked<MACRO_INPUT_LANES, MACRO_OUTPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH, BASE_B_WIDTH,
                       BASE_C_WIDTH, MACRO_WRITE_INPUT_LANES, MAC_LATENCY, MODE, A_WIDTH,
                       B_WIDTH, SIGNED>;

 public:
  static constexpr int CIM_MODE_BIT_SERIAL_VALUE =
      PackedElement::CIM_MODE_BIT_SERIAL_VALUE;
  static constexpr int INPUT_LANES = PackedElement::INPUT_LANES;
  static constexpr int WRITE_INPUT_LANES = PackedElement::WRITE_INPUT_LANES;
  static constexpr int SUM_GUARD_WIDTH = PackedElement::SUM_GUARD_WIDTH;
  static constexpr int WEIGHT_SLICES = PackedElement::WEIGHT_SLICES;
  static constexpr int OUTPUT_LANES = PackedElement::OUTPUT_LANES;
  static constexpr int C_WIDTH = PackedElement::C_WIDTH;
  static constexpr int INPUT_INDEX_WIDTH = PackedElement::INPUT_INDEX_WIDTH;
  static constexpr int SET_INDEX_WIDTH = PackedElement::SET_INDEX_WIDTH;
  static constexpr int A_BUS_WIDTH = PackedElement::A_BUS_WIDTH;
  static constexpr int B_BUS_WIDTH = PackedElement::B_BUS_WIDTH;
  static constexpr int C_BUS_WIDTH = PackedElement::C_BUS_WIDTH;

  // A and C span the element input and output lanes. Each weight block
  // spans WRITE_INPUT_LANES input positions and every output lane.
  using WeightSet = typename PackedElement::WeightSet;
  using AData = Pack1D<ac_int<A_WIDTH, false>, INPUT_LANES>;
  using BData = Pack1D<Pack1D<ac_int<B_WIDTH, false>, OUTPUT_LANES>, WRITE_INPUT_LANES>;
  using CData = Pack1D<ac_int<C_WIDTH, false>, OUTPUT_LANES>;

  // Return the minimum mclk edge spacing between accepted element issues
  static constexpr int issue_window() { return PackedElement::issue_window(); }

  // Return the number of mclk cycles from an accepted issue to its retirement
  static constexpr int operation_latency() {
    return PackedElement::operation_latency();
  }

  // CIMElement clock and reset interface
  sc_in<bool> CCS_INIT_S1(wclk);
  sc_in<bool> CCS_INIT_S1(mclk);
  sc_in<bool> CCS_INIT_S1(rstn);

  // Logical A vector; a and compute_set remain stable throughout an accepted issue
  // window
  sc_in<AData> CCS_INIT_S1(a);

  // Weight block spanning WRITE_INPUT_LANES inputs and all output lanes
  sc_in<BData> CCS_INIT_S1(b);
  sc_in<bool> CCS_INIT_S1(wen);
  sc_in<ac_int<INPUT_INDEX_WIDTH, false>> CCS_INIT_S1(write_input_index);
  sc_in<WeightSet> CCS_INIT_S1(write_set);

  // A committed upstream request arrives as mac_issue one cycle later
  sc_in<bool> CCS_INIT_S1(mac_issue);
  sc_in<WeightSet> CCS_INIT_S1(compute_set);

  // mac_ready advertises upstream commit timing while mac_busy covers operand use
  sc_out<CData> CCS_INIT_S1(c);
  sc_out<bool> CCS_INIT_S1(c_retire);
  sc_out<bool> CCS_INIT_S1(mac_ready);
  sc_out<bool> CCS_INIT_S1(mac_busy);

 private:
  PackedElement packed;
  sc_signal<ac_int<A_BUS_WIDTH, false>> a_bus;
  sc_signal<ac_int<B_BUS_WIDTH, false>> b_bus;
  sc_signal<ac_int<C_BUS_WIDTH, false>> c_bus;

 public:
  // Construct the public array-port adapter around the packed blackbox boundary
  SC_CTOR(CIMElement) : packed("packed") {
    packed.wclk(wclk);
    packed.mclk(mclk);
    packed.rstn(rstn);
    packed.a_bus(a_bus);
    packed.b_bus(b_bus);
    packed.wen(wen);
    packed.write_input_index(write_input_index);
    packed.write_set(write_set);
    packed.mac_issue(mac_issue);
    packed.compute_set(compute_set);
    packed.c_bus(c_bus);
    packed.c_retire(c_retire);
    packed.mac_ready(mac_ready);
    packed.mac_busy(mac_busy);

    SC_METHOD(pack_inputs);
    sensitive << a << b;

    SC_METHOD(unpack_outputs);
    sensitive << c_bus;
  }

 private:
  // Return the packed B bus bit offset for one logical B value
  static constexpr int b_bus_offset(int write_input_offset, int output_index) {
    return ((write_input_offset * OUTPUT_LANES) + output_index) * B_WIDTH;
  }

  // Pack array-shaped public A/B ports into stable vector ports for Catapult
  void pack_inputs() {
    ac_int<A_BUS_WIDTH, false> packed_a = 0;
    ac_int<B_BUS_WIDTH, false> packed_b = 0;
    const AData a_data = a.read();
    const BData b_data = b.read();

    for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
      packed_a.set_slc(input_index * A_WIDTH, a_data[input_index]);
    }

    for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) {
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
        packed_b.set_slc(b_bus_offset(write_input_offset, output_index), b_data[write_input_offset][output_index]);
      }
    }

    a_bus.write(packed_a);
    b_bus.write(packed_b);
  }

  // Unpack the packed result bus back into the native CIMElement result ports
  void unpack_outputs() {
    const ac_int<C_BUS_WIDTH, false> packed_c = c_bus.read();
    CData c_data;
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
      c_data[output_index] = packed_c.template slc<C_WIDTH>(output_index * C_WIDTH);
    }
    c.write(c_data);
  }
};
