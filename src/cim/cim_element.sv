// CIM element adapter around the CIM macro wrapper
//
// This module adds two pieces of behavior around the macro wrapper:
//   1. CIM element-level A_WIDTH can be wider than the macro wrapper's BASE_A_WIDTH.
//      The element sends A to the macro wrapper MSB slice first, then lower slices, and
//      combines the returned results in an external accumulator.
//   2. CIM element-level B_WIDTH can be wider than the macro wrapper's BASE_B_WIDTH.
//      The element splits each B value across multiple macro wrapper output lanes, then
//      combines the partial results from those channels in the external accumulator.
//
// Element geometry:
//
//                                    OUTPUT_LANES
//                                 +------------------+
//                    INPUT_LANES  | B[input][output] |
//                                 +------------------+
//
//                 INPUT_LANES                      OUTPUT_LANES
//          +-------------------+              +--------------------+
// requests | A[request][input] |  x  B  =      | C[request][output] |
//          +-------------------+              +--------------------+
//
// Requests advance in time: each compute request consumes one A vector and
// produces one C vector using the weight set selected by compute_set.
// A weight write supplies WRITE_INPUT_LANES consecutive input positions for
// every output lane, starting at write_input_index in write_set.
// Each element output lane combines WEIGHT_SLICES native macro output lanes.

`include "cim_typedefs.svh"

module CIMIntElement #(
    parameter int unsigned MACRO_INPUT_LANES = 64,
    parameter int unsigned MACRO_OUTPUT_LANES = 8,
    parameter int unsigned WEIGHT_SETS = 18,

    // BASE_* parameters describe the base shape of the macro wrapper
    // In bit-serial mode, the macro wrapper A width may be widened up to BASE_C_WIDTH capacity.
    parameter int unsigned BASE_A_WIDTH = 4,
    parameter int unsigned BASE_B_WIDTH = 4,
    parameter int unsigned BASE_C_WIDTH = 20,
    parameter int unsigned MACRO_WRITE_INPUT_LANES = 1,
    parameter int unsigned MAC_LATENCY = 1,
    parameter cim_mode_t MODE = CIM_MODE_BIT_SERIAL,
    parameter cim_macro_wrapper_impl_t MACRO_IMPL = CIM_MACRO_WRAPPER_IMPL_MODEL,

    // A_WIDTH and B_WIDTH are the logical operand widths implemented by this element
    // SIGNED applies to both logical A and logical B at design time
    parameter int unsigned A_WIDTH = 8,
    parameter int unsigned B_WIDTH = 8,
    parameter bit SIGNED = 1'b0,

    localparam int unsigned INPUT_LANES = MACRO_INPUT_LANES,
    localparam int unsigned WRITE_INPUT_LANES = MACRO_WRITE_INPUT_LANES,
    localparam int unsigned SUM_GUARD_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES),

    // One B value is split across WEIGHT_SLICES physical macro wrapper output lanes
    localparam int unsigned WEIGHT_SLICES = B_WIDTH / BASE_B_WIDTH,
    localparam int unsigned OUTPUT_LANES = MACRO_OUTPUT_LANES / WEIGHT_SLICES,

    // C_WIDTH is the result width depending on the operand widths and the vector length
    localparam int unsigned C_WIDTH = A_WIDTH + B_WIDTH + SUM_GUARD_WIDTH,
    localparam int unsigned INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES),
    localparam int unsigned SET_INDEX_WIDTH = (WEIGHT_SETS <= 1) ? 1 : $clog2(WEIGHT_SETS)
) (
    input  logic                       wclk,
    input  logic                       mclk,
    input  logic                       rstn,

    // Logical A operand is one vector with INPUT_LANES values. a and compute_set must remain
    // stable while mac_busy is high
    input  logic [A_WIDTH-1:0]         a [INPUT_LANES],

    // One B write covers consecutive input lanes for all output lanes
    input  logic [B_WIDTH-1:0]         b [WRITE_INPUT_LANES][OUTPUT_LANES],
    input  logic                       wen,
    input  logic [INPUT_INDEX_WIDTH-1:0]          write_input_index,
    input  logic [SET_INDEX_WIDTH-1:0]        write_set,

    // mac_issue high on an eligible edge starts one MAC. A request committed
    // when mac_ready is high reaches this port on the following edge
    input  logic                       mac_issue,
    input  logic [SET_INDEX_WIDTH-1:0]        compute_set,

    output logic [C_WIDTH-1:0]         c [OUTPUT_LANES],  // registered result, stable until the next retire
    output logic                       c_retire,   // one-cycle pulse per retired result, aligned with c
    output logic                       mac_ready,  // high when upstream may commit a request on the next edge
    output logic                       mac_busy    // high while the current MAC consumes A and B
);

  // CIM element walks operand A slice by slice. The width of a slice depends on the selected macro wrapper mode:
  // For bit-parallel, the slice is the fixed BASE_A_WIDTH fed into the macro wrapper. For bit-serial, the internal
  // accumulator width may allow the macro wrapper to process a wider slice than BASE_A_WIDTH, so it performs an
  // additional "intra-slice" (or window) walking.
  // We refer the base width the macro wrapper processes at a time as a "window", and the max width the macro wrapper
  // can process a "slice". A slice contains one or more window.

  // ---------------------------------------------------------------------------
  // Slice Walking
  // ---------------------------------------------------------------------------

  logic issue_ready_now;
  logic start_mac;
  // The final A slice result reaches the accumulator on the retire edge
  logic retire_op;

  // Max operand A width supported by the macro wrapper in serial mode
  localparam int unsigned SERIAL_MAX_SLICE_WIDTH = BASE_C_WIDTH - BASE_B_WIDTH - SUM_GUARD_WIDTH;

  // The slice width may just be A_WIDTH if the A_WIDTH is smaller than what the macro wrapper can handle
  localparam int unsigned SERIAL_SLICE_WIDTH = (A_WIDTH < SERIAL_MAX_SLICE_WIDTH) ? A_WIDTH : SERIAL_MAX_SLICE_WIDTH;
  localparam int unsigned SLICE_WIDTH = (MODE == CIM_MODE_BIT_SERIAL) ? SERIAL_SLICE_WIDTH : BASE_A_WIDTH;
  // Number of A slices needed for A_WIDTH; + SLICE_WIDTH - 1 implements ceiling division to cover any partial final slice
  localparam int unsigned NUM_SLICES = (A_WIDTH + SLICE_WIDTH - 1) / SLICE_WIDTH;
  localparam int unsigned BITS_SLICE = (NUM_SLICES <= 1) ? 1 : $clog2(NUM_SLICES);

  // SLICE_LAUNCH_INTERVAL is the number of cycles between launching slices into the macro wrapper
  // Bit-parallel consumes a slice in one cycle; bit-serial consumes one bit per cycle, padded to whole windows
  localparam int unsigned SERIAL_SLICE_INTERVAL = (SERIAL_SLICE_WIDTH + BASE_A_WIDTH - 1) / BASE_A_WIDTH * BASE_A_WIDTH;
  localparam int unsigned SLICE_LAUNCH_INTERVAL = (MODE == CIM_MODE_BIT_SERIAL) ? SERIAL_SLICE_INTERVAL : 1;
  localparam int unsigned BITS_SLICE_LAUNCH_INTERVAL =
    (SLICE_LAUNCH_INTERVAL <= 1) ? 1 : $clog2(SLICE_LAUNCH_INTERVAL + 1);
  // The actual MAC latency for a slice; for bit-serial, including the tail latency for completing the last bit of the slice
  localparam int unsigned SLICE_MAC_CYCLES = SLICE_LAUNCH_INTERVAL + MAC_LATENCY - 1;

  // Slice driven into the window walker this cycle. Slice 0 contains the MSB bits of operand A.
  logic [BITS_SLICE-1:0] issue_a_slice_idx;
  // Next slice that can be issued once the current slice is retiring
  logic [BITS_SLICE-1:0] next_a_slice_idx;
  // All slices have entered the macro wrapper; results may still be retiring through the macro wrapper pipeline
  logic issued_all_slices;
  // Cycle counter for the current slice; it is set to 1 because the launch cycle already feeds the macro wrapper
  logic [BITS_SLICE_LAUNCH_INTERVAL-1:0] slice_cycle;
  // Registered slice walk state after the launch cycle
  logic slice_walk_active;
  assign slice_walk_active = (slice_cycle != '0);

  // The issue window closes once the final slice has fed its last window; the
  // retire pipeline may still be draining while a new issue is accepted
  logic issue_window_open;
  assign issue_window_open = slice_walk_active &&
    !(issued_all_slices && (slice_cycle == BITS_SLICE_LAUNCH_INTERVAL'(SLICE_LAUNCH_INTERVAL)));
  assign issue_ready_now = rstn && !issue_window_open;
  assign start_mac = mac_issue && issue_ready_now;

  // issue_slice marks the first cycle of a slice
  logic issue_slice;
  assign issue_slice = start_mac || (rstn && !issued_all_slices &&
    (slice_cycle == BITS_SLICE_LAUNCH_INTERVAL'(SLICE_LAUNCH_INTERVAL)));

  always_comb begin
    // eagerly update the issue slice index
    issue_a_slice_idx = next_a_slice_idx;

    // A new mac operation issues the first slice
    if (start_mac) begin
      issue_a_slice_idx = '0;
    end
  end

  // The final use cycle lets upstream commit the next request one edge early
  logic final_operand_use;
  assign final_operand_use =
    ((SLICE_LAUNCH_INTERVAL == 1) && issue_slice &&
     (issue_a_slice_idx == BITS_SLICE'(NUM_SLICES - 1))) ||
    (slice_walk_active && issued_all_slices &&
     (slice_cycle ==
      BITS_SLICE_LAUNCH_INTERVAL'(SLICE_LAUNCH_INTERVAL - 1)));
  assign mac_ready = rstn &&
    (final_operand_use || (issue_ready_now && !start_mac));
  assign mac_busy = rstn && (issue_window_open || start_mac);

  always_ff @(posedge mclk or negedge rstn) begin
    if (!rstn) begin
      slice_cycle <= '0;
    end else if (issue_slice) begin
      slice_cycle <= BITS_SLICE_LAUNCH_INTERVAL'(1);
    end else if (slice_walk_active &&
                 (slice_cycle <
                  BITS_SLICE_LAUNCH_INTERVAL'(SLICE_LAUNCH_INTERVAL))) begin
      slice_cycle <= slice_cycle + BITS_SLICE_LAUNCH_INTERVAL'(1);
    end
  end

  always_ff @(posedge mclk or negedge rstn) begin
    if (!rstn) begin
      next_a_slice_idx <= '0;
      issued_all_slices <= 1'b0;
    end else if (issue_slice) begin
      if (issue_a_slice_idx == BITS_SLICE'(NUM_SLICES - 1)) begin
        next_a_slice_idx <= '0;
        issued_all_slices <= 1'b1;
      end else begin
        next_a_slice_idx <= issue_a_slice_idx + BITS_SLICE'(1);
        issued_all_slices <= 1'b0;
      end
    end
  end

  // first marks the signed/MSB A slice, and last marks completion of the element op.
  logic slice_valid_pipe [SLICE_MAC_CYCLES];
  logic slice_is_first_pipe [SLICE_MAC_CYCLES];
  logic slice_is_last_pipe [SLICE_MAC_CYCLES];

  always_ff @(posedge mclk or negedge rstn) begin
    if (!rstn) begin
      for (int stage = 0; stage < SLICE_MAC_CYCLES; stage++) begin
        slice_valid_pipe[stage] <= 1'b0;
        slice_is_first_pipe[stage] <= 1'b0;
        slice_is_last_pipe[stage] <= 1'b0;
      end
    end else begin
      slice_valid_pipe[0] <= issue_slice;
      slice_is_first_pipe[0] <= issue_slice && (issue_a_slice_idx == '0);
      slice_is_last_pipe[0] <= issue_slice && (issue_a_slice_idx == BITS_SLICE'(NUM_SLICES - 1));
      for (int stage = 1; stage < SLICE_MAC_CYCLES; stage++) begin
        slice_valid_pipe[stage] <= slice_valid_pipe[stage-1];
        slice_is_first_pipe[stage] <= slice_is_first_pipe[stage-1];
        slice_is_last_pipe[stage] <= slice_is_last_pipe[stage-1];
      end
    end
  end

  // Indicates the current slice has reached the external accumulator.
  logic slice_result_ready;
  assign slice_result_ready = slice_valid_pipe[SLICE_MAC_CYCLES-1];

  // Indicates the current retiring slice is the first/final A slice.
  logic retiring_first_a_slice, retiring_last_a_slice;
  assign retiring_first_a_slice = slice_is_first_pipe[SLICE_MAC_CYCLES-1];
  assign retiring_last_a_slice = slice_is_last_pipe[SLICE_MAC_CYCLES-1];

  // The final slice result is written to the output stage on the retire edge
  assign retire_op = slice_result_ready && retiring_last_a_slice;

  // ---------------------------------------------------------------------------
  // Window Walking
  // ---------------------------------------------------------------------------

  // Number of windows in a slice
  localparam int unsigned A_WINDOWS_PER_SLICE = (SLICE_WIDTH + BASE_A_WIDTH - 1) / BASE_A_WIDTH;
  localparam int unsigned BITS_A_WINDOW = (A_WINDOWS_PER_SLICE <= 1) ? 1 : $clog2(A_WINDOWS_PER_SLICE);

  // Active slice stays registered after launch because the issue index immediately advances
  logic [BITS_SLICE-1:0] a_window_slice_idx;
  logic [BITS_SLICE-1:0] macro_wrapper_a_slice_idx;
  assign macro_wrapper_a_slice_idx = issue_slice ? issue_a_slice_idx : a_window_slice_idx;

  always_ff @(posedge mclk or negedge rstn) begin
    if (!rstn) begin
      a_window_slice_idx <= '0;
    end else if (issue_slice) begin
      a_window_slice_idx <= issue_a_slice_idx;
    end
  end

  // The mac signal of the macro wrapper is asserted for SLICE_LAUNCH_INTERVAL cycles so that all bits are fed to the wrapper
  // This does not include the tail latency for the MAC op to finish
  logic macro_wrapper_mac;
  assign macro_wrapper_mac = rstn && (issue_slice ||
    (slice_walk_active && (slice_cycle < BITS_SLICE_LAUNCH_INTERVAL'(SLICE_LAUNCH_INTERVAL))));

  // Each window remains stable while the serial macro wrapper consumes its BASE_A_WIDTH bits
  logic [BITS_A_WINDOW-1:0] issue_a_window_idx;

  always_comb begin
    issue_a_window_idx = '0;
    if (macro_wrapper_mac) begin
        if (issue_slice)
          issue_a_window_idx = '0;
        else
          issue_a_window_idx = BITS_A_WINDOW'(slice_cycle / BASE_A_WIDTH);
    end
  end

  logic macro_wrapper_a_signed;
  assign macro_wrapper_a_signed = SIGNED && (macro_wrapper_a_slice_idx == '0) && (issue_a_window_idx == '0);

  logic macro_wrapper_init;
  assign macro_wrapper_init = (MODE == CIM_MODE_BIT_SERIAL) ? issue_slice : macro_wrapper_mac;

  // ---------------------------------------------------------------------------
  // Macro-Wrapper-Facing Data Buses
  // ---------------------------------------------------------------------------

  // Repack A/B operands into the fixed macro wrapper shape. B signedness is also
  // expanded here because only the MSB physical B slice should be signed. These
  // are declared ahead of the functions/always_comb below because reduce_b_slices
  // reads macro_wrapper_c; VCS rejects referencing a module variable declared
  // later in the module (Verilator accepts the forward reference).
  logic [BASE_A_WIDTH-1:0] macro_wrapper_a [MACRO_INPUT_LANES];
  logic [BASE_B_WIDTH-1:0] macro_wrapper_b [MACRO_WRITE_INPUT_LANES][MACRO_OUTPUT_LANES];
  logic macro_wrapper_b_signed [MACRO_OUTPUT_LANES];
  logic [BASE_C_WIDTH-1:0] macro_wrapper_c [MACRO_OUTPUT_LANES];

  // Select one macro-wrapper-width A window from a logical A slice
  function automatic logic [BASE_A_WIDTH-1:0] select_a_window(
      input logic [A_WIDTH-1:0] a,
      input logic [BITS_SLICE-1:0] slice_idx,
      input logic [BITS_A_WINDOW-1:0] window_idx
  );
    int unsigned slice_lower_bit;
    int unsigned window_bit_offset;
    int unsigned slice_bit_idx;
    int unsigned a_bit_idx;
    logic sign_bit;
    begin
      // Slice numbering is MSB first. The expression below finds the LSB of the slice in the A operand
      slice_lower_bit = (NUM_SLICES - 1 - int'(slice_idx)) * SLICE_WIDTH;

      // Similarly, find the LSB of the window in the slice
      window_bit_offset = (A_WINDOWS_PER_SLICE - 1 - int'(window_idx)) * BASE_A_WIDTH;

      // Pre-fill with sign bits so a narrow MSB slice is sign- or zero-extended
      sign_bit = SIGNED && (slice_idx == 0) && a[A_WIDTH - 1];
      select_a_window = {BASE_A_WIDTH{sign_bit}};

      // Copy only the real bits in this slice; any remaining high bits keep
      // the sign/zero fill from above.
      for (int bit_idx = 0; bit_idx < BASE_A_WIDTH; bit_idx++) begin
        slice_bit_idx = window_bit_offset + bit_idx;
        a_bit_idx = slice_lower_bit + slice_bit_idx;
        if ((slice_bit_idx < SLICE_WIDTH) && (a_bit_idx < A_WIDTH)) begin
          select_a_window[bit_idx] = a[a_bit_idx];
        end
      end
    end
  endfunction

  function automatic logic [BASE_B_WIDTH-1:0] select_b_slice(
      input logic [B_WIDTH-1:0] b,
      input int unsigned slice_idx
  );
    int unsigned lower_bit;
    begin
      lower_bit = (WEIGHT_SLICES - 1 - slice_idx) * BASE_B_WIDTH;
      select_b_slice = b[lower_bit +: BASE_B_WIDTH];
    end
  endfunction

  function automatic logic [C_WIDTH-1:0] extend_macro_wrapper_c(
      input logic [BASE_C_WIDTH-1:0] value,
      input logic result_is_signed
  );
    logic signed [BASE_C_WIDTH-1:0] signed_value;
    begin
      signed_value = value;
      extend_macro_wrapper_c = result_is_signed ? C_WIDTH'(signed_value) : C_WIDTH'(value);
    end
  endfunction

  function automatic logic [C_WIDTH-1:0] reduce_b_slices(
      input int unsigned output_index
  );
    int unsigned macro_output_index;
    int unsigned b_shift;
    logic result_is_signed;
    begin
      reduce_b_slices = '0;
      for (int b_slice = 0; b_slice < WEIGHT_SLICES; b_slice++) begin
        macro_output_index = output_index * WEIGHT_SLICES + b_slice;
        b_shift = (WEIGHT_SLICES - 1 - b_slice) * BASE_B_WIDTH;
        result_is_signed = SIGNED && (retiring_first_a_slice || (b_slice == 0));
        reduce_b_slices += extend_macro_wrapper_c(macro_wrapper_c[macro_output_index], result_is_signed) << b_shift;
      end
    end
  endfunction

  always_comb begin
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
      macro_wrapper_a[input_index] = select_a_window(a[input_index], macro_wrapper_a_slice_idx, issue_a_window_idx);
    end
  end

  // Weight slices occupy consecutive native macro output lanes:
  // macro_output_index = output_index * WEIGHT_SLICES + slice_idx.
  // Each write covers the same input positions in every output lane.
  //
  // Example: OUTPUT_LANES=2, WRITE_INPUT_LANES=2, WEIGHT_SLICES=2.
  //
  // b (element):             output 0       output 1
  //   write offset 0   b[0][0]        b[0][1]
  //   write offset 1   b[1][0]        b[1][1]
  //
  // macro_wrapper_b:         native output lanes
  //                          0             1             2             3
  //   write offset 0   b[0][0].sl0   b[0][0].sl1   b[0][1].sl0   b[0][1].sl1
  //   write offset 1   b[1][0].sl0   b[1][0].sl1   b[1][1].sl0   b[1][1].sl1
  //
  // sl0 is the most significant weight slice; sl1 is the least significant.
  always_comb begin
    for (int macro_output_index = 0; macro_output_index < MACRO_OUTPUT_LANES; macro_output_index++) begin
      int unsigned output_index;
      int unsigned slice_idx;

      output_index = macro_output_index / WEIGHT_SLICES;
      slice_idx = macro_output_index % WEIGHT_SLICES;
      // Only the MSB slice needs to be signed
      macro_wrapper_b_signed[macro_output_index] = SIGNED && (slice_idx == 0);

      // Position within the current WRITE_INPUT_LANES-wide write block
      for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) begin
        macro_wrapper_b[write_input_offset][macro_output_index] = select_b_slice(b[write_input_offset][output_index], slice_idx);
      end
    end
  end


  // ---------------------------------------------------------------------------
  // CIM Macro Wrapper
  // ---------------------------------------------------------------------------

  CIMIntMacroWrapper #(
      .INPUT_LANES(MACRO_INPUT_LANES),
      .OUTPUT_LANES(MACRO_OUTPUT_LANES),
      .WEIGHT_SETS(WEIGHT_SETS),
      .A_WIDTH(BASE_A_WIDTH),
      .B_WIDTH(BASE_B_WIDTH),
      .C_WIDTH(BASE_C_WIDTH),
      .WRITE_INPUT_LANES(MACRO_WRITE_INPUT_LANES),
      .MAC_LATENCY(MAC_LATENCY),
      .MODE(MODE),
      .IMPL(MACRO_IMPL)
  ) macro_wrapper (
      .wclk(wclk),
      .mclk(mclk),
      .a(macro_wrapper_a),
      .b(macro_wrapper_b),
      .wen(wen),
      .mac(macro_wrapper_mac),
      .init(macro_wrapper_init),
      .a_signed(macro_wrapper_a_signed),
      .b_signed(macro_wrapper_b_signed),
      .write_input_index(write_input_index),
      .write_set(write_set),
      .compute_set(compute_set),
      .c(macro_wrapper_c)
  );

  // ---------------------------------------------------------------------------
  // External Accumulator and Outputs
  // ---------------------------------------------------------------------------

  // Hold partial results while the element walks across multiple A slices
  logic [C_WIDTH-1:0] acc [OUTPUT_LANES];
  // Next accumulator value shared by the accumulator and the retire capture
  logic [C_WIDTH-1:0] acc_next [OUTPUT_LANES];
  // Registered output stage keeps a retired result stable while the next op accumulates
  logic [C_WIDTH-1:0] c_out [OUTPUT_LANES];
  assign c = c_out;

  always_comb begin
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      // The first retiring slice restarts the accumulation, so back-to-back ops need no clear
      acc_next[output_index] = (retiring_first_a_slice ? {C_WIDTH{1'b0}} : (acc[output_index] << SLICE_WIDTH)) + reduce_b_slices(output_index);
    end
  end

  always_ff @(posedge mclk or negedge rstn) begin
    if (!rstn) begin
      c_retire <= 1'b0;
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        acc[output_index] <= '0;
        c_out[output_index] <= '0;
      end
    end else begin
      // Assigned every cycle so the retire indication is a one-cycle pulse
      // rather than a level a consumer would have to remember the phase of
      c_retire <= slice_result_ready && retire_op;
      if (slice_result_ready) begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
          acc[output_index] <= acc_next[output_index];
        end
        if (retire_op) begin
          for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
            c_out[output_index] <= acc_next[output_index];
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Static Parameter Checks
  // ---------------------------------------------------------------------------
  // Check static element parameters during elaboration
  generate
    if (A_WIDTH == 0) begin : gen_invalid_a_width
      $fatal(1, "CIMIntElement: A_WIDTH must be positive");
    end
    if (B_WIDTH == 0) begin : gen_invalid_b_width
      $fatal(1, "CIMIntElement: B_WIDTH must be positive");
    end
    if (BASE_B_WIDTH != 0) begin : gen_check_base_b_width
      if ((B_WIDTH % BASE_B_WIDTH) != 0) begin : gen_invalid_b_width_base_b_width
        $fatal(1, "CIMIntElement: B_WIDTH must be a multiple of BASE_B_WIDTH");
      end
    end
    if (WEIGHT_SLICES == 0) begin : gen_invalid_weight_slices
      $fatal(1, "CIMIntElement: B_WIDTH must be at least BASE_B_WIDTH");
    end
    if (WEIGHT_SLICES != 0) begin : gen_check_weight_slices
      if ((MACRO_OUTPUT_LANES % WEIGHT_SLICES) != 0) begin : gen_invalid_output_lanes_weight_slices
        $fatal(1, "CIMIntElement: MACRO_OUTPUT_LANES must be divisible by WEIGHT_SLICES for wider B grouping");
      end
    end
    if (SLICE_WIDTH == 0) begin : gen_invalid_slice_width
      $fatal(1, "CIMIntElement: selected macro wrapper A slice width must be positive");
    end
  endgenerate

endmodule

// CIMIntElementPacked adapts Catapult-friendly packed buses to the native array RTL
module CIMIntElementPacked #(
    parameter int unsigned MACRO_INPUT_LANES = 64,
    parameter int unsigned MACRO_OUTPUT_LANES = 8,
    parameter int unsigned WEIGHT_SETS = 18,

    // BASE_* parameters describe the base shape of the macro wrapper
    parameter int unsigned BASE_A_WIDTH = 4,
    parameter int unsigned BASE_B_WIDTH = 4,
    parameter int unsigned BASE_C_WIDTH = 20,
    parameter int unsigned MACRO_WRITE_INPUT_LANES = 1,
    parameter int unsigned MAC_LATENCY = 1,
    parameter cim_mode_t MODE = CIM_MODE_BIT_SERIAL,
    parameter cim_macro_wrapper_impl_t MACRO_IMPL = CIM_MACRO_WRAPPER_IMPL_MODEL,

    // A_WIDTH and B_WIDTH are the logical operand widths implemented by this element
    parameter int unsigned A_WIDTH = 8,
    parameter int unsigned B_WIDTH = 8,
    parameter bit SIGNED = 1'b0,

    localparam int unsigned INPUT_LANES = MACRO_INPUT_LANES,
    localparam int unsigned WRITE_INPUT_LANES = MACRO_WRITE_INPUT_LANES,
    localparam int unsigned SUM_GUARD_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES),
    localparam int unsigned WEIGHT_SLICES = B_WIDTH / BASE_B_WIDTH,
    localparam int unsigned OUTPUT_LANES = MACRO_OUTPUT_LANES / WEIGHT_SLICES,
    localparam int unsigned C_WIDTH = A_WIDTH + B_WIDTH + SUM_GUARD_WIDTH,
    localparam int unsigned INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES),
    localparam int unsigned SET_INDEX_WIDTH = (WEIGHT_SETS <= 1) ? 1 : $clog2(WEIGHT_SETS),
    localparam int unsigned A_BUS_WIDTH = INPUT_LANES * A_WIDTH,
    localparam int unsigned B_BUS_WIDTH = OUTPUT_LANES * WRITE_INPUT_LANES * B_WIDTH,
    localparam int unsigned C_BUS_WIDTH = OUTPUT_LANES * C_WIDTH
) (
    input  logic                         wclk,
    input  logic                         mclk,
    input  logic                         rstn,

    input  logic [A_BUS_WIDTH-1:0]       a_bus,
    input  logic [B_BUS_WIDTH-1:0]       b_bus,
    input  logic                         wen,
    input  logic [INPUT_INDEX_WIDTH-1:0]            write_input_index,
    input  logic [SET_INDEX_WIDTH-1:0]          write_set,

    input  logic                         mac_issue,
    input  logic [SET_INDEX_WIDTH-1:0]          compute_set,

    output logic [C_BUS_WIDTH-1:0]       c_bus,
    output logic                         c_retire,
    output logic                         mac_ready,
    output logic                         mac_busy
);

  logic [A_WIDTH-1:0] a [INPUT_LANES];
  logic [B_WIDTH-1:0] b [WRITE_INPUT_LANES][OUTPUT_LANES];
  logic [C_WIDTH-1:0] c [OUTPUT_LANES];

  always_comb begin
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
      a[input_index] = a_bus[input_index * A_WIDTH +: A_WIDTH];
    end

    for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        b[write_input_offset][output_index] = b_bus[((write_input_offset * OUTPUT_LANES) + output_index) * B_WIDTH +: B_WIDTH];
      end
    end

    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      c_bus[output_index * C_WIDTH +: C_WIDTH] = c[output_index];
    end
  end

  CIMIntElement #(
      .MACRO_INPUT_LANES(MACRO_INPUT_LANES),
      .MACRO_OUTPUT_LANES(MACRO_OUTPUT_LANES),
      .WEIGHT_SETS(WEIGHT_SETS),
      .BASE_A_WIDTH(BASE_A_WIDTH),
      .BASE_B_WIDTH(BASE_B_WIDTH),
      .BASE_C_WIDTH(BASE_C_WIDTH),
      .MACRO_WRITE_INPUT_LANES(MACRO_WRITE_INPUT_LANES),
      .MAC_LATENCY(MAC_LATENCY),
      .MODE(MODE),
      .MACRO_IMPL(MACRO_IMPL),
      .A_WIDTH(A_WIDTH),
      .B_WIDTH(B_WIDTH),
      .SIGNED(SIGNED)
  ) core (
      .wclk(wclk),
      .mclk(mclk),
      .rstn(rstn),
      .a(a),
      .b(b),
      .wen(wen),
      .write_input_index(write_input_index),
      .write_set(write_set),
      .mac_issue(mac_issue),
      .compute_set(compute_set),
      .c(c),
      .c_retire(c_retire),
      .mac_ready(mac_ready),
      .mac_busy(mac_busy)
  );

endmodule
