// CIM macro wrapper and behavioral implementations for stored-B matrix-vector MAC operations
//
// The model is intentionally resetless to match the physical macro.
// The public CIMIntMacroWrapper interface stays canonical while implementations sit behind it

`include "cim_typedefs.svh"

`ifndef SYNTHESIS
// CIMExceptionPkg records simulation-only DUT exception types for self-checking illegal tests
package CIMExceptionPkg;
  localparam string EXCEPTION_TYPE_SERIAL_INIT_WITHOUT_MAC = "SERIAL_INIT_WITHOUT_MAC";
  localparam string EXCEPTION_TYPE_SERIAL_MAC_DROP = "SERIAL_MAC_DROP";
  localparam string EXCEPTION_TYPE_ROW_WRITE_MAC_COLLISION = "ROW_WRITE_MAC_COLLISION";

  bit seen_serial_init_without_mac;
  bit seen_serial_mac_drop;
  bit seen_row_write_mac_collision;
  int unsigned exception_count;
  int unsigned unknown_exception_type_count;

  // Record one collected exception type and keep the simulator running
  task automatic collect_exception(input string exception_type, input string message);
    begin
      exception_count++;
      if (exception_type == EXCEPTION_TYPE_SERIAL_INIT_WITHOUT_MAC) begin
        seen_serial_init_without_mac = 1'b1;
      end else if (exception_type == EXCEPTION_TYPE_SERIAL_MAC_DROP) begin
        seen_serial_mac_drop = 1'b1;
      end else if (exception_type == EXCEPTION_TYPE_ROW_WRITE_MAC_COLLISION) begin
        seen_row_write_mac_collision = 1'b1;
      end else begin
        unknown_exception_type_count++;
      end
      $display("[exception:%s] %s", exception_type, message);
    end
  endtask

  // Return whether a collected exception type was observed
  function automatic bit exception_type_seen(input string exception_type);
    begin
      if (exception_type == EXCEPTION_TYPE_SERIAL_INIT_WITHOUT_MAC) begin
        exception_type_seen = seen_serial_init_without_mac;
      end else if (exception_type == EXCEPTION_TYPE_SERIAL_MAC_DROP) begin
        exception_type_seen = seen_serial_mac_drop;
      end else if (exception_type == EXCEPTION_TYPE_ROW_WRITE_MAC_COLLISION) begin
        exception_type_seen = seen_row_write_mac_collision;
      end else begin
        exception_type_seen = 1'b0;
      end
    end
  endfunction

  // Return the number of collected exceptions
  function automatic int unsigned collected_exception_count();
    begin
      collected_exception_count = exception_count;
    end
  endfunction

  // Return the number of collected exception strings not recognized by the package
  function automatic int unsigned collected_unknown_exception_type_count();
    begin
      collected_unknown_exception_type_count = unknown_exception_type_count;
    end
  endfunction
endpackage
`endif

// CIMIntMacroWrapper keeps the macro wrapper interface stable while selecting an implementation
module CIMIntMacroWrapper #(
    parameter int unsigned INPUT_LANES,
    parameter int unsigned OUTPUT_LANES,
    parameter int unsigned WEIGHT_SETS,
    parameter int unsigned A_WIDTH,                  // Bit-width of A operand (the streaming operand)
    parameter int unsigned B_WIDTH,                  // Base bit-width of B operand (the stored operand)
    parameter int unsigned WRITE_INPUT_LANES,                 // Number of input lanes written per cycle when writing B
    parameter int unsigned C_WIDTH,                  // Bit-width of the output result; This determines the max supported precision for A
    parameter int unsigned MAC_LATENCY,              // Latency to produce C; for bit-serial, this is the latency for a single bit's MAC, the total latency would be this plus A_WIDTH-1
    parameter cim_mode_t MODE = CIM_MODE_BIT_SERIAL, // Select bit-parallel or bit-serial MAC behavior
    parameter cim_macro_wrapper_impl_t IMPL = CIM_MACRO_WRAPPER_IMPL_MODEL,     // Select the implementation behind this wrapper
    localparam int unsigned INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES),      // Number of bits needed to index the B operand when writing; note when WRITE_INPUT_LANES > 1, contiguous addresses are written
    localparam int unsigned SET_INDEX_WIDTH  = (WEIGHT_SETS <= 1) ? 1 : $clog2(WEIGHT_SETS)  // Number of bits needed to index the B sets
) (
    input  logic                  wclk,
    input  logic                  mclk,
    input  logic [A_WIDTH-1:0]    a [INPUT_LANES],               // Activation operand sampled directly while mac is high
    input  logic [B_WIDTH-1:0]    b [WRITE_INPUT_LANES][OUTPUT_LANES],    // Row-major stored-operand block to be written
    input  logic                  wen,
    input  logic                  mac,                     // Signals the start of an MAC op, no need to stay high for the mac pipeline; for bit-serial, this performs one-bit mac; for bit-parallel, this performs an A_WIDTHxB_WIDTH mac
    input  logic                  init,                    // Bit-serial: marks the first partial result when mac is high; ignored by bit-parallel
    input  logic                  a_signed,
    input  logic                  b_signed [OUTPUT_LANES],       // Each output lane can have different signedness, which is needed to support wider signed B computation
    input  logic [INPUT_INDEX_WIDTH-1:0] write_input_index,                    // First input lane written across all output lanes
    input  logic [SET_INDEX_WIDTH-1:0]   write_set,                    // Index of the B set to write
    input  logic [SET_INDEX_WIDTH-1:0]   compute_set,                    // Index of the B set used for MAC computation
    output logic [C_WIDTH-1:0]    c [OUTPUT_LANES]
);
`ifndef SYNTHESIS
  import CIMExceptionPkg::*;
`endif

  generate
    if (IMPL == CIM_MACRO_WRAPPER_IMPL_MODEL) begin : gen_model_impl
      CIMIntMacroModel #(
          .INPUT_LANES(INPUT_LANES),
          .OUTPUT_LANES(OUTPUT_LANES),
          .WEIGHT_SETS(WEIGHT_SETS),
          .A_WIDTH(A_WIDTH),
          .B_WIDTH(B_WIDTH),
          .C_WIDTH(C_WIDTH),
          .WRITE_INPUT_LANES(WRITE_INPUT_LANES),
          .MAC_LATENCY(MAC_LATENCY),
          .MODE(MODE)
      ) model (
          .wclk(wclk),
          .mclk(mclk),
          .a(a),
          .b(b),
          .wen(wen),
          .mac(mac),
          .init(init),
          .a_signed(a_signed),
          .b_signed(b_signed),
          .write_input_index(write_input_index),
          .write_set(write_set),
          .compute_set(compute_set),
          .c(c)
      );
    end else if (IMPL == CIM_MACRO_WRAPPER_IMPL_CIM_MACRO_1) begin : gen_cim_macro_1_impl
      CIMVanillaMacroAdapter #(
          .INPUT_LANES(INPUT_LANES),
          .OUTPUT_LANES(OUTPUT_LANES),
          .WEIGHT_SETS(WEIGHT_SETS),
          .A_WIDTH(A_WIDTH),
          .B_WIDTH(B_WIDTH),
          .C_WIDTH(C_WIDTH),
          .WRITE_INPUT_LANES(WRITE_INPUT_LANES),
          .MAC_LATENCY(MAC_LATENCY),
          .MODE(MODE)
      ) macro_1 (
          .wclk(wclk),
          .mclk(mclk),
          .a(a),
          .b(b),
          .wen(wen),
          .mac(mac),
          .init(init),
          .a_signed(a_signed),
          .b_signed(b_signed),
          .write_input_index(write_input_index),
          .write_set(write_set),
          .compute_set(compute_set),
          .c(c)
      );
    end else begin : gen_unsupported_impl
      always_comb begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
          c[output_index] = '0;
        end
      end

`ifndef SYNTHESIS
      initial begin
        $fatal(1, "CIMIntMacroWrapper: unsupported macro wrapper implementation selector");
      end
`endif
    end
  endgenerate

  localparam int unsigned MACRO_SUM_GUARD_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES);

  // Check static macro parameters during elaboration
  generate
    if (INPUT_LANES == 0) begin : gen_invalid_input_lanes
      $fatal(1, "CIMIntMacroWrapper: INPUT_LANES must be positive");
    end
    if (OUTPUT_LANES == 0) begin : gen_invalid_output_lanes
      $fatal(1, "CIMIntMacroWrapper: OUTPUT_LANES must be positive");
    end
    if (WEIGHT_SETS == 0) begin : gen_invalid_weight_sets
      $fatal(1, "CIMIntMacroWrapper: WEIGHT_SETS must be positive");
    end
    if (A_WIDTH == 0) begin : gen_invalid_a_width
      $fatal(1, "CIMIntMacroWrapper: A_WIDTH must be positive");
    end
    if (B_WIDTH == 0) begin : gen_invalid_b_width
      $fatal(1, "CIMIntMacroWrapper: B_WIDTH must be positive");
    end
    if (C_WIDTH == 0) begin : gen_invalid_c_width
      $fatal(1, "CIMIntMacroWrapper: C_WIDTH must be positive");
    end
    if (WRITE_INPUT_LANES == 0) begin : gen_invalid_write_bw
      $fatal(1, "CIMIntMacroWrapper: WRITE_INPUT_LANES must be positive");
    end
    if (WRITE_INPUT_LANES != 0) begin : gen_check_write_bw
      if ((INPUT_LANES % WRITE_INPUT_LANES) != 0) begin : gen_invalid_input_lanes_write_bw
        $fatal(1, "CIMIntMacroWrapper: INPUT_LANES must be divisible by WRITE_INPUT_LANES");
      end
    end
    if (MAC_LATENCY == 0) begin : gen_invalid_mac_latency
      $fatal(1, "CIMIntMacroWrapper: MAC_LATENCY must be positive");
    end
    if (C_WIDTH < (A_WIDTH + B_WIDTH + MACRO_SUM_GUARD_WIDTH)) begin : gen_invalid_c_width_small
      $fatal(1, "CIMIntMacroWrapper: C_WIDTH is too small to hold results of the macro wrapper A/B operand width");
    end
  endgenerate

`ifndef SYNTHESIS
  // Report macro protocol violations as fatal unless a collection test asks to log and keep going
  task automatic report_macro_violation(input string exception_type, input string message);
    begin
      if ($test$plusargs("collect_exception_types")) begin
        collect_exception(exception_type, message);
      end else begin
        $fatal(1, "%s", message);
      end
    end
  endtask

  // Internal debug-only completion marker for standalone macro simulation
  localparam int unsigned BITS_A_COUNT = (A_WIDTH <= 1) ? 1 : $clog2(A_WIDTH + 1);
  localparam int unsigned C_VALID_DELAY = (MAC_LATENCY <= 1) ? 0 : MAC_LATENCY - 1;

  logic c_valid;

  logic [BITS_A_COUNT-1:0] serial_bits_seen;
  logic serial_result_input_valid;
  logic result_input_valid;

  always_comb begin
    serial_result_input_valid = 1'b0;
    if (mac) begin
      if (init) begin
        serial_result_input_valid = (A_WIDTH == 1);
      end else begin
        serial_result_input_valid = (serial_bits_seen == BITS_A_COUNT'(A_WIDTH - 1));
      end
    end
  end

  assign result_input_valid = (MODE == CIM_MODE_BIT_SERIAL) ? serial_result_input_valid : mac;

  initial begin
    serial_bits_seen = '0;
  end

  always @(posedge mclk) begin
    if (mac) begin
      if (init || (serial_bits_seen == BITS_A_COUNT'(A_WIDTH))) begin
        serial_bits_seen <= BITS_A_COUNT'(1);
      end else begin
        serial_bits_seen <= serial_bits_seen + BITS_A_COUNT'(1);
      end
    end
  end

  generate
    if (C_VALID_DELAY == 0) begin : gen_debug_valid_no_delay
      initial begin
        c_valid = 1'b0;
      end

      always @(posedge mclk) begin
        c_valid <= result_input_valid;
      end
    end else begin : gen_debug_valid_pipe
      logic c_valid_pipe [C_VALID_DELAY];

      initial begin
        c_valid = 1'b0;
        for (int stage = 0; stage < C_VALID_DELAY; stage++) begin
          c_valid_pipe[stage] = 1'b0;
        end
      end

      always @(posedge mclk) begin
        c_valid_pipe[0] <= result_input_valid;
        for (int stage = 1; stage < C_VALID_DELAY; stage++) begin
          c_valid_pipe[stage] <= c_valid_pipe[stage-1];
        end
        c_valid <= c_valid_pipe[C_VALID_DELAY-1];
      end
    end

    if (MODE == CIM_MODE_BIT_SERIAL) begin : gen_serial_protocol_checker
      logic mac_checker_prev;
      logic mac_checker_active;
      int unsigned mac_checker_cycles_left;

      initial begin
        mac_checker_prev = 1'b0;
        mac_checker_active = 1'b0;
        mac_checker_cycles_left = 0;
      end

      // Bit-serial streams must present one consecutive mac cycle per A bit
      always @(posedge mclk) begin
        mac_checker_prev <= mac;

        if (init && mac) begin
          mac_checker_active <= (A_WIDTH > 1);
          mac_checker_cycles_left <= (A_WIDTH > 1) ? A_WIDTH - 1 : 0;
        end else if (mac_checker_active) begin
          if (!mac) begin
            report_macro_violation(EXCEPTION_TYPE_SERIAL_MAC_DROP, $sformatf("CIMIntMacroWrapper bit-serial protocol violation: mac deasserted before A_WIDTH (%0d) consecutive cycles completed", A_WIDTH));
            mac_checker_active <= 1'b0;
            mac_checker_cycles_left <= 0;
          end else if (mac_checker_cycles_left <= 1) begin
            mac_checker_active <= 1'b0;
            mac_checker_cycles_left <= 0;
          end else begin
            mac_checker_cycles_left <= mac_checker_cycles_left - 1;
          end
        end else if (mac && !mac_checker_prev) begin
          mac_checker_active <= (A_WIDTH > 1);
          mac_checker_cycles_left <= (A_WIDTH > 1) ? A_WIDTH - 1 : 0;
        end
      end
    end
  endgenerate

  // init is a marker attached to a valid MAC input, not a standalone reset
  always @(posedge mclk) begin
    if ((MODE == CIM_MODE_BIT_SERIAL) && init && !mac) begin
      report_macro_violation(EXCEPTION_TYPE_SERIAL_INIT_WITHOUT_MAC, "CIMIntMacroWrapper protocol violation: init asserted while mac is low");
    end
  end
`endif
endmodule
