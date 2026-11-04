// Behavioral CIM macro model for CIMIntMacroWrapper
`include "cim_typedefs.svh"

// CIMIntMacroModel stores weight sets and computes bit-parallel or bit-serial MAC results against A
module CIMIntMacroModel #(
    parameter int unsigned INPUT_LANES = 64,
    parameter int unsigned OUTPUT_LANES = 8,
    parameter int unsigned WEIGHT_SETS = 18,
    parameter int unsigned A_WIDTH = 4,              // Bit-width of A operand (the streaming operand)
    parameter int unsigned B_WIDTH = 4,              // Base bit-width of B operand (the stored operand)
    parameter int unsigned C_WIDTH = 20,             // Bit-width of the output result; This determines the max supported precision for A
    parameter int unsigned WRITE_INPUT_LANES = 1,          // Number of input lanes written per cycle when writing B
    parameter int unsigned MAC_LATENCY = 1,          // Latency to produce C; for bit-serial, this is the latency for a single bit's MAC, the total latency would be this plus A_WIDTH-1
    parameter cim_mode_t MODE = CIM_MODE_BIT_SERIAL, // Select bit-parallel or bit-serial MAC behavior
    localparam int unsigned INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES),  // Number of bits needed to index the B operand when writing; note when WRITE_INPUT_LANES > 1, contiguous addresses are written
    localparam int unsigned SET_INDEX_WIDTH  = (WEIGHT_SETS <= 1) ? 1 : $clog2(WEIGHT_SETS)               // Number of bits needed to index weight sets
) (
    input  logic                  wclk,
    input  logic                  mclk,
    input  logic [A_WIDTH-1:0]    a [INPUT_LANES],               // Activation operand sampled directly while mac is high
    input  logic [B_WIDTH-1:0]    b [WRITE_INPUT_LANES][OUTPUT_LANES],    // Row-major stored-operand block to be written
    input  logic                  wen,
    input  logic                  mac,                     // Signals the start of an MAC op, no need to stay high for the mac pipeline; for bit-serial, this performs one-bit mac; for bit-parallel, this performs an A_WIDTHxB_WIDTH mac
    input  logic                  init,                    // Serial: marks the first partial result when mac is high; ignored by bit-parallel
    input  logic                  a_signed,
    input  logic                  b_signed [OUTPUT_LANES],       // Each output lane can have different signedness, which is needed to support wider signed B computation
    input  logic [INPUT_INDEX_WIDTH-1:0] write_input_index,                    // Input channel written across all output lanes
    input  logic [SET_INDEX_WIDTH-1:0]   write_set,                    // Weight set selected for writing B
    input  logic [SET_INDEX_WIDTH-1:0]   compute_set,                    // Weight set selected for MAC computation
    output logic [C_WIDTH-1:0]    c [OUTPUT_LANES]
);
// synthesis translate_off
  import CIMExceptionPkg::*;

  // Stored B data
  logic [B_WIDTH-1:0] b_mem [WEIGHT_SETS][INPUT_LANES][OUTPUT_LANES];

  // -- Write logic for B
  always_ff @(posedge wclk) begin
    if (wen) begin
      for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) begin : chan_in
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin : chan_out
          b_mem[write_set][write_input_index + INPUT_INDEX_WIDTH'(write_input_offset)][output_index] <= b[write_input_offset][output_index];
        end
      end
    end
  end

  // Weight set selected for the current MAC operation
  logic [B_WIDTH-1:0] b_mac [INPUT_LANES][OUTPUT_LANES];
  assign b_mac = b_mem[compute_set];

  // Report model protocol violations as fatal unless a collection test asks to log and keep going
  task automatic report_model_violation(input string exception_type, input string message);
    begin
      if ($test$plusargs("collect_exception_types")) begin
        collect_exception(exception_type, message);
      end else begin
        $fatal(1, "%s", message);
      end
    end
  endtask

  // Simulation checker for weight-set write/MAC exclusion in the behavioral model
  always @(posedge wclk) begin
    if (wen && mac && (write_set == compute_set)) begin
      report_model_violation(EXCEPTION_TYPE_ROW_WRITE_MAC_COLLISION, $sformatf("CIMIntMacroModel weight-set protocol violation: write and MAC target set %0d while both enables are high", write_set));
    end
  end

  // -- MAC logic
  generate
    if (MODE == CIM_MODE_BIT_PARALLEL) begin : gen_bit_parallel

      // Keep one extra bit so unsigned*unsigned and mixed signedness products
      // can share one signed internal representation without losing the top bit.
      localparam int unsigned MAC_RES_WIDTH = A_WIDTH + B_WIDTH + $clog2(INPUT_LANES) + 1;

      function automatic logic [MAC_RES_WIDTH-1:0] mac_product(
          input logic [A_WIDTH-1:0] a_val,
          input logic [B_WIDTH-1:0] b_val,
          input logic b_is_signed
      );
        logic signed [A_WIDTH-1:0] a_signed_value;
        logic signed [B_WIDTH-1:0] b_signed_value;
        logic signed [MAC_RES_WIDTH-1:0] a_ext;
        logic signed [MAC_RES_WIDTH-1:0] b_ext;
        logic signed [MAC_RES_WIDTH-1:0] product;
        begin
          a_signed_value = a_val;
          b_signed_value = b_val;
          // Extend each operand according to its own signedness before the multiply.
          // In the mixed signed/unsigned case, the signed operand must keep
          // its sign bit while the unsigned operand must zero-extend so its
          // MSB is not interpreted as a sign.
          a_ext = a_signed ? MAC_RES_WIDTH'(a_signed_value) : MAC_RES_WIDTH'(a_val);
          b_ext = b_is_signed ? MAC_RES_WIDTH'(b_signed_value) : MAC_RES_WIDTH'(b_val);
          product = a_ext * b_ext;
          mac_product = product;
        end
      endfunction

      function automatic logic [C_WIDTH-1:0] extend_mac_result(
          input logic [MAC_RES_WIDTH-1:0] value,
          input logic result_is_signed
      );
        logic signed [MAC_RES_WIDTH-1:0] value_signed;
        begin
          value_signed = value;
          extend_mac_result = result_is_signed ? C_WIDTH'(value_signed) : C_WIDTH'(value);
        end
      endfunction

      logic [MAC_RES_WIDTH-1:0] mac_res [OUTPUT_LANES];
      // Compute MAC
      always_comb begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin : mac_output_lanes
          mac_res[output_index] = '0;
          for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin : mac_input_lanes
            mac_res[output_index] += mac_product(a[input_index], b_mac[input_index][output_index], b_signed[output_index]);
          end
        end
      end

      logic [MAC_RES_WIDTH-1:0] mac_pipe [MAC_LATENCY][OUTPUT_LANES];
      // Signedness is sampled with each issued MAC and used when the delayed
      // result is extended, so it must be pipelined with mac_pipe.
      logic mac_signed_pipe [MAC_LATENCY][OUTPUT_LANES];

      always_ff @(posedge mclk) begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
          mac_pipe[0][output_index] <= mac_res[output_index];
          mac_signed_pipe[0][output_index] <= a_signed || b_signed[output_index];
        end

        for (int stage = 1; stage < MAC_LATENCY; stage++) begin
          for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
            mac_pipe[stage][output_index] <= mac_pipe[stage-1][output_index];
            mac_signed_pipe[stage][output_index] <= mac_signed_pipe[stage-1][output_index];
          end
        end
      end

      // Assign outputs
      for (genvar output_index = 0; output_index < OUTPUT_LANES; output_index++) begin : assign_c
        // When either operand is signed, the MAC result is treated as signed
        assign c[output_index] = extend_mac_result(mac_pipe[MAC_LATENCY-1][output_index], mac_signed_pipe[MAC_LATENCY-1][output_index]);
      end

    end else if (MODE == CIM_MODE_BIT_SERIAL) begin : gen_bit_serial

      // The number of pipeline stages between the start of MAC operation and when the result is ready for accumulation
      localparam int unsigned MAC_NUM_STAGES = MAC_LATENCY - 1;
      // In bit-serial mode, adder tree accumulates B_WIDTH bits of multiply result across all input lanes.
      // Keep one extra bit so the signed MSB partial sum can be negated without overflowing.
      localparam int unsigned MAC_RES_WIDTH = B_WIDTH + $clog2(INPUT_LANES) + 1;

      localparam int unsigned BITS_A_BIT_IDX = (A_WIDTH <= 1) ? 1 : $clog2(A_WIDTH);
      localparam logic [BITS_A_BIT_IDX-1:0] A_MSB_IDX = BITS_A_BIT_IDX'(A_WIDTH - 1);
      logic [BITS_A_BIT_IDX-1:0] bit_idx;
      logic [BITS_A_BIT_IDX-1:0] active_bit_idx;

      // init selects the MSB for the current cycle and marks this MAC input as
      // the first partial result of a new accumulation stream.
      assign active_bit_idx = (init && mac) ? A_MSB_IDX : bit_idx;

      // When init and mac overlap, the counter restarts for the accepted stream
      // and immediately advances to the next bit.
      always_ff @(posedge mclk) begin
        if (mac) begin
          if (init) begin
            if (A_MSB_IDX != '0) begin
              bit_idx <= A_MSB_IDX - BITS_A_BIT_IDX'(1);
            end else begin
              bit_idx <= A_MSB_IDX;
            end
          end else if (bit_idx == '0) begin
            bit_idx <= A_MSB_IDX;
          end else begin
            bit_idx <= bit_idx - BITS_A_BIT_IDX'(1);
          end
        end
      end

      function automatic logic [MAC_RES_WIDTH-1:0] extend_b_for_mac(
          input logic [B_WIDTH-1:0] b_val,
          input logic b_is_signed
      );
        logic signed [B_WIDTH-1:0] b_signed_value;
        begin
          b_signed_value = b_val;
          extend_b_for_mac = b_is_signed ? MAC_RES_WIDTH'(b_signed_value) : MAC_RES_WIDTH'(b_val);
        end
      endfunction

      // For signed A, the MSB has negative weight. Compute the same partial sum
      // as the other bits, then negate the whole sum before the shared adder path.
      function automatic logic [MAC_RES_WIDTH-1:0] apply_a_sign_bit(
          input logic [MAC_RES_WIDTH-1:0] partial_sum,
          input logic is_msb
      );
        logic signed [MAC_RES_WIDTH-1:0] partial_sum_signed;
        begin
          partial_sum_signed = partial_sum;
          apply_a_sign_bit = (a_signed && is_msb) ? -partial_sum_signed : partial_sum;
        end
      endfunction

      function automatic logic [C_WIDTH-1:0] extend_acc_in(
          input logic [MAC_RES_WIDTH-1:0] value,
          input logic result_is_signed
      );
        logic signed [MAC_RES_WIDTH-1:0] value_signed;
        begin
          value_signed = value;
          extend_acc_in = result_is_signed ? C_WIDTH'(value_signed) : C_WIDTH'(value);
        end
      endfunction

      // Partial MAC result for the current bit position of A
      logic [MAC_RES_WIDTH-1:0] mac_res [OUTPUT_LANES];
      // Signedness must travel with the partial result because a new stream can
      // enter before the previous stream has retired through the MAC pipeline.
      logic mac_res_signed [OUTPUT_LANES];

      always_comb begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
          mac_res_signed[output_index] = a_signed || b_signed[output_index];
        end
      end

      // Compute MAC
      always_comb begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin : mac_output_lanes
          mac_res[output_index] = '0;
          for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin : mac_input_lanes
            if (a[input_index][active_bit_idx]) begin
              mac_res[output_index] += extend_b_for_mac(b_mac[input_index][output_index], b_signed[output_index]);
            end
          end
          mac_res[output_index] = apply_a_sign_bit(mac_res[output_index], active_bit_idx == A_MSB_IDX);
        end
      end

      // Accumulator
      logic [C_WIDTH-1:0] acc [OUTPUT_LANES];

      // Case when the MAC result is accumulated in the same cycle
      if (MAC_NUM_STAGES == 0) begin : gen_no_mac_pipe

        always_ff @(posedge mclk) begin
          for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
            if (mac) begin
              if (init) begin
                acc[output_index] <= extend_acc_in(mac_res[output_index], mac_res_signed[output_index]);
              end else begin
                acc[output_index] <= (acc[output_index] << 1) + extend_acc_in(mac_res[output_index], mac_res_signed[output_index]);
              end
            end
          end
        end

      end

      // Case when multiply + adder tree has a >=1 cycle latency. The first
      // marker travels with the partial sum so the accumulator overwrites when
      // the first delayed partial result retires.
      else begin : gen_mac_pipe
        logic [MAC_RES_WIDTH-1:0] mac_pipe [MAC_NUM_STAGES][OUTPUT_LANES];
        // acc_en_pipe is the delayed MAC input-enable. It lets the accumulator
        // accept the pipeline tail after mac deasserts; callers must still avoid
        // bubbles inside the serial input stream.
        logic acc_en_pipe [MAC_NUM_STAGES];
        logic mac_init_pipe [MAC_NUM_STAGES];
        // Signedness belongs to the partial result, not the current interface
        // controls; it is pipelined so overlapping streams extend correctly.
        logic mac_signed_pipe [MAC_NUM_STAGES][OUTPUT_LANES];

        // Propagate the partial-result pipeline. init travels with the partial
        // result so it initializes the accumulator when that result retires.
        always_ff @(posedge mclk) begin
          acc_en_pipe[0] <= mac;
          mac_init_pipe[0] <= init && mac;
          for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
            mac_pipe[0][output_index] <= mac_res[output_index];
            mac_signed_pipe[0][output_index] <= mac_res_signed[output_index];
          end

          for (int stage = 1; stage < MAC_NUM_STAGES; stage++) begin
            acc_en_pipe[stage] <= acc_en_pipe[stage-1];
            mac_init_pipe[stage] <= mac_init_pipe[stage-1];
            for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
              mac_pipe[stage][output_index] <= mac_pipe[stage-1][output_index];
              mac_signed_pipe[stage][output_index] <= mac_signed_pipe[stage-1][output_index];
            end
          end
        end

        always_ff @(posedge mclk) begin
          for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
            if (acc_en_pipe[MAC_NUM_STAGES-1]) begin
              if (mac_init_pipe[MAC_NUM_STAGES-1]) begin
                acc[output_index] <= extend_acc_in(mac_pipe[MAC_NUM_STAGES-1][output_index], mac_signed_pipe[MAC_NUM_STAGES-1][output_index]);
              end else begin
                acc[output_index] <= (acc[output_index] << 1) + extend_acc_in(mac_pipe[MAC_NUM_STAGES-1][output_index], mac_signed_pipe[MAC_NUM_STAGES-1][output_index]);
              end
            end
          end
        end
      end

      // Assign outputs
      assign c = acc;
    end
  endgenerate

// synthesis translate_on
endmodule
