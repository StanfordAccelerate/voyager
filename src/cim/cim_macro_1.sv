// CIM macro 1 implementation and adapter for CIMIntMacroWrapper
`include "cim_typedefs.svh"

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
// CIMVanillaMacro models the concrete bit-serial SRAM-backed macro used by CIM macro 1
module CIMVanillaMacro #(
    parameter int unsigned INPUT_LANES  = 256,      //MAC input dimension
    parameter int unsigned OUTPUT_LANES = 64,       //MAC output dimension
    parameter int unsigned W_BITS        = 4,
    parameter int unsigned A_BITS        = 4,
    parameter int unsigned DIN_BITS     = OUTPUT_LANES*W_BITS,
    parameter int unsigned ADD_BITS     = $clog2(INPUT_LANES) + W_BITS,
    parameter int unsigned PSUM_BITS    = ADD_BITS + A_BITS,
    parameter int unsigned DOUT_BITS    = PSUM_BITS * OUTPUT_LANES
)(
    input wire [INPUT_LANES-1:0] IN,      //bit-serial input activations
    input wire [DIN_BITS-1:0] DIN,     //SRAM write bit-width
    input wire [$clog2(INPUT_LANES)-1:0]   WADDR,
    input wire CLK,
    input wire WEN,               //Write enable, low active
    input wire CEN,               //Compute enable, low active
    output reg [DOUT_BITS-1:0] DOUT
);
// synthesis translate_off
    logic [$clog2(A_BITS+1)-1:0] bit_cnt;

    logic [DIN_BITS-1:0] WSRAM [INPUT_LANES];
    logic [DOUT_BITS-1:0] PSUM;
    always @ (posedge CLK) begin
        if (~WEN) begin
            WSRAM[WADDR] <= DIN;
        end
    end

    always @ (posedge CLK) begin
        if (~CEN) begin
            bit_cnt <= (bit_cnt == A_BITS)? 1: bit_cnt + 1;
        end
        else begin
            bit_cnt <= A_BITS;
        end
    end

    logic [ADD_BITS-1:0] ADD_RES [OUTPUT_LANES];
    genvar gv_i;
    generate
    for (gv_i = 0; gv_i < OUTPUT_LANES; gv_i++) begin : g_add
        always_comb begin
            ADD_RES[gv_i] = '0;
            for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
                if (IN[input_index]) begin
                    ADD_RES[gv_i] += WSRAM[input_index][gv_i*W_BITS +: W_BITS];   //assume unsigned addition for simplicity
                end
            end
        end
    end
    endgenerate

    generate
    for (gv_i = 0; gv_i < OUTPUT_LANES; gv_i++) begin : g_dout
        always @ (posedge CLK) begin
            if (~CEN) begin
                PSUM[gv_i*PSUM_BITS +: PSUM_BITS] <=
                    (bit_cnt == A_BITS)?
                        ADD_RES[gv_i] << (A_BITS-1):
                        (ADD_RES[gv_i] << (A_BITS-1)) +
                        (PSUM[gv_i*PSUM_BITS +: PSUM_BITS] >> 1);
			DOUT[gv_i*PSUM_BITS +: PSUM_BITS] <= (bit_cnt == A_BITS)? PSUM[gv_i*PSUM_BITS +: PSUM_BITS]: DOUT[gv_i*PSUM_BITS +: PSUM_BITS];
            end
        end
    end
    endgenerate
// synthesis translate_on
endmodule
/* verilator lint_on WIDTHTRUNC */
/* verilator lint_on WIDTHEXPAND */


// CIMVanillaMacroAdapter maps the canonical wrapper interface toward a generic real macro shape
module CIMVanillaMacroAdapter #(
    parameter int unsigned INPUT_LANES = 64,
    parameter int unsigned OUTPUT_LANES = 8,
    parameter int unsigned WEIGHT_SETS = 1,
    parameter int unsigned A_WIDTH = 4,
    parameter int unsigned B_WIDTH = 4,
    parameter int unsigned C_WIDTH = 20,
    parameter int unsigned WRITE_INPUT_LANES = 1,
    parameter int unsigned MAC_LATENCY = 1,
    parameter cim_mode_t MODE = CIM_MODE_BIT_SERIAL,
    localparam int unsigned INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES),
    localparam int unsigned SET_INDEX_WIDTH = (WEIGHT_SETS <= 1) ? 1 : $clog2(WEIGHT_SETS)
) (
    input  logic                  wclk,
    input  logic                  mclk,
    input  logic [A_WIDTH-1:0]    a [INPUT_LANES],
    input  logic [B_WIDTH-1:0]    b [WRITE_INPUT_LANES][OUTPUT_LANES],
    input  logic                  wen,
    input  logic                  mac,
    input  logic                  init,
    input  logic                  a_signed,
    input  logic                  b_signed [OUTPUT_LANES],
    input  logic [INPUT_INDEX_WIDTH-1:0] write_input_index,
    input  logic [SET_INDEX_WIDTH-1:0]   write_set,
    input  logic [SET_INDEX_WIDTH-1:0]   compute_set,
    output logic [C_WIDTH-1:0]    c [OUTPUT_LANES]
);

  localparam int unsigned GENERIC_DIN_BITS = OUTPUT_LANES * B_WIDTH;
  localparam int unsigned GENERIC_ADD_BITS = $clog2(INPUT_LANES) + B_WIDTH;
  localparam int unsigned GENERIC_PSUM_BITS = GENERIC_ADD_BITS + A_WIDTH;
  localparam int unsigned GENERIC_DOUT_BITS = GENERIC_PSUM_BITS * OUTPUT_LANES;
  localparam int unsigned BITS_A_BIT_IDX = (A_WIDTH <= 1) ? 1 : $clog2(A_WIDTH);
  localparam logic [BITS_A_BIT_IDX-1:0] A_LAST_IDX = BITS_A_BIT_IDX'(A_WIDTH - 1);

  logic [BITS_A_BIT_IDX-1:0] active_bit_idx;
  logic [BITS_A_BIT_IDX-1:0] bit_idx = '0;
  logic flush_pending = 1'b0;
  logic internal_compute;
  logic [INPUT_LANES-1:0] generic_in;
  logic [GENERIC_DIN_BITS-1:0] generic_din;
  logic [GENERIC_DOUT_BITS-1:0] generic_dout;
  logic [INPUT_INDEX_WIDTH-1:0] generic_write_input_index;
  logic generic_clk;
  logic generic_wen_n;
  logic generic_cen_n;

  assign active_bit_idx = (init && mac) ? '0 : bit_idx;
  assign internal_compute = mac || flush_pending;
  assign generic_clk = mclk | wclk;  // Vanilla macro has one shared write/compute clock
  assign generic_write_input_index = write_input_index;
  assign generic_wen_n = ~wen;
  assign generic_cen_n = ~internal_compute;

  always_ff @(posedge mclk) begin
    if (mac) begin
      if (init) begin
        if (A_WIDTH > 1) begin
          bit_idx <= BITS_A_BIT_IDX'(1);
        end else begin
          bit_idx <= '0;
        end
        flush_pending <= (A_WIDTH == 1);
      end else if (bit_idx == A_LAST_IDX) begin
        bit_idx <= '0;
        flush_pending <= 1'b1;
      end else begin
        bit_idx <= bit_idx + BITS_A_BIT_IDX'(1);
        flush_pending <= 1'b0;
      end
    end else if (flush_pending) begin
      bit_idx <= '0;
      flush_pending <= 1'b0;
    end else begin
      flush_pending <= 1'b0;
    end
  end

  always_comb begin
    generic_in = '0;
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
      if (mac) begin
        generic_in[input_index] = a[input_index][active_bit_idx];
      end
    end
  end

  always_comb begin
    generic_din = '0;
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      generic_din[output_index*B_WIDTH +: B_WIDTH] = b[0][output_index];
    end
  end

  CIMVanillaMacro #(
      .INPUT_LANES(INPUT_LANES),
      .OUTPUT_LANES(OUTPUT_LANES),
      .W_BITS(B_WIDTH),
      .A_BITS(A_WIDTH)
  ) cim_macro (
      .IN(generic_in),
      .DIN(generic_din),
      .WADDR(generic_write_input_index),
      .CLK(generic_clk),
      .WEN(generic_wen_n),
      .CEN(generic_cen_n),
      .DOUT(generic_dout)
  );

  always_comb begin
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      c[output_index] = C_WIDTH'(generic_dout[output_index*GENERIC_PSUM_BITS +: GENERIC_PSUM_BITS]);
    end
  end

`ifndef SYNTHESIS
  initial begin
    if (WEIGHT_SETS != 1) begin
      $fatal(1, "CIMVanillaMacroAdapter: CIM macro 1 adapter requires WEIGHT_SETS == 1");
    end
    if (WRITE_INPUT_LANES != 1) begin
      $fatal(1, "CIMVanillaMacroAdapter: CIM macro 1 adapter requires WRITE_INPUT_LANES == 1");
    end
    if (MODE != CIM_MODE_BIT_SERIAL) begin
      $fatal(1, "CIMVanillaMacroAdapter: CIM macro 1 adapter requires bit-serial mode");
    end
    if (MAC_LATENCY < 3) begin
      $fatal(1, "CIMVanillaMacroAdapter: CIM macro 1 adapter requires MAC_LATENCY >= 3");
    end
  end

  // Signed canonical controls are unsupported by the generic macro shape
  always @(posedge mclk) begin
    if (mac && a_signed) begin
      $error("CIMVanillaMacroAdapter: a_signed must stay low");
    end
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      if (mac && b_signed[output_index]) begin
        $error("CIMVanillaMacroAdapter: b_signed[%0d] must stay low", output_index);
      end
    end
    if (mac && (compute_set != '0)) begin
      $error("CIMVanillaMacroAdapter: compute_set must stay zero because the vanilla macro has a single B set");
    end
  end

  // The generic macro has one CLK, so write traffic is assumed synchronous to mclk
  always @(posedge wclk) begin
    if (wen && (write_set != '0)) begin
      $error("CIMVanillaMacroAdapter: write_set must stay zero because the vanilla macro has a single B set");
    end
  end
`endif

endmodule
