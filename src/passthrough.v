module passthrough #(
    parameter WIDTH = 64
)(
    input clk,
    input rstn,
    input [WIDTH-1:0] bitsIn_dat,
    output bitsIn_rdy,
    input bitsIn_vld,
    output [WIDTH-1:0] typeOut_dat,
    input typeOut_rdy,
    output typeOut_vld
);

assign typeOut_vld = bitsIn_vld;
assign bitsIn_rdy = typeOut_rdy;
assign typeOut_dat = bitsIn_dat;

endmodule
