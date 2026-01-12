module passthrough #(
    parameter WIDTH = 64
)(
    input clk,
    input rstn,
    input [WIDTH-1:0] bits_in_dat,
    output bits_in_rdy,
    input bits_in_vld,
    output [WIDTH-1:0] type_out_dat,
    input type_out_rdy,
    output type_out_vld
);

assign type_out_vld = bits_in_vld;
assign bits_in_rdy = type_out_rdy;
assign type_out_dat = bits_in_dat;

endmodule
