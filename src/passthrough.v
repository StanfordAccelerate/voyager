module passthrough #(
  parameter WIDTH = 64
)(
  input logic clk,
  input logic rstn,
  input logic [WIDTH-1:0] bitsIn_dat,
  output logic bitsIn_rdy,
  input logic bitsIn_vld,
  output logic [WIDTH-1:0] typeOut_dat,
  input logic typeOut_rdy,
  output logic typeOut_vld
);

assign typeOut_vld = bitsIn_vld;
assign bitsIn_rdy = typeOut_rdy;
assign typeOut_dat = bitsIn_dat;

endmodule
