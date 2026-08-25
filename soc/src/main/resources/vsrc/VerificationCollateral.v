`ifdef GL_SIM
// TODO: replace this with the correct path in the GL netlist
`define VOYAGER(sig) TestDriver.testHarness.chiptop0.system_voyager``sig
`define MATRIX_UNIT_START_VLD        `VOYAGER(__accelerator_matrix_unit_start_vld)
`define MATRIX_UNIT_DONE_VLD         `VOYAGER(__accelerator_matrix_unit_done_vld)
`define VECTOR_UNIT_START_VLD        `VOYAGER(__accelerator_vector_unit_start_vld)
`define VECTOR_UNIT_DONE_VLD         `VOYAGER(__accelerator_vector_unit_done_vld)
`define MATRIX_VECTOR_UNIT_START_VLD `VOYAGER(__accelerator_matrix_vector_unit_start_vld)
`define MATRIX_VECTOR_UNIT_DONE_VLD  `VOYAGER(__accelerator_matrix_vector_unit_done_vld)
`define SPMM_UNIT_START_VLD          `VOYAGER(__accelerator_spmm_unit_start_vld)
`define SPMM_UNIT_DONE_VLD           `VOYAGER(__accelerator_spmm_unit_done_vld)
`else
`define VOYAGER TestDriver.testHarness.chiptop0.system.voyager
`define MATRIX_UNIT_START_VLD        `VOYAGER.accelerator.matrix_unit.start_vld
`define MATRIX_UNIT_DONE_VLD         `VOYAGER.accelerator.matrix_unit.done_vld
`define VECTOR_UNIT_START_VLD        `VOYAGER.accelerator.vector_unit.start_vld
`define VECTOR_UNIT_DONE_VLD         `VOYAGER.accelerator.vector_unit.done_vld
`define MATRIX_VECTOR_UNIT_START_VLD `VOYAGER.accelerator.matrix_vector_unit.start_vld
`define MATRIX_VECTOR_UNIT_DONE_VLD  `VOYAGER.accelerator.matrix_vector_unit.done_vld
`define SPMM_UNIT_START_VLD          `VOYAGER.accelerator.spmm_unit.start_vld
`define SPMM_UNIT_DONE_VLD           `VOYAGER.accelerator.spmm_unit.done_vld
`endif

import "DPI-C" context function void load_memory();
// unit encoding matches Step::Unit: 0=matrix, 1=vector, 2=mvm, 3=spmm
import "DPI-C" context function void check_outputs(input int unit);
import "DPI-C" context function void unit_started(input int unit);

module VoyagerVerification (
    input clock,
    input reset
);

    wire _source_clk = TestDriver.testHarness.source.clk;
    logic matrix_unit_start_vld_q;
    logic vector_unit_start_vld_q;
`ifdef SUPPORT_MVM
    logic matrix_vector_unit_start_vld_q;
`endif
`ifdef SUPPORT_SPMM
    logic spmm_unit_start_vld_q;
`endif

    initial begin
        @(posedge clock);  // move past time 0 to avoid signal glitch
        wait (reset === 1'b1);
        wait (reset === 1'b0);
        $display("[%0t] reset deassertion", $time);
        load_memory();
    end

    task automatic check_unit_status(
        input string name,
        input int    unit,
        input logic  start_vld,
        input logic  done_vld,
        input logic  start_vld_q
    );
        // GL start_rdy is hard to probe, so use start_vld falling edge as the
        // vld/rdy accept event.
        if (start_vld_q && !start_vld) begin
            $display("[%0t] %s started", $time, name);

            fork
                begin
                    // Off-edge for the same reason as check_outputs below:
                    // the grant pump answers by depositing the next start
                    // credit into the semaphore registers via VPI.
                    @(negedge _source_clk);
                    unit_started(unit);
                end
            join_none
        end

        if (done_vld) begin
            $display("[%0t] %s done", $time, name);

            fork
                begin
                    // Wait for data to be written to memory.
                    repeat (100) @(posedge _source_clk);
                    // Deliver the tick off-edge: check_outputs() deposits into
                    // the semaphore registers via VPI, and a deposit in the
                    // same time slot as the register's posedge NBA update is
                    // scheduler-order dependent.
                    @(negedge _source_clk);
                    check_outputs(unit);
                end
            join_none
        end
    endtask

    always @(posedge _source_clk) begin
        if (reset) begin
            matrix_unit_start_vld_q <= 1'b0;
            vector_unit_start_vld_q <= 1'b0;
`ifdef SUPPORT_MVM
            matrix_vector_unit_start_vld_q <= 1'b0;
`endif
`ifdef SUPPORT_SPMM
            spmm_unit_start_vld_q <= 1'b0;
`endif
        end else begin
            matrix_unit_start_vld_q <= `MATRIX_UNIT_START_VLD;
            vector_unit_start_vld_q <= `VECTOR_UNIT_START_VLD;
`ifdef SUPPORT_MVM
            matrix_vector_unit_start_vld_q <= `MATRIX_VECTOR_UNIT_START_VLD;
`endif
`ifdef SUPPORT_SPMM
            spmm_unit_start_vld_q <= `SPMM_UNIT_START_VLD;
`endif

            check_unit_status("matrix unit", 0,
                              `MATRIX_UNIT_START_VLD,
                              `MATRIX_UNIT_DONE_VLD,
                              matrix_unit_start_vld_q);

            check_unit_status("vector unit", 1,
                              `VECTOR_UNIT_START_VLD,
                              `VECTOR_UNIT_DONE_VLD,
                              vector_unit_start_vld_q);
`ifdef SUPPORT_MVM
            check_unit_status("matrix vector unit", 2,
                              `MATRIX_VECTOR_UNIT_START_VLD,
                              `MATRIX_VECTOR_UNIT_DONE_VLD,
                              matrix_vector_unit_start_vld_q);
`endif
`ifdef SUPPORT_SPMM
            check_unit_status("spmm unit", 3,
                              `SPMM_UNIT_START_VLD,
                              `SPMM_UNIT_DONE_VLD,
                              spmm_unit_start_vld_q);
`endif
        end
    end

endmodule
