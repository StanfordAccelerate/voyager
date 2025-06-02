#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../ParamsDeserializer.h"
#include "Accumulator.h"
#include "Broadcaster.h"
#include "OutputController.h"
#include "Quantizer.h"
#include "Reducer.h"
#include "VectorFetch.h"
#include "VectorOps.h"
#include "VectorPipeline.h"

template <typename VectorType, typename BufferType, typename ScaleType,
          int Width>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  VectorParamsDeserializer CCS_INIT_S1(param_deserializer);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::In<Pack1D<BufferType, Width>> accumulation_buffer_read_data[2];
  Connections::SyncOut accumulation_buffer_done[2];
  Connections::Out<BufferWriteRequest<Pack1D<BufferType, Width>>>
      accumulation_buffer_write_request[2];
  Connections::Combinational<Pack1D<BufferType, Width>>
      accumulation_buffer_output;
#endif

#if SUPPORT_MVM
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(
      matrix_vector_unit_data);
#endif

  Connections::In<Pack1D<BufferType, Width>> CCS_INIT_S1(matrix_unit_output);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_params);
  Connections::Combinational<VectorInstructionConfig> CCS_INIT_S1(
      vector_instruction);

  // Instruction channels
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(pipeline_instr);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(accumulator_instr);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(reducer_instr);

  // Vector fetch ports
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_0_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_1_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_2_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_3_req);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(vector_fetch_3_resp);

  // Output
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_unit_output);
  Connections::Combinational<ScaleType> CCS_INIT_S1(mx_scale);

  // External memory output
  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_address_out);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(scale_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(scale_address_out);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  // Submodules
  VectorFetchUnit<VectorType, BufferType, Width, VU_INPUT_TYPES> CCS_INIT_S1(
      fetcher);
  VectorPipeline<VectorType, BufferType, Width> CCS_INIT_S1(pipeline);
  VectorReducer<VectorType, Width> CCS_INIT_S1(reducer);
  VectorAccumulator<VectorType, Width> CCS_INIT_S1(accumulator);
  VectorQuantizer<VectorType, ScaleType, Width> CCS_INIT_S1(quantizer);
  OutputController<VectorType, ScaleType, Width, OUTPUT_DATATYPES> CCS_INIT_S1(
      output_controller);

  Broadcaster<Pack1D<VectorType, Width>> CCS_INIT_S1(broadcaster_0);
  Broadcaster<Pack1D<VectorType, Width>> CCS_INIT_S1(broadcaster_1);

  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_fetch_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      output_controller_params);

  // Internal connections between submodules
  Connections::Combinational<VectorInstructions> quantizer_instr;
  Connections::Combinational<Pack1D<VectorType, Width>> quantizer_input;
  Connections::Combinational<Pack1D<VectorType, Width>> quantizer_scale;

  Connections::Combinational<Pack1D<VectorType, Width>> reducer_input;
  Connections::Combinational<Pack1D<VectorType, Width>> accumulator_input;
  Connections::Combinational<Pack1D<VectorType, Width>> accumulator_output;

  Connections::Combinational<Pack1D<VectorType, Width>> reducer_output_0;
  Connections::Combinational<Pack1D<VectorType, Width>> reducer_output_1;

  Connections::Combinational<ac_int<16, false>> broadcast_count_0;
  Connections::Combinational<ac_int<16, false>> broadcast_count_1;
  Connections::Combinational<Pack1D<VectorType, Width>> broadcast_input_0;
  Connections::Combinational<Pack1D<VectorType, Width>> broadcast_input_1;

  SC_CTOR(VectorUnit)
      : pipeline("pipeline"),
        reducer("reducer"),
        accumulator("accumulator"),
        quantizer("quantizer"),
        fetcher("fetcher"),
        output_controller("output_controller") {
    // Param deserializer
    param_deserializer.clk(clk);
    param_deserializer.rstn(rstn);
    param_deserializer.serialParamsIn(serial_params_in);
    param_deserializer.vectorParamsOut(vector_params);
    param_deserializer.vectorInstructionsOut(vector_instruction);

    // Vector fetcher
    fetcher.clk(clk);
    fetcher.rstn(rstn);
    fetcher.params_in(vector_fetch_params);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    fetcher.accumulation_buffer_read_address[0](
        accumulation_buffer_read_address[0]);
    fetcher.accumulation_buffer_read_address[1](
        accumulation_buffer_read_address[1]);
    fetcher.accumulation_buffer_done[0](accumulation_buffer_done[0]);
    fetcher.accumulation_buffer_done[1](accumulation_buffer_done[1]);
    fetcher.accumulation_buffer_read_data[0](accumulation_buffer_read_data[0]);
    fetcher.accumulation_buffer_read_data[1](accumulation_buffer_read_data[1]);
    fetcher.accumulation_buffer_output(accumulation_buffer_output);
    fetcher.accumulation_buffer_write_request[0](
        accumulation_buffer_write_request[0]);
    fetcher.accumulation_buffer_write_request[1](
        accumulation_buffer_write_request[1]);
#endif
    fetcher.vector_fetch_0_req(vector_fetch_0_req);
    fetcher.vector_fetch_0_resp(vector_fetch_0_resp);
    fetcher.vector_fetch_0_data(vector_fetch_0_data);

    fetcher.vector_fetch_1_req(vector_fetch_1_req);
    fetcher.vector_fetch_1_resp(vector_fetch_1_resp);
    fetcher.vector_fetch_1_data(vector_fetch_1_data);

    fetcher.vector_fetch_2_req(vector_fetch_2_req);
    fetcher.vector_fetch_2_resp(vector_fetch_2_resp);
    fetcher.vector_fetch_2_data(vector_fetch_2_data);

    // Main pipeline
    pipeline.clk(clk);
    pipeline.rstn(rstn);
    pipeline.instr(pipeline_instr);
    pipeline.matrix_unit_output(matrix_unit_output);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    pipeline.accumulation_buffer_output(accumulation_buffer_output);
#endif
#if SUPPORT_MVM
    pipeline.matrix_vector_unit_data(matrix_vector_unit_data);
#endif
    pipeline.vector_fetch_0_data(vector_fetch_0_data);
    pipeline.vector_fetch_1_data(vector_fetch_1_data);
    pipeline.vector_fetch_2_data(vector_fetch_2_data);
    pipeline.vector_fetch_3_req(vector_fetch_3_req);
    pipeline.vector_fetch_3_resp(vector_fetch_3_resp);
    pipeline.accumulator_output(accumulator_output);
    pipeline.reducer_output_0(reducer_output_0);
    pipeline.reducer_output_1(reducer_output_1);

    pipeline.quantizer_instr(quantizer_instr);
    pipeline.quantizer_input(quantizer_input);
    pipeline.quantizer_scale(quantizer_scale);

    pipeline.reducer_input(reducer_input);
    pipeline.accumulator_input(accumulator_input);

    // Reducer
    reducer.clk(clk);
    reducer.rstn(rstn);
    reducer.instr(reducer_instr);
    reducer.input(reducer_input);

    reducer.count_0(broadcast_count_0);
    reducer.output_0(broadcast_input_0);
    reducer.count_1(broadcast_count_1);
    reducer.output_1(broadcast_input_1);

    broadcaster_0.clk(clk);
    broadcaster_0.rstn(rstn);
    broadcaster_0.dataIn(broadcast_input_0);
    broadcaster_0.count(broadcast_count_0);
    broadcaster_0.dataOut(reducer_output_0);

    broadcaster_1.clk(clk);
    broadcaster_1.rstn(rstn);
    broadcaster_1.dataIn(broadcast_input_1);
    broadcaster_1.count(broadcast_count_1);
    broadcaster_1.dataOut(reducer_output_1);

    // Accumulator
    accumulator.clk(clk);
    accumulator.rstn(rstn);
    accumulator.instr(accumulator_instr);
    accumulator.input(accumulator_input);
    accumulator.output(accumulator_output);

    // Quantizer
    quantizer.clk(clk);
    quantizer.rstn(rstn);
    quantizer.instr(quantizer_instr);
    quantizer.input(quantizer_input);
    quantizer.scale(quantizer_scale);
    quantizer.output(vector_unit_output);
    quantizer.mx_scale(mx_scale);

    // Output controller
    output_controller.clk(clk);
    output_controller.rstn(rstn);
    output_controller.params_in(output_controller_params);
    output_controller.vector_in(vector_unit_output);
    output_controller.scale_in(mx_scale);
    output_controller.vector_out(vector_out);
    output_controller.vector_address_out(vector_address_out);
    output_controller.scale_out(scale_out);
    output_controller.scale_address_out(scale_address_out);
    output_controller.done(done);

    // Param / Instruction handling
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_instructions);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    vector_params.ResetRead();
    vector_fetch_params.ResetWrite();
    output_controller_params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vector_params.Pop();
      vector_fetch_params.Push(params);
      output_controller_params.Push(params);
    }
  }

  void send_instructions() {
    vector_instruction.ResetRead();
    pipeline_instr.ResetWrite();
    accumulator_instr.ResetWrite();
    reducer_instr.ResetWrite();

    start.Reset();

    wait();

    while (true) {
      VectorInstructionConfig vector_inst_config = vector_instruction.Pop();
      start.SyncPush();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int loop = 0; loop < vector_inst_config.instLoopCount; loop++) {
        for (int i = 0; i < 8; i++) {
          VectorInstructions inst = vector_inst_config.inst[i];

          for (int count = 0; count < vector_inst_config.instCount[i];
               count++) {
            if (inst.op_type == VectorInstructions::vector) {
              pipeline_instr.Push(inst);
            } else if (inst.op_type == VectorInstructions::accumulation) {
              accumulator_instr.Push(inst);
            } else {
              reducer_instr.Push(inst);
            }
          }

          if (i >= vector_inst_config.instLen - 1) {
            break;
          }
        }
      }
    }
  }
};
