#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../ParamsDeserializer.h"
#include "Accumulator.h"
#include "OutputController.h"
#include "Reducer.h"
#include "VectorFetch.h"
#include "VectorOps.h"
#include "VectorPipeline.h"

template <typename VectorType, typename BufferType, typename ScaleType,
          int Width, int OcDimension>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  VectorParamsDeserializer CCS_INIT_S1(param_deserializer);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::In<Pack1D<BufferType, OcDimension>>
      accumulation_buffer_read_data[2];
  Connections::SyncOut accumulation_buffer_done[2];
  Connections::Out<BufferWriteRequest<Pack1D<BufferType, OcDimension>>>
      accumulation_buffer_write_request[2];
  Connections::Combinational<Pack1D<BufferType, Width>>
      accumulation_buffer_output;
#endif

#if SUPPORT_MVM
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(
      matrix_vector_unit_data);
#endif

  Connections::In<Pack1D<BufferType, OcDimension>> CCS_INIT_S1(
      matrix_unit_output);
  Connections::Combinational<Pack1D<BufferType, Width>> CCS_INIT_S1(
      matrix_unit_output_unpacked);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_params);
  Connections::Combinational<VectorInstructionConfig> CCS_INIT_S1(
      vector_instruction);
  Connections::Combinational<VectorInstructionConfig> CCS_INIT_S1(
      output_instruction);

  // Instruction channels
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(pipeline_instr);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(accumulator_instr);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(reducer_instr);

  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_fetch_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      output_controller_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(send_output_params);

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

  // Vector Pipeline
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_3_req);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(vector_fetch_3_resp);

  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      pipeline_to_memory);
  Connections::Combinational<ScaleType> CCS_INIT_S1(mx_scale);

  // Internal connections between submodules
  Connections::Combinational<Pack1D<VectorType, Width>> reducer_input;
  Connections::Combinational<Pack1D<VectorType, Width>> reducer_output_0;
  Connections::Combinational<Pack1D<VectorType, Width>> reducer_output_1;
  Connections::Combinational<Pack1D<VectorType, Width>> reducer_to_memory;

  Connections::Combinational<Pack1D<VectorType, Width>> accumulator_input;
  Connections::Combinational<Pack1D<VectorType, Width>> accumulator_to_pipeline;
  Connections::Combinational<Pack1D<VectorType, Width>> accumulator_to_memory;

  Connections::Combinational<Pack1D<VectorType, OcDimension>> vector_unit_output;

  // Output Controller
  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_address_out);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(scale_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(scale_address_out);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  // Submodules
  VectorFetchUnit<VectorType, BufferType, Width, OcDimension, VU_INPUT_TYPES>
      CCS_INIT_S1(fetcher);
  VectorPipeline<VectorType, BufferType, ScaleType, Width, OcDimension>
      CCS_INIT_S1(pipeline);
  VectorReducer<VectorType, Width> CCS_INIT_S1(reducer);
  VectorAccumulator<VectorType, Width> CCS_INIT_S1(accumulator);
  OutputController<VectorType, ScaleType, OcDimension, OUTPUT_DATATYPES>
      CCS_INIT_S1(output_controller);

  SC_CTOR(VectorUnit)
      : pipeline("pipeline"),
        reducer("reducer"),
        accumulator("accumulator"),
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
    pipeline.matrix_unit_output(matrix_unit_output_unpacked);
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
    pipeline.accumulator_output(accumulator_to_pipeline);
    pipeline.reducer_output_0(reducer_output_0);
    pipeline.reducer_output_1(reducer_output_1);
    pipeline.mx_scale(mx_scale);
    pipeline.vector_unit_output(pipeline_to_memory);
    pipeline.reducer_input(reducer_input);
    pipeline.accumulator_input(accumulator_input);

    // Reducer
    reducer.clk(clk);
    reducer.rstn(rstn);
    reducer.instr(reducer_instr);
    reducer.input(reducer_input);
    reducer.output_to_stage0(reducer_output_0);
    reducer.output_to_stage2(reducer_output_1);
    reducer.output_to_memory(reducer_to_memory);

    // Accumulator
    accumulator.clk(clk);
    accumulator.rstn(rstn);
    accumulator.instr(accumulator_instr);
    accumulator.input(accumulator_input);
    accumulator.output_to_pipeline(accumulator_to_pipeline);
    accumulator.output_to_memory(accumulator_to_memory);

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

    SC_THREAD(unpack_matrix_output);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_outputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    vector_params.ResetRead();
    vector_fetch_params.ResetWrite();
    output_controller_params.ResetWrite();
    send_output_params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vector_params.Pop();
      vector_fetch_params.Push(params);
      output_controller_params.Push(params);
      send_output_params.Push(params);
    }
  }

  void send_instructions() {
    vector_instruction.ResetRead();
    pipeline_instr.ResetWrite();
    accumulator_instr.ResetWrite();
    reducer_instr.ResetWrite();
    output_instruction.ResetWrite();

    start.Reset();

    wait();

    while (true) {
      VectorInstructionConfig vector_inst_config = vector_instruction.Pop();
      output_instruction.Push(vector_inst_config);
      start.SyncPush();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(vector_inst_config.instLoopCount) loop = 0;; loop++) {
        for (decltype(vector_inst_config.instLen) i = 0;; i++) {
          VectorInstructions inst = vector_inst_config.inst[i];

          for (ac_int<20, false> count = 0;; count++) {
            if (inst.op_type == VectorInstructions::vector) {
              pipeline_instr.Push(inst);
            } else if (inst.op_type == VectorInstructions::accumulation) {
              accumulator_instr.Push(inst);
            } else {
              reducer_instr.Push(inst);
            }

            if (count == vector_inst_config.instCount[i] - 1) {
              break;
            }
          }
          if (i == vector_inst_config.instLen - 1) {
            break;
          }
        }
        if (loop == vector_inst_config.instLoopCount - 1) {
          break;
        }
      }
    }
  }

  void unpack_matrix_output() {
    matrix_unit_output.Reset();
    matrix_unit_output_unpacked.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<BufferType, OcDimension> full_response = matrix_unit_output.Pop();

      if constexpr (OcDimension == Width) {
        matrix_unit_output_unpacked.Push(full_response);
      } else {
        for (int i = 0; i < OcDimension / Width; i++) {
          Pack1D<BufferType, Width> unpacked_data;
#pragma hls_unroll yes
          for (int j = 0; j < Width; j++) {
            unpacked_data[j] = full_response[i * Width + j];
          }
          matrix_unit_output_unpacked.Push(unpacked_data);
        }
      }
    }
  }

  void send_outputs() {
    output_instruction.ResetRead();
    send_output_params.ResetRead();
    pipeline_to_memory.ResetRead();
    reducer_to_memory.ResetRead();
    accumulator_to_memory.ResetRead();
    vector_unit_output.ResetWrite();

    wait();

    while (1) {
      VectorInstructionConfig vector_inst_config = output_instruction.Pop();
      VectorParams params = send_output_params.Pop();

      constexpr int packing_factor = OcDimension / Width;

      ac_int<32, false> num_outputs =
          params.output_loops[0][0] * params.output_loops[0][1] *
          params.output_loops[0][2] * params.output_loops[1][0] *
          params.output_loops[1][1] * params.output_loops[1][2];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(vector_inst_config.instLen) i = 0;; i++) {
        VectorInstructions inst = vector_inst_config.inst[i];

        if (inst.vdest == VectorInstructions::to_output ||
            inst.rdest == VectorInstructions::to_memory) {
          for (ac_int<32, false> count = 0;; count++) {
            Pack1D<VectorType, OcDimension> packed_outputs;

            for (ac_int<4, false> pack = 0;; pack++) {
              Pack1D<VectorType, Width> outputs;
              if (inst.op_type == VectorInstructions::vector) {
                outputs = pipeline_to_memory.Pop();
              } else if (inst.op_type == VectorInstructions::accumulation) {
                outputs = accumulator_to_memory.Pop();
              } else if (inst.op_type == VectorInstructions::reduction) {
                outputs = reducer_to_memory.Pop();
              }

#pragma hls_unroll yes
              for (int i = 0; i < Width; i++) {
                packed_outputs[pack * Width + i] = outputs[i];
              }

              if (pack == packing_factor - 1) {
                break;
              }
            }
            vector_unit_output.Push(packed_outputs);

            if (count == num_outputs - 1) {
              break;
            }
          }
          break;
        }
        if (i == vector_inst_config.instLen - 1) {
          break;
        }
      }
    }
  }
};
