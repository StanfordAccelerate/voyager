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
          int width, int mu_width>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<BufferType, mu_width>> CCS_INIT_S1(matrix_unit_output);
  Connections::Combinational<Pack1D<BufferType, width>> CCS_INIT_S1(
      matrix_unit_output_unpacked);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::In<Pack1D<BufferType, mu_width>>
      accumulation_buffer_read_data[2];
  Connections::SyncOut accumulation_buffer_done[2];
  Connections::Combinational<Pack1D<BufferType, width>>
      accumulation_buffer_output;
#endif

#if SUPPORT_MVM
  Connections::In<Pack1D<BufferType, width>> CCS_INIT_S1(
      matrix_vector_unit_output);
#endif

#if SUPPORT_SPMM
  Connections::In<Pack1D<VectorType, width>> CCS_INIT_S1(spmm_unit_output);
#endif

#if SUPPORT_DWC
  Connections::In<Pack1D<BufferType, width>> CCS_INIT_S1(dwc_unit_in);
  Connections::In<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(dwc_address_in);
#endif

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
  Connections::Combinational<Pack1D<VectorType, width>> CCS_INIT_S1(
      vector_fetch_0_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::Combinational<Pack1D<VectorType, width>> CCS_INIT_S1(
      vector_fetch_1_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);
  Connections::Combinational<Pack1D<VectorType, width>> CCS_INIT_S1(
      vector_fetch_2_data);

  // Vector Pipeline
  Connections::Combinational<Pack1D<VectorType, width>> CCS_INIT_S1(
      pipeline_to_memory);
  Connections::Combinational<ScaleType> CCS_INIT_S1(mx_scale);

  Connections::Combinational<ApproxUnitConfig> CCS_INIT_S1(approx_unit_config);

  // Internal connections between submodules
  Connections::Combinational<Pack1D<VectorType, width>> reducer_input;
  Connections::Combinational<Pack1D<VectorType, width>> reducer_output_0;
  Connections::Combinational<Pack1D<VectorType, width>> reducer_output_1;
  Connections::Combinational<Pack1D<VectorType, width>> reducer_to_memory;

  Connections::Combinational<Pack1D<VectorType, width>> accumulator_input;
  Connections::Combinational<Pack1D<VectorType, width>> accumulator_to_pipeline;
  Connections::Combinational<Pack1D<VectorType, width>> accumulator_to_memory;

  Connections::Combinational<Pack1D<VectorType, mu_width>> vector_unit_output;

#if SUPPORT_SPMM
  using Meta = SPMM_META_DATATYPE;

  Connections::Combinational<OutlierFilterConfig> CCS_INIT_S1(
      outlier_filter_config);
  Connections::Combinational<CsrDataAndIndices<VectorType, Meta, width>>
      CCS_INIT_S1(csr_data_and_indices);
  Connections::Combinational<Pack1D<Meta, width>> CCS_INIT_S1(csr_indptr);
#endif

  // Outputs
  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_addr);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(
      mx_scale_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(mx_scale_output_addr);
  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_addr);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  // Submodules
  VectorParamsDeserializer CCS_INIT_S1(param_deserializer);
  VectorFetchUnit<VectorType, BufferType, width, mu_width, VU_INPUT_TYPES>
      CCS_INIT_S1(fetcher);
  VectorPipeline<VectorType, BufferType, ScaleType, width, mu_width>
      CCS_INIT_S1(pipeline);
  VectorReducer<VectorType, width> CCS_INIT_S1(reducer);
  VectorAccumulator<VectorType, width> CCS_INIT_S1(accumulator);
  OutputController<VectorType, ScaleType, mu_width, OUTPUT_DATATYPES>
      CCS_INIT_S1(output_controller);

  SC_CTOR(VectorUnit) {
    // Param deserializer
    param_deserializer.clk(clk);
    param_deserializer.rstn(rstn);
    param_deserializer.serial_params_in(serial_params_in);
    param_deserializer.vector_params_out(vector_params);
    param_deserializer.vector_instructions_out(vector_instruction);

    // Vector fetcher
    fetcher.clk(clk);
    fetcher.rstn(rstn);
    fetcher.params_in(vector_fetch_params);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    fetcher.accumulation_buffer_read_address[0](
        accumulation_buffer_read_address[0]);
    fetcher.accumulation_buffer_read_address[1](
        accumulation_buffer_read_address[1]);
    fetcher.accumulation_buffer_read_data[0](accumulation_buffer_read_data[0]);
    fetcher.accumulation_buffer_read_data[1](accumulation_buffer_read_data[1]);
    fetcher.accumulation_buffer_done[0](accumulation_buffer_done[0]);
    fetcher.accumulation_buffer_done[1](accumulation_buffer_done[1]);
    fetcher.accumulation_buffer_output(accumulation_buffer_output);
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
    pipeline.matrix_vector_unit_output(matrix_vector_unit_output);
#endif
#if SUPPORT_SPMM
    pipeline.spmm_unit_output(spmm_unit_output);
#endif
#if SUPPORT_DWC
    pipeline.dwc_unit_in(dwc_unit_in);
#endif
    pipeline.vector_fetch_0_data(vector_fetch_0_data);
    pipeline.vector_fetch_1_data(vector_fetch_1_data);
    pipeline.vector_fetch_2_data(vector_fetch_2_data);
    pipeline.accumulator_output(accumulator_to_pipeline);
    pipeline.reducer_output_0(reducer_output_0);
    pipeline.reducer_output_1(reducer_output_1);
    pipeline.mx_scale(mx_scale);
    pipeline.vector_unit_output(pipeline_to_memory);
    pipeline.reducer_input(reducer_input);
    pipeline.accumulator_input(accumulator_input);
    pipeline.approx_unit_config(approx_unit_config);
#if SUPPORT_SPMM
    pipeline.outlier_filter_config(outlier_filter_config);
    pipeline.csr_data_and_indices(csr_data_and_indices);
    pipeline.csr_indptr(csr_indptr);
#endif

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
#if SUPPORT_SPMM
    output_controller.csr_indptr(csr_indptr);
    output_controller.csr_data_and_indices(csr_data_and_indices);
#endif
    output_controller.vector_output_data(vector_output_data);
    output_controller.vector_output_addr(vector_output_addr);
    output_controller.mx_scale_output_data(mx_scale_output_data);
    output_controller.mx_scale_output_addr(mx_scale_output_addr);
    output_controller.sparse_tensor_output_data(sparse_tensor_output_data);
    output_controller.sparse_tensor_output_addr(sparse_tensor_output_addr);
    output_controller.done(done);
#if SUPPORT_DWC
    output_controller.dwc_address_in(dwc_address_in);
#endif

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

  void send_instructions() {
    vector_instruction.ResetRead();
    pipeline_instr.ResetWrite();
    accumulator_instr.ResetWrite();
    reducer_instr.ResetWrite();
    approx_unit_config.ResetWrite();
    output_instruction.ResetWrite();
#if SUPPORT_SPMM
    outlier_filter_config.ResetWrite();
#endif

    vector_params.ResetRead();
    vector_fetch_params.ResetWrite();
    output_controller_params.ResetWrite();
    send_output_params.ResetWrite();

    start.Reset();

    wait();

    while (true) {
      VectorParams params = vector_params.Pop();
      VectorInstructionConfig instruction_config = vector_instruction.Pop();

      start.SyncPush();

      vector_fetch_params.Push(params);
      output_controller_params.Push(params);
      send_output_params.Push(params);
      output_instruction.Push(instruction_config);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(instruction_config.config_loop_count) i = 0;; i++) {
        for (decltype(instruction_config.num_inst) j = 0;; j++) {
          VectorInstructions inst = instruction_config.inst[j];

          if (inst.op_type == VectorInstructions::vector) {
            pipeline_instr.Push(inst);
            approx_unit_config.Push(instruction_config.approx);
#if SUPPORT_SPMM
            if (params.has_sparse_output) {
              outlier_filter_config.Push(instruction_config.outlier_filter);
            }
#endif
          } else if (inst.op_type == VectorInstructions::accumulation) {
            accumulator_instr.Push(inst);
          } else if (inst.op_type == VectorInstructions::reduction) {
            reducer_instr.Push(inst);
          }

          if (j == instruction_config.num_inst - 1) break;
        }
        if (i == instruction_config.config_loop_count - 1) break;
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
      auto full_response = matrix_unit_output.Pop();
      for (int i = 0; i < mu_width / width; i++) {
        Pack1D<BufferType, width> unpacked_data;
#pragma hls_unroll yes
        for (int j = 0; j < width; j++) {
          unpacked_data[j] = full_response[i * width + j];
        }
        matrix_unit_output_unpacked.Push(unpacked_data);
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
      VectorParams params = send_output_params.Pop();
      VectorInstructionConfig instruction_config = output_instruction.Pop();

      ac_int<2, false> op_type;
      VectorInstructions output_inst;
#pragma hls_unroll yes
      for (int i = 0; i < 8; i++) {
        VectorInstructions inst = instruction_config.inst[i];
        if (inst.vdest == VectorInstructions::to_output ||
            inst.rdest == VectorInstructions::to_memory) {
          op_type = inst.op_type;
          output_inst = inst;
          break;
        }
      }

      ac_int<32, false> num_outputs =
          output_inst.inst_loop_count * instruction_config.config_loop_count;
#if VECTOR_UNIT_WIDTH != OC_DIMENSION
      if (output_inst.op_type == VectorInstructions::reduction &&
          output_inst.rduplicate) {
        num_outputs *= (mu_width / vu_width);
      }
#endif

#if SUPPORT_CODEBOOK_QUANT
      VectorType midpoints[NUM_CODEBOOK_ENTRIES];
      if (params.use_output_codebook) {
#pragma hls_unroll yes
        for (int i = 1; i < NUM_CODEBOOK_ENTRIES; i++) {
          midpoints[i] =
              typename VectorType::ac_float_rep(params.output_code[i - 1]);
          midpoints[i].adjust_exponent(-1);
        }
      }
#endif

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
      for (ac_int<32, false> count = 0;; count++) {
        Pack1D<VectorType, mu_width> packed_outputs;

        for (int pack = 0; pack < mu_width / width; pack++) {
          auto outputs = Pack1D<VectorType, width>::zero();

          if (op_type == VectorInstructions::vector) {
            outputs = pipeline_to_memory.Pop();
          } else if (op_type == VectorInstructions::accumulation) {
            outputs = accumulator_to_memory.Pop();
          } else if (op_type == VectorInstructions::reduction) {
            outputs = reducer_to_memory.Pop();
          }

#if SUPPORT_CODEBOOK_QUANT
          if (params.use_output_codebook) {
#pragma hls_unroll yes
            for (int i = 0; i < width; i++) {
              auto index = find_codebook_index(outputs[i], midpoints);
              outputs[i].set_bits(index);
            }
          }
#endif

#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            packed_outputs[pack * width + i] = outputs[i];
          }
        }

        vector_unit_output.Push(packed_outputs);

        if (count == num_outputs - 1) break;
      }
    }
  }
};
