#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../ParamsDeserializer.h"
#include "Accumulator.h"
#include "OutputController.h"
#include "Reducer.h"
#include "Utils.h"
#include "VectorFetch.h"
#include "VectorOps.h"
#include "VectorPipeline.h"

template <typename VectorType, typename BufferType, typename ScaleType,
          int vec_unit_width, int vec_reducer_width, int vec_accum_width,
          int mu_width, int port_width>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<BufferType, mu_width>> CCS_INIT_S1(matrix_unit_output);
  Connections::Combinational<Pack1D<BufferType, vec_unit_width>> CCS_INIT_S1(
      matrix_unit_output_n);

#if SUPPORT_MVM
  Connections::In<Pack1D<BufferType, vec_unit_width>> CCS_INIT_S1(
      matrix_vector_unit_output);
#endif

#if SUPPORT_SPMM
  Connections::In<Pack1D<VectorType, vec_unit_width>> CCS_INIT_S1(
      spmm_unit_output);
#endif

#if SUPPORT_DWC
  Connections::In<Pack1D<BufferType, vec_unit_width>> CCS_INIT_S1(dwc_unit_in);
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

  Connections::Combinational<ApproxUnitConfig> CCS_INIT_S1(approx_unit_config);
  Connections::Combinational<CodebookQuantizationConfig> CCS_INIT_S1(
      codebook_config);

  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_fetch_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      output_controller_params);

  // Vector fetch ports
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(vector_fetch_0_resp);
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>> CCS_INIT_S1(
      vector_fetch_0_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(vector_fetch_1_resp);
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>> CCS_INIT_S1(
      vector_fetch_1_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(vector_fetch_2_resp);
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>> CCS_INIT_S1(
      vector_fetch_2_data);

  // 1. Pipeline Interfaces (Standard Wide Width)
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>>
      pipeline_to_reducer_w;
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>>
      pipeline_to_accum_w;
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>>
      reducer_to_pipeline_w;
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>>
      accum_to_pipeline_w;

  // 2. Output Controller Interfaces (Standard Wide Width)
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>>
      reducer_to_memory_w;
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>>
      accum_to_memory_w;
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>>
      pipeline_to_memory_w;

  // 3. Narrow Island Signals (For Reducer/Accumulator internals)
  Connections::Combinational<Pack1D<VectorType, vec_reducer_width>>
      reducer_in_n;
  Connections::Combinational<Pack1D<VectorType, vec_reducer_width>>
      reducer_to_pipeline_n;
  Connections::Combinational<Pack1D<VectorType, vec_reducer_width>>
      reducer_to_memory_n;

  Connections::Combinational<Pack1D<VectorType, vec_accum_width>> accum_in_n;
  Connections::Combinational<Pack1D<VectorType, vec_accum_width>>
      accum_to_pipeline_n;
  Connections::Combinational<Pack1D<VectorType, vec_accum_width>>
      accum_to_memory_n;

  Connections::Combinational<ScaleType> CCS_INIT_S1(mx_scale);
  Connections::Combinational<Pack1D<VectorType, vec_unit_width>> CCS_INIT_S1(
      vector_unit_output);

#if SUPPORT_SPMM
  using Meta = SPMM_META_DATATYPE;

  Connections::Combinational<OutlierFilterConfig> CCS_INIT_S1(
      outlier_filter_config);
  Connections::Combinational<
      CsrDataAndIndices<VectorType, Meta, vec_unit_width>>
      CCS_INIT_S1(csr_data_and_indices);
  Connections::Combinational<Pack1D<Meta, vec_unit_width>> CCS_INIT_S1(
      csr_indptr);
#endif

  // Outputs
  Connections::Out<ac_int<port_width, false>> CCS_INIT_S1(vector_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_addr);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(
      mx_scale_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      mx_scale_output_addr);
  Connections::Out<ac_int<port_width, false>> CCS_INIT_S1(
      sparse_tensor_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_addr);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  // Submodules
  VectorParamsDeserializer CCS_INIT_S1(param_deserializer);
  VectorFetchUnit<VectorType, BufferType, vec_unit_width, mu_width,
                  VU_INPUT_TYPES>
      CCS_INIT_S1(fetcher);
  VectorPipeline<VectorType, BufferType, ScaleType, vec_unit_width, mu_width>
      CCS_INIT_S1(pipeline);
  VectorReducer<VectorType, vec_reducer_width> CCS_INIT_S1(reducer);
  VectorAccumulator<VectorType, vec_accum_width> CCS_INIT_S1(accumulator);
  OutputController<VectorType, ScaleType, mu_width, OUTPUT_DATATYPES>
      CCS_INIT_S1(output_controller);

  // Reducer Adapters
  Serializer<VectorType, vec_unit_width, vec_reducer_width> CCS_INIT_S1(
      pipeline_reducer_serializer);
  Deserializer<VectorType, vec_reducer_width, vec_unit_width> CCS_INIT_S1(
      reducer_pipeline_deserializer);
  Deserializer<VectorType, vec_reducer_width, vec_unit_width> CCS_INIT_S1(
      reducer_memory_deserializer);

  // Accumulator Adapters
  Slicer<VectorType, vec_unit_width, vec_accum_width> CCS_INIT_S1(
      pipeline_accum_slicer);
  Deserializer<VectorType, vec_accum_width, vec_unit_width> CCS_INIT_S1(
      accum_pipeline_deserializer);
  Deserializer<VectorType, vec_accum_width, vec_unit_width> CCS_INIT_S1(
      accum_memory_deserializer);

  Serializer<BufferType, mu_width, vec_unit_width> CCS_INIT_S1(
      matrix_unit_output_serializer);

  SC_CTOR(VectorUnit) {
    param_deserializer.clk(clk);
    param_deserializer.rstn(rstn);
    param_deserializer.serial_params_in(serial_params_in);
    param_deserializer.vector_params_out(vector_params);
    param_deserializer.vector_instructions_out(vector_instruction);

    fetcher.clk(clk);
    fetcher.rstn(rstn);
    fetcher.params_in(vector_fetch_params);
    fetcher.vector_fetch_0_req(vector_fetch_0_req);
    fetcher.vector_fetch_0_resp(vector_fetch_0_resp);
    fetcher.vector_fetch_0_data(vector_fetch_0_data);
    fetcher.vector_fetch_1_req(vector_fetch_1_req);
    fetcher.vector_fetch_1_resp(vector_fetch_1_resp);
    fetcher.vector_fetch_1_data(vector_fetch_1_data);
    fetcher.vector_fetch_2_req(vector_fetch_2_req);
    fetcher.vector_fetch_2_resp(vector_fetch_2_resp);
    fetcher.vector_fetch_2_data(vector_fetch_2_data);

    pipeline.clk(clk);
    pipeline.rstn(rstn);
    pipeline.instr(pipeline_instr);
    pipeline.approx_unit_config(approx_unit_config);
    pipeline.codebook_config(codebook_config);
    pipeline.matrix_unit_output(matrix_unit_output_n);
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
    pipeline.reducer_input(pipeline_to_reducer_w);
    pipeline.reducer_output(reducer_to_pipeline_w);
    pipeline.accumulator_input(pipeline_to_accum_w);
    pipeline.accumulator_output(accum_to_pipeline_w);
    pipeline.mx_scale(mx_scale);
    pipeline.vector_unit_output(pipeline_to_memory_w);
#if SUPPORT_SPMM
    pipeline.outlier_filter_config(outlier_filter_config);
    pipeline.csr_data_and_indices(csr_data_and_indices);
    pipeline.csr_indptr(csr_indptr);
#endif

    reducer.clk(clk);
    reducer.rstn(rstn);
    reducer.instr(reducer_instr);
    reducer.input(reducer_in_n);
    reducer.output_to_pipeline(reducer_to_pipeline_n);
    reducer.output_to_memory(reducer_to_memory_n);

    accumulator.clk(clk);
    accumulator.rstn(rstn);
    accumulator.instr(accumulator_instr);
    accumulator.input(accum_in_n);
    accumulator.output_to_pipeline(accum_to_pipeline_n);
    accumulator.output_to_memory(accum_to_memory_n);

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

    matrix_unit_output_serializer.clk(clk);
    matrix_unit_output_serializer.rstn(rstn);
    matrix_unit_output_serializer.in(matrix_unit_output);
    matrix_unit_output_serializer.out(matrix_unit_output_n);

    pipeline_reducer_serializer.clk(clk);
    pipeline_reducer_serializer.rstn(rstn);
    pipeline_reducer_serializer.in(pipeline_to_reducer_w);
    pipeline_reducer_serializer.out(reducer_in_n);

    pipeline_accum_slicer.clk(clk);
    pipeline_accum_slicer.rstn(rstn);
    pipeline_accum_slicer.in(pipeline_to_accum_w);
    pipeline_accum_slicer.out(accum_in_n);

    reducer_pipeline_deserializer.clk(clk);
    reducer_pipeline_deserializer.rstn(rstn);
    reducer_pipeline_deserializer.in(reducer_to_pipeline_n);
    reducer_pipeline_deserializer.out(reducer_to_pipeline_w);

    reducer_memory_deserializer.clk(clk);
    reducer_memory_deserializer.rstn(rstn);
    reducer_memory_deserializer.in(reducer_to_memory_n);
    reducer_memory_deserializer.out(reducer_to_memory_w);

    accum_pipeline_deserializer.clk(clk);
    accum_pipeline_deserializer.rstn(rstn);
    accum_pipeline_deserializer.in(accum_to_pipeline_n);
    accum_pipeline_deserializer.out(accum_to_pipeline_w);

    accum_memory_deserializer.clk(clk);
    accum_memory_deserializer.rstn(rstn);
    accum_memory_deserializer.in(accum_to_memory_n);
    accum_memory_deserializer.out(accum_to_memory_w);

    SC_THREAD(send_instructions);
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
    codebook_config.ResetWrite();
    output_instruction.ResetWrite();
#if SUPPORT_SPMM
    outlier_filter_config.ResetWrite();
#endif
    vector_params.ResetRead();
    vector_fetch_params.ResetWrite();
    output_controller_params.ResetWrite();

    start.Reset();

    wait();

    while (true) {
      VectorParams params = vector_params.Pop();
      VectorInstructionConfig instruction_config = vector_instruction.Pop();

      start.SyncPush();

      vector_fetch_params.Push(params);
      output_controller_params.Push(params);
      output_instruction.Push(instruction_config);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(instruction_config.config_loop_count) i = 0;; i++) {
        for (int j = 0; j < 8; j++) {
          VectorInstructions inst = instruction_config.inst[j];

          if (inst.op_type == VectorInstructions::vector) {
            pipeline_instr.Push(inst);
            approx_unit_config.Push(instruction_config.approx_config);
            if (inst.vdest == VectorInstructions::to_output) {
              codebook_config.Push(instruction_config.codebook_config);
            }

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

  void send_outputs() {
    output_instruction.ResetRead();
    pipeline_to_memory_w.ResetRead();
    reducer_to_memory_w.ResetRead();
    accum_to_memory_w.ResetRead();
    vector_unit_output.ResetWrite();

    wait();

    while (1) {
      VectorInstructionConfig instruction_config = output_instruction.Pop();

      VectorInstructions output_inst;
#pragma hls_unroll yes
      for (int i = 0; i < 8; i++) {
        VectorInstructions inst = instruction_config.inst[i];
        if (inst.vdest == VectorInstructions::to_output ||
            inst.rdest == VectorInstructions::to_memory) {
          output_inst = inst;
          break;
        }
      }

      auto op_type = output_inst.op_type;
      ac_int<32, false> num_outputs =
          output_inst.inst_loop_count * instruction_config.config_loop_count;

      if (op_type == VectorInstructions::vector) {
        num_outputs = num_outputs / (mu_width / vec_unit_width);
      } else if (op_type == VectorInstructions::accumulation) {
        num_outputs = num_outputs / (mu_width / vec_accum_width);
      } else if (op_type == VectorInstructions::reduction &&
                 !output_inst.rduplicate) {
        num_outputs = num_outputs / (mu_width / vec_reducer_width);
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> count = 0;; count++) {
        auto outputs = Pack1D<VectorType, vec_unit_width>::zero();

        if (op_type == VectorInstructions::vector) {
          outputs = pipeline_to_memory_w.Pop();
        } else if (op_type == VectorInstructions::accumulation) {
          outputs = accum_to_memory_w.Pop();
        } else if (op_type == VectorInstructions::reduction) {
          outputs = reducer_to_memory_w.Pop();
        }

        vector_unit_output.Push(outputs);

        if (count == num_outputs - 1) break;
      }
    }
  }
};
