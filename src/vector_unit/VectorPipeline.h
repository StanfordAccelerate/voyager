#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "OutlierFilter2.h"
#include "VectorOps.h"

#if SUPPORT_MX && VECTOR_UNIT_WIDTH != OC_DIMENSION
#define MX_SPLIT_MODE 1
#else
#define MX_SPLIT_MODE 0
#endif

template <typename VectorType, typename BufferType, typename ScaleType,
          int vu_width, int mu_width>
SC_MODULE(VectorPipeline) {
  using VectorPack = Pack1D<VectorType, vu_width>;

  sc_in<bool> clk;
  sc_in<bool> rstn;

  Connections::In<VectorInstructions> instr;
  Connections::In<ApproxUnitConfig> approx_unit_config;

  Connections::In<Pack1D<BufferType, vu_width>> matrix_unit_output;
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::In<Pack1D<BufferType, vu_width>> accumulation_buffer_output;
#endif
#if SUPPORT_MVM
  Connections::In<Pack1D<BufferType, vu_width>> matrix_vector_unit_output;
#endif
#if SUPPORT_SPMM
  Connections::In<VectorPack> spmm_unit_output;
#endif
#if SUPPORT_DWC
  Connections::In<Pack1D<BufferType, vu_width>> dwc_unit_in;
#endif

  Connections::In<VectorPack> vector_fetch_0_data;
  Connections::In<VectorPack> vector_fetch_1_data;
  Connections::In<VectorPack> vector_fetch_2_data;
  Connections::In<VectorPack> reducer_output;
  Connections::In<VectorPack> accumulator_output;

  // Outputs to other submodules
  Connections::Out<ScaleType> mx_scale;
  Connections::Out<VectorPack> vector_unit_output;

  Connections::Out<VectorPack> reducer_input;
  Connections::Out<VectorPack> accumulator_input;

  Connections::Combinational<VectorInstructions> stage_0_inst;
  Connections::Combinational<VectorInstructions> stage_1_inst;
  Connections::Combinational<VectorInstructions> stage_2_inst;
  Connections::Combinational<VectorInstructions> stage_3_inst;

  Connections::Combinational<Pack1D<VectorPack, 4>> stage_0_input;
  Connections::Combinational<Pack1D<VectorPack, 3>> stage_1_input;
  Connections::Combinational<Pack1D<VectorPack, 3>> stage_2_input;
  Connections::Combinational<Pack1D<VectorPack, 2>> stage_3_input;

#if MX_SPLIT_MODE
  Connections::Fifo<Pack1D<VectorPack, 2>, mu_width / vu_width + 8>
      stage_3_input_fifo;
  Connections::Combinational<Pack1D<VectorPack, 2>> stage_3_input_fifo_in;

  Connections::Combinational<VectorPack> calculate_qparam_inputs;
  Connections::Combinational<VectorType> stage_3_amax;
#endif

// TODO: define new flags for outlier filtering
#if SUPPORT_SPMM
  using Meta = SPMM_META_DATATYPE;

  Connections::In<OutlierFilterConfig> outlier_filter_config;
  Connections::Combinational<VectorInstructions> outlier_filter_inst;
  Connections::Combinational<Pack1D<VectorPack, 2>> stage_3_payload;

  Connections::Combinational<VectorPack> outlier_filter_input;
  Connections::Combinational<VectorPack> outlier_filter_output;
  Connections::Out<CsrDataAndIndices<VectorType, Meta, vu_width>>
      csr_data_and_indices;
  Connections::Out<Pack1D<Meta, vu_width>> csr_indptr;

  OutlierFilter<VectorType, SPMM_META_DATATYPE, vu_width> CCS_INIT_S1(
      outlier_filter);
#endif

  SC_CTOR(VectorPipeline) {
    SC_THREAD(router);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_stage_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_stage_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_stage_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_stage_3);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#if MX_SPLIT_MODE
    SC_THREAD(compute_mx_qparams);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    stage_3_input_fifo.clk(clk);
    stage_3_input_fifo.rst(rstn);
    stage_3_input_fifo.enq(stage_3_input_fifo_in);
    stage_3_input_fifo.deq(stage_3_input);
#endif
#if SUPPORT_SPMM
    SC_THREAD(run_outlier_filter);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    outlier_filter.clk(clk);
    outlier_filter.rstn(rstn);
    outlier_filter.config_in(outlier_filter_config);
    outlier_filter.data_in(outlier_filter_input);
    outlier_filter.data_out(outlier_filter_output);
    outlier_filter.csr_data_and_indices_out(csr_data_and_indices);
    outlier_filter.csr_indptr_out(csr_indptr);
#endif
  }

  void router() {
    instr.Reset();
    approx_unit_config.Reset();
    matrix_unit_output.Reset();
    vector_fetch_0_data.Reset();
    vector_fetch_1_data.Reset();
    vector_fetch_2_data.Reset();
    reducer_output.Reset();
    accumulator_output.Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_output.Reset();
#endif
#if SUPPORT_MVM
    matrix_vector_unit_output.Reset();
#endif
#if SUPPORT_DWC
    dwc_unit_in.Reset();
#endif
    stage_0_inst.ResetWrite();
    stage_1_inst.ResetWrite();
    stage_2_inst.ResetWrite();
    stage_3_inst.ResetWrite();
    stage_0_input.ResetWrite();
#if SUPPORT_SPMM
    outlier_filter_inst.ResetWrite();
#endif

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = instr.Pop();
      ApproxUnitConfig approx_config = approx_unit_config.Pop();

      stage_0_inst.Push(inst);
      stage_1_inst.Push(inst);
      stage_2_inst.Push(inst);

      if (inst.vdest == VectorInstructions::to_output) {
        stage_3_inst.Push(inst);
#if SUPPORT_SPMM
        outlier_filter_inst.Push(inst);
#endif
      }

      for (decltype(inst.inst_loop_count) i = 0;; i++) {
        VectorPack op0_src0, op0_src1, op2_src1, op3_src1;

        Pack1D<BufferType, vu_width> raw_dq_input;
        bool dq_active = false;
        bool dq_is_src0 = false;

        if (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit ||
            inst.vector_op0_src1 == VectorInstructions::from_matrix_unit) {
          raw_dq_input = matrix_unit_output.Pop();
          dq_active = true;
          dq_is_src0 =
              (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit);
        }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
        else if (inst.vector_op0_src0 ==
                     VectorInstructions::from_accum_buffer ||
                 inst.vector_op0_src1 ==
                     VectorInstructions::from_accum_buffer) {
          raw_dq_input = accumulation_buffer_output.Pop();
          dq_active = true;
          dq_is_src0 =
              (inst.vector_op0_src0 == VectorInstructions::from_accum_buffer);
        }
#endif
#if SUPPORT_MVM
        else if (inst.vector_op0_src0 ==
                     VectorInstructions::from_matrix_vector_unit ||
                 inst.vector_op0_src1 ==
                     VectorInstructions::from_matrix_vector_unit) {
          raw_dq_input = matrix_vector_unit_output.Pop();
          dq_active = true;
          dq_is_src0 = (inst.vector_op0_src0 ==
                        VectorInstructions::from_matrix_vector_unit);
        }
#endif
#if SUPPORT_DWC
        else if (inst.vector_op0_src0 == VectorInstructions::from_dwc_unit ||
                 inst.vector_op0_src1 == VectorInstructions::from_dwc_unit) {
          raw_dq_input = dwc_unit_in.Pop();
          dq_active = true;
          dq_is_src0 =
              (inst.vector_op0_src0 == VectorInstructions::from_dwc_unit);
        }
#endif
        if (dq_active) {
          VectorPack processed_dq;
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            processed_dq[i] = raw_dq_input[i];
          }

          if (dq_is_src0) {
            op0_src0 = processed_dq;
          } else {
            op0_src1 = processed_dq;
          }
        }
#if SUPPORT_SPMM
        if (inst.vector_op0_src0 == VectorInstructions::from_spmm_unit ||
            inst.vector_op0_src1 == VectorInstructions::from_spmm_unit) {
          VectorPack temp = spmm_unit_output.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_spmm_unit) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }
#endif
        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0 ||
            inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_0) {
          VectorPack temp = vector_fetch_0_data.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1 ||
            inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_1) {
          VectorPack temp = vector_fetch_1_data.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }

        if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2 ||
            inst.vector_op3_src1 == VectorInstructions::from_vector_fetch_2) {
          VectorPack temp = vector_fetch_2_data.Pop();
          if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2) {
            op2_src1 = temp;
          } else {
            op3_src1 = temp;
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_accumulator ||
            inst.vector_op0_src1 == VectorInstructions::from_accumulator ||
            inst.vector_op2_src1 == VectorInstructions::from_accumulator) {
          VectorPack temp = accumulator_output.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_accumulator) {
            op0_src0 = temp;
          } else if (inst.vector_op0_src1 ==
                     VectorInstructions::from_accumulator) {
            op0_src1 = temp;
          } else {
            op2_src1 = temp;
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_vector_reducer ||
            inst.vector_op0_src1 == VectorInstructions::from_vector_reducer ||
            inst.vector_op2_src1 == VectorInstructions::from_vector_reducer) {
          VectorPack temp = reducer_output.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_vector_reducer) {
            op0_src0 = temp;
          } else if (inst.vector_op0_src1 ==
                     VectorInstructions::from_vector_reducer) {
            op0_src1 = temp;
          } else {
            op2_src1 = temp;
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_immediate_0 ||
            inst.vector_op0_src1 == VectorInstructions::from_immediate_0) {
          VectorPack temp;
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            temp[i].set_bits(inst.immediate0);
          }

          if (inst.vector_op0_src0 == VectorInstructions::from_immediate_0) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }

        if (inst.vector_op0 == VectorInstructions::op0_poly) {
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            auto coef = vpoly_coef<VectorType>(op0_src0[i], approx_config);
            auto& [c0, c1, c2] = coef;
            op0_src1[i] = c2;
            op2_src1[i] = c1;
            op3_src1[i] = c0;
          }
        }

        if (inst.vdequantize) {
          op0_src0 = vdequantize<VectorType, VectorType, vu_width>(
              op0_src0, inst.vector_dq_scale);
        }

        auto payloads = Pack1D<VectorPack, 4>::create(
            {op0_src0, op0_src1, op2_src1, op3_src1});
        stage_0_input.Push(payloads);

        if (i == inst.inst_loop_count - 1) break;
      }
    }
  }

  void run_stage_0() {
    stage_0_inst.ResetRead();
    stage_0_input.ResetRead();
    stage_1_input.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      VectorInstructions inst = stage_0_inst.Pop();
      auto op0 = inst.vector_op0;
      bool is_poly = (op0 == VectorInstructions::op0_poly);
      bool is_mac = (op0 == VectorInstructions::op0_mac);
      VectorType immediate = VectorType::from_bits(inst.immediate0);

      for (decltype(inst.inst_loop_count) i = 0;; i++) {
        auto payloads = stage_0_input.Pop();
        auto op0_src0 = payloads[0];
        auto op0_src1 = payloads[1];

        if (op0 == VectorInstructions::op0_sub) {
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            op0_src1[i] = op0_src1[i].negate();
          }
        }

        // op0_mac: op0_src0 + op0_src1 * immediate0
        // op0_vpoly: op0_src0 * op0_src1 + op2_src1

        VectorPack mul_lhs = is_mac ? VectorPack::fill(immediate) : op0_src0;
        VectorPack mul_rhs = op0_src1;
        VectorPack mul_result = vmul<VectorType, vu_width>(mul_lhs, mul_rhs);

        VectorPack add_lhs = is_poly ? mul_result : op0_src0;
        VectorPack add_rhs =
            is_poly ? payloads[2] : (is_mac ? mul_result : op0_src1);
        VectorPack add_result = vadd<VectorType, vu_width>(add_lhs, add_rhs);

        // Stage 0: add, sub, mult
        VectorPack stage0_output;
        if (op0 == VectorInstructions::op0_add ||
            op0 == VectorInstructions::op0_sub || is_poly || is_mac) {
          stage0_output = add_result;
        } else if (op0 == VectorInstructions::op0_mul) {
          stage0_output = mul_result;
        } else {
          stage0_output = op0_src0;
        }

        // For polynomial ops, inputs need to be multiplied again
        VectorPack payload2 = is_poly ? op0_src0 : payloads[2];

        auto payloads_next = Pack1D<VectorPack, 3>::create(
            {stage0_output, payload2, payloads[3]});
        stage_1_input.Push(payloads_next);

        if (i == inst.inst_loop_count - 1) break;
      }
    }
  }

  void run_stage_1() {
    stage_1_inst.ResetRead();

    stage_1_input.ResetRead();
    stage_2_input.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      VectorInstructions inst = stage_1_inst.Pop();

      auto op1 = inst.vector_op1;

      for (decltype(inst.inst_loop_count) i = 0;; i++) {
        auto payloads = stage_1_input.Pop();
        auto op1_src0 = payloads[0];

        // Stage 1: exp, abs, relu
        VectorPack stage1_output;
        if (op1 == VectorInstructions::op1_abs) {
          stage1_output = vabs<VectorType, vu_width>(op1_src0);
        } else if (op1 == VectorInstructions::op1_exp) {
          stage1_output = vexp_fp8<VectorType, vu_width>(op1_src0);
        } else if (op1 == VectorInstructions::op1_relu) {
          stage1_output = vrelu<VectorType, vu_width>(op1_src0);
        } else {
          stage1_output = op1_src0;
        }

        auto payloads_next = Pack1D<VectorPack, 3>::create(
            {stage1_output, payloads[1], payloads[2]});
        stage_2_input.Push(payloads_next);

        if (i == inst.inst_loop_count - 1) break;
      }
    }
  }

  void push_stage_3_inputs(const ac_int<2, false> op3,
                           const Pack1D<VectorPack, 2>& payloads) {
#if MX_SPLIT_MODE
    if (op3 == VectorInstructions::op3_quantize_mx ||
        op3 == VectorInstructions::op3_quantize_mx_outlier) {
      calculate_qparam_inputs.Push(payloads[0]);
    }
    stage_3_input_fifo_in.Push(payloads);
#else
    stage_3_input.Push(payloads);
#endif
  }

  void run_stage_2() {
    stage_2_inst.ResetRead();
    stage_2_input.ResetRead();
    reducer_input.Reset();
    accumulator_input.Reset();
#if SUPPORT_SPMM
    outlier_filter_input.ResetWrite();
    stage_3_payload.ResetWrite();
#endif
#if MX_SPLIT_MODE
    calculate_qparam_inputs.ResetWrite();
    stage_3_input_fifo_in.ResetWrite();
#else
    stage_3_input.ResetWrite();
#endif

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      VectorInstructions inst = stage_2_inst.Pop();
      auto op2 = inst.vector_op2;
      auto op3 = inst.vector_op3;
      auto vdest = inst.vdest;
      bool is_poly = (op2 == VectorInstructions::op2_poly);
      bool is_mac = (op2 == VectorInstructions::op2_mac);
      VectorType immediate = VectorType::from_bits(inst.immediate1);

      for (decltype(inst.inst_loop_count) i = 0;; i++) {
        auto payloads = stage_2_input.Pop();
        auto op2_src0 = payloads[0];
        auto op2_src1 = payloads[1];

        if (op2 == VectorInstructions::op2_sqr) {
          op2_src1 = op2_src0;
        }

        if (inst.vector_op2_src1 == VectorInstructions::from_immediate_1) {
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            op2_src1[i].set_bits(inst.immediate1);
          }
        }

        VectorPack mul_lhs = is_mac ? VectorPack::fill(immediate) : op2_src0;
        VectorPack mul_rhs = op2_src1;

        VectorPack mul_result = vmul<VectorType, vu_width>(mul_lhs, mul_rhs);

        VectorPack add_lhs = is_poly ? mul_result : op2_src0;
        VectorPack add_rhs =
            is_poly ? payloads[2] : (is_mac ? mul_result : op2_src1);
        VectorPack add_result = vadd<VectorType, vu_width>(add_lhs, add_rhs);

        // Stage 2: add, mult, square
        VectorPack stage2_output;
        if (op2 == VectorInstructions::op2_add || is_poly || is_mac) {
          stage2_output = add_result;
        } else if (op2 == VectorInstructions::op2_mul ||
                   op2 == VectorInstructions::op2_sqr) {
          stage2_output = mul_result;
        } else {
          stage2_output = op2_src0;
        }

        // Write outputs
        if (vdest == VectorInstructions::to_output) {
          auto payloads_next =
              Pack1D<VectorPack, 2>::create({stage2_output, payloads[2]});
#if SUPPORT_SPMM
          if (op3 == VectorInstructions::vquantize_mx_outlier) {
            outlier_filter_input.Push(stage2_output);
          }
          stage_3_payload.Push(payloads_next);
#else
          push_stage_3_inputs(op3, payloads_next);
#endif
        } else if (vdest == VectorInstructions::to_reduce) {
          reducer_input.Push(stage2_output);
        } else if (vdest == VectorInstructions::to_accumulate) {
          accumulator_input.Push(stage2_output);
        }

        if (i == inst.inst_loop_count - 1) break;
      }
    }
  }

#if SUPPORT_SPMM
  void run_outlier_filter() {
    outlier_filter_inst.ResetRead();
    stage_3_payload.ResetRead();
    outlier_filter_output.ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      VectorInstructions inst = outlier_filter_inst.Pop();
      auto op3 = inst.vector_op3;

      for (decltype(inst.inst_loop_count) i = 0;; i++) {
        auto payload = stage_3_payload.Pop();

        if (op3 == VectorInstructions::vquantize_mx_outlier) {
          payload[0] = outlier_filter_output.Pop();
        }

        push_stage_3_inputs(op3, payload);

        if (i == inst.inst_loop_count - 1) break;
      }
    }
  }
#endif

#if MX_SPLIT_MODE
  void compute_mx_qparams() {
    calculate_qparam_inputs.ResetRead();
    stage_3_amax.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      VectorType amax_history = VectorType::zero();

      for (int i = 0; i < mu_width / vu_width; i++) {
        auto op3_src0 = calculate_qparam_inputs.Pop();

        VectorPack temp;
#pragma hls_unroll yes
        for (int j = 0; j < vu_width; j++) {
          temp[j] = op3_src0[j].abs();
        }
        VectorType amax = tree_max(temp);

        amax_history = std::max(amax, amax_history);
      }

      stage_3_amax.Push(amax_history);
    }
  }
#endif

  void run_stage_3() {
    stage_3_inst.ResetRead();
    mx_scale.Reset();
    vector_unit_output.Reset();
#if MX_SPLIT_MODE
    stage_3_amax.ResetRead();
#else
    stage_3_input.ResetRead();
#endif

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      auto inst = stage_3_inst.Pop();
      auto op3 = inst.vector_op3;
      auto qparam = inst.immediate2;

      bool is_mx_quantization =
          (op3 == VectorInstructions::op3_quantize_mx ||
           op3 == VectorInstructions::op3_quantize_mx_outlier);

#if MX_SPLIT_MODE
      constexpr int ratio = mu_width / vu_width;
      ScaleType scale;
#endif

      for (decltype(inst.inst_loop_count) i = 0;; i++) {
        auto payloads = stage_3_input.Pop();
        auto op3_src0 = payloads[0];
        auto op3_src1 = payloads[1];

        if (inst.vector_op3_src1 == VectorInstructions::from_immediate_2) {
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            op3_src1[i].set_bits(inst.immediate2);
          }
        }

        // Cast the second operand to ScaleType for division
        Pack1D<ScaleType, vu_width> div_op1;
#pragma hls_unroll yes
        for (int i = 0; i < vu_width; i++) {
          div_op1[i] = op3_src1[i];
        }

#if SUPPORT_MX
        if (is_mx_quantization) {
#if MX_SPLIT_MODE
          if (i % ratio == 0) {
            VectorType amax = stage_3_amax.Pop();
            scale = calculate_mx_scale<VectorType, ScaleType>(amax, qparam);
            mx_scale.Push(scale);
          }
#else
          ScaleType scale = calculate_mx_scale<VectorType, ScaleType, vu_width>(
              op3_src0, qparam);
          mx_scale.Push(scale);
#endif

#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            div_op1[i] = scale;
          }
        }
#endif
        // Stage 3: quantize
        VectorPack stage3_output;
        if (op3 == VectorInstructions::op3_div || is_mx_quantization) {
          stage3_output =
              vquantize<VectorType, ScaleType, vu_width>(op3_src0, div_op1);
        } else {
          stage3_output = op3_src0;
        }

        vector_unit_output.Push(stage3_output);

        if (i == inst.inst_loop_count - 1) break;
      }
    }
  }
};
