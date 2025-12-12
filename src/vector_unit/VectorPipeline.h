#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
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

  // Inputs
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

  Connections::In<VectorPack> accumulator_output;
  Connections::In<VectorPack> reducer_output_0;
  Connections::In<VectorPack> reducer_output_1;

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
  Connections::Combinational<Pack1D<VectorPack, 2>> stage_3_input_fifo_out;

  Connections::Combinational<VectorPack> calculate_qparam_inputs;
  Connections::Combinational<VectorType> stage_3_amax;
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
    stage_3_input_fifo.deq(stage_3_input_fifo_out);
#endif
  }

  void router() {
    instr.Reset();
    matrix_unit_output.Reset();
    vector_fetch_0_data.Reset();
    vector_fetch_1_data.Reset();
    vector_fetch_2_data.Reset();
    accumulator_output.Reset();
    reducer_output_0.Reset();
    reducer_output_1.Reset();
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

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = instr.Pop();
      stage_0_inst.Push(inst);
      stage_1_inst.Push(inst);
      stage_2_inst.Push(inst);
      stage_3_inst.Push(inst);

      for (decltype(inst.inst_count) i = 0;; i++) {
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
                     VectorInstructions::from_accumulation_buffer ||
                 inst.vector_op0_src1 ==
                     VectorInstructions::from_accumulation_buffer) {
          raw_dq_input = accumulation_buffer_output.Pop();
          dq_active = true;
          dq_is_src0 = (inst.vector_op0_src0 ==
                        VectorInstructions::from_accumulation_buffer);
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
          if (inst.vdequantize) {
            vdequantize<BufferType, VectorType, vu_width>(
                raw_dq_input, processed_dq, inst.vector_dq_scale);
          } else {
#pragma hls_unroll yes
            for (int i = 0; i < vu_width; i++) {
              processed_dq[i] = raw_dq_input[i];
            }
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

        if (inst.vector_op0_src0 == VectorInstructions::from_accumulation ||
            inst.vector_op0_src1 == VectorInstructions::from_accumulation ||
            inst.vector_op2_src1 == VectorInstructions::from_accumulation) {
          VectorPack temp = accumulator_output.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_accumulation) {
            op0_src0 = temp;
          } else if (inst.vector_op0_src1 ==
                     VectorInstructions::from_accumulation) {
            op0_src1 = temp;
          } else {
            op2_src1 = temp;
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_reduction_0 ||
            inst.vector_op0_src1 == VectorInstructions::from_reduction_0 ||
            inst.vector_op2_src1 == VectorInstructions::from_reduction_0) {
          VectorPack temp = reducer_output_0.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_reduction_0) {
            op0_src0 = temp;
          } else if (inst.vector_op0_src1 ==
                     VectorInstructions::from_reduction_0) {
            op0_src1 = temp;
          } else {
            op2_src1 = temp;
          }
        }

        if (inst.vector_op2_src1 == VectorInstructions::from_reduction_1) {
          op2_src1 = reducer_output_1.Pop();
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

        if (inst.vector_op2_src1 == VectorInstructions::from_immediate_1) {
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            op2_src1[i].set_bits(inst.immediate1);
          }
        }

        if (inst.vector_op3_src1 == VectorInstructions::from_immediate_2) {
#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            op3_src1[i].set_bits(inst.immediate2);
          }
        }

        auto payloads = Pack1D<VectorPack, 4>::create(
            {op0_src0, op0_src1, op2_src1, op3_src1});
        stage_0_input.Push(payloads);

        if (i == inst.inst_count - 1) break;
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

      for (decltype(inst.inst_count) i = 0;; i++) {
        auto payloads = stage_0_input.Pop();
        auto op0_src0 = payloads[0];
        auto op0_src1 = payloads[1];
        VectorPack res0;

        // Stage 0: add, sub, mult
        if (op0 == VectorInstructions::vadd ||
            op0 == VectorInstructions::vsub) {
          if (op0 == VectorInstructions::vsub) {
#pragma hls_unroll yes
            for (int i = 0; i < vu_width; i++) {
              op0_src1[i] = op0_src1[i].negate();
            }
          }
          res0 = vadd<VectorType, vu_width>(op0_src0, op0_src1);
        } else if (op0 == VectorInstructions::vmult) {
          res0 = vmul<VectorType, vu_width>(op0_src0, op0_src1);
        } else {
          res0 = op0_src0;
        }

        auto payloads_next =
            Pack1D<VectorPack, 3>::create({res0, payloads[2], payloads[3]});
        stage_1_input.Push(payloads_next);

        if (i == inst.inst_count - 1) break;
      }
    }
  }

  void run_stage_1() {
    stage_1_inst.ResetRead();
    approx_unit_config.Reset();
    stage_1_input.ResetRead();
    stage_2_input.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      VectorInstructions inst = stage_1_inst.Pop();
      ApproxUnitConfig config = approx_unit_config.Pop();
      auto op1 = inst.vector_op1;

      for (decltype(inst.inst_count) i = 0;; i++) {
        auto payloads = stage_1_input.Pop();
        auto op1_src0 = payloads[0];
        VectorPack res1;

        // Stage 1: exp, abs, activations
        if (op1 == VectorInstructions::vpoly) {
          res1 =
              vpoly<VectorType, vu_width>(op1_src0, config.maxes, config.ranges,
                                          config.clamp_min, config.clamp_max);
        } else if (op1 == VectorInstructions::vabs) {
          res1 = vabs<VectorType, vu_width>(op1_src0);
        } else if (op1 == VectorInstructions::vrelu) {
          res1 = vrelu<VectorType, vu_width>(op1_src0);
        } else {
          res1 = op1_src0;
        }

        auto payloads_next =
            Pack1D<VectorPack, 3>::create({res1, payloads[1], payloads[2]});
        stage_2_input.Push(payloads_next);

        if (i == inst.inst_count - 1) break;
      }
    }
  }

  void run_stage_2() {
    stage_2_inst.ResetRead();
    stage_2_input.ResetRead();
    reducer_input.Reset();
    accumulator_input.Reset();
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

      for (decltype(inst.inst_count) i = 0;; i++) {
        auto payloads = stage_2_input.Pop();
        auto op2_src0 = payloads[0];
        auto op2_src1 = payloads[1];
        VectorPack res2;

        // Stage 2: add, mult, square
        if (op2 == VectorInstructions::vadd) {
          res2 = vadd<VectorType, vu_width>(op2_src0, op2_src1);
        } else if (op2 == VectorInstructions::vmult ||
                   op2 == VectorInstructions::vsquare) {
          if (op2 == VectorInstructions::vsquare) {
            op2_src1 = op2_src0;
          }
          res2 = vmul<VectorType, vu_width>(op2_src0, op2_src1);
        } else {
          res2 = op2_src0;
        }

        // Write outputs
        if (vdest == VectorInstructions::to_output) {
          auto payloads_next =
              Pack1D<VectorPack, 2>::create({res2, payloads[2]});
#if MX_SPLIT_MODE
          if (op3 == VectorInstructions::vquantize_mx) {
            calculate_qparam_inputs.Push(res2);
          }
          stage_3_input_fifo_in.Push(payloads_next);
#else
          stage_3_input.Push(payloads_next);
#endif
        } else if (vdest == VectorInstructions::to_reduce) {
          reducer_input.Push(res2);
        } else if (vdest == VectorInstructions::to_accumulate) {
          accumulator_input.Push(res2);
        }
        if (i == inst.inst_count - 1) break;
      }
    }
  }
#if MX_SPLIT_MODE
  void compute_mx_qparams() {
    calculate_qparam_inputs.ResetRead();
    stage_3_amax.ResetWrite();

    wait();

    VectorType amax_history;
    ac_int<4, false> count = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      auto op3_src0 = calculate_qparam_inputs.Pop();

      VectorPack temp;
#pragma hls_unroll yes
      for (int i = 0; i < vu_width; i++) {
        temp[i] = op3_src0[i].abs();
      }
      VectorType amax = tree_max(temp);
      amax_history = count == 0 ? amax : std::max(amax, amax_history);

      if (count == mu_width / vu_width - 1) {
        stage_3_amax.Push(amax_history);
        count = 0;
      } else {
        count++;
      }
    }
  }
#endif
  void run_stage_3() {
    stage_3_inst.ResetRead();
    mx_scale.Reset();
    vector_unit_output.Reset();
#if MX_SPLIT_MODE
    stage_3_input_fifo_out.ResetRead();
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

      if (inst.vdest != VectorInstructions::to_output) {
        continue;
      }

#if MX_SPLIT_MODE
      ac_int<4, false> count = 0;
      ScaleType scale;
#endif

      for (decltype(inst.inst_count) i = 0;; i++) {
#if MX_SPLIT_MODE
        auto payloads = stage_3_input_fifo_out.Pop();
#else
        auto payloads = stage_3_input.Pop();
#endif
        auto op3_src0 = payloads[0];
        auto op3_src1 = payloads[1];
        VectorPack res3;

#if SUPPORT_MX
        if (op3 == VectorInstructions::vquantize_mx) {
#if VECTOR_UNIT_WIDTH != OC_DIMENSION
          if (count == 0) {
            VectorType amax = stage_3_amax.Pop();
            scale =
                compute_scale<VectorType, ScaleType, vu_width>(amax, qparam);
            mx_scale.Push(scale);
          }

          if (count == mu_width / vu_width - 1) {
            count = 0;
          } else {
            count++;
          }
#else
          ScaleType scale = calculate_mx_scale<VectorType, ScaleType, vu_width>(
              op3_src0, qparam);
          mx_scale.Push(scale);
#endif

#pragma hls_unroll yes
          for (int i = 0; i < vu_width; i++) {
            op3_src1[i] = scale;
          }
        }
#endif

        // Stage 3: div, quantize
        if (op3 == VectorInstructions::vdiv ||
            op3 == VectorInstructions::vquantize_mx) {
          res3 = vdiv<VectorType, vu_width>(op3_src0, op3_src1);
        } else {
          res3 = op3_src0;
        }

        vector_unit_output.Push(res3);

        if (i == inst.inst_count - 1) break;
      }
    }
  }
};
