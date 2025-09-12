#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename BufferType, typename ScaleType,
          int Width, int OcDimension>
SC_MODULE(VectorPipeline) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<ApproxUnitConfig> approx_unit_config;
  Connections::In<Pack1D<BufferType, Width>> matrix_unit_output;

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::In<Pack1D<BufferType, Width>> accumulation_buffer_output;
#endif

#if SUPPORT_MVM
  Connections::In<Pack1D<VectorType, Width>> matrix_vector_unit_data;
#endif

#if SUPPORT_DWC
  Connections::In<Pack1D<BufferType, Width>> dwc_unit_in;
#endif

  Connections::In<Pack1D<VectorType, Width>> vector_fetch_0_data;
  Connections::In<Pack1D<VectorType, Width>> vector_fetch_1_data;
  Connections::In<Pack1D<VectorType, Width>> vector_fetch_2_data;

  Connections::In<Pack1D<VectorType, Width>> accumulator_output;
  Connections::In<Pack1D<VectorType, Width>> reducer_output_0;
  Connections::In<Pack1D<VectorType, Width>> reducer_output_1;

  // Outputs to other submodules
  Connections::Out<ScaleType> mx_scale;
  Connections::Out<Pack1D<VectorType, Width>> vector_unit_output;

  Connections::Out<Pack1D<VectorType, Width>> reducer_input;
  Connections::Out<Pack1D<VectorType, Width>> accumulator_input;

  using StageInput = Transaction<VectorType, Width>;
  Connections::Combinational<Pack1D<StageInput, 4>> stage0_input;
  Connections::Combinational<Pack1D<StageInput, 3>> stage1_input;
  Connections::Combinational<Pack1D<StageInput, 3>> stage2_input;
  Connections::Combinational<Pack1D<StageInput, 2>> stage3_input;

  Connections::Combinational<VectorInstructions> stage2_inst;

#if SUPPORT_MX && VECTOR_UNIT_WIDTH != OC_DIMENSION
  Connections::Fifo<Pack1D<StageInput, 2>, OcDimension / Width>
      stage3_input_fifo;
  Connections::Combinational<Pack1D<StageInput, 2>> stage3_input_fifo_in;
  Connections::Combinational<Pack1D<StageInput, 2>> stage3_input_fifo_out;

  Connections::Combinational<ScaleType> stage3_scale;
#endif

  SC_CTOR(VectorPipeline) {
    SC_THREAD(router);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(stage0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(stage1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(stage2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(stage3);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#if SUPPORT_MX && VECTOR_UNIT_WIDTH != OC_DIMENSION
    SC_THREAD(compute_mx_qparams);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    stage3_input_fifo.clk(clk);
    stage3_input_fifo.rst(rstn);
    stage3_input_fifo.enq(stage3_input_fifo_in);
    stage3_input_fifo.deq(stage3_input_fifo_out);
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
    matrix_vector_unit_data.Reset();
#endif
#if SUPPORT_DWC
    dwc_unit_in.Reset();
#endif
    stage0_input.ResetWrite();
    stage2_inst.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = instr.Pop();
      decltype(inst.inst_count) total_values = inst.inst_count;
      decltype(inst.inst_count) counter = 0;

      stage2_inst.Push(inst);

      while (counter++ < total_values) {
        // Vector unit inputs
        Pack1D<VectorType, Width> op0_src0;
        Pack1D<VectorType, Width> op0_src1;
        Pack1D<VectorType, Width> op2_src1;
        Pack1D<VectorType, Width> op3_src1;

        if (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit ||
            inst.vector_op0_src1 == VectorInstructions::from_matrix_unit) {
          Pack1D<BufferType, Width> sa_output = matrix_unit_output.Pop();

          Pack1D<VectorType, Width> temp;
          if (inst.vdequantize) {
            vdequantize<BufferType, VectorType, Width>(sa_output, temp,
                                                       inst.vector_dq_scale);
          } else {
#pragma hls_unroll yes
            for (int i = 0; i < Width; i++) {
              temp[i] = sa_output[i];
            }
          }

          if (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
        if (inst.vector_op0_src0 ==
                VectorInstructions::from_accumulation_buffer ||
            inst.vector_op0_src1 ==
                VectorInstructions::from_accumulation_buffer) {
          Pack1D<BufferType, Width> sa_output =
              accumulation_buffer_output.Pop();

          Pack1D<VectorType, Width> temp;
          if (inst.vdequantize) {
            vdequantize<BufferType, VectorType, Width>(sa_output, temp,
                                                       inst.vector_dq_scale);
          } else {
#pragma hls_unroll yes
            for (int i = 0; i < Width; i++) {
              temp[i] = sa_output[i];
            }
          }

          if (inst.vector_op0_src0 ==
              VectorInstructions::from_accumulation_buffer) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }
#endif

#if SUPPORT_MVM
        if (inst.vector_op0_src0 ==
                VectorInstructions::from_matrix_vector_unit ||
            inst.vector_op0_src1 ==
                VectorInstructions::from_matrix_vector_unit) {
          Pack1D<VectorType, Width> temp = matrix_vector_unit_data.Pop();
          if (inst.vector_op0_src0 ==
              VectorInstructions::from_matrix_vector_unit) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }
#endif

#if SUPPORT_DWC
        if (inst.vector_op0_src0 ==
                VectorInstructions::from_dwc_unit ||
            inst.vector_op0_src1 ==
                VectorInstructions::from_dwc_unit) {
          Pack1D<BufferType, Width> sa_output = dwc_unit_in.Pop();
          Pack1D<VectorType, Width> temp;
          if (inst.vdequantize) {
            vdequantize<BufferType, VectorType, Width>(sa_output, temp,
                                                       inst.vector_dq_scale);
          } else {
#pragma hls_unroll yes
            for (int i = 0; i < Width; i++) {
              temp[i] = sa_output[i];
            }
          }

          if (inst.vector_op0_src0 ==
              VectorInstructions::from_dwc_unit) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }
#endif

        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0 ||
            inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_0) {
          Pack1D<VectorType, Width> temp = vector_fetch_0_data.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1 ||
            inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_1) {
          Pack1D<VectorType, Width> temp = vector_fetch_1_data.Pop();
          if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1) {
            op0_src0 = temp;
          } else {
            op0_src1 = temp;
          }
        }

        if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2 ||
            inst.vector_op3_src1 == VectorInstructions::from_vector_fetch_2) {
          Pack1D<VectorType, Width> temp = vector_fetch_2_data.Pop();
          if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2) {
            op2_src1 = temp;
          } else {
            op3_src1 = temp;
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_accumulation ||
            inst.vector_op0_src1 == VectorInstructions::from_accumulation ||
            inst.vector_op2_src1 == VectorInstructions::from_accumulation) {
          Pack1D<VectorType, Width> temp = accumulator_output.Pop();
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
          Pack1D<VectorType, Width> temp = reducer_output_0.Pop();
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
          Pack1D<VectorType, Width> temp;
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
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
          for (int i = 0; i < Width; i++) {
            op2_src1[i].set_bits(inst.immediate1);
          }
        }

        if (inst.vector_op3_src1 == VectorInstructions::from_immediate_2) {
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            op3_src1[i].set_bits(inst.immediate2);
          }
        }

        // HACK: We are storing op0_src1 in the second transaction
        Pack1D<StageInput, 4> transactions;
        transactions[0] = {inst.vector_op0, inst.immediate0, op0_src0};
        transactions[1] = {inst.vector_op1, inst.immediate0, op0_src1};
        transactions[2] = {inst.vector_op2, inst.vdest, op2_src1};
        transactions[3] = {inst.vector_op3, inst.immediate2, op3_src1};
        stage0_input.Push(transactions);
      }
    }
  }

  void stage0() {
    stage0_input.ResetRead();
    stage1_input.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      Pack1D<StageInput, 4> transactions = stage0_input.Pop();
      decltype(VectorInstructions::vector_op0) op0 = transactions[0].op;
      Pack1D<VectorType, Width> op0_src0 = transactions[0].payload;
      Pack1D<VectorType, Width> op0_src1 = transactions[1].payload;
      Pack1D<VectorType, Width> res0;

      // Stage 0: add, sub, mult
      if (op0 == VectorInstructions::vadd || op0 == VectorInstructions::vsub) {
        if (op0 == VectorInstructions::vsub) {
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            op0_src1[i] = op0_src1[i].negate();
          }
        }
        res0 = vadd<VectorType, Width>(op0_src0, op0_src1);
      } else if (op0 == VectorInstructions::vmult) {
        res0 = vmul<VectorType, Width>(op0_src0, op0_src1);
      } else {
        res0 = op0_src0;
      }

      transactions[1].payload = res0;

      Pack1D<StageInput, 3> stage1_data;
      stage1_data[0] = transactions[1];
      stage1_data[1] = transactions[2];
      stage1_data[2] = transactions[3];
      stage1_input.Push(stage1_data);
    }
  }

  void stage1() {
    stage1_input.ResetRead();
    stage2_input.ResetWrite();
    approx_unit_config.Reset();
    stage2_inst.ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      ApproxUnitConfig config = approx_unit_config.Pop();
      VectorInstructions inst = stage2_inst.Pop();
      decltype(inst.inst_count) total_values = inst.inst_count;
      decltype(inst.inst_count) counter = 0;

      while (counter++ < total_values) {
        Pack1D<StageInput, 3> transactions = stage1_input.Pop();
        decltype(VectorInstructions::vector_op1) op1 = transactions[0].op;
        Pack1D<VectorType, Width> op1_src0 = transactions[0].payload;
        Pack1D<VectorType, Width> res1;

        // Stage 1: exp, abs, activations
        if (op1 == VectorInstructions::vpoly) {
          res1 = vpoly<VectorType, Width>(op1_src0, config.maxes, config.ranges,
                                          config.clamp_min, config.clamp_max);
        } else if (op1 == VectorInstructions::vabs) {
          res1 = vabs<VectorType, Width>(op1_src0);
        } else if (op1 == VectorInstructions::vrelu) {
          res1 = vrelu<VectorType, Width>(op1_src0);
        } else {
          res1 = op1_src0;
        }

        transactions[0].payload = res1;
        stage2_input.Push(transactions);
      }
    }
  }

  void stage2() {
    stage2_input.ResetRead();
    stage3_input.ResetWrite();
    reducer_input.Reset();
    accumulator_input.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      Pack1D<StageInput, 3> transactions = stage2_input.Pop();
      decltype(VectorInstructions::vector_op2) op2 = transactions[1].op;
      decltype(VectorInstructions::vdest) vdest = transactions[1].immediate;
      Pack1D<VectorType, Width> op2_src0 = transactions[0].payload;
      Pack1D<VectorType, Width> op2_src1 = transactions[1].payload;
      Pack1D<VectorType, Width> res2;

      // Stage 2: add, mult, square
      if (op2 == VectorInstructions::vadd) {
        res2 = vadd<VectorType, Width>(op2_src0, op2_src1);
      } else if (op2 == VectorInstructions::vmult ||
                 op2 == VectorInstructions::vsquare) {
        if (op2 == VectorInstructions::vsquare) {
          op2_src1 = op2_src0;
        }
        res2 = vmul<VectorType, Width>(op2_src0, op2_src1);
      } else {
        res2 = op2_src0;
      }

      // Write outputs
      if (vdest == VectorInstructions::to_output) {
        Pack1D<StageInput, 2> stage3_data;
        stage3_data[0] = {0, 0, res2};
        stage3_data[1] = transactions[2];
        stage3_input.Push(stage3_data);
      } else if (vdest == VectorInstructions::to_reduce) {
        reducer_input.Push(res2);
      } else if (vdest == VectorInstructions::to_accumulate) {
        accumulator_input.Push(res2);
      }
    }
  }
#if SUPPORT_MX && VECTOR_UNIT_WIDTH != OC_DIMENSION
  void compute_mx_qparams() {
    stage3_input.ResetRead();
    stage3_input_fifo_in.ResetWrite();
    stage3_scale.ResetWrite();

    wait();

    VectorType amax_history;
    ac_int<4, false> count = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      Pack1D<StageInput, 2> transactions = stage3_input.Pop();
      decltype(VectorInstructions::vector_op3) op3 = transactions[1].op;
      decltype(VectorInstructions::immediate2) qparam =
          transactions[1].immediate;
      Pack1D<VectorType, Width> op3_src0 = transactions[0].payload;

      if (op3 == VectorInstructions::vquantize_mx) {
        Pack1D<VectorType, Width> temp;
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          temp[i] = op3_src0[i].abs();
        }
        VectorType amax = tree_max(temp);
        amax_history = count == 0 ? amax : std::max(amax, amax_history);
        count = count + 1;

        if (count == OcDimension / Width) {
          ScaleType scale =
              compute_scale<VectorType, ScaleType, Width>(amax_history, qparam);
          stage3_scale.Push(scale);
          count = 0;
        }
      }

      stage3_input_fifo_in.Push(transactions);
    }
  }
#endif
  void stage3() {
#if SUPPORT_MX && VECTOR_UNIT_WIDTH != OC_DIMENSION
    stage3_input_fifo_out.ResetRead();
    stage3_scale.ResetRead();
#else
    stage3_input.ResetRead();
#endif
    mx_scale.Reset();
    vector_unit_output.Reset();

    wait();

#if SUPPORT_MX && VECTOR_UNIT_WIDTH != OC_DIMENSION
    ac_int<4, false> count = 0;
    ScaleType scale;
#endif

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
#if SUPPORT_MX && VECTOR_UNIT_WIDTH != OC_DIMENSION
      Pack1D<StageInput, 2> transactions = stage3_input_fifo_out.Pop();
#else
      Pack1D<StageInput, 2> transactions = stage3_input.Pop();
#endif
      decltype(VectorInstructions::vector_op3) op3 = transactions[1].op;
      decltype(VectorInstructions::immediate2) qparam =
          transactions[1].immediate;
      Pack1D<VectorType, Width> op3_src0 = transactions[0].payload;
      Pack1D<VectorType, Width> op3_src1 = transactions[1].payload;
      Pack1D<VectorType, Width> res3;

#if SUPPORT_MX
      if (op3 == VectorInstructions::vquantize_mx) {
#if VECTOR_UNIT_WIDTH != OC_DIMENSION
        if (count == 0) {
          scale = stage3_scale.Pop();
          mx_scale.Push(scale);
        }

        count = count + 1;
        if (count == OcDimension / Width) {
          count = 0;
        }
#else
        ScaleType scale =
            calculate_mx_scale<VectorType, ScaleType, Width>(op3_src0, qparam);
        mx_scale.Push(scale);
#endif

#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          op3_src1[i] = scale;
        }
      }
#endif

      // Stage 3: div, quantize
      if (op3 == VectorInstructions::vdiv ||
          op3 == VectorInstructions::vquantize_mx) {
        res3 = vdiv<VectorType, Width>(op3_src0, op3_src1);
      } else {
        res3 = op3_src0;
      }

      vector_unit_output.Push(res3);
    }
  }
};
