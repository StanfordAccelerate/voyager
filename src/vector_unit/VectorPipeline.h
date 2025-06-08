#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "Broadcaster.h"
#include "VectorOps.h"

template <typename VectorType, typename BufferType, typename ScaleType,
          int Width>
SC_MODULE(VectorPipeline) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<BufferType, Width>> matrix_unit_output;

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::In<Pack1D<BufferType, Width>> accumulation_buffer_output;
#endif

#if SUPPORT_MVM
  Connections::In<Pack1D<VectorType, Width>> matrix_vector_unit_data;
#endif

  Connections::In<Pack1D<VectorType, Width>> vector_fetch_0_data;
  Connections::In<Pack1D<VectorType, Width>> vector_fetch_1_data;
  Connections::In<Pack1D<VectorType, Width>> vector_fetch_2_data;

  Connections::Out<MemoryRequest> vector_fetch_3_req;
  Connections::In<ac_int<16, false>> vector_fetch_3_resp;

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
  }

  void router() {
    instr.Reset();
    matrix_unit_output.Reset();
    vector_fetch_0_data.Reset();
    vector_fetch_1_data.Reset();
    vector_fetch_2_data.Reset();
    vector_fetch_3_req.Reset();
    vector_fetch_3_resp.Reset();
    accumulator_output.Reset();
    reducer_output_0.Reset();
    reducer_output_1.Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_output.Reset();
#endif
#if SUPPORT_MVM
    matrix_vector_unit_data.Reset();
#endif
    stage0_input.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = instr.Pop();
      decltype(inst.inst_count) total_values = inst.inst_count;
      decltype(inst.inst_count) counter = 0;

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

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      Pack1D<StageInput, 3> transactions = stage1_input.Pop();
      decltype(VectorInstructions::vector_op1) op1 = transactions[0].op;
      Pack1D<VectorType, Width> op1_src0 = transactions[0].payload;
      Pack1D<VectorType, Width> res1;

      // Stage 1: exp, abs, activations
      if (op1 == VectorInstructions::vexp) {
        res1 = vexp<VectorType, Width>(op1_src0);
      } else if (op1 == VectorInstructions::vabs) {
        res1 = vabs<VectorType, Width>(op1_src0);
      } else if (op1 == VectorInstructions::vrelu) {
        res1 = vrelu<VectorType, Width>(op1_src0);
      } else if (op1 == VectorInstructions::vgelu) {
        res1 = vgelu<VectorType, Width>(op1_src0);
      } else if (op1 == VectorInstructions::vsilu) {
        res1 = vsilu<VectorType, Width>(op1_src0);
        // } else if (op1 == VectorInstructions::vmap) {
        //   for (int i = 0; i < Width; i++) {
        //     DataTypes::bfloat16 value = op1_src0[i];

        //     ac_int<32, false> address = value.bits_rep() * 2;
        //     MemoryRequest request = {input.VMAP_OFFSET + address, 2};
        //     vector_fetch_3_req.Push(request);

        //     value.set_bits(vector_fetch_3_resp.Pop());
        //     res1[i] = value;
        //   }
      } else {
        res1 = op1_src0;
      }

      transactions[0].payload = res1;
      stage2_input.Push(transactions);
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

  void stage3() {
    stage3_input.ResetRead();
    mx_scale.Reset();
    vector_unit_output.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (1) {
      Pack1D<StageInput, 2> transactions = stage3_input.Pop();
      decltype(VectorInstructions::vector_op3) op3 = transactions[1].op;
      decltype(VectorInstructions::immediate2) qparam =
          transactions[1].immediate;
      Pack1D<VectorType, Width> op3_src0 = transactions[0].payload;
      Pack1D<VectorType, Width> op3_src1 = transactions[1].payload;
      Pack1D<VectorType, Width> res3;

#if SUPPORT_MX
      if (op3 == VectorInstructions::vquantize_mx) {
        ScaleType scale =
            calculate_mx_scale<VectorType, ScaleType, Width>(op3_src0, qparam);
        mx_scale.Push(scale);

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
