#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "Repeater.h"

template <typename VectorType, int Width>
SC_MODULE(VectorReducer) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<VectorType, Width>> input;

  Connections::Out<Pack1D<VectorType, Width>> output_to_stage0;
  Connections::Out<Pack1D<VectorType, Width>> output_to_stage2;
  Connections::Out<Pack1D<VectorType, Width>> output_to_memory;

  Repeater<Pack1D<VectorType, Width>> CCS_INIT_S1(repeater_0);
  Connections::Combinational<ac_int<16, false>> repeat_count_0;
  Connections::Combinational<Pack1D<VectorType, Width>> repeat_input_0;

  Repeater<Pack1D<VectorType, Width>> CCS_INIT_S1(repeater_1);
  Connections::Combinational<ac_int<16, false>> repeat_count_1;
  Connections::Combinational<Pack1D<VectorType, Width>> repeat_input_1;

#if VECTOR_UNIT_WIDTH != OC_DIMENSION
  Repeater<Pack1D<VectorType, Width>> CCS_INIT_S1(repeater_2);
  Connections::Combinational<ac_int<16, false>> repeat_count_2;
  Connections::Combinational<Pack1D<VectorType, Width>> repeat_input_2;
#endif

  static constexpr int N = 2;
  static constexpr int last = N - 1;

  static_assert(N > 0, "Pipeline size N must be greater than 0");

  SC_CTOR(VectorReducer) {
    repeater_0.clk(clk);
    repeater_0.rstn(rstn);
    repeater_0.data_in(repeat_input_0);
    repeater_0.count(repeat_count_0);
    repeater_0.data_out(output_to_stage0);

    repeater_1.clk(clk);
    repeater_1.rstn(rstn);
    repeater_1.data_in(repeat_input_1);
    repeater_1.count(repeat_count_1);
    repeater_1.data_out(output_to_stage2);

#if VECTOR_UNIT_WIDTH != OC_DIMENSION
    repeater_2.clk(clk);
    repeater_2.rstn(rstn);
    repeater_2.data_in(repeat_input_2);
    repeater_2.count(repeat_count_2);
    repeater_2.data_out(output_to_memory);
#endif

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    instr.Reset();
    input.Reset();
    repeat_count_0.ResetWrite();
    repeat_input_0.ResetWrite();
    repeat_count_1.ResetWrite();
    repeat_input_1.ResetWrite();
#if VECTOR_UNIT_WIDTH != OC_DIMENSION
    repeat_count_2.ResetWrite();
    repeat_input_2.ResetWrite();
#else
    output_to_memory.Reset();
#endif

    wait();

    while (true) {
      VectorInstructions inst = instr.Pop();
      decltype(inst.inst_count) total_values = inst.inst_count;
      decltype(inst.inst_count) counter = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (counter++ < total_values) {
        Pack1D<VectorType, Width> res;
        for (ac_int<8, false> i = 0;; i++) {
          Pack1D<VectorType, N> acc_old;

#pragma hls_unroll yes
          for (int j = 0; j < N; j++) {
            if (inst.reduce_op == VectorInstructions::radd) {
              acc_old[j] = VectorType::zero();
            } else {  // rmax
              acc_old[j] = VectorType::min();
            }
          }

          for (decltype(inst.reduce_count) j = 0;; j++) {
            Pack1D<VectorType, Width> reduce_input = input.Pop();

            VectorType acc;
            if (inst.reduce_op == VectorInstructions::radd) {
              VectorType sum = tree_sum(reduce_input);
              acc = j < N ? sum : acc_old[last] + sum;
            } else {  // rmax
              VectorType max = tree_max(reduce_input);
              acc = j < N ? max : std::max(acc_old[last], max);
            }

#pragma hls_unroll yes
            for (int k = last; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (j == inst.reduce_count - 1) {
              break;
            }
          }

          VectorType output;
          if (inst.reduce_op == VectorInstructions::radd) {
            output = tree_sum(acc_old);
          } else {  // rmax
            output = tree_max(acc_old);
          }

          if (inst.rsqrt) {
            output = output.sqrt();
          }

          if (inst.rreciprocal) {
            output =
                output.is_zero() ? VectorType::zero() : output.reciprocal();
          }

          if (inst.rscale) {
            VectorType scale;
            scale.set_bits(inst.immediate1);
            output = output * scale;
          }

          res[i] = output;
          if (inst.rduplicate || i == Width - 1) {
            break;
          }
        }

        if (inst.rduplicate) {
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            res[i] = res[0];
          }
        }

        if (inst.rdest == VectorInstructions::to_op0) {
          repeat_count_0.Push(inst.immediate0);
          repeat_input_0.Push(res);
        } else if (inst.rdest == VectorInstructions::to_op2) {
          repeat_count_1.Push(inst.immediate0);
          repeat_input_1.Push(res);
        } else if (inst.rdest == VectorInstructions::to_memory) {
#if VECTOR_UNIT_WIDTH == OC_DIMENSION
          output_to_memory.Push(res);
#else
          repeat_count_2.Push(inst.rduplicate ? OC_DIMENSION / Width : 1);
          repeat_input_2.Push(res);
#endif
        }
      }
    }
  }
};
