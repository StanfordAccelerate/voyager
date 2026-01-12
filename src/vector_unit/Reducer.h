#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "Repeater.h"

template <typename T, int width>
SC_MODULE(VectorReducer) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<T, width>> input;

  Connections::Out<Pack1D<T, width>> output_to_stage0;
  Connections::Out<Pack1D<T, width>> output_to_stage2;
  Connections::Out<Pack1D<T, width>> output_to_memory;

  Repeater<Pack1D<T, width>> CCS_INIT_S1(repeater_0);
  Connections::Combinational<ac_int<16, false>> repeat_count_0;
  Connections::Combinational<Pack1D<T, width>> repeat_input_0;

  Repeater<Pack1D<T, width>> CCS_INIT_S1(repeater_1);
  Connections::Combinational<ac_int<16, false>> repeat_count_1;
  Connections::Combinational<Pack1D<T, width>> repeat_input_1;

#if VECTOR_UNIT_WIDTH != OC_DIMENSION
  Repeater<Pack1D<T, width>> CCS_INIT_S1(repeater_2);
  Connections::Combinational<ac_int<16, false>> repeat_count_2;
  Connections::Combinational<Pack1D<T, width>> repeat_input_2;
#endif

  static constexpr int N = 2;
  static constexpr int LAST = N - 1;

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

      const bool is_sum_reduction = inst.reduce_op == VectorInstructions::radd;
      const T fill_value = is_sum_reduction ? T::zero() : T::min();

      ac_int<16, false> repeat_times =
          inst.rduplicate ? OC_DIMENSION / width : 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(inst.inst_loop_count) count = 0;; count++) {
        Pack1D<T, width> res;

        for (int i = 0; i < width; i++) {
          auto acc_old = Pack1D<T, N>::fill(fill_value);

          for (decltype(inst.reduce_count) j = 0;; j++) {
            Pack1D<T, width> reduce_input = input.Pop();

            T acc;
            if (is_sum_reduction) {
              T sum = tree_sum(reduce_input);
              acc = acc_old[LAST] + sum;
            } else {
              T max = tree_max(reduce_input);
              acc = std::max(acc_old[LAST], max);
            }

#pragma hls_unroll yes
            for (int k = LAST; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (j == inst.reduce_count - 1) break;
          }

          T output;
          if (is_sum_reduction) {
            output = tree_sum(acc_old);
          } else {
            output = tree_max(acc_old);
          }

          if (inst.rsqrt) {
            output = output.sqrt();
          }

          if (inst.rreciprocal) {
            output = output.is_zero() ? T::zero() : output.reciprocal();
          }

          if (inst.rscale) {
            T scale;
            scale.set_bits(inst.immediate1);
            output = output * scale;
          }

          res[i] = output;

          if (inst.rduplicate) {
            res = Pack1D<T, width>::fill(output);
            break;
          }
        }

        if (inst.rdest == VectorInstructions::to_op0) {
          repeat_count_0.Push(inst.immediate0);
          repeat_input_0.Push(res);
        } else if (inst.rdest == VectorInstructions::to_op2) {
          repeat_count_1.Push(inst.immediate0);
          repeat_input_1.Push(res);
        } else if (inst.rdest == VectorInstructions::to_memory) {
#if VECTOR_UNIT_WIDTH != OC_DIMENSION
          repeat_count_2.Push(repeat_times);
          repeat_input_2.Push(res);
#else
          output_to_memory.Push(res);
#endif
        }

        if (count == inst.inst_loop_count - 1) break;
      }
    }
  }
};
