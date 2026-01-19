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

  Connections::Out<Pack1D<T, width>> output_to_pipeline;
  Connections::Out<Pack1D<T, width>> output_to_memory;

  Repeater<Pack1D<T, width>> CCS_INIT_S1(pipeline_repeater);
  Connections::Combinational<ac_int<16, false>> pipeline_repeat_count;
  Connections::Combinational<Pack1D<T, width>> pipeline_repeat_data;

#if VECTOR_UNIT_WIDTH != OC_DIMENSION
  Repeater<Pack1D<T, width>> CCS_INIT_S1(output_repeater);
  Connections::Combinational<ac_int<16, false>> output_repeat_count;
  Connections::Combinational<Pack1D<T, width>> output_repeat_data;
#endif

  static constexpr int N = 2;
  static constexpr int LAST = N - 1;

  static_assert(N > 0, "Pipeline size N must be greater than 0");

  SC_CTOR(VectorReducer) {
    pipeline_repeater.clk(clk);
    pipeline_repeater.rstn(rstn);
    pipeline_repeater.data_in(pipeline_repeat_data);
    pipeline_repeater.count(pipeline_repeat_count);
    pipeline_repeater.data_out(output_to_pipeline);

#if VECTOR_UNIT_WIDTH != OC_DIMENSION
    output_repeater.clk(clk);
    output_repeater.rstn(rstn);
    output_repeater.data_in(output_repeat_data);
    output_repeater.count(output_repeat_count);
    output_repeater.data_out(output_to_memory);
#endif

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    instr.Reset();
    input.Reset();
    pipeline_repeat_count.ResetWrite();
    pipeline_repeat_data.ResetWrite();
#if VECTOR_UNIT_WIDTH != OC_DIMENSION
    output_repeat_count.ResetWrite();
    output_repeat_data.ResetWrite();
#else
    output_to_memory.Reset();
#endif

    wait();

    while (true) {
      VectorInstructions inst = instr.Pop();

      bool is_sum_op = (inst.reduce_op == VectorInstructions::radd);
      T fill_value = is_sum_op ? T::zero() : T::min();

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
            T sum = tree_sum(reduce_input);
            T max = tree_max(reduce_input);
            T acc = is_sum_op ? (acc_old[LAST] + sum)
                              : std::max(acc_old[LAST], max);

#pragma hls_unroll yes
            for (int k = LAST; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (j == inst.reduce_count - 1) break;
          }

          T output = is_sum_op ? tree_sum(acc_old) : tree_max(acc_old);

          if (inst.rsqrt) {
            output = output.sqrt();
          }

          if (inst.rreciprocal) {
            output = output.is_zero() ? T::zero() : output.reciprocal();
          }

          if (inst.rscale) {
            T scale = T::from_bits(inst.immediate1);
            output = output * scale;
          }

          res[i] = output;

          if (inst.rduplicate) {
            res = Pack1D<T, width>::fill(output);
            break;
          }
        }

        if (inst.rdest == VectorInstructions::to_memory) {
#if VECTOR_UNIT_WIDTH != OC_DIMENSION
          output_repeat_count.Push(repeat_times);
          output_repeat_data.Push(res);
#else
          output_to_memory.Push(res);
#endif
        } else {
          pipeline_repeat_count.Push(inst.immediate0);
          pipeline_repeat_data.Push(res);
        }

        if (count == inst.inst_loop_count - 1) break;
      }
    }
  }
};
