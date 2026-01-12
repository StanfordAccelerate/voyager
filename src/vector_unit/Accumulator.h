#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "VectorOps.h"

template <typename T, int width>
SC_MODULE(VectorAccumulator) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<T, width>> input;

  // Outputs
  Connections::Out<Pack1D<T, width>> output_to_pipeline;
  Connections::Out<Pack1D<T, width>> output_to_memory;

#ifdef CLOCK_PERIOD
  static constexpr double clock_period = CLOCK_PERIOD;
#else
  static constexpr double clock_period = 5.0;  // Default to 5 ns if not defined
#endif

  // Number of feedback delay stages based on clock period
  static constexpr int SUM_N = (clock_period < 5) ? 4 : 2;
  static constexpr int SUM_LAST = SUM_N - 1;
  static constexpr int MAX_N = (clock_period < 5) ? 2 : 1;
  static constexpr int MAX_LAST = MAX_N - 1;

  static_assert(SUM_N > 0, "Pipeline size SUM_N must be greater than 0");
  static_assert(MAX_N > 0, "Pipeline size MAX_N must be greater than 0");

  SC_CTOR(VectorAccumulator) {
    SC_THREAD(run_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run_accumulation() {
    instr.Reset();
    input.Reset();
    output_to_pipeline.Reset();
    output_to_memory.Reset();

    wait();

    while (true) {
      VectorInstructions inst = instr.Pop();
      decltype(inst.inst_loop_count) total_values = inst.inst_loop_count;
      decltype(inst.inst_loop_count) counter = 0;

      if (inst.reduce_op == VectorInstructions::radd) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (counter++ < total_values) {
          Pack1D<T, width> acc_old[SUM_N];

#pragma hls_unroll yes
          for (int i = 0; i < SUM_N; i++) {
            acc_old[i] = Pack1D<T, width>::zero();
          }

          for (decltype(inst.reduce_count) i = 0;; i++) {
            Pack1D<T, width> accum_input = input.Pop();
            Pack1D<T, width> acc = vadd(acc_old[SUM_LAST], accum_input);

#pragma hls_unroll yes
            for (int k = SUM_LAST; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (i == inst.reduce_count - 1) break;
          }

          Pack1D<T, width> outputs;

#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            Pack1D<T, SUM_N> col;
#pragma hls_unroll yes
            for (int j = 0; j < SUM_N; j++) {
              col[j] = acc_old[j][i];
            }
            outputs[i] = tree_sum(col);
          }

          if (inst.rdest == VectorInstructions::to_memory) {
            output_to_memory.Push(outputs);
          } else {
            output_to_pipeline.Push(outputs);
          }
        }
      } else if (inst.reduce_op == VectorInstructions::rmax) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (counter++ < total_values) {
          Pack1D<T, width> acc_old[MAX_N];

#pragma hls_unroll yes
          for (int i = 0; i < MAX_N; i++) {
            acc_old[i] = Pack1D<T, width>::fill(T::min());
          }

          for (decltype(inst.reduce_count) i = 0;; i++) {
            Pack1D<T, width> accum_input = input.Pop();
            Pack1D<T, width> acc = vmax(acc_old[MAX_LAST], accum_input);

#pragma hls_unroll yes
            for (int k = MAX_LAST; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (i == inst.reduce_count - 1) break;
          }

          Pack1D<T, width> outputs;

#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            Pack1D<T, MAX_N> col;
#pragma hls_unroll yes
            for (int j = 0; j < MAX_N; j++) {
              col[j] = acc_old[j][i];
            }
            outputs[i] = tree_max(col);
          }

          if (inst.rdest == VectorInstructions::to_memory) {
            output_to_memory.Push(outputs);
          } else {
            output_to_pipeline.Push(outputs);
          }
        }
      }
    }
  }
};
