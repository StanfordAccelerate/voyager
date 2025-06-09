#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "VectorOps.h"

template <typename VectorType, int Width>
SC_MODULE(VectorAccumulator) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<VectorType, Width>> input;

  // Outputs
  Connections::Out<Pack1D<VectorType, Width>> output_to_pipeline;
  Connections::Out<Pack1D<VectorType, Width>> output_to_memory;

#ifdef CLOCK_PERIOD
  static constexpr double clock_period = CLOCK_PERIOD;
#else
  static constexpr double clock_period = 5.0;  // Default to 5 ns if not defined
#endif
  static constexpr int sum_n = (clock_period < 1)   ? 5
                               : (clock_period < 5) ? 4
                                                    : 2;
  static constexpr int sum_last = sum_n - 1;
  static constexpr int max_n = (clock_period < 5) ? 2 : 1;
  static constexpr int max_last = max_n - 1;

  static_assert(sum_n > 0, "Pipeline size sum_n must be greater than 0");
  static_assert(max_n > 0, "Pipeline size max_n must be greater than 0");

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
      decltype(inst.inst_count) total_values = inst.inst_count;
      decltype(inst.inst_count) counter = 0;

      if (inst.reduce_op == VectorInstructions::radd) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (counter++ < total_values) {
          Pack1D<VectorType, Width> acc_old[sum_n];

#pragma hls_unroll yes
          for (int i = 0; i < sum_n; i++) {
#pragma hls_unroll yes
            for (int j = 0; j < Width; j++) {
              acc_old[i][j] = VectorType::zero();
            }
          }

          for (decltype(inst.reduce_count) i = 0;; i++) {
            Pack1D<VectorType, Width> reduce_input = input.Pop();

            Pack1D<VectorType, Width> acc =
                i < sum_n ? reduce_input
                          : vadd(acc_old[sum_last], reduce_input);

#pragma hls_unroll yes
            for (int k = sum_last; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (i == inst.reduce_count - 1) {
              break;
            }
          }

          Pack1D<VectorType, Width> outputs;

#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            Pack1D<VectorType, sum_n> col;
#pragma hls_unroll yes
            for (int j = 0; j < sum_n; j++) {
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
          Pack1D<VectorType, Width> acc_old[max_n];

#pragma hls_unroll yes
          for (int i = 0; i < max_n; i++) {
#pragma hls_unroll yes
            for (int j = 0; j < Width; j++) {
              acc_old[i][j] = VectorType::min();
            }
          }

          for (decltype(inst.reduce_count) i = 0;; i++) {
            Pack1D<VectorType, Width> reduce_input = input.Pop();

            Pack1D<VectorType, Width> acc =
                i < max_n ? reduce_input
                          : vmax(acc_old[max_last], reduce_input);

#pragma hls_unroll yes
            for (int k = max_last; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (i == inst.reduce_count - 1) {
              break;
            }
          }

          Pack1D<VectorType, Width> outputs;

#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            Pack1D<VectorType, max_n> col;
#pragma hls_unroll yes
            for (int j = 0; j < max_n; j++) {
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
