#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "VectorOps.h"

template <typename VectorType, size_t Width>
SC_MODULE(VectorAccumulator) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<VectorType, Width>> input;

  // Outputs
  Connections::Out<Pack1D<VectorType, Width>> output;

  static constexpr int N = 2;
  static constexpr int last = N - 1;

  static_assert(N > 0, "Pipeline size N must be greater than 0");

  SC_CTOR(VectorAccumulator) {
    SC_THREAD(run_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run_accumulation() {
    instr.Reset();
    input.Reset();
    output.Reset();

    wait();

    while (true) {
      VectorInstructions inst = instr.Pop();

      Pack1D<VectorType, Width> acc_old[N];

#pragma hls_unroll yes
      for (int i = 0; i < N; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < Width; j++) {
          if (inst.reduce_op == VectorInstructions::radd) {
            acc_old[i][j] = VectorType::zero();
          } else if (inst.reduce_op == VectorInstructions::rmax) {
            acc_old[i][j] = VectorType::min();
          }
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(inst.reduce_count) i = 0;; i++) {
        Pack1D<VectorType, Width> reduce_input = input.Pop();

        Pack1D<VectorType, Width> acc;
        if (inst.reduce_op == VectorInstructions::radd) {
          acc = i < N ? reduce_input : vadd(acc_old[last], reduce_input);
        } else if (inst.reduce_op == VectorInstructions::rmax) {
          acc = i < N ? reduce_input : vmax(acc_old[last], reduce_input);
        }

        for (int k = last; k > 0; k--) {
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
        Pack1D<VectorType, N> col;
#pragma hls_unroll yes
        for (int j = 0; j < N; j++) {
          col[j] = acc_old[j][i];
        }

        if (inst.reduce_op == VectorInstructions::radd) {
          outputs[i] = tree_sum(col);
        } else if (inst.reduce_op == VectorInstructions::rmax) {
          outputs[i] = tree_max(col);
        }
      }

      DLOG("accumulation finished: " << outputs);
      output.Push(outputs);
    }
  }
};
