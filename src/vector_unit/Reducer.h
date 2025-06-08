#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename VectorType, int Width>
SC_MODULE(VectorReducer) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<VectorType, Width>> input;

  // Outputs
  Connections::Out<ac_int<16, false>> count_0;
  Connections::Out<ac_int<16, false>> count_1;
  Connections::Out<Pack1D<VectorType, Width>> output_0;
  Connections::Out<Pack1D<VectorType, Width>> output_1;
  Connections::Out<Pack1D<VectorType, Width>> output_to_memory;

  static constexpr int N = 2;
  static constexpr int last = N - 1;

  static_assert(N > 0, "Pipeline size N must be greater than 0");

  SC_CTOR(VectorReducer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    instr.Reset();
    input.Reset();
    count_0.Reset();
    count_1.Reset();
    output_0.Reset();
    output_1.Reset();
    output_to_memory.Reset();

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
          VectorType acc;
          Pack1D<VectorType, N> acc_old;

#pragma hls_unroll yes
          for (int j = 0; j < N; j++) {
            if (inst.reduce_op == VectorInstructions::radd) {
              acc_old[j] = VectorType::zero();
            } else if (inst.reduce_op == VectorInstructions::rmax) {
              acc_old[j] = VectorType::min();
            }
          }

          for (decltype(inst.reduce_count) j = 0;; j++) {
            Pack1D<VectorType, Width> reduce_input = input.Pop();

            if (inst.reduce_op == VectorInstructions::radd) {
              VectorType sum = tree_sum(reduce_input);
              acc = j < N ? sum : acc_old[last] + sum;
            } else if (inst.reduce_op == VectorInstructions::rmax) {
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
          } else if (inst.reduce_op == VectorInstructions::rmax) {
            output = tree_max(acc_old);
          }

          if (inst.rsqrt) {
            output = output.sqrt();
          }

          if (inst.rreciprocal) {
            output = output.reciprocal();
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
          count_0.Push(inst.immediate0);
          output_0.Push(res);
        } else if (inst.rdest == VectorInstructions::to_op2) {
          count_1.Push(inst.immediate0);
          output_1.Push(res);
        } else if (inst.rdest == VectorInstructions::to_memory) {
          output_to_memory.Push(res);
        }
      }
    }
  }
};
