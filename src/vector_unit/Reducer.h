#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename T, int width>
SC_MODULE(VectorReducer) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<T, width>> input;

  Connections::Out<Pack1D<T, width>> output_to_pipeline;
  Connections::Out<Pack1D<T, width>> output_to_memory;

  Connections::Combinational<VectorInstructions> repeat_inst;
  Connections::Combinational<Pack1D<T, width>> repeat_data;

#ifdef CLOCK_PERIOD
  static constexpr double clock_period = CLOCK_PERIOD;
#else
  static constexpr double clock_period = 5.0;  // Default to 5 ns if not defined
#endif

  static constexpr int FEEDBACK_DELAY = (clock_period < 0.8) ? 3 : 2;
  static constexpr int LAST = FEEDBACK_DELAY - 1;
  static constexpr int ratio = VECTOR_UNIT_WIDTH / REDUCER_WIDTH;

  static_assert(FEEDBACK_DELAY > 0,
                "Pipeline size FEEDBACK_DELAY must be greater than 0");

  SC_CTOR(VectorReducer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(repeat_output);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    instr.Reset();
    repeat_inst.ResetWrite();
    input.Reset();
    repeat_data.ResetWrite();
    output_to_pipeline.Reset();

    wait();

    while (true) {
      VectorInstructions inst = instr.Pop();

      if (inst.rdest == VectorInstructions::to_memory) {
        repeat_inst.Push(inst);
      }

      bool is_sum_op = (inst.reduce_op == VectorInstructions::radd);
      T fill_value = is_sum_op ? T::zero() : T::min();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(inst.inst_loop_count) count = 0;; count++) {
        Pack1D<T, width> res;

        for (int i = 0; i < width; i++) {
          auto acc_old = Pack1D<T, FEEDBACK_DELAY>::fill(fill_value);

          for (decltype(inst.reduce_count) j = 0;; j++) {
            Pack1D<T, width> reduce_input = input.Pop();
            T sum = fused_add_tree(reduce_input);
            T max = max_tree(reduce_input);
            T acc = is_sum_op ? (acc_old[LAST] + sum)
                              : std::max(acc_old[LAST], max);

#pragma hls_unroll yes
            for (int k = LAST; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }

            acc_old[0] = acc;

            if (j == inst.reduce_count - 1) break;
          }

          T output = is_sum_op ? fused_add_tree(acc_old) : max_tree(acc_old);

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
          repeat_data.Push(res);
        } else {
          output_to_pipeline.Push(res);
        }

        if (count == inst.inst_loop_count - 1) break;
      }
    }
  }

  void repeat_output() {
    repeat_inst.ResetRead();
    repeat_data.ResetRead();
    output_to_memory.Reset();

    wait();

    while (true) {
      VectorInstructions inst = repeat_inst.Pop();
      ac_int<16, false> repeat = inst.rduplicate ? ratio : 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (decltype(inst.inst_loop_count) count = 0;; count++) {
        Pack1D<T, width> data = repeat_data.Pop();

        for (ac_int<16, false> i = 0;; i++) {
          output_to_memory.Push(data);
          if (i == repeat - 1) break;
        }

        if (count == inst.inst_loop_count - 1) break;
      }
    }
  }
};
