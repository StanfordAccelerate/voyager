#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename VectorType, typename ScaleType, int Width>
SC_MODULE(VectorQuantizer) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<VectorType, Width>> input;
  Connections::In<Pack1D<VectorType, Width>> scale;

  // Outputs
  Connections::Out<Pack1D<VectorType, Width>> output;
  Connections::Out<ScaleType> mx_scale;

  SC_CTOR(VectorQuantizer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    instr.Reset();
    input.Reset();
    scale.Reset();
    output.Reset();
    mx_scale.Reset();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = instr.Pop();
      Pack1D<VectorType, Width> res2 = input.Pop();
      Pack1D<VectorType, Width> res3;
      Pack1D<VectorType, Width> op3_src1;

      if (inst.vector_op3_src1 == VectorInstructions::from_vector_fetch_2) {
        op3_src1 = scale.Pop();
      }

      if (inst.vector_op3_src1 == VectorInstructions::from_immediate_2) {
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          op3_src1[i].set_bits(inst.immediate2);
        }
      }

#if SUPPORT_MX
      if (inst.vector_op3 == VectorInstructions::vquantize_mx) {
        ScaleType scale = calculate_mx_scale<VectorType, ScaleType, Width>(
            res2, inst.immediate2);

#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          op3_src1[i] = scale;
        }

        mx_scale.Push(scale);
      }
#endif

      // Stage 3: div, quantize
      if (inst.vector_op3 == VectorInstructions::vdiv ||
          inst.vector_op3 == VectorInstructions::vquantize_mx) {
        res3 = vdiv<VectorType, Width>(res2, op3_src1);
      } else {
        res3 = res2;
      }

      output.Push(res3);
    }
  }
};
