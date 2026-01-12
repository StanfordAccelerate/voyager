#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../AccelTypes.h"

template <typename Input, typename Weight, typename Psum, typename Scale,
          typename Output, int width, int bs>
SC_MODULE(VectorMacUnit) {
 private:
  static constexpr int num_blocks = width / bs;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<Input, width>> CCS_INIT_S1(inputs_in);
  Connections::In<Pack1D<Weight, width>> CCS_INIT_S1(weights_in);
#if SUPPORT_MX
  Connections::In<Pack1D<Scale, num_blocks>> CCS_INIT_S1(input_scales_in);
  Connections::In<Pack1D<Scale, num_blocks>> CCS_INIT_S1(weight_scales_in);
#endif
  Connections::Out<Output> CCS_INIT_S1(psum_out);

  SC_CTOR(VectorMacUnit) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    inputs_in.Reset();
    weights_in.Reset();
    psum_out.Reset();
#if SUPPORT_MX
    input_scales_in.Reset();
    weight_scales_in.Reset();
#endif

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<Input, width> inputs = inputs_in.Pop();
      Pack1D<Weight, width> weights = weights_in.Pop();
#if SUPPORT_MX
      Pack1D<Scale, num_blocks> input_scales = input_scales_in.Pop();
      Pack1D<Scale, num_blocks> weight_scales = weight_scales_in.Pop();
#endif

      Pack1D<Output, num_blocks> psums;
#pragma hls_unroll yes
      for (int c1 = 0; c1 < num_blocks; c1++) {
        Pack1D<Psum, bs> products;
#pragma hls_unroll yes
        for (int i = 0; i < bs; i++) {
          products[i] = (Psum)inputs[c1 * bs + i] * (Psum)weights[c1 * bs + i];
        }
        Output psum = tree_sum(products);
#if SUPPORT_MX
        psum *= (Output)input_scales[c1] * (Output)weight_scales[c1];
#endif
        psums[c1] = psum;
      }

      Output psum = tree_sum(psums);
      psum_out.Push(psum);
    }
  }
};
