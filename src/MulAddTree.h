#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename Input, typename Weight, typename Psum, typename Output>
SC_MODULE(MulAddTree) {
 private:
#if SUPPORT_MX
  Output mul_results[DWC_KERNEL_SIZE];
  Output mul_scale[DWC_KERNEL_SIZE];
  Psum mul_unscaled_results[DWC_KERNEL_SIZE];
  Output out_partial[7];
#else
  Psum mul_results[DWC_KERNEL_SIZE];
  Psum out_partial[7];
#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<Weight, DWC_KERNEL_SIZE>> CCS_INIT_S1(weight_in);
  Connections::In<Pack1D<Input, DWC_KERNEL_SIZE>> CCS_INIT_S1(input_in);
#if SUPPORT_MX
  Connections::In<Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE>> CCS_INIT_S1(
      weight_scale_in);
  Connections::In<Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE>> CCS_INIT_S1(
      input_scale_in);
#endif
  Connections::In<ac_int<1, false>> CCS_INIT_S1(update_weight);
  Connections::In<Pack1D<ac_int<1, false>, DWC_KERNEL_SIZE>> CCS_INIT_S1(
      weight_mask);
  Connections::In<Output> CCS_INIT_S1(bias_in);

  Connections::Out<Output> CCS_INIT_S1(adder_out);

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(MulAddTree);
  MulAddTree()
      : sc_module(sc_gen_unique_name("MulAddTree"))
#else
  SC_CTOR(MulAddTree)
#endif
  {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    weight_in.Reset();
    input_in.Reset();
    weight_mask.Reset();
    bias_in.Reset();
    adder_out.Reset();
    update_weight.Reset();
#if SUPPORT_MX
    weight_scale_in.Reset();
    input_scale_in.Reset();
#endif

    wait();

    Pack1D<Weight, DWC_KERNEL_SIZE> weight_reg;
    Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE> weight_scale_reg;
    Output bias_reg;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<Input, DWC_KERNEL_SIZE> input = input_in.Pop();
      ac_int<1, false> update_weight_sig = update_weight.Pop();
      Pack1D<ac_int<1, false>, DWC_KERNEL_SIZE> mask = weight_mask.Pop();
      Pack1D<Weight, DWC_KERNEL_SIZE> weight = weight_in.Pop();
#if SUPPORT_MX
      Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE> weight_scale =
          weight_scale_in.Pop();
      Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE> input_scale =
          input_scale_in.Pop();
      if (update_weight_sig) {
        weight_scale_reg = weight_scale;
      }
#endif
      Output bias = bias_in.Pop();

      if (update_weight_sig) {
        weight_reg = weight;
        bias_reg = bias;
      }

#pragma hls_unroll yes
      for (int i = 0; i < 9; i++) {
#if SUPPORT_MX
        if (mask[i]) {
          mul_unscaled_results[i] =
              static_cast<Psum>(input[i]) * static_cast<Psum>(weight_reg[i]);
          mul_scale[i] = static_cast<Output>(input_scale[i]) *
                         static_cast<Output>(weight_scale_reg[i]);
          mul_results[i] =
              static_cast<Output>(mul_unscaled_results[i]) * mul_scale[i];
        } else {
          mul_results[i] = Output::zero();
        }
#else
        if (mask[i]) {
          mul_results[i] = static_cast<Psum>(input[i]) * weight_reg[i];
        } else {
          mul_results[i] = Psum::zero();
        }
#endif
      }
#pragma hls_unroll yes
      for (int i = 0; i < 4; i++) {
        out_partial[i] = mul_results[i * 2] + mul_results[i * 2 + 1];
      }
      out_partial[4] = out_partial[0] + out_partial[1];
      out_partial[5] = out_partial[2] + out_partial[3];
      out_partial[6] = out_partial[4] + out_partial[5] + mul_results[8];

      Output output = bias_reg + static_cast<Output>(out_partial[6]);
      adder_out.Push(output);
    }
  }
};
