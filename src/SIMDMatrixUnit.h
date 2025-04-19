#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename InputTypeTuple, typename WeightTypeTuple, typename Input,
          typename Weight, typename Psum, typename AccumType, typename Scale,
          int Width>
struct SIMDMatrixUnit;

template <typename... InputTypes, typename... WeightTypes, typename Input,
          typename Weight, typename Psum, typename AccumType, typename Scale,
          int Width>
struct SIMDMatrixUnit<std::tuple<InputTypes...>, std::tuple<WeightTypes...>,
                      Input, Weight, Psum, AccumType, Scale, Width>
    : public sc_module {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<MatrixParam> CCS_INIT_S1(params_in);

  Connections::Combinational<Pack1D<Weight, Width>> CCS_INIT_S1(inputs_in);
  Connections::Combinational<Pack1D<Weight, Width>> CCS_INIT_S1(weights_in);

  SC_CTOR(SIMDMatrixUnit) {
    SC_THREAD(fetch_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetch_inputs() {}

  void process_inputs() {}

  void fetch_weights() {}

  void process_weights() {}

  void run_accumulation() {
    params_in.Reset();
    weights_in.ReserRead();

    wait();

    while (true) {
      MatrixParams inst = params_in.Pop();

      int K1, C1, C0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k1 = 0; k1 < K1; k1++) {
        Pack1D<AccumType, Width> outputs;

#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          outputs[i] = AccumType::zero();
        }

        for (int c1 = 0; c1 < C1; c1++) {
          Pack1D<Psum, Width> psums;

#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            psums[i] = Psum::zero();
          }

          Pack1D<Input, Width> inputs = inputs_in.Pop();

          for (int c0 = 0; c0 < C0; c0++) {
            const auto weights = weights_in.Pop();
#pragma hls_unroll yes
            for (int k0 = 0; k0 < Width; k0++) {
              psums[k0] += inputs[c0] * weights[k0];
            }
          }

          // Rescale the psums
          const auto scales = weight_scales.Pop();

#pragma hls_unroll yes
          for (int k0 = 0; k0 < Width; k0++) {
            outputs[k0] += static_cast<AccumType>(psums[k0]) *
                           static_cast<AccumType>(scales[k0]);
          }
        }
      }
    }
  }
};
