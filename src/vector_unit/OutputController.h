#pragma once

#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename ScaleType, int Width,
          typename... OutputTypes>
SC_MODULE(OutputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::Combinational<VectorParams> CCS_INIT_S1(send_data_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(quantizer_params);

  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_in);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      quantized_data);

  Connections::Fifo<ScaleType, 16> CCS_INIT_S1(scale_fifo);
  Connections::In<ScaleType> CCS_INIT_S1(scale_in);
  Connections::Combinational<ScaleType> CCS_INIT_S1(scale_deq);

#if SUPPORT_DWC
  Connections::In<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(dwc_address_in);
#endif

  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_address_out);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(scale_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(scale_address_out);

  Connections::SyncOut CCS_INIT_S1(done);

  static constexpr int LOOP_WIDTH = 16;

  SC_CTOR(OutputController) {
    scale_fifo.clk(clk);
    scale_fifo.rst(rstn);
    scale_fifo.enq(scale_in);
    scale_fifo.deq(scale_deq);

    SC_THREAD(send_address);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_data);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(quantizer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void quantizer() {
    vector_in.Reset();
    quantizer_params.ResetRead();
    quantized_data.ResetWrite();

    wait();

    while (true) {
      VectorParams params = quantizer_params.Pop();

      ac_int<32, false> num_outputs =
          params.output_loops[0][0] * params.output_loops[0][1] *
          params.output_loops[0][2] * params.output_loops[1][0] *
          params.output_loops[1][1] * params.output_loops[1][2];

      ac_int<32, false> counter = 0;

#if SUPPORT_CODEBOOK_QUANT
      if (params.use_output_codebook) {
        using ac_float_t = typename StdFloat<7, 5>::ac_float_rep;

        ac_float_t midpoints[NUM_CODEBOOK_ENTRIES - 1];

#pragma hls_unroll yes
        for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
          StdFloat<7, 5> value = ac_float_t(params.output_code[i]);
          value.adjust_exponent(-1);
          midpoints[i] = value.float_val;
        }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (counter++ < num_outputs) {
          Pack1D<VectorType, Width> outputs = vector_in.Pop();

          Pack1D<ac_float_t, Width> ac_float_outputs;
          Pack1D<ac_int<4, false>, Width> indices;

#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            ac_float_outputs[i] = ac_float_t(outputs[i].float_val);
            indices[i] = 0;
          }

#pragma hls_unroll 4
          for (int j = 0; j < NUM_CODEBOOK_ENTRIES - 1; j++) {
#pragma hls_unroll yes
            for (int i = 0; i < Width; i++) {
              indices[i] += ac_float_outputs[i] > midpoints[j];
            }
          }

          Pack1D<VectorType, Width> quantized_outputs;
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            quantized_outputs[i].set_bits(indices[i]);
          }
          quantized_data.Push(quantized_outputs);
        }
      } else
#endif
      {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (counter++ < num_outputs) {
          // Pass through
          Pack1D<VectorType, Width> outputs = vector_in.Pop();
          quantized_data.Push(outputs);
        }
      }
    }
  }

  void send_data() {
    send_data_params.ResetRead();
    scale_deq.ResetRead();
    scale_out.Reset();
    quantized_data.ResetRead();
    vector_out.Reset();
    done.Reset();

    wait();

    while (true) {
      VectorParams params = send_data_params.Pop();

      ac_int<32, false> num_outputs =
          params.output_loops[0][0] * params.output_loops[0][1] *
          params.output_loops[0][2] * params.output_loops[1][0] *
          params.output_loops[1][1] * params.output_loops[1][2];

      ac_int<32, false> counter = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (counter++ < num_outputs) {
#if SUPPORT_MX
        if (params.quantize_output_mx) {
          ScaleType scale = scale_deq.Pop();
          scale_out.Push(scale.bits_rep());
        }
#endif

        Pack1D<VectorType, Width> outputs = quantized_data.Pop();

        bool found = (send_output_data<OutputTypes, Width, VectorType,
                                       OC_PORT_WIDTH, OutputTypes...>(
                          params.output_dtype, params.use_output_codebook,
                          outputs, vector_out) ||
                      ...);

#ifndef __SYNTHESIS__
        if (!found) {
          std::cerr << "Error: output type '" << params.output_dtype
                    << "' is not valid.\n";
        }
#endif
      }

      done.SyncPush();
    }
  }

  void send_address() {
    params_in.Reset();
    send_data_params.ResetWrite();
    quantizer_params.ResetWrite();
    vector_address_out.Reset();
    scale_address_out.Reset();
#if SUPPORT_DWC
    dwc_address_in.Reset();
#endif

    wait();

    while (true) {
      VectorParams params = params_in.Pop();
      send_data_params.Push(params);
      quantizer_params.Push(params);

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.output_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y1 =
          params.output_loops[0][params.output_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.output_loops[0][params.output_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.output_loops[0][params.output_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.output_loops[1][params.output_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.output_loops[1][params.output_x_loop_idx[1]];
      ac_int<16, false> K1 =
          params.output_loops[1][params.output_k_loop_idx[1]] * Width;

      ac_int<16, false> X = X0 * X1;
      ac_int<16, false> K = K2 * K1;

      ac_int<16, false> loop_bound_4 = params.output_loops[1][2];
      ac_int<16, false> loop_bound_3 = params.output_loops[1][1] * loop_bound_4;
      ac_int<16, false> loop_bound_2 = params.output_loops[1][0] * loop_bound_3;
      ac_int<16, false> loop_bound_1 = params.output_loops[0][2] * loop_bound_2;
      ac_int<16, false> loop_bound_0 = params.output_loops[0][1] * loop_bound_1;

#if SUPPORT_DWC
      if (params.addr_from_dwc) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (true) {
          ac_int<ADDRESS_WIDTH, false> address = dwc_address_in.Pop();
          bool found =
            (send_output_address<OutputTypes, Width, OC_PORT_WIDTH,
                                OutputTypes...>(
                params.output_dtype, params.vector_output_offset,
                address, vector_address_out) ||
            ...);

#ifndef __SYNTHESIS__
      if (!found) {
        std::cerr << "Error: output type '" << params.output_dtype
                  << "' is not valid.\n";
      }
#endif
#if SUPPORT_MX
          if (params.quantize_output_mx) {
            scale_address_out.Push(params.mx_scale_offset +
                                    address / Width * ScaleType::width /
                                        8);
          }
#endif
        }
      }
#endif

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
              for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                  ac_int<32, false> address;
                  if (params.output_mode == 1) {
                    ac_int<LOOP_WIDTH, false> x0 =
                        loop_counters[1][params.output_x_loop_idx[1]];
                    ac_int<LOOP_WIDTH, false> x1 =
                        loop_counters[0][params.output_x_loop_idx[0]];
                    ac_int<LOOP_WIDTH, false> y0 =
                        loop_counters[1][params.output_y_loop_idx[1]];
                    ac_int<LOOP_WIDTH, false> y1 =
                        loop_counters[0][params.output_y_loop_idx[0]];
                    ac_int<LOOP_WIDTH, false> k1 =
                        loop_counters[1][params.output_k_loop_idx[1]];
                    ac_int<LOOP_WIDTH, false> k2 =
                        loop_counters[0][params.output_k_loop_idx[0]];

                    ac_int<32, false> y = y1 * Y0 + y0;
                    ac_int<32, false> x = x1 * X0 + x0;
                    ac_int<32, false> k = k2 * K1 + k1 * Width;

                    ac_int<8, false> head_size = params.head_size_power_of_two;
                    ac_int<16, false> mask = (1 << head_size) - 1;

                    if (params.has_attn_head_permute) {
                      // k / head_size * (X * head_size) + x * head_size
                      // + k % head_size
                      address = (((k >> head_size) * X) << head_size) +
                                (x << head_size) + (k & mask);
                    } else {
                      address = y * X * K + x * K + k;
                    }
                  } else if (params.output_mode == 2) {
                    address = (loop_counters[0][0] * loop_bound_0 +
                               loop_counters[0][1] * loop_bound_1 +
                               loop_counters[0][2] * loop_bound_2 +
                               loop_counters[1][0] * loop_bound_3 +
                               loop_counters[1][1] * loop_bound_4 +
                               loop_counters[1][2]) *
                              Width;
                  }

#if SUPPORT_MX
                  if (params.quantize_output_mx) {
                    scale_address_out.Push(params.mx_scale_offset +
                                           address / Width * ScaleType::width /
                                               8);
                  }
#endif

                  bool found =
                      (send_output_address<OutputTypes, Width, OC_PORT_WIDTH,
                                           OutputTypes...>(
                           params.output_dtype, params.vector_output_offset,
                           address, vector_address_out) ||
                       ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: output type '" << params.output_dtype
                              << "' is not valid.\n";
                  }
#endif

                  if (loop_counters[1][2] == loop_bounds[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] == loop_bounds[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] == loop_bounds[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] == loop_bounds[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) {
          break;
        }
      }
    }
  }
};
