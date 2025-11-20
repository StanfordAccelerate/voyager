#pragma once

#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename ScaleType, int width,
          typename... OutputTypes>
SC_MODULE(OutputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      quantize_codebook_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(write_address_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      write_scale_address_params);
  Connections::Combinational<ac_int<32, false>> CCS_INIT_S1(write_scale_params);

  Connections::In<Pack1D<VectorType, width>> CCS_INIT_S1(vector_in);
  Connections::In<ScaleType> CCS_INIT_S1(scale_in);

  Connections::Combinational<Pack1D<VectorType, width>> CCS_INIT_S1(
      quantize_output);

#if SUPPORT_DWC
  Connections::In<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(dwc_address_in);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      dwc_scale_address);
#endif

  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_address_out);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(scale_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(scale_address_out);

  Connections::SyncOut CCS_INIT_S1(done);

  static constexpr int LOOP_WIDTH = 16;

  SC_CTOR(OutputController) {
    SC_THREAD(quantize_codebook);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_data);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_address);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#if SUPPORT_MX
    SC_THREAD(write_mx_scale);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_scale_address);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
  }

  void quantize_codebook() {
    vector_in.Reset();
    quantize_codebook_params.ResetRead();
    quantize_output.ResetWrite();

    wait();

    while (true) {
      VectorParams params = quantize_codebook_params.Pop();

      ac_int<32, false> loop_bound =
          params.output_loops[0][0] * params.output_loops[0][1] *
              params.output_loops[0][2] * params.output_loops[1][0] *
              params.output_loops[1][1] * params.output_loops[1][2] -
          1;

#if SUPPORT_CODEBOOK_QUANT
      if (params.use_output_codebook) {
        VectorType midpoints[NUM_CODEBOOK_ENTRIES];
#pragma hls_unroll yes
        for (int i = 1; i < NUM_CODEBOOK_ENTRIES; i++) {
          midpoints[i] =
              typename VectorType::ac_float_rep(params.output_code[i - 1]);
          midpoints[i].adjust_exponent(-1);
        }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (ac_int<32, false> i = 0;; i++) {
          auto outputs = vector_in.Pop();
          Pack1D<VectorType, width> indices;
#pragma hls_unroll yes
          for (int j = 0; j < width; j++) {
            auto index = quantize16_iter(outputs[j], midpoints);
            indices[j].set_bits(index);
          }
          quantize_output.Push(indices);
          if (i == loop_bound) break;
        }
      } else
#endif
      {
        // Pass through
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (ac_int<32, false> i = 0;; i++) {
          quantize_output.Push(vector_in.Pop());
          if (i == loop_bound) break;
        }
      }
    }
  }

  void write_data() {
    params_in.Reset();
    quantize_codebook_params.ResetWrite();
    write_address_params.ResetWrite();
    write_scale_params.ResetWrite();
    write_scale_address_params.ResetWrite();
    quantize_output.ResetRead();
    vector_out.Reset();
    done.Reset();

    wait();

    while (true) {
      VectorParams params = params_in.Pop();
      quantize_codebook_params.Push(params);
      write_address_params.Push(params);

      ac_int<32, false> loop_bound =
          params.output_loops[0][0] * params.output_loops[0][1] *
              params.output_loops[0][2] * params.output_loops[1][0] *
              params.output_loops[1][1] * params.output_loops[1][2] -
          1;

      if (params.quantize_output_mx) {
        write_scale_params.Push(loop_bound);
        write_scale_address_params.Push(params);
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> i = 0;; i++) {
        Pack1D<VectorType, width> outputs = quantize_output.Pop();

        bool found = (send_output_data<OutputTypes, width, VectorType,
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
        if (i == loop_bound) break;
      }

      done.SyncPush();
    }
  }

  void write_address() {
    write_address_params.ResetRead();
    vector_address_out.Reset();
#if SUPPORT_DWC
    dwc_address_in.Reset();
    dwc_scale_address.ResetWrite();
#endif

    wait();

    while (true) {
      VectorParams params = write_address_params.Pop();

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
          params.output_loops[1][params.output_k_loop_idx[1]] * width;

      ac_int<16, false> X = X0 * X1;
      ac_int<16, false> K = K2 * K1;

      ac_int<16, false> loop_bound_4 = params.output_loops[1][2];
      ac_int<16, false> loop_bound_3 = params.output_loops[1][1] * loop_bound_4;
      ac_int<16, false> loop_bound_2 = params.output_loops[1][0] * loop_bound_3;
      ac_int<16, false> loop_bound_1 = params.output_loops[0][2] * loop_bound_2;
      ac_int<16, false> loop_bound_0 = params.output_loops[0][1] * loop_bound_1;

#if SUPPORT_DWC
      // TODO: remove this and handle properly using vector params
      if (params.is_dwc) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (true) {
          ac_int<ADDRESS_WIDTH, false> address = dwc_address_in.Pop();
          bool found = (send_output_address<OutputTypes, width, OC_PORT_WIDTH,
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
            dwc_scale_address.Push(params.mx_scale_offset +
                                   address / width * ScaleType::width / 8);
          }
#endif
        }
      } else
#endif
      {
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
                      ac_int<LOOP_WIDTH, false> y1 =
                          loop_counters[0][params.output_y_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> x1 =
                          loop_counters[0][params.output_x_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> k2 =
                          loop_counters[0][params.output_k_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> y0 =
                          loop_counters[1][params.output_y_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> x0 =
                          loop_counters[1][params.output_x_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> k1 =
                          loop_counters[1][params.output_k_loop_idx[1]];

                      ac_int<32, false> y = y1 * Y0 + y0;
                      ac_int<32, false> x = x1 * X0 + x0;
                      ac_int<32, false> k = k2 * K1 + k1 * width;

                      if (params.transpose_for_scores) {
                        // k / head_size * (X * head_size) + x * head_size
                        // + k % head_size
                        ac_int<16, false> mask =
                            (1 << params.head_size_lg2) - 1;
                        address = (((k >> params.head_size_lg2) * X)
                                   << params.head_size_lg2) +
                                  (x << params.head_size_lg2) + (k & mask);
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
                                width;
                    }

                    bool found =
                        (send_output_address<OutputTypes, width, OC_PORT_WIDTH,
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
                    if (loop_counters[1][2] == loop_bounds[1][2]) break;
                  }
                  if (loop_counters[1][1] == loop_bounds[1][1]) break;
                }
                if (loop_counters[1][0] == loop_bounds[1][0]) break;
              }
              if (loop_counters[0][2] == loop_bounds[0][2]) break;
            }
            if (loop_counters[0][1] == loop_bounds[0][1]) break;
          }
          if (loop_counters[0][0] == loop_bounds[0][0]) break;
        }
      }
    }
  }

  void write_mx_scale() {
    write_scale_params.ResetRead();
    scale_in.Reset();
    scale_out.Reset();

    wait();

    while (true) {
      ac_int<32, false> loop_bound = write_scale_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> i = 0;; i++) {
        ScaleType scale = scale_in.Pop();
        scale_out.Push(scale.bits_rep());
        if (i == loop_bound) break;
      }
    }
  }

  void write_scale_address() {
    write_scale_address_params.ResetRead();
    scale_address_out.Reset();
#if SUPPORT_DWC
    dwc_scale_address.ResetRead();
#endif

    wait();

    while (true) {
      VectorParams params = write_scale_address_params.Pop();

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
          params.output_loops[1][params.output_k_loop_idx[1]];

      ac_int<16, false> X = X0 * X1;
      ac_int<16, false> K = K2 * K1;

      ac_int<16, false> loop_bound_4 = params.output_loops[1][2];
      ac_int<16, false> loop_bound_3 = params.output_loops[1][1] * loop_bound_4;
      ac_int<16, false> loop_bound_2 = params.output_loops[1][0] * loop_bound_3;
      ac_int<16, false> loop_bound_1 = params.output_loops[0][2] * loop_bound_2;
      ac_int<16, false> loop_bound_0 = params.output_loops[0][1] * loop_bound_1;

#if SUPPORT_DWC
      // TODO: remove this and handle properly using vector params
      if (params.is_dwc) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (true) {
          ac_int<ADDRESS_WIDTH, false> address = dwc_scale_address.Pop();
          scale_address_out.Push(address);
        }
      } else
#endif
      {
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
                      ac_int<LOOP_WIDTH, false> y1 =
                          loop_counters[0][params.output_y_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> x1 =
                          loop_counters[0][params.output_x_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> k2 =
                          loop_counters[0][params.output_k_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> y0 =
                          loop_counters[1][params.output_y_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> x0 =
                          loop_counters[1][params.output_x_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> k1 =
                          loop_counters[1][params.output_k_loop_idx[1]];

                      ac_int<32, false> y = y1 * Y0 + y0;
                      ac_int<32, false> x = x1 * X0 + x0;
                      ac_int<32, false> k = k2 * K1 + k1;
                      address = y * X * K + x * K + k;
                    } else if (params.output_mode == 2) {
                      address = loop_counters[0][0] * loop_bound_0 +
                                loop_counters[0][1] * loop_bound_1 +
                                loop_counters[0][2] * loop_bound_2 +
                                loop_counters[1][0] * loop_bound_3 +
                                loop_counters[1][1] * loop_bound_4 +
                                loop_counters[1][2];
                    }

                    scale_address_out.Push(params.mx_scale_offset +
                                           address * ScaleType::width / 8);

                    if (loop_counters[1][2] == loop_bounds[1][2]) break;
                  }
                  if (loop_counters[1][1] == loop_bounds[1][1]) break;
                }
                if (loop_counters[1][0] == loop_bounds[1][0]) break;
              }
              if (loop_counters[0][2] == loop_bounds[0][2]) break;
            }
            if (loop_counters[0][1] == loop_bounds[0][1]) break;
          }
          if (loop_counters[0][0] == loop_bounds[0][0]) break;
        }
      }
    }
  }
};
