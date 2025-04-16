#pragma once

#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename ScaleType, int Width,
          typename... OutputTypes>
SC_MODULE(OutputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_in);
  Connections::In<ScaleType> CCS_INIT_S1(scale_in);

  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_address_out);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(scale_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(scale_address_out);

  Connections::SyncOut CCS_INIT_S1(done);

  SC_CTOR(OutputController) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    params_in.Reset();
    vector_in.Reset();
    scale_in.Reset();
    vector_out.Reset();
    vector_address_out.Reset();
    scale_out.Reset();
    scale_address_out.Reset();
    done.Reset();

    wait();

    while (true) {
      VectorParams params = params_in.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.output_loops[i][j];
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            for (loop_counters[1][0] = 0;
                 loop_counters[1][0] < loop_bounds[1][0];
                 loop_counters[1][0]++) {
              for (loop_counters[1][1] = 0;
                   loop_counters[1][1] < loop_bounds[1][1];
                   loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;
                     loop_counters[1][2] < loop_bounds[1][2];
                     loop_counters[1][2]++) {
                  ac_int<32, false> address;
                  if (params.output_mode == 1) {
                    ac_int<11, false> x0 =
                        loop_counters[1][params.output_x_loop_idx[1]];
                    ac_int<11, false> x1 =
                        loop_counters[0][params.output_x_loop_idx[0]];
                    ac_int<11, false> y0 =
                        loop_counters[1][params.output_y_loop_idx[1]];
                    ac_int<11, false> y1 =
                        loop_counters[0][params.output_y_loop_idx[0]];
                    ac_int<11, false> k1 =
                        loop_counters[1][params.output_k_loop_idx[1]];
                    ac_int<11, false> k2 =
                        loop_counters[0][params.output_k_loop_idx[0]];

                    ac_int<11, false> X0 =
                        loop_bounds[1][params.output_x_loop_idx[1]];
                    ac_int<11, false> X1 =
                        loop_bounds[0][params.output_x_loop_idx[0]];
                    ac_int<11, false> Y0 =
                        loop_bounds[1][params.output_y_loop_idx[1]];
                    ac_int<11, false> Y1 =
                        loop_bounds[0][params.output_y_loop_idx[0]];
                    ac_int<11, false> K2 =
                        loop_bounds[0][params.output_k_loop_idx[0]];
                    ac_int<11, false> K1 =
                        loop_bounds[1][params.output_k_loop_idx[1]];

                    ac_int<16, false> k = k2 * K1 * Width + k1 * Width;
                    ac_int<16, false> K = K2 * K1 * Width;

                    ac_int<16, false> x = x0 + x1 * X0;
                    ac_int<16, false> X = X0 * X1;

                    ac_int<16, false> y = y0 + y1 * Y0;
                    ac_int<16, false> Y = Y0 * Y1;

                    ac_int<8, false> head_size = params.head_size_power_of_two;
                    ac_int<16, false> mask = (1 << head_size) - 1;

                    if (params.has_attn_head_permute) {
                      address = (((k >> head_size) * X) << head_size) +
                                (x << head_size) + (k & mask);
                    } else if (params.has_output_permute) {
                      address = ((k >> head_size) * K) + ((y & mask) * K * 4) +
                                (k & mask) + (y >> head_size * K / 4);
                    } else {
                      address = y * X * K + x * K + k;
                    }
                  } else if (params.output_mode == 2) {
                    ac_int<11, false> loop_0 = loop_counters[0][0];
                    ac_int<11, false> loop_1 = loop_counters[0][1];
                    ac_int<11, false> loop_2 = loop_counters[0][2];
                    ac_int<11, false> loop_3 = loop_counters[1][0];
                    ac_int<11, false> loop_4 = loop_counters[1][1];
                    ac_int<11, false> loop_5 = loop_counters[1][2];

                    ac_int<11, false> loop_bound_0 = loop_bounds[0][0];
                    ac_int<11, false> loop_bound_1 = loop_bounds[0][1];
                    ac_int<11, false> loop_bound_2 = loop_bounds[0][2];
                    ac_int<11, false> loop_bound_3 = loop_bounds[1][0];
                    ac_int<11, false> loop_bound_4 = loop_bounds[1][1];
                    ac_int<11, false> loop_bound_5 = loop_bounds[1][2];

                    address =
                        (loop_0 * loop_bound_1 * loop_bound_2 * loop_bound_3 *
                             loop_bound_4 * loop_bound_5 +
                         loop_1 * loop_bound_2 * loop_bound_3 * loop_bound_4 *
                             loop_bound_5 +
                         loop_2 * loop_bound_3 * loop_bound_4 * loop_bound_5 +
                         loop_3 * loop_bound_4 * loop_bound_5 +
                         loop_4 * loop_bound_5 + loop_5) *
                        Width;
                  }

#if SUPPORT_MX
                  if (params.quantize_output_mx) {
                    ScaleType scale = scale_in.Pop();
                    scale_out.Push(scale.bits_rep());
                    scale_address_out.Push(params.SCALE_OFFSET +
                                           address / Width * ScaleType::width /
                                               8);
                  }
#endif

                  Pack1D<VectorType, Width> outputs = vector_in.Pop();

                  bool found =
                      ((get_type_index<OutputTypes, OutputTypes...>() ==
                                params.output_dtype
                            ? (vwrite_out<VectorType, OutputTypes, Width>(
                                   outputs, address,
                                   params.VECTOR_OUTPUT_OFFSET, vector_out,
                                   vector_address_out),
                               true)
                            : false) ||
                       ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: Index '" << params.output_dtype
                              << "' is not valid.\n";
                  }
#endif

                  if (loop_counters[1][2] >= loop_bounds[1][2] - 1) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_bounds[1][1] - 1) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_bounds[1][0] - 1) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_bounds[0][2] - 1) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_bounds[0][1] - 1) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_bounds[0][0] - 1) {
          break;
        }
      }

      done.SyncPush();
    }
  }
};
