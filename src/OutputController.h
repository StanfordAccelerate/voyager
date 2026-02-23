#pragma once

#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

template <typename T, size_t N, typename BufferType, unsigned port_width,
          unsigned int addr_width, typename... Ts>
bool send_output_data_and_addr(
    ac_int<4, false> dtype, Pack1D<BufferType, N> inputs,
    ac_int<ADDRESS_WIDTH, false> offset, ac_int<ADDRESS_WIDTH, false> address,
    Connections::Out<ac_int<port_width, false>>& output_channel,
    Connections::Out<ac_int<addr_width, false>>& address_channel) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int num_words = (T::width * N + port_width - 1) / port_width;

  Pack1D<T, N> outputs;
#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs[i] = inputs[i];
  }

  ac_int<T::width * N, false> bits;
  bits = BitsToType<decltype(bits)>(TypeToBits(outputs));

  for (int i = 0; i < num_words; i++) {
    output_channel.Push(bits.template slc<port_width>(i * port_width));
    address_channel.Push(offset + address * T::width / 8 + i * port_width / 8);
  }

  return true;
}

template <typename BufferType, int width, int port_width,
          typename... OutputTypes>
SC_MODULE(MatrixUnitOutputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      read_accum_buffer_params);

  Connections::In<Pack1D<BufferType, width>> CCS_INIT_S1(
      matrix_processor_output);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::In<Pack1D<BufferType, width>> accumulation_buffer_read_data[2];
  Connections::SyncOut accumulation_buffer_done[2];
#endif

  Connections::Out<Pack1D<BufferType, width>> CCS_INIT_S1(
      vector_unit_input_data);
  Connections::Out<ac_int<port_width, false>> CCS_INIT_S1(
      matrix_unit_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      matrix_unit_output_addr);

  Connections::SyncOut CCS_INIT_S1(done);

  static constexpr int LOOP_WIDTH = 16;

  SC_CTOR(MatrixUnitOutputController) {
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    SC_THREAD(read_accumulation_buffer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
    SC_THREAD(write_matrix_output);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  void read_accumulation_buffer() {
    read_accum_buffer_params.ResetRead();
    accumulation_buffer_read_address[0].Reset();
    accumulation_buffer_read_address[1].Reset();
    accumulation_buffer_done[0].Reset();
    accumulation_buffer_done[1].Reset();

    bool accumulation_buffer_bank = 0;

    while (true) {
      MatrixParams params = read_accum_buffer_params.Pop();

      auto Y0 = params.loops[1][params.y_loop_idx[1]];
      auto X0 = params.loops[1][params.x_loop_idx[1]];
      auto K1 = params.loops[1][params.weight_loop_idx[1]];

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      // set irrelevant loop bounds to 0
      loop_bounds[1][params.reduction_loop_idx[1]] = 0;
      loop_bounds[1][params.fy_loop_idx[1]] = 0;
      loop_bounds[1][params.fx_loop_idx] = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
              for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                  for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                    for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                      for (loop_counters[1][5] = 0;; loop_counters[1][5]++) {
                        ac_int<LOOP_WIDTH, false> y1 =
                            loop_counters[0][params.y_loop_idx[0]];
                        ac_int<LOOP_WIDTH, false> x1 =
                            loop_counters[0][params.x_loop_idx[0]];
                        ac_int<LOOP_WIDTH, false> k1 =
                            loop_counters[0][params.weight_loop_idx[0]];
                        ac_int<LOOP_WIDTH, false> y0 =
                            loop_counters[1][params.y_loop_idx[1]];
                        ac_int<LOOP_WIDTH, false> x0 =
                            loop_counters[1][params.x_loop_idx[1]];
                        ac_int<LOOP_WIDTH, false> k0 =
                            loop_counters[1][params.weight_loop_idx[1]];

                        ac_int<32, false> address = k0 * Y0 * X0 + y0 * X0 + x0;

                        accumulation_buffer_read_address
                            [accumulation_buffer_bank]
                                .Push(address);

                        if (k0 == K1 - 1 && y0 == Y0 - 1 && x0 == X0 - 1) {
                          accumulation_buffer_done[accumulation_buffer_bank]
                              .SyncPush();
                          accumulation_buffer_bank = !accumulation_buffer_bank;
                        }

                        if (loop_counters[1][5] == loop_bounds[1][5]) break;
                      }
                      if (loop_counters[1][4] == loop_bounds[1][4]) break;
                    }
                    if (loop_counters[1][3] == loop_bounds[1][3]) break;
                  }
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
#endif
  void write_matrix_output() {
    params_in.Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    read_accum_buffer_params.ResetWrite();
    accumulation_buffer_read_data[0].Reset();
    accumulation_buffer_read_data[1].Reset();
#endif
    matrix_processor_output.Reset();
    vector_unit_input_data.Reset();
    matrix_unit_output_data.Reset();
    matrix_unit_output_addr.Reset();
    done.Reset();

    wait();

    bool accumulation_buffer_bank = 0;

    while (true) {
      MatrixParams params = params_in.Pop();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      if (params.write_output_to_accum_buffer) {
        read_accum_buffer_params.Push(params);
      }
#endif
      auto Y1 = params.loops[0][params.y_loop_idx[0]];
      auto X1 = params.loops[0][params.x_loop_idx[0]];
      auto K2 = params.loops[0][params.weight_loop_idx[0]];
      auto Y0 = params.loops[1][params.y_loop_idx[1]];
      auto X0 = params.loops[1][params.x_loop_idx[1]];
      auto K1 = params.loops[1][params.weight_loop_idx[1]] * width;

      ac_int<16, false> X = X0 * X1;
      ac_int<16, false> K = K2 * K1;

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      // set irrelevant loop bounds to 0
      loop_bounds[1][params.reduction_loop_idx[1]] = 0;
      loop_bounds[1][params.fy_loop_idx[1]] = 0;
      loop_bounds[1][params.fx_loop_idx] = 0;

      ac_int<LOOP_WIDTH, false> y0_max = loop_bounds[1][params.y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> x0_max = loop_bounds[1][params.x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> k1_max =
          loop_bounds[1][params.weight_loop_idx[1]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
              for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                  for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                    for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                      for (loop_counters[1][5] = 0;; loop_counters[1][5]++) {
                        ac_int<LOOP_WIDTH, false> y1 =
                            loop_counters[0][params.y_loop_idx[0]];
                        ac_int<LOOP_WIDTH, false> x1 =
                            loop_counters[0][params.x_loop_idx[0]];
                        ac_int<LOOP_WIDTH, false> k2 =
                            loop_counters[0][params.weight_loop_idx[0]];
                        ac_int<LOOP_WIDTH, false> y0 =
                            loop_counters[1][params.y_loop_idx[1]];
                        ac_int<LOOP_WIDTH, false> x0 =
                            loop_counters[1][params.x_loop_idx[1]];
                        ac_int<LOOP_WIDTH, false> k1 =
                            loop_counters[1][params.weight_loop_idx[1]];

                        Pack1D<BufferType, width> outputs;
#if DOUBLE_BUFFERED_ACCUM_BUFFER
                        if (params.write_output_to_accum_buffer) {
                          outputs = accumulation_buffer_read_data
                                        [accumulation_buffer_bank]
                                            .Pop();
                          ac_int<LOOP_WIDTH, false> x0 =
                              loop_counters[1][params.x_loop_idx[1]];
                          ac_int<LOOP_WIDTH, false> y0 =
                              loop_counters[1][params.y_loop_idx[1]];
                          ac_int<LOOP_WIDTH, false> k1 =
                              loop_counters[1][params.weight_loop_idx[1]];

                          if (k1 == k1_max && y0 == y0_max && x0 == x0_max) {
                            accumulation_buffer_bank =
                                !accumulation_buffer_bank;
                          }
                        } else
#endif
                        {
                          outputs = matrix_processor_output.Pop();
                        }

                        if (params.output_to_memory) {
                          ac_int<16, false> y = y1 * Y0 + y0;
                          ac_int<16, false> x = x1 * X0 + x0;
                          ac_int<16, false> k = k2 * K1 + k1 * width;
                          ac_int<32, false> address = y * X * K + x * K + k;

                          bool found = (send_output_data_and_addr<
                                            OutputTypes, width, BufferType,
                                            OC_PORT_WIDTH, ADDRESS_WIDTH,
                                            OutputTypes...>(
                                            params.output_dtype, outputs,
                                            params.output_offset, address,
                                            matrix_unit_output_data,
                                            matrix_unit_output_addr) ||
                                        ...);
#ifndef __SYNTHESIS__
                          if (!found) {
                            std::cerr << "Error: output type '"
                                      << params.output_dtype
                                      << "' is not valid.\n";
                          }
#endif
                        } else {
                          vector_unit_input_data.Push(outputs);
                        }

                        if (loop_counters[1][5] == loop_bounds[1][5]) break;
                      }
                      if (loop_counters[1][4] == loop_bounds[1][4]) break;
                    }
                    if (loop_counters[1][3] == loop_bounds[1][3]) break;
                  }
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

      done.SyncPush();
    }
  }
};
