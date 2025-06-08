#pragma once

#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename BufferType, int Width,
          typename... InputTypes>
SC_MODULE(VectorFetchUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  sc_fifo<ac_int<2, false>> vector_fetch_0_done;
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_fetch_0_data);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::In<Pack1D<BufferType, Width>> accumulation_buffer_read_data[2];
  Connections::Out<BufferWriteRequest<Pack1D<BufferType, Width>>>
      accumulation_buffer_write_request[2];
  Connections::Out<Pack1D<BufferType, Width>> CCS_INIT_S1(
      accumulation_buffer_output);
  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::SyncOut accumulation_buffer_done[2];
#endif

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  sc_fifo<bool> vector_fetch_1_done;
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_fetch_1_data);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);
  sc_fifo<bool> vector_fetch_2_done;
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_fetch_2_data);

  Connections::Combinational<VectorParams> CCS_INIT_S1(addr_gen0_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addr_gen1_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addr_gen2_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(data_resp_0_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(data_resp_1_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(data_resp_2_params);

  static constexpr int LOOP_WIDTH = 16;
  static constexpr int BUFSIZE = 1024 / Width < Width ? 1024 / Width : Width;

  SC_CTOR(VectorFetchUnit) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_address_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_response_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_address_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_response_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_address_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_response_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetch_address_0() {
    addr_gen0_params.ResetRead();
    vector_fetch_0_req.Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_read_address[0].Reset();
    accumulation_buffer_read_address[1].Reset();
    accumulation_buffer_done[0].Reset();
    accumulation_buffer_done[1].Reset();
#endif

    bool accumulation_buffer_bank = 0;

    wait();

    while (true) {
      VectorParams params = addr_gen0_params.Pop();

      ac_int<LOOP_WIDTH, false> Y1 =
          params.addr_gen0_loops[0][params.addr_gen0_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.addr_gen0_loops[0][params.addr_gen0_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.addr_gen0_loops[0][params.addr_gen0_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.addr_gen0_loops[1][params.addr_gen0_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.addr_gen0_loops[1][params.addr_gen0_x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.addr_gen0_loops[1][params.addr_gen0_k_loop_idx[1]];

      if (params.has_transpose && BUFSIZE != Width) {
        X1 = X1 * BUFSIZE / Width;
      }

      ac_int<16, false> Y = Y1 * Y0;
      ac_int<16, false> X = X1 * X0;
      ac_int<16, false> K = K2 * K1 * Width;

      ac_int<LOOP_WIDTH, false> stride_y = Y0;
      ac_int<LOOP_WIDTH, false> stride_x = X0;
      ac_int<16, false> stride_k = K1 * Width;

      ac_int<4, false> padding_y = params.padding[0];
      ac_int<4, false> padding_x = params.padding[1];

      // stride = 0 means default behavior, stride = inner tile size
      if (params.stride[0]) {
        stride_y = params.stride[0];
        Y = (Y1 - 1) * stride_y + Y0 - padding_y;
      }

      if (params.stride[1]) {
        stride_x = params.stride[1];
        X = (X1 - 1) * stride_x + X0 - padding_x;
      }

      ac_int<8, false> offset_x = (padding_x + 1) / 2;
      ac_int<8, false> offset_y = (padding_y + 1) / 2;

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_starts[2][3];
      ac_int<LOOP_WIDTH, false> loop_ends[2][3];
      ac_int<LOOP_WIDTH, false> loop_steps[2][3];

      ac_int<LOOP_WIDTH, false> loop_bounds[6] = {
          params.addr_gen0_loops[0][0], params.addr_gen0_loops[0][1],
          params.addr_gen0_loops[0][2], params.addr_gen0_loops[1][0],
          params.addr_gen0_loops[1][1], params.addr_gen0_loops[1][2],
      };

      ac_int<LOOP_WIDTH, false> loop_starts_1d[6] = {
          0, 0, 0, 0, 0, 0,
      };
      ac_int<LOOP_WIDTH, false> loop_ends_1d[6] = {
          params.addr_gen0_loops[0][0], params.addr_gen0_loops[0][1],
          params.addr_gen0_loops[0][2], params.addr_gen0_loops[1][0],
          params.addr_gen0_loops[1][1], params.addr_gen0_loops[1][2],
      };
      ac_int<LOOP_WIDTH, false> loop_steps_1d[6] = {
          1, 1, 1, 1, 1, 1,
      };

      if (params.has_slicing) {
        loop_starts_1d[params.addr_gen0_dim] = params.addr_gen0_start;
        loop_ends_1d[params.addr_gen0_dim] = params.addr_gen0_end;
        loop_steps_1d[params.addr_gen0_dim] = params.addr_gen0_step;
      }

      if (params.has_permute) {
#pragma hls_unroll yes
        for (int dim = 0; dim < 6; dim++) {
          loop_ends_1d[dim] = loop_bounds[params.addr_gen0_dims[dim]];
        }
      }

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          int index = i * 3 + j;
          loop_starts[i][j] = loop_starts_1d[index];
          loop_ends[i][j] = loop_ends_1d[index] - loop_steps_1d[index];
          loop_steps[i][j] = loop_steps_1d[index];
        }
      }

      // Pre-compute loop bounds
#pragma hls_unroll yes
      for (int i = 0; i < 6; i++) {
        if (params.addr_gen0_broadcast[i]) {
          loop_bounds[i] = i == 5 ? Width : 1;
        }
      }

      ac_int<16, false> loop_bound_4 = loop_bounds[5];
      ac_int<16, false> loop_bound_3 = loop_bounds[4] * loop_bound_4;
      ac_int<16, false> loop_bound_2 = loop_bounds[3] * loop_bound_3;
      ac_int<16, false> loop_bound_1 = loop_bounds[2] * loop_bound_2;
      ac_int<16, false> loop_bound_0 = loop_bounds[1] * loop_bound_1;

      ac_int<16, false> X_K = X * K;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    LOOP_0_0:
      for (loop_counters[0][0] = loop_starts[0][0];;
           loop_counters[0][0] += loop_steps[0][0]) {
      LOOP_0_1:
        for (loop_counters[0][1] = loop_starts[0][1];;
             loop_counters[0][1] += loop_steps[0][1]) {
        LOOP_0_2:
          for (loop_counters[0][2] = loop_starts[0][2];;
               loop_counters[0][2] += loop_steps[0][2]) {
          LOOP_1_0:
            for (loop_counters[1][0] = loop_starts[1][0];;
                 loop_counters[1][0] += loop_steps[1][0]) {
            LOOP_1_1:
              for (loop_counters[1][1] = loop_starts[1][1];;
                   loop_counters[1][1] += loop_steps[1][1]) {
              LOOP_1_2:
                for (loop_counters[1][2] = loop_starts[1][2];;
                     loop_counters[1][2] += loop_steps[1][2]) {
                  ac_int<32, false> address;
                  bool in_bound = true;
                  bool switch_accumulation_buffer_bank = false;

                  ac_int<11, false> y1 =
                      loop_counters[0][params.addr_gen0_y_loop_idx[0]];
                  ac_int<11, false> x1 =
                      loop_counters[0][params.addr_gen0_x_loop_idx[0]];
                  ac_int<11, false> k1 =
                      loop_counters[0][params.addr_gen0_k_loop_idx[0]];
                  ac_int<11, false> y0 =
                      loop_counters[1][params.addr_gen0_y_loop_idx[1]];
                  ac_int<11, false> x0 =
                      loop_counters[1][params.addr_gen0_x_loop_idx[1]];
                  ac_int<11, false> k0 =
                      loop_counters[1][params.addr_gen0_k_loop_idx[1]];

                  if (params.addr_gen0_mode == 1) {
                    ac_int<16, true> y = y1 * stride_y + y0 - offset_y;
                    ac_int<16, true> x = x1 * stride_x + x0 - offset_x;
                    ac_int<16, false> k = k1 * stride_k + k0 * Width;

                  ADDRESS_GEN0_MODE_1:
                    if (params.has_transpose) {
                      address = y * X_K + (k + x0) * X + x1 * BUFSIZE;
                    } else {
                      address = y * X_K + x * K + k;
                    }

                    in_bound = x >= 0 && x < X && y >= 0 && y < Y;
                  } else if (params.addr_gen0_mode == 2) {
                    ac_int<LOOP_WIDTH, false> indices[6] = {
                        loop_counters[0][0], loop_counters[0][1],
                        loop_counters[0][2], loop_counters[1][0],
                        loop_counters[1][1], loop_counters[1][2],
                    };

#pragma hls_unroll yes
                    for (int i = 0; i < 6; i++) {
                      if (params.addr_gen0_broadcast[i]) {
                        indices[i] = 0;
                      }
                    }

                    if (params.has_permute) {
                      ac_int<LOOP_WIDTH, false> permuted_indices[6];
#pragma hls_unroll yes
                      for (int i = 0; i < 6; i++) {
                        permuted_indices[params.addr_gen0_dims[i]] = indices[i];
                      }
#pragma hls_unroll yes
                      for (int i = 0; i < 6; i++) {
                        indices[i] = permuted_indices[i];
                      }
                    }

                  ADDRESS_GEN0_MODE_2:
                    address =
                        (indices[0] * loop_bound_0 + indices[1] * loop_bound_1 +
                         indices[2] * loop_bound_2 + indices[3] * loop_bound_3 +
                         indices[4] * loop_bound_4 + indices[5]) *
                        Width;
                  } else if (params.addr_gen0_mode == 3) {
                  // read from accumulation buffer
                  ADDRESS_GEN0_MODE_3:
                    address = k0 * Y0 * X0 + y0 * X0 + x0;
                    switch_accumulation_buffer_bank =
                        k0 == K1 - 1 && y0 == Y0 - 1 && x0 == X0 - 1;
                  }

                  if (params.addr_gen0_mode == 1 ||
                      params.addr_gen0_mode == 2) {
                    if (params.has_transpose || in_bound) {
                    FETCH_ADDRESS_0:
                      bool found =
                          (fetch_vector_input<InputTypes, Width, InputTypes...>(
                               params.addr_gen0_dtype, address,
                               params.ADDRESS_GEN0_OFFSET,
                               vector_fetch_0_req) ||
                           ...);

#ifndef __SYNTHESIS__
                      if (!found) {
                        std::cerr << "Error: vector input 0 dtype '"
                                  << params.addr_gen0_dtype
                                  << "' is not valid.\n";
                      }
#endif
                    }

                    if (!params.has_transpose) {
                      vector_fetch_0_done.write(in_bound);
                    }
                  }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
                  else {
                    accumulation_buffer_read_address[accumulation_buffer_bank]
                        .Push(address);
                    if (switch_accumulation_buffer_bank) {
                      accumulation_buffer_done[accumulation_buffer_bank]
                          .SyncPush();
                      accumulation_buffer_bank = !accumulation_buffer_bank;
                    }
                  }
#endif

                  if (loop_counters[1][2] >= loop_ends[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0]) {
          break;
        }
      }

      vector_fetch_0_done.write(2);
    }
  }

  void feed_data_response_0() {
    data_resp_0_params.ResetRead();
    vector_fetch_0_resp.Reset();
    vector_fetch_0_data.Reset();

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_read_data[0].Reset();
    accumulation_buffer_read_data[1].Reset();
    accumulation_buffer_write_request[0].Reset();
    accumulation_buffer_write_request[1].Reset();
    accumulation_buffer_output.Reset();
#endif

    bool accumulation_buffer_bank = 0;

    wait();

    while (true) {
      VectorParams params = data_resp_0_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_ends[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_ends[i][j] = params.addr_gen0_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y0 =
          params.addr_gen0_loops[1][params.addr_gen0_y_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> X0 =
          params.addr_gen0_loops[1][params.addr_gen0_x_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> K1 =
          params.addr_gen0_loops[1][params.addr_gen0_k_loop_idx[1]] - 1;

      if (params.has_transpose) {
        VectorType buffer[BUFSIZE][Width];

#ifndef __SYNTHESIS__
        assert(params.addr_gen0_loops[1][2] == Width);
#endif

        ac_int<32, false> total_values =
            params.addr_gen0_loops[0][0] * params.addr_gen0_loops[0][1] *
            params.addr_gen0_loops[0][2] * params.addr_gen0_loops[1][0] *
            params.addr_gen0_loops[1][1];
        ac_int<32, false> counter = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      TRANSPOSE_OUTER:
        while (counter++ < total_values) {
        TRANSPOSE_READ:
          for (ac_int<10, false> col = 0;; col++) {
            Pack1D<VectorType, Width> full_response;

            bool found = (process_vector_input<InputTypes, Width, VectorType,
                                               InputTypes...>(
                              params.addr_gen0_dtype, vector_fetch_0_resp,
                              full_response) ||
                          ...);

#ifndef __SYNTHESIS__
            if (!found) {
              std::cerr << "Error: vector input 0 dtype '"
                        << params.addr_gen0_dtype << "' is not valid.\n";
            }
#endif

            Pack1D<VectorType, Width> dequantized;
            vdequantize<VectorType, VectorType, Width>(
                full_response, dequantized, params.addr_gen0_dq_scale);

            // We may not use all the data in the response
#pragma hls_unroll yes
            for (int row = 0; row < BUFSIZE; row++) {
              buffer[row][col] = dequantized[row];
            }

            if (col == Width - 1) {
              break;
            }
          }

        TRANSPOSE_WRITE:
          for (ac_int<10, false> row = 0;; row++) {
            Pack1D<VectorType, Width> transposed;
#pragma hls_unroll yes
            for (int col = 0; col < Width; col++) {
              transposed[col] = buffer[row][col];
            }

            vector_fetch_0_data.Push(transposed);

            if (row == BUFSIZE - 1) {
              break;
            }
          }
        }
      }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      else if (params.addr_gen0_mode == 3) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
              for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                  for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                    Pack1D<BufferType, Width> read_data =
                        accumulation_buffer_read_data[accumulation_buffer_bank]
                            .Pop();
                    accumulation_buffer_output.Push(read_data);

                    ac_int<LOOP_WIDTH, false> x0 =
                        loop_counters[1][params.addr_gen0_x_loop_idx[1]];
                    ac_int<LOOP_WIDTH, false> y0 =
                        loop_counters[1][params.addr_gen0_y_loop_idx[1]];
                    ac_int<LOOP_WIDTH, false> k1 =
                        loop_counters[1][params.addr_gen0_k_loop_idx[1]];

                    if (k1 == K1 && y0 == Y0 && x0 == X0) {
                      accumulation_buffer_bank = !accumulation_buffer_bank;
                    }

                    if (loop_counters[1][2] == loop_ends[1][2]) {
                      break;
                    }
                  }
                  if (loop_counters[1][1] == loop_ends[1][1]) {
                    break;
                  }
                }
                if (loop_counters[1][0] == loop_ends[1][0]) {
                  break;
                }
              }
              if (loop_counters[0][2] == loop_ends[0][2]) {
                break;
              }
            }
            if (loop_counters[0][1] == loop_ends[0][1]) {
              break;
            }
          }
          if (loop_counters[0][0] == loop_ends[0][0]) {
            break;
          }
        }
      }
#endif
      else {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (true) {
          ac_int<2, false> in_bound = vector_fetch_0_done.read();
          if (in_bound == 2) {
            break;
          }

          Pack1D<VectorType, Width> full_response;

          if (!in_bound) {
#pragma hls_unroll yes
            for (int i = 0; i < Width; i++) {
              if (params.is_maxpool) {
                full_response[i] = VectorType::min();
              } else {
                full_response[i] = VectorType::zero();
              }
            }
          } else {
            bool found = (process_vector_input<InputTypes, Width, VectorType,
                                               InputTypes...>(
                              params.addr_gen0_dtype, vector_fetch_0_resp,
                              full_response) ||
                          ...);
#ifndef __SYNTHESIS__
            if (!found) {
              std::cerr << "Error: vector input 0 type index '"
                        << params.addr_gen0_dtype << "' is not valid.\n ";
            }
#endif
          }

#if !SUPPORT_MX
          Pack1D<VectorType, Width> dequantized;
          vdequantize<VectorType, VectorType, Width>(full_response, dequantized,
                                                     params.addr_gen0_dq_scale);
          vector_fetch_0_data.Push(dequantized);
#else
          vector_fetch_0_data.Push(full_response);
#endif
        }
      }
    }
  }

  void fetch_address_1() {
    addr_gen1_params.ResetRead();
    vector_fetch_1_req.Reset();

    wait();

    while (true) {
      VectorParams params = addr_gen1_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_ends[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_ends[i][j] = params.addr_gen1_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y1 =
          params.addr_gen1_loops[0][params.addr_gen1_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.addr_gen1_loops[0][params.addr_gen1_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.addr_gen1_loops[0][params.addr_gen1_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.addr_gen1_loops[1][params.addr_gen1_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.addr_gen1_loops[1][params.addr_gen1_x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.addr_gen1_loops[1][params.addr_gen1_k_loop_idx[1]];

      ac_int<16, false> X = X1 * X0;
      ac_int<16, false> K = K2 * K1 * Width;

      if (params.addr_gen1_broadcast[1]) {
        X = 1;
      }

      if (params.addr_gen1_broadcast[2]) {
        K = Width;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
              for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                  ac_int<LOOP_WIDTH, false> x0 =
                      loop_counters[1][params.addr_gen1_x_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> x1 =
                      loop_counters[0][params.addr_gen1_x_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> y0 =
                      loop_counters[1][params.addr_gen1_y_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> y1 =
                      loop_counters[0][params.addr_gen1_y_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> k0 =
                      loop_counters[1][params.addr_gen1_k_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> k1 =
                      loop_counters[0][params.addr_gen1_k_loop_idx[0]];

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> k = k1 * K1 * Width + k0 * Width;

                  if (params.addr_gen1_broadcast[0]) {
                    y = 0;
                  }

                  if (params.addr_gen1_broadcast[1]) {
                    x = 0;
                  }

                  if (params.addr_gen1_broadcast[2]) {
                    k = 0;
                  }

                  ac_int<32, false> address = y * X * K + x * K + k;

                  bool found =
                      (fetch_vector_input<InputTypes, Width, InputTypes...>(
                           params.addr_gen1_dtype, address,
                           params.ADDRESS_GEN1_OFFSET, vector_fetch_1_req) ||
                       ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: vector input 1 dtype '"
                              << params.addr_gen1_dtype << "' is not valid.\n";
                  }
#endif

                  vector_fetch_1_done.write(false);

                  if (loop_counters[1][2] == loop_ends[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] == loop_ends[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] == loop_ends[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] == loop_ends[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] == loop_ends[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] == loop_ends[0][0]) {
          break;
        }
      }
      vector_fetch_1_done.write(true);
    }
  }

  void feed_data_response_1() {
    data_resp_1_params.ResetRead();
    vector_fetch_1_resp.Reset();
    vector_fetch_1_data.Reset();

    wait();

    while (true) {
      VectorParams params = data_resp_1_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!vector_fetch_1_done.read()) {
        Pack1D<VectorType, Width> full_response;

        bool found =
            (process_vector_input<InputTypes, Width, VectorType, InputTypes...>(
                 params.addr_gen1_dtype, vector_fetch_1_resp, full_response) ||
             ...);

#ifndef __SYNTHESIS__
        if (!found) {
          std::cerr << "Error: vector input 1 dtype '" << params.addr_gen1_dtype
                    << "' is not valid.\n";
        }
#endif

#if !SUPPORT_MX
        Pack1D<VectorType, Width> dequantized;
        vdequantize<VectorType, VectorType, Width>(full_response, dequantized,
                                                   params.addr_gen1_dq_scale);
        vector_fetch_1_data.Push(dequantized);
#else
        vector_fetch_1_data.Push(full_response);
#endif
      }
    }
  }

  void fetch_address_2() {
    addr_gen2_params.ResetRead();
    vector_fetch_2_req.Reset();

    wait();

    while (true) {
      VectorParams params = addr_gen2_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_ends[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_ends[i][j] = params.addr_gen2_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y1 =
          params.addr_gen2_loops[0][params.addr_gen2_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.addr_gen2_loops[0][params.addr_gen2_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.addr_gen2_loops[0][params.addr_gen2_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.addr_gen2_loops[1][params.addr_gen2_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.addr_gen2_loops[1][params.addr_gen2_x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.addr_gen2_loops[1][params.addr_gen2_k_loop_idx[1]];

      ac_int<16, false> X = X1 * X0;
      ac_int<16, false> K = K2 * K1 * Width;

      if (params.addr_gen2_broadcast[1]) {
        X = 1;
      }

      if (params.addr_gen2_broadcast[2]) {
        K = Width;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
              for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                  ac_int<LOOP_WIDTH, false> x0 =
                      loop_counters[1][params.addr_gen2_x_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> x1 =
                      loop_counters[0][params.addr_gen2_x_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> y0 =
                      loop_counters[1][params.addr_gen2_y_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> y1 =
                      loop_counters[0][params.addr_gen2_y_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> k0 =
                      loop_counters[1][params.addr_gen2_k_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> k1 =
                      loop_counters[0][params.addr_gen2_k_loop_idx[0]];

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> k = k1 * K1 * Width + k0 * Width;

                  if (params.addr_gen2_broadcast[0]) {
                    y = 0;
                  }

                  if (params.addr_gen2_broadcast[1]) {
                    x = 0;
                  }

                  if (params.addr_gen2_broadcast[2]) {
                    k = 0;
                  }

                  ac_int<32, false> address = y * X * K + x * K + k;

                  bool found =
                      (fetch_vector_input<InputTypes, Width, InputTypes...>(
                           params.addr_gen2_dtype, address,
                           params.ADDRESS_GEN2_OFFSET, vector_fetch_2_req) ||
                       ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: vector input 2 dtype '"
                              << params.addr_gen2_dtype << "' is not valid.\n";
                  }
#endif

                  vector_fetch_2_done.write(false);

                  if (loop_counters[1][2] == loop_ends[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] == loop_ends[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] == loop_ends[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] == loop_ends[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] == loop_ends[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] == loop_ends[0][0]) {
          break;
        }
      }
      vector_fetch_2_done.write(true);
    }
  }

  void feed_data_response_2() {
    data_resp_2_params.ResetRead();
    vector_fetch_2_resp.Reset();
    vector_fetch_2_data.Reset();

    wait();

    while (true) {
      VectorParams params = data_resp_2_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!vector_fetch_2_done.read()) {
        Pack1D<VectorType, Width> full_response;

        bool found =
            (process_vector_input<InputTypes, Width, VectorType, InputTypes...>(
                 params.addr_gen2_dtype, vector_fetch_2_resp, full_response) ||
             ...);

#ifndef __SYNTHESIS__
        if (!found) {
          std::cerr << "Error: vector input 2 dtype '" << params.addr_gen2_dtype
                    << "' is not valid.\n";
        }
#endif

#if !SUPPORT_MX
        Pack1D<VectorType, Width> dequantized;
        vdequantize<VectorType, VectorType, Width>(full_response, dequantized,
                                                   params.addr_gen2_dq_scale);
        vector_fetch_2_data.Push(dequantized);
#else
        vector_fetch_2_data.Push(full_response);
#endif
      }
    }
  }

  void read_params() {
    params_in.Reset();

    addr_gen0_params.ResetWrite();
    addr_gen1_params.ResetWrite();
    addr_gen2_params.ResetWrite();

    data_resp_0_params.ResetWrite();
    data_resp_1_params.ResetWrite();
    data_resp_2_params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = params_in.Pop();

      if (params.addr_gen0_mode != 0) {
        addr_gen0_params.Push(params);
        data_resp_0_params.Push(params);
      }

      if (params.addr_gen1_mode != 0) {
        addr_gen1_params.Push(params);
        data_resp_1_params.Push(params);
      }

      if (params.addr_gen2_mode != 0) {
        addr_gen2_params.Push(params);
        data_resp_2_params.Push(params);
      }
    }
  }
};
