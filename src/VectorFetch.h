#pragma once

#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename BufferType, int Width,
          typename... InputTypes>
SC_MODULE(VectorFetchUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_request_out);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_request_out);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_request_out);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp_in);
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_0_data_out);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::In<Pack1D<BufferType, Width>> accumulation_buffer_read_data[2];
  Connections::Out<BufferWriteRequest<Pack1D<BufferType, Width>>>
      accumulation_buffer_write_request[2];
  Connections::Out<Pack1D<BufferType, Width>> CCS_INIT_S1(
      accumulationBufferOutput);
  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::SyncOut accumulation_buffer_done[2];
#endif

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp_in);
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_1_data_out);

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp_in);
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_2_data_out);

  Connections::Combinational<VectorParams> CCS_INIT_S1(addr_gen0_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addr_gen1_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addr_gen2_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(data_resp_0_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(data_resp_1_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(data_resp_2_params);

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
    vector_fetch_0_request_out.Reset();
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

      ac_int<11, false> Y1 =
          params.addr_gen0_loops[0][params.addr_gen0_y_loop_idx[0]];
      ac_int<11, false> X1 =
          params.addr_gen0_loops[0][params.addr_gen0_x_loop_idx[0]];
      ac_int<11, false> K1 =
          params.addr_gen0_loops[0][params.addr_gen0_k_loop_idx[0]];
      ac_int<11, false> Y0 =
          params.addr_gen0_loops[1][params.addr_gen0_y_loop_idx[1]];
      ac_int<11, false> X0 =
          params.addr_gen0_loops[1][params.addr_gen0_x_loop_idx[1]];
      ac_int<11, false> K0 =
          params.addr_gen0_loops[1][params.addr_gen0_k_loop_idx[1]];

      if (params.has_transpose && BUFSIZE != Width) {
        X1 = X1 * BUFSIZE / Width;
      }

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addr_gen0_loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

      if (params.has_slicing) {
        int slice_dim = params.addr_gen0_dim;
        int i = slice_dim >= 3 ? 1 : 0;
        int j = slice_dim >= 3 ? slice_dim - 3 : slice_dim;
        loop_starts[i][j] = params.addr_gen0_start;
        loop_ends[i][j] = params.addr_gen0_end;
        loop_steps[i][j] = params.addr_gen0_step;
      } else if (params.has_permute) {
#pragma hls_unroll yes
        for (int dim = 0; dim < 6; dim++) {
          int i = dim >= 3 ? 1 : 0;
          int j = dim >= 3 ? dim - 3 : dim;
          int new_dim = params.addr_gen0_dims[dim];
          int new_i = new_dim >= 3 ? 1 : 0;
          int new_j = new_dim >= 3 ? new_dim - 3 : new_dim;
          loop_ends[i][j] = params.addr_gen0_loops[new_i][new_j];
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  ac_int<32, false> address;
                  bool switch_accumulation_buffer_bank = false;
                  if (params.addr_gen0_mode == 1) {
                    ac_int<11, false> x0 =
                        loop_counters[1][params.addr_gen0_x_loop_idx[1]];
                    ac_int<11, false> x1 =
                        loop_counters[0][params.addr_gen0_x_loop_idx[0]];
                    ac_int<11, false> y0 =
                        loop_counters[1][params.addr_gen0_y_loop_idx[1]];
                    ac_int<11, false> y1 =
                        loop_counters[0][params.addr_gen0_y_loop_idx[0]];
                    ac_int<11, false> k0 =
                        loop_counters[1][params.addr_gen0_k_loop_idx[1]];
                    ac_int<11, false> k1 =
                        loop_counters[0][params.addr_gen0_k_loop_idx[0]];

                    ac_int<16, false> k = k1 * K0 * Width + k0 * Width;
                    ac_int<16, false> K = K1 * K0 * Width;

                    ac_int<16, false> x = x1 * X0 + x0;
                    ac_int<16, false> X = X1 * X0;

                    ac_int<16, false> y = y1 * Y0 + y0;
                    ac_int<16, false> Y = Y1 * Y0;

                    if (params.has_transpose) {
                      address = y * K * X + (k + x0) * X + x1 * BUFSIZE;
                    } else {
                      address = y * X * K + x * K + k;
                    }
                  } else if (params.addr_gen0_mode == 2) {
                    ac_int<11, false> indices[6] = {
                        loop_counters[0][0], loop_counters[0][1],
                        loop_counters[0][2], loop_counters[1][0],
                        loop_counters[1][1], loop_counters[1][2]};

                    ac_int<11, false> loop_bounds[6] = {
                        params.addr_gen0_loops[0][0],
                        params.addr_gen0_loops[0][1],
                        params.addr_gen0_loops[0][2],
                        params.addr_gen0_loops[1][0],
                        params.addr_gen0_loops[1][1],
                        params.addr_gen0_loops[1][2]};

#pragma hls_unroll yes
                    for (int i = 0; i < 6; i++) {
                      if (params.addr_gen0_broadcast[i]) {
                        indices[i] = 0;
                        loop_bounds[i] = 1;
                      }
                    }

                    if (params.has_permute) {
                      ac_int<11, false> permuted_indices[6];
#pragma hls_unroll yes
                      for (int i = 0; i < 6; i++) {
                        permuted_indices[params.addr_gen0_dims[i]] = indices[i];
                      }

#pragma hls_unroll yes
                      for (int i = 0; i < 6; i++) {
                        indices[i] = permuted_indices[i];
                      }
                    }

                    address =
                        (indices[0] * loop_bounds[1] * loop_bounds[2] *
                             loop_bounds[3] * loop_bounds[4] * loop_bounds[5] +
                         indices[1] * loop_bounds[2] * loop_bounds[3] *
                             loop_bounds[4] * loop_bounds[5] +
                         indices[2] * loop_bounds[3] * loop_bounds[4] *
                             loop_bounds[5] +
                         indices[3] * loop_bounds[4] * loop_bounds[5] +
                         indices[4] * loop_bounds[5] + indices[5]) *
                        Width;
                  } else if (params.addr_gen0_mode == 3) {
                    // read from accumulation buffer
                    ac_int<11, false> x0 =
                        loop_counters[1][params.addr_gen0_x_loop_idx[1]];
                    ac_int<11, false> x1 =
                        loop_counters[0][params.addr_gen0_x_loop_idx[0]];
                    ac_int<11, false> y0 =
                        loop_counters[1][params.addr_gen0_y_loop_idx[1]];
                    ac_int<11, false> y1 =
                        loop_counters[0][params.addr_gen0_y_loop_idx[0]];
                    ac_int<11, false> k0 =
                        loop_counters[1][params.addr_gen0_k_loop_idx[1]];
                    ac_int<11, false> k1 =
                        loop_counters[0][params.addr_gen0_k_loop_idx[0]];

                    address = k0 * Y0 * X0 + y0 * X0 + x0;

                    if (k0 == K0 - 1 && y0 == Y0 - 1 && x0 == X0 - 1) {
                      switch_accumulation_buffer_bank = true;
                    }
                  }

                  if (params.addr_gen0_mode == 1 ||
                      params.addr_gen0_mode == 2) {
                    bool found =
                        (fetch_vector_input<InputTypes, Width, InputTypes...>(
                             params.addr_gen0_dtype, address,
                             params.ADDRESS_GEN0_OFFSET,
                             vector_fetch_0_request_out) ||
                         ...);

#ifndef __SYNTHESIS__
                    if (!found) {
                      std::cerr << "Error: vector input 0 dtype '"
                                << params.addr_gen0_dtype
                                << "' is not valid.\n";
                    }
#endif
                  } else {
#if DOUBLE_BUFFERED_ACCUM_BUFFER
                    accumulation_buffer_read_address[accumulation_buffer_bank]
                        .Push(address);
                    if (switch_accumulation_buffer_bank) {
                      accumulation_buffer_done[accumulation_buffer_bank]
                          .SyncPush();
                      accumulation_buffer_bank = !accumulation_buffer_bank;
                    }
#endif
                  }

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
        }
      }
    }
  }

  void feed_data_response_0() {
    data_resp_0_params.ResetRead();
    vector_fetch_0_resp_in.Reset();
    vector_fetch_0_data_out.Reset();

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_read_data[0].Reset();
    accumulation_buffer_read_data[1].Reset();
    accumulation_buffer_write_request[0].Reset();
    accumulation_buffer_write_request[1].Reset();
    accumulationBufferOutput.Reset();
#endif

    bool accumulation_buffer_bank = 0;

    wait();

    while (true) {
      VectorParams params = data_resp_0_params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addr_gen0_loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

      if (params.has_slicing) {
        int slice_dim = params.addr_gen0_dim;
        int i = slice_dim >= 3 ? 1 : 0;
        int j = slice_dim >= 3 ? slice_dim - 3 : slice_dim;
        loop_starts[i][j] = params.addr_gen0_start;
        loop_ends[i][j] = params.addr_gen0_end;
        loop_steps[i][j] = params.addr_gen0_step;
      }

      if (params.has_transpose) {
        VectorType buffer[BUFSIZE][Width];

        assert(loop_ends[1][2] == Width);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_ends[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_ends[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_ends[0][2];
                 loop_counters[0][2]++) {
              for (loop_counters[1][0] = 0;
                   loop_counters[1][0] < loop_ends[1][0];
                   loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;
                     loop_counters[1][1] < loop_ends[1][1];
                     loop_counters[1][1]++) {
                  for (int col = 0; col < Width; col++) {
                    Pack1D<VectorType, Width> full_response;

                    bool found =
                        (process_vector_input<InputTypes, Width, VectorType,
                                              InputTypes...>(
                             params.addr_gen0_dtype, vector_fetch_0_resp_in,
                             full_response) ||
                         ...);

#ifndef __SYNTHESIS__
                    if (!found) {
                      std::cerr << "Error: vector input 0 dtype '"
                                << params.addr_gen0_dtype
                                << "' is not valid.\n";
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
                  }

                  // Write out from transpose buffer
                  for (int row = 0; row < BUFSIZE; row++) {
                    Pack1D<VectorType, Width> transposed;
#pragma hls_unroll yes
                    for (int col = 0; col < Width; col++) {
                      transposed[col] = buffer[row][col];
                    }

                    vector_fetch_0_data_out.Push(transposed);
                  }
                }
              }
            }
          }
        }
      } else if (params.addr_gen0_mode == 3) {
#if DOUBLE_BUFFERED_ACCUM_BUFFER
        ac_int<11, false> Y1 =
            params.addr_gen0_loops[0][params.addr_gen0_y_loop_idx[0]];
        ac_int<11, false> X1 =
            params.addr_gen0_loops[0][params.addr_gen0_x_loop_idx[0]];
        ac_int<11, false> K1 =
            params.addr_gen0_loops[0][params.addr_gen0_k_loop_idx[0]];
        ac_int<11, false> Y0 =
            params.addr_gen0_loops[1][params.addr_gen0_y_loop_idx[1]];
        ac_int<11, false> X0 =
            params.addr_gen0_loops[1][params.addr_gen0_x_loop_idx[1]];
        ac_int<11, false> K0 =
            params.addr_gen0_loops[1][params.addr_gen0_k_loop_idx[1]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = loop_starts[0][0];
             loop_counters[0][0] < loop_ends[0][0];
             loop_counters[0][0] += loop_steps[0][0]) {
          for (loop_counters[0][1] = loop_starts[0][1];
               loop_counters[0][1] < loop_ends[0][1];
               loop_counters[0][1] += loop_steps[0][1]) {
            for (loop_counters[0][2] = loop_starts[0][2];
                 loop_counters[0][2] < loop_ends[0][2];
                 loop_counters[0][2] += loop_steps[0][2]) {
              for (loop_counters[1][0] = loop_starts[1][0];
                   loop_counters[1][0] < loop_ends[1][0];
                   loop_counters[1][0] += loop_steps[1][0]) {
                for (loop_counters[1][1] = loop_starts[1][1];
                     loop_counters[1][1] < loop_ends[1][1];
                     loop_counters[1][1] += loop_steps[1][1]) {
                  for (loop_counters[1][2] = loop_starts[1][2];
                       loop_counters[1][2] < loop_ends[1][2];
                       loop_counters[1][2] += loop_steps[1][2]) {
                    Pack1D<BufferType, Width> read_data =
                        accumulation_buffer_read_data[accumulation_buffer_bank]
                            .Pop();
                    accumulationBufferOutput.Push(read_data);

                    ac_int<11, false> x0 =
                        loop_counters[1][params.addr_gen0_x_loop_idx[1]];
                    ac_int<11, false> x1 =
                        loop_counters[0][params.addr_gen0_x_loop_idx[0]];
                    ac_int<11, false> y0 =
                        loop_counters[1][params.addr_gen0_y_loop_idx[1]];
                    ac_int<11, false> y1 =
                        loop_counters[0][params.addr_gen0_y_loop_idx[0]];
                    ac_int<11, false> k0 =
                        loop_counters[1][params.addr_gen0_k_loop_idx[1]];
                    ac_int<11, false> k1 =
                        loop_counters[0][params.addr_gen0_k_loop_idx[0]];

                    if (k0 == K0 - 1 && y0 == Y0 - 1 && x0 == X0 - 1) {
                      accumulation_buffer_bank = !accumulation_buffer_bank;
                    }
                    if (loop_counters[1][2] >=
                        loop_ends[1][2] - loop_steps[1][2]) {
                      break;
                    }
                  }
                  if (loop_counters[1][1] >=
                      loop_ends[1][1] - loop_steps[1][1]) {
                    break;
                  }
                }
                if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                  break;
                }
              }
              if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
                break;
              }
            }
            if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
              break;
            }
          }
          if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
            break;
          }
        }
#endif
      } else {
        // passthrough
        ac_int<32, false> num_writes = loop_ends[0][0] * loop_ends[0][1] *
                                       loop_ends[0][2] * loop_ends[1][0] *
                                       loop_ends[1][1] * loop_ends[1][2];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < num_writes; i++) {
          Pack1D<VectorType, Width> full_response;

          bool found = (process_vector_input<InputTypes, Width, VectorType,
                                             InputTypes...>(
                            params.addr_gen0_dtype, vector_fetch_0_resp_in,
                            full_response) ||
                        ...);

#ifndef __SYNTHESIS__
          if (!found) {
            std::cerr << "Error: vector input 0 dtype '"
                      << params.addr_gen0_dtype << "' is not valid.\n";
          }
#endif

          Pack1D<VectorType, Width> dequantized;
          vdequantize<VectorType, VectorType, Width>(full_response, dequantized,
                                                     params.addr_gen0_dq_scale);

          vector_fetch_0_data_out.Push(dequantized);
        }
      }
    }
  }

  void fetch_address_1() {
    addr_gen1_params.ResetRead();
    vector_fetch_1_request_out.Reset();

    wait();

    while (true) {
      VectorParams params = addr_gen1_params.Pop();

      ac_int<11, false> Y1 =
          params.addr_gen1_loops[0][params.addr_gen1_y_loop_idx[0]];
      ac_int<11, false> X1 =
          params.addr_gen1_loops[0][params.addr_gen1_x_loop_idx[0]];
      ac_int<11, false> K1 =
          params.addr_gen1_loops[0][params.addr_gen1_k_loop_idx[0]];
      ac_int<11, false> Y0 =
          params.addr_gen1_loops[1][params.addr_gen1_y_loop_idx[1]];
      ac_int<11, false> X0 =
          params.addr_gen1_loops[1][params.addr_gen1_x_loop_idx[1]];
      ac_int<11, false> K0 =
          params.addr_gen1_loops[1][params.addr_gen1_k_loop_idx[1]];

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addr_gen1_loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  ac_int<11, false> x0 =
                      loop_counters[1][params.addr_gen1_x_loop_idx[1]];
                  ac_int<11, false> x1 =
                      loop_counters[0][params.addr_gen1_x_loop_idx[0]];
                  ac_int<11, false> y0 =
                      loop_counters[1][params.addr_gen1_y_loop_idx[1]];
                  ac_int<11, false> y1 =
                      loop_counters[0][params.addr_gen1_y_loop_idx[0]];
                  ac_int<11, false> k0 =
                      loop_counters[1][params.addr_gen1_k_loop_idx[1]];
                  ac_int<11, false> k1 =
                      loop_counters[0][params.addr_gen1_k_loop_idx[0]];

                  ac_int<16, false> k = k1 * K0 * Width + k0 * Width;
                  ac_int<16, false> K = K1 * K0 * Width;

                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> X = X1 * X0;

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> Y = Y1 * Y0;

                  if (params.addr_gen1_broadcast[0]) {
                    y = 0;
                  }

                  if (params.addr_gen1_broadcast[1]) {
                    x = 0;
                    X = 1;
                  }

                  if (params.addr_gen1_broadcast[2]) {
                    k = 0;
                    K = 1;
                  }

                  ac_int<32, false> address = y * X * K + x * K + k;

                  bool found =
                      (fetch_vector_input<InputTypes, Width, InputTypes...>(
                           params.addr_gen1_dtype, address,
                           params.ADDRESS_GEN1_OFFSET,
                           vector_fetch_1_request_out) ||
                       ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: vector input 1 dtype '"
                              << params.addr_gen1_dtype << "' is not valid.\n";
                  }
#endif

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
        }
      }
    }
  }

  void feed_data_response_1() {
    data_resp_1_params.ResetRead();
    vector_fetch_1_resp_in.Reset();
    vector_fetch_1_data_out.Reset();

    wait();

    while (true) {
      VectorParams params = data_resp_1_params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addr_gen1_loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  Pack1D<VectorType, Width> full_response;

                  bool found = (process_vector_input<InputTypes, Width,
                                                     VectorType, InputTypes...>(
                                    params.addr_gen1_dtype,
                                    vector_fetch_1_resp_in, full_response) ||
                                ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: vector input 1 dtype '"
                              << params.addr_gen1_dtype << "' is not valid.\n";
                  }
#endif

                  Pack1D<VectorType, Width> dequantized;
                  vdequantize<VectorType, VectorType, Width>(
                      full_response, dequantized, params.addr_gen1_dq_scale);

                  vector_fetch_1_data_out.Push(dequantized);

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
        }
      }
    }
  }

  void fetch_address_2() {
    addr_gen2_params.ResetRead();
    vector_fetch_2_request_out.Reset();

    wait();

    while (true) {
      VectorParams params = addr_gen2_params.Pop();

      ac_int<11, false> Y1 =
          params.addr_gen2_loops[0][params.addr_gen2_y_loop_idx[0]];
      ac_int<11, false> X1 =
          params.addr_gen2_loops[0][params.addr_gen2_x_loop_idx[0]];
      ac_int<11, false> K1 =
          params.addr_gen2_loops[0][params.addr_gen2_k_loop_idx[0]];
      ac_int<11, false> Y0 =
          params.addr_gen2_loops[1][params.addr_gen2_y_loop_idx[1]];
      ac_int<11, false> X0 =
          params.addr_gen2_loops[1][params.addr_gen2_x_loop_idx[1]];
      ac_int<11, false> K0 =
          params.addr_gen2_loops[1][params.addr_gen2_k_loop_idx[1]];

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addr_gen2_loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  ac_int<11, false> x0 =
                      loop_counters[1][params.addr_gen2_x_loop_idx[1]];
                  ac_int<11, false> x1 =
                      loop_counters[0][params.addr_gen2_x_loop_idx[0]];
                  ac_int<11, false> y0 =
                      loop_counters[1][params.addr_gen2_y_loop_idx[1]];
                  ac_int<11, false> y1 =
                      loop_counters[0][params.addr_gen2_y_loop_idx[0]];
                  ac_int<11, false> k0 =
                      loop_counters[1][params.addr_gen2_k_loop_idx[1]];
                  ac_int<11, false> k1 =
                      loop_counters[0][params.addr_gen2_k_loop_idx[0]];

                  ac_int<16, false> k = k1 * K0 * Width + k0 * Width;
                  ac_int<16, false> K = K1 * K0 * Width;

                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> X = X1 * X0;

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> Y = Y1 * Y0;

                  if (params.addr_gen2_broadcast[0]) {
                    y = 0;
                  }

                  if (params.addr_gen2_broadcast[1]) {
                    x = 0;
                    X = 1;
                  }

                  if (params.addr_gen2_broadcast[2]) {
                    k = 0;
                    K = 1;
                  }

                  ac_int<32, false> address = y * X * K + x * K + k;

                  bool found =
                      (fetch_vector_input<InputTypes, Width, InputTypes...>(
                           params.addr_gen2_dtype, address,
                           params.ADDRESS_GEN2_OFFSET,
                           vector_fetch_2_request_out) ||
                       ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: vector input 2 dtype '"
                              << params.addr_gen2_dtype << "' is not valid.\n";
                  }
#endif

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
        }
      }
    }
  }

  void feed_data_response_2() {
    data_resp_2_params.ResetRead();
    vector_fetch_2_resp_in.Reset();
    vector_fetch_2_data_out.Reset();

    wait();

    while (true) {
      VectorParams params = data_resp_2_params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addr_gen2_loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  Pack1D<VectorType, Width> full_response;

                  bool found = (process_vector_input<InputTypes, Width,
                                                     VectorType, InputTypes...>(
                                    params.addr_gen2_dtype,
                                    vector_fetch_2_resp_in, full_response) ||
                                ...);

#ifndef __SYNTHESIS__
                  if (!found) {
                    std::cerr << "Error: vector input 2 dtype '"
                              << params.addr_gen2_dtype << "' is not valid.\n";
                  }
#endif

                  Pack1D<VectorType, Width> dequantized;
                  vdequantize<VectorType, VectorType, Width>(
                      full_response, dequantized, params.addr_gen2_dq_scale);

                  vector_fetch_2_data_out.Push(dequantized);

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
        }
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
