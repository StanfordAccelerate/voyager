#pragma once

#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename BufferType, int Width, int OcDimension,
          typename... InputTypes>
SC_MODULE(VectorFetchUnit) {
  static constexpr int LOOP_WIDTH = 11;
  static constexpr int MAX_BUFSIZE =
      1024 / OcDimension < OcDimension ? 1024 / OcDimension : OcDimension;
  static constexpr int BUFSIZE = Width < MAX_BUFSIZE ? Width : MAX_BUFSIZE;
  static constexpr int MAX_DTYPE_WIDTH = std::max({InputTypes::width...});
  static constexpr int MAX_RESPONSE_WIDTH = MAX_DTYPE_WIDTH * OcDimension;

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_req);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::In<Pack1D<BufferType, OcDimension>>
      accumulation_buffer_read_data[2];
  Connections::Out<BufferWriteRequest<Pack1D<BufferType, OcDimension>>>
      accumulation_buffer_write_request[2];
  Connections::Out<Pack1D<BufferType, Width>> CCS_INIT_S1(
      accumulation_buffer_output);
  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::SyncOut accumulation_buffer_done[2];
#endif

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_resp);
  Connections::Combinational<ac_int<MAX_RESPONSE_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_packed_bits);
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_fetch_0_data);
  sc_fifo<bool> vector_fetch_0_packer_done;
  sc_fifo<ac_int<2, false>> vector_fetch_0_data_done;

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_resp);
  Connections::Combinational<ac_int<MAX_RESPONSE_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_packed_bits);
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_fetch_1_data);
  sc_fifo<bool> vector_fetch_1_packer_done;
  sc_fifo<bool> vector_fetch_1_data_done;

  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_resp);
  Connections::Combinational<ac_int<MAX_RESPONSE_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_packed_bits);
  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_fetch_2_data);
  sc_fifo<bool> vector_fetch_2_packer_done;
  sc_fifo<bool> vector_fetch_2_data_done;

  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_fetch_0_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_fetch_1_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_fetch_2_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vector_fetch_0_packer_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vector_fetch_1_packer_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vector_fetch_2_packer_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vector_fetch_0_data_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vector_fetch_1_data_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vector_fetch_2_data_params);

  SC_CTOR(VectorFetchUnit) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_request_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(pack_response_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_request_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(pack_response_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_request_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(pack_response_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void send_request_0() {
    vector_fetch_0_params.ResetRead();
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
      VectorParams params = vector_fetch_0_params.Pop();

      ac_int<LOOP_WIDTH, false> Y1 =
          params.vector_fetch_0_loops[0][params.vector_fetch_0_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.vector_fetch_0_loops[0][params.vector_fetch_0_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.vector_fetch_0_loops[0][params.vector_fetch_0_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.vector_fetch_0_loops[1][params.vector_fetch_0_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.vector_fetch_0_loops[1][params.vector_fetch_0_x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.vector_fetch_0_loops[1][params.vector_fetch_0_k_loop_idx[1]];

      ac_int<LOOP_WIDTH, false> stride_y = Y0;
      ac_int<LOOP_WIDTH, false> stride_x = X0;
      ac_int<16, false> stride_k = Width * params.vector_fetch_0_packing_factor;

      if (params.has_transpose) {
        X1 = X1 * BUFSIZE / OcDimension;
        stride_k = stride_k * OcDimension / Width;
      }

      ac_int<16, false> Y = Y1 * Y0;
      ac_int<16, false> X = X1 * X0;
      ac_int<16, false> K = K2 * K1 * stride_k;

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

      ac_int<LOOP_WIDTH, false> loop_starts_flat[6] = {
          0, 0, 0, 0, 0, 0,
      };
      ac_int<LOOP_WIDTH, false> loop_steps_flat[6] = {
          1, 1, 1, 1, 1, 1,
      };
      ac_int<LOOP_WIDTH, false> loop_ends_flat[6] = {
          params.vector_fetch_0_loops[0][0], params.vector_fetch_0_loops[0][1],
          params.vector_fetch_0_loops[0][2], params.vector_fetch_0_loops[1][0],
          params.vector_fetch_0_loops[1][1], params.vector_fetch_0_loops[1][2],
      };
      ac_int<LOOP_WIDTH, false> loop_bounds[6] = {
          params.vector_fetch_0_loops[0][0], params.vector_fetch_0_loops[0][1],
          params.vector_fetch_0_loops[0][2], params.vector_fetch_0_loops[1][0],
          params.vector_fetch_0_loops[1][1], params.vector_fetch_0_loops[1][2],
      };

      if (params.has_slicing) {
        loop_starts_flat[params.vector_fetch_0_dim] =
            params.vector_fetch_0_start;
        loop_ends_flat[params.vector_fetch_0_dim] = params.vector_fetch_0_end;
        loop_steps_flat[params.vector_fetch_0_dim] = params.vector_fetch_0_step;
      }

      if (params.has_permute) {
#pragma hls_unroll yes
        for (int dim = 0; dim < 6; dim++) {
          loop_ends_flat[dim] = loop_bounds[params.vector_fetch_0_dims[dim]];
        }
      }

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          int index = i * 3 + j;
          loop_starts[i][j] = loop_starts_flat[index];
          loop_ends[i][j] = loop_ends_flat[index] - loop_steps_flat[index];
          loop_steps[i][j] = loop_steps_flat[index];
        }
      }

      // Pre-compute loop bounds
#pragma hls_unroll yes
      for (int i = 0; i < 6; i++) {
        if (params.vector_fetch_0_broadcast[i]) {
          loop_bounds[i] = i == 5 ? Width : 1;
        }
      }

      ac_int<16, false> loop_bound_4 = loop_bounds[5];
      ac_int<16, false> loop_bound_3 = loop_bounds[4] * loop_bound_4;
      ac_int<16, false> loop_bound_2 = loop_bounds[3] * loop_bound_3;
      ac_int<16, false> loop_bound_1 = loop_bounds[2] * loop_bound_2;
      ac_int<16, false> loop_bound_0 = loop_bounds[1] * loop_bound_1;

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
                      loop_counters[0][params.vector_fetch_0_y_loop_idx[0]];
                  ac_int<11, false> x1 =
                      loop_counters[0][params.vector_fetch_0_x_loop_idx[0]];
                  ac_int<11, false> k1 =
                      loop_counters[0][params.vector_fetch_0_k_loop_idx[0]];
                  ac_int<11, false> y0 =
                      loop_counters[1][params.vector_fetch_0_y_loop_idx[1]];
                  ac_int<11, false> x0 =
                      loop_counters[1][params.vector_fetch_0_x_loop_idx[1]];
                  ac_int<11, false> k0 =
                      loop_counters[1][params.vector_fetch_0_k_loop_idx[1]];

                  if (params.vector_fetch_0_mode == 1) {
                    ac_int<16, true> y = y1 * stride_y + y0 - offset_y;
                    ac_int<16, true> x = x1 * stride_x + x0 - offset_x;
                    ac_int<16, false> k = (k1 * K1 + k0) * stride_k;

                  ADDRESS_GEN0_MODE_1:
                    if (params.has_transpose) {
                      address = y * X * K + (k + x0) * X + x1 * BUFSIZE;
                    } else {
                      address = y * X * K + x * K + k;
                    }

                    in_bound = x >= 0 && x < X && y >= 0 && y < Y;
                  } else if (params.vector_fetch_0_mode == 2) {
                    ac_int<LOOP_WIDTH, false> indices[6] = {
                        loop_counters[0][0], loop_counters[0][1],
                        loop_counters[0][2], loop_counters[1][0],
                        loop_counters[1][1], loop_counters[1][2],
                    };

#pragma hls_unroll yes
                    for (int i = 0; i < 6; i++) {
                      if (params.vector_fetch_0_broadcast[i]) {
                        indices[i] = 0;
                      }
                    }

                    if (params.has_permute) {
                      ac_int<LOOP_WIDTH, false> permuted_indices[6];
#pragma hls_unroll yes
                      for (int i = 0; i < 6; i++) {
                        permuted_indices[params.vector_fetch_0_dims[i]] =
                            indices[i];
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
                        stride_k;
                  } else if (params.vector_fetch_0_mode == 3) {
                  // read from accumulation buffer
                  ADDRESS_GEN0_MODE_3:
                    address = k0 * Y0 * X0 + y0 * X0 + x0;
                    switch_accumulation_buffer_bank =
                        k0 == K1 - 1 && y0 == Y0 - 1 && x0 == X0 - 1;
                  }

                  if (params.vector_fetch_0_mode == 1 ||
                      params.vector_fetch_0_mode == 2) {
                    if (params.has_transpose || in_bound) {
                      fetch_vector_input<InputTypes...>(
                          params.vector_fetch_0_dtype,
                          params.vector_fetch_0_offset, address,
                          params.vector_fetch_0_burst_size, vector_fetch_0_req);
                      vector_fetch_0_packer_done.write(false);
                    }

                    if (!params.has_transpose) {
                      vector_fetch_0_data_done.write(in_bound);
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

      vector_fetch_0_packer_done.write(true);
      vector_fetch_0_data_done.write(2);
    }
  }

  void pack_response_0() {
    vector_fetch_0_packer_params.ResetRead();
    vector_fetch_0_resp.Reset();
    vector_fetch_0_packed_bits.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vector_fetch_0_packer_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!vector_fetch_0_packer_done.read()) {
        ac_int<MAX_RESPONSE_WIDTH, false> packed_bits;

        for (ac_int<4, false> i = 0;; i++) {
          packed_bits.set_slc(i * OC_PORT_WIDTH, vector_fetch_0_resp.Pop());
          if (i == params.vector_fetch_0_num_beats - 1) {
            break;
          }
        }

        vector_fetch_0_packed_bits.Push(packed_bits);
      }
    }
  }

  void feed_data_0() {
    vector_fetch_0_data_params.ResetRead();
    vector_fetch_0_packed_bits.ResetRead();
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
      VectorParams params = vector_fetch_0_data_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_ends[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_ends[i][j] = params.vector_fetch_0_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y0 =
          params.vector_fetch_0_loops[1][params.vector_fetch_0_y_loop_idx[1]] -
          1;
      ac_int<LOOP_WIDTH, false> X0 =
          params.vector_fetch_0_loops[1][params.vector_fetch_0_x_loop_idx[1]] -
          1;
      ac_int<LOOP_WIDTH, false> K1 =
          params.vector_fetch_0_loops[1][params.vector_fetch_0_k_loop_idx[1]] -
          1;

      if (params.has_transpose) {
        VectorType buffer[BUFSIZE][OcDimension];

#ifndef __SYNTHESIS__
        assert(params.vector_fetch_0_loops[1][2] == OcDimension);
#endif

        ac_int<32, false> total_values = params.vector_fetch_0_loops[0][0] *
                                         params.vector_fetch_0_loops[0][1] *
                                         params.vector_fetch_0_loops[0][2] *
                                         params.vector_fetch_0_loops[1][0] *
                                         params.vector_fetch_0_loops[1][1];
        ac_int<32, false> counter = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      TRANSPOSE_OUTER:
        while (counter++ < total_values) {
        TRANSPOSE_READ:
          for (ac_int<10, false> col = 0;; col++) {
            auto bits = vector_fetch_0_packed_bits.Pop();

            Pack1D<VectorType, Width> outputs;
            bool found = (unpack_vector_data<InputTypes, VectorType, Width,
                                             MAX_RESPONSE_WIDTH, InputTypes...>(
                              params.vector_fetch_0_dtype, bits, outputs, 0) ||
                          ...);

#ifndef __SYNTHESIS__
            if (!found) {
              std::cerr << "Error: vector fetch 0 input dtype '"
                        << params.vector_fetch_0_dtype << "' is not valid.\n";
            }
#endif

            Pack1D<VectorType, Width> dequantized;
            vdequantize<VectorType, VectorType, Width>(
                outputs, dequantized, params.vector_fetch_0_dq_scale);

            // We may not use all the data in the response
#pragma hls_unroll yes
            for (int row = 0; row < BUFSIZE; row++) {
              buffer[row][col] = dequantized[row];
            }

            if (col == OcDimension - 1) {
              break;
            }
          }

        TRANSPOSE_WRITE:
          for (ac_int<10, false> row = 0;; row++) {
            Pack1D<VectorType, OcDimension> transposed;
#pragma hls_unroll yes
            for (int col = 0; col < OcDimension; col++) {
              transposed[col] = buffer[row][col];
            }

            if constexpr (OcDimension == Width) {
              vector_fetch_0_data.Push(transposed);
            } else {
              for (ac_int<4, false> pack = 0;; pack++) {
                Pack1D<VectorType, Width> unpacked;
#pragma hls_unroll yes
                for (int i = 0; i < Width; i++) {
                  unpacked[i] = transposed[pack * Width + i];
                }
                vector_fetch_0_data.Push(unpacked);

                if (pack == OcDimension / Width - 1) {
                  break;
                }
              }
            }

            if (row == BUFSIZE - 1) {
              break;
            }
          }
        }
      }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      else if (params.vector_fetch_0_mode == 3) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
              for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                  for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                    auto read_data =
                        accumulation_buffer_read_data[accumulation_buffer_bank]
                            .Pop();
                    for (int i = 0; i < OcDimension / Width; i++) {
                      Pack1D<BufferType, Width> output_data;
#pragma hls_unroll yes
                      for (int j = 0; j < Width; j++) {
                        output_data[j] = read_data[i * Width + j];
                      }
                      accumulation_buffer_output.Push(output_data);
                    }

                    ac_int<LOOP_WIDTH, false> x0 =
                        loop_counters[1][params.vector_fetch_0_x_loop_idx[1]];
                    ac_int<LOOP_WIDTH, false> y0 =
                        loop_counters[1][params.vector_fetch_0_y_loop_idx[1]];
                    ac_int<LOOP_WIDTH, false> k1 =
                        loop_counters[1][params.vector_fetch_0_k_loop_idx[1]];

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
          ac_int<2, false> in_bound = vector_fetch_0_data_done.read();
          if (in_bound[1]) {
            break;
          }

          ac_int<MAX_RESPONSE_WIDTH, false> bits;
          if (in_bound[0]) {
            bits = vector_fetch_0_packed_bits.Pop();
          }

          for (ac_int<4, false> i = 0;; i++) {
            Pack1D<VectorType, Width> outputs;
#pragma hls_unroll yes
            for (int i = 0; i < Width; i++) {
              outputs[i] =
                  params.is_maxpool ? VectorType::min() : VectorType::zero();
            }

            if (in_bound[0]) {
              bool found =
                  (unpack_vector_data<InputTypes, VectorType, Width,
                                      MAX_RESPONSE_WIDTH, InputTypes...>(
                       params.vector_fetch_0_dtype, bits, outputs, i) ||
                   ...);

#ifndef __SYNTHESIS__
              if (!found) {
                std::cerr << "Error: vector fetch 0 input dtype '"
                          << params.vector_fetch_0_dtype << "' is not valid.\n";
              }
#endif
            }

#if !SUPPORT_MX
            Pack1D<VectorType, Width> dequantized;
            vdequantize<VectorType, VectorType, Width>(
                outputs, dequantized, params.vector_fetch_0_dq_scale);
            vector_fetch_0_data.Push(dequantized);
#else
            vector_fetch_0_data.Push(outputs);
#endif

            if (i == params.vector_fetch_0_packing_factor - 1) {
              break;
            }
          }
        }
      }
    }
  }

  void send_request_1() {
    vector_fetch_1_params.ResetRead();
    vector_fetch_1_req.Reset();

    wait();

    while (true) {
      VectorParams params = vector_fetch_1_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_ends[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_ends[i][j] = params.vector_fetch_1_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y1 =
          params.vector_fetch_1_loops[0][params.vector_fetch_1_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.vector_fetch_1_loops[0][params.vector_fetch_1_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.vector_fetch_1_loops[0][params.vector_fetch_1_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.vector_fetch_1_loops[1][params.vector_fetch_1_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.vector_fetch_1_loops[1][params.vector_fetch_1_x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.vector_fetch_1_loops[1][params.vector_fetch_1_k_loop_idx[1]];

      ac_int<16, false> X = X1 * X0;
      ac_int<16, false> k_stride = Width * params.vector_fetch_1_packing_factor;
      ac_int<16, false> K = K2 * K1 * k_stride;

      if (params.vector_fetch_1_broadcast[1]) {
        X = 1;
      }

      if (params.vector_fetch_1_broadcast[2]) {
        K = k_stride;
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
                      loop_counters[1][params.vector_fetch_1_x_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> x1 =
                      loop_counters[0][params.vector_fetch_1_x_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> y0 =
                      loop_counters[1][params.vector_fetch_1_y_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> y1 =
                      loop_counters[0][params.vector_fetch_1_y_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> k0 =
                      loop_counters[1][params.vector_fetch_1_k_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> k1 =
                      loop_counters[0][params.vector_fetch_1_k_loop_idx[0]];

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> k = (k1 * K1 + k0) * k_stride;

                  if (params.vector_fetch_1_broadcast[0]) {
                    y = 0;
                  }

                  if (params.vector_fetch_1_broadcast[1]) {
                    x = 0;
                  }

                  if (params.vector_fetch_1_broadcast[2]) {
                    k = 0;
                  }

                  ac_int<32, false> address = y * X * K + x * K + k;

                  fetch_vector_input<InputTypes...>(
                      params.vector_fetch_1_dtype, params.vector_fetch_1_offset,
                      address, params.vector_fetch_1_burst_size,
                      vector_fetch_1_req);
                  vector_fetch_1_packer_done.write(false);
                  vector_fetch_1_data_done.write(false);

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
      vector_fetch_1_packer_done.write(true);
      vector_fetch_1_data_done.write(true);
    }
  }

  void pack_response_1() {
    vector_fetch_1_packer_params.ResetRead();
    vector_fetch_1_resp.Reset();
    vector_fetch_1_packed_bits.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vector_fetch_1_packer_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!vector_fetch_1_packer_done.read()) {
        ac_int<MAX_RESPONSE_WIDTH, false> packed_bits;

        for (ac_int<4, false> i = 0;; i++) {
          packed_bits.set_slc(i * OC_PORT_WIDTH, vector_fetch_1_resp.Pop());
          if (i == params.vector_fetch_1_num_beats - 1) {
            break;
          }
        }

        vector_fetch_1_packed_bits.Push(packed_bits);
      }
    }
  }

  void feed_data_1() {
    vector_fetch_1_data_params.ResetRead();
    vector_fetch_1_packed_bits.ResetRead();
    vector_fetch_1_data.Reset();

    wait();

    while (true) {
      VectorParams params = vector_fetch_1_data_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!vector_fetch_1_data_done.read()) {
        ac_int<MAX_RESPONSE_WIDTH, false> bits =
            vector_fetch_1_packed_bits.Pop();

        for (ac_int<4, false> i = 0;; i++) {
          Pack1D<VectorType, Width> outputs;
          bool found = (unpack_vector_data<InputTypes, VectorType, Width,
                                           MAX_RESPONSE_WIDTH, InputTypes...>(
                            params.vector_fetch_1_dtype, bits, outputs, i) ||
                        ...);

#ifndef __SYNTHESIS__
          if (!found) {
            std::cerr << "Error: vector fetch 1 input dtype '"
                      << params.vector_fetch_1_dtype << "' is not valid.\n";
          }
#endif

#if !SUPPORT_MX
          Pack1D<VectorType, Width> dequantized;
          vdequantize<VectorType, VectorType, Width>(
              outputs, dequantized, params.vector_fetch_1_dq_scale);
          vector_fetch_1_data.Push(dequantized);
#else
          vector_fetch_1_data.Push(outputs);
#endif

          if (i == params.vector_fetch_1_packing_factor - 1) {
            break;
          }
        }
      }
    }
  }

  void send_request_2() {
    vector_fetch_2_params.ResetRead();
    vector_fetch_2_req.Reset();

    wait();

    while (true) {
      VectorParams params = vector_fetch_2_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_ends[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_ends[i][j] = params.vector_fetch_2_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y1 =
          params.vector_fetch_2_loops[0][params.vector_fetch_2_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.vector_fetch_2_loops[0][params.vector_fetch_2_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.vector_fetch_2_loops[0][params.vector_fetch_2_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.vector_fetch_2_loops[1][params.vector_fetch_2_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.vector_fetch_2_loops[1][params.vector_fetch_2_x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.vector_fetch_2_loops[1][params.vector_fetch_2_k_loop_idx[1]];

      ac_int<16, false> X = X1 * X0;
      ac_int<16, false> k_stride = Width * params.vector_fetch_2_packing_factor;
      ac_int<16, false> K = K2 * K1 * k_stride;

      if (params.vector_fetch_2_broadcast[1]) {
        X = 1;
      }

      if (params.vector_fetch_2_broadcast[2]) {
        K = k_stride;
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
                      loop_counters[1][params.vector_fetch_2_x_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> x1 =
                      loop_counters[0][params.vector_fetch_2_x_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> y0 =
                      loop_counters[1][params.vector_fetch_2_y_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> y1 =
                      loop_counters[0][params.vector_fetch_2_y_loop_idx[0]];
                  ac_int<LOOP_WIDTH, false> k0 =
                      loop_counters[1][params.vector_fetch_2_k_loop_idx[1]];
                  ac_int<LOOP_WIDTH, false> k1 =
                      loop_counters[0][params.vector_fetch_2_k_loop_idx[0]];

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> k = (k1 * K1 + k0) * k_stride;

                  if (params.vector_fetch_2_broadcast[0]) {
                    y = 0;
                  }

                  if (params.vector_fetch_2_broadcast[1]) {
                    x = 0;
                  }

                  if (params.vector_fetch_2_broadcast[2]) {
                    k = 0;
                  }

                  ac_int<32, false> address = y * X * K + x * K + k;

                  fetch_vector_input<InputTypes...>(
                      params.vector_fetch_2_dtype, params.vector_fetch_2_offset,
                      address, params.vector_fetch_2_burst_size,
                      vector_fetch_2_req);
                  vector_fetch_2_packer_done.write(false);
                  vector_fetch_2_data_done.write(false);

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
      vector_fetch_2_packer_done.write(true);
      vector_fetch_2_data_done.write(true);
    }
  }

  void pack_response_2() {
    vector_fetch_2_packer_params.ResetRead();
    vector_fetch_2_resp.Reset();
    vector_fetch_2_packed_bits.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vector_fetch_2_packer_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!vector_fetch_2_packer_done.read()) {
        ac_int<MAX_RESPONSE_WIDTH, false> packed_bits;
        for (ac_int<4, false> i = 0;; i++) {
          packed_bits.set_slc(i * OC_PORT_WIDTH, vector_fetch_2_resp.Pop());
          if (i == params.vector_fetch_2_num_beats - 1) {
            break;
          }
        }
        vector_fetch_2_packed_bits.Push(packed_bits);
      }
    }
  }

  void feed_data_2() {
    vector_fetch_2_data_params.ResetRead();
    vector_fetch_2_packed_bits.ResetRead();
    vector_fetch_2_data.Reset();

    wait();

    while (true) {
      VectorParams params = vector_fetch_2_data_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!vector_fetch_2_data_done.read()) {
        ac_int<MAX_RESPONSE_WIDTH, false> bits =
            vector_fetch_2_packed_bits.Pop();

        for (ac_int<4, false> i = 0;; i++) {
          Pack1D<VectorType, Width> outputs;
          bool found = (unpack_vector_data<InputTypes, VectorType, Width,
                                           MAX_RESPONSE_WIDTH, InputTypes...>(
                            params.vector_fetch_2_dtype, bits, outputs, i) ||
                        ...);

#ifndef __SYNTHESIS__
          if (!found) {
            std::cerr << "Error: vector fetch 2 input dtype '"
                      << params.vector_fetch_2_dtype << "' is not valid.\n";
          }
#endif

#if !SUPPORT_MX
          Pack1D<VectorType, Width> dequantized;
          vdequantize<VectorType, VectorType, Width>(
              outputs, dequantized, params.vector_fetch_2_dq_scale);
          vector_fetch_2_data.Push(dequantized);
#else
          vector_fetch_2_data.Push(outputs);
#endif

          if (i == params.vector_fetch_2_packing_factor - 1) {
            break;
          }
        }
      }
    }
  }

  void read_params() {
    params_in.Reset();
    vector_fetch_0_params.ResetWrite();
    vector_fetch_1_params.ResetWrite();
    vector_fetch_2_params.ResetWrite();
    vector_fetch_0_packer_params.ResetWrite();
    vector_fetch_1_packer_params.ResetWrite();
    vector_fetch_2_packer_params.ResetWrite();
    vector_fetch_0_data_params.ResetWrite();
    vector_fetch_1_data_params.ResetWrite();
    vector_fetch_2_data_params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = params_in.Pop();

      if (params.vector_fetch_0_mode != 0) {
        vector_fetch_0_params.Push(params);
        vector_fetch_0_packer_params.Push(params);
        vector_fetch_0_data_params.Push(params);
      }

      if (params.vector_fetch_1_mode != 0) {
        vector_fetch_1_params.Push(params);
        vector_fetch_1_packer_params.Push(params);
        vector_fetch_1_data_params.Push(params);
      }

      if (params.vector_fetch_2_mode != 0) {
        vector_fetch_2_params.Push(params);
        vector_fetch_2_packer_params.Push(params);
        vector_fetch_2_data_params.Push(params);
      }
    }
  }
};
