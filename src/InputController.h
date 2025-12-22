#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "Utils.h"

template <typename InputTypeTuple, int rows, int port_width, int buffer_width>
struct InputController;

template <typename... InputTypes, int rows, int port_width, int buffer_width>
struct InputController<std::tuple<InputTypes...>, rows, port_width,
                       buffer_width> : public sc_module {
  static constexpr int LOOP_WIDTH = 10;
  static constexpr int DATA_WIDTH = buffer_width / rows;
  static constexpr int MAX_FETCH_WIDTH = std::max(
      {dtype_fetch_config<InputTypes, rows, port_width>::max_fetch_width...});

  // num x values packed in a word for replication
  static constexpr int packing_factor = (rows == 4)    ? 1
                                        : (rows == 8)  ? 2
                                        : (rows == 16) ? 4
                                        : (rows == 32) ? 8
                                        : (rows == 64) ? 8
                                                       : 0;

  // num words needed to store the boundary pixels. essentially
  // ceil(3/packing_factor)
  static constexpr int boundary_words = (rows == 4)    ? 3
                                        : (rows == 8)  ? 2
                                        : (rows == 16) ? 1
                                        : (rows == 32) ? 1
                                        : (rows == 64) ? 1
                                                       : 0;

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_resp);
  sc_fifo<bool> fetcher_done;
  sc_fifo<bool> fetcher_done_2;

  Connections::Out<BufferWriteRequest<ac_int<buffer_width, false>>>
      write_request[2];
  Connections::Out<BufferReadRequest> read_request[2];

  Connections::In<ac_int<buffer_width, false>> CCS_INIT_S1(window_buffer_in);
  Connections::Out<ac_int<buffer_width, false>> CCS_INIT_S1(window_buffer_out);

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(reader_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(winder_buffer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(input_packer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposer_params);

  Connections::Combinational<ac_int<MAX_FETCH_WIDTH, false>> packed_bits;
  Connections::Combinational<ac_int<buffer_width, false>> transpose_out;

  SC_CTOR(InputController) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetcher);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(reader);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(writer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(window_buffer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(input_packer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(transposer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetcher() {
    fetcher_params.ResetRead();
    input_req.Reset();

    wait();

    while (true) {
      const MatrixParams params = fetcher_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weight_loop_idx[1]] = 1;
      loop_bounds[1][params.fx_loop_idx] = 1;
      loop_bounds[1][params.fy_loop_idx[1]] = 1;

      ac_int<LOOP_WIDTH, false> Y1 = params.loops[0][params.y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 = params.loops[0][params.x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> C2 =
          params.loops[0][params.reduction_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reduction_loop_idx[1]];
      ac_int<16, false> Y0 = params.loops[1][params.y_loop_idx[1]];
      ac_int<16, false> X0 = params.loops[1][params.x_loop_idx[1]];
      ac_int<4, false> FX = params.loops[1][params.fx_loop_idx];
      ac_int<4, false> FY0 = params.loops[1][params.fy_loop_idx[1]];
      ac_int<8, false> STRIDE = params.stride;

      if (params.is_resnet_replication) {
        FX = 7;
      }

      ac_int<16, false> IY0 = Y0 * STRIDE;
      ac_int<16, false> IX0 = X0 * STRIDE;

      loop_bounds[1][params.y_loop_idx[1]] = (FY0 == 1 ? Y0 : IY0) + FY0 - 1;

      if (params.is_resnet_replication) {
        loop_bounds[1][params.x_loop_idx[1]] =
            IX0 / packing_factor + 2 * boundary_words;
      } else if (params.is_generic_replication) {
        loop_bounds[1][params.x_loop_idx[1]] = IX0 >> params.fx_unrolling_lg2;
      } else {
        loop_bounds[1][params.x_loop_idx[1]] = (FX == 1 ? X0 : IX0) + FX - 1;
      }

      // reduce the number of iterations by packing factor
      C1 = C1 >> params.input_pack_factor_lg2;
      loop_bounds[1][params.reduction_loop_idx[1]] = C1;

      ac_int<16, false> Y = params.input_y;
      ac_int<16, false> X = params.input_x;
      ac_int<16, false> c_stride = rows << params.input_pack_factor_lg2;
      ac_int<16, false> C = C2 * C1 * c_stride;

      if (params.is_generic_replication) {
        Y = Y1 * IY0;
        X = X1 * IX0;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          for (loop_counters[1][5] = 0;;
                               loop_counters[1][5]++) {
                            ac_int<LOOP_WIDTH, false> y1 =
                                loop_counters[0][params.y_loop_idx[0]];
                            ac_int<LOOP_WIDTH, false> x1 =
                                loop_counters[0][params.x_loop_idx[0]];
                            ac_int<LOOP_WIDTH, false> c2 =
                                loop_counters[0][params.reduction_loop_idx[0]];
                            ac_int<LOOP_WIDTH, false> y0 =
                                loop_counters[1][params.y_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> x0 =
                                loop_counters[1][params.x_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> c1 =
                                loop_counters[1][params.reduction_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> fy1 =
                                loop_counters[0][params.fy_loop_idx[0]];

                            // adjust address for stride
                            if (FX == 1) {
                              x0 = x0 * STRIDE;
                            }
                            if (FY0 == 1) {
                              y0 = y0 * STRIDE + fy1;
                            }

                            if (params.is_resnet_replication) {
                              if (x0 != 0 && x1 != 0) {
                                x0 = (x0 - boundary_words) * packing_factor +
                                     params.padding;
                              } else {
                                x0 = x0 * packing_factor;
                              }
                            } else if (params.is_generic_replication) {
                              x0 = x0 << params.fx_unrolling_lg2;
                            }

                            ac_int<16, true> y = y1 * IY0 + y0 - params.padding;
                            ac_int<16, true> x = x1 * IX0 + x0 - params.padding;
                            ac_int<16, false> c = (c2 * C1 + c1) * c_stride;

                            if (x >= 0 && y >= 0 && x < X && y < Y) {
                              ac_int<32, false> address = y * X * C + x * C + c;

                              if (params.is_resnet_replication) {
                                address = y * (X / packing_factor) * rows +
                                          (x / packing_factor) * rows + c;
                              } else if (params.is_generic_replication) {
                                address =
                                    y * (X >> params.fx_unrolling_lg2) * rows +
                                    (x >> params.fx_unrolling_lg2) * rows + c;
                              } else if (params.merge_heads) {
                                // (c / head_size) * (X * head_size) + (x *
                                // head_size) + (c % head_size)
                                ac_int<16, false> mask =
                                    (1 << params.head_size_lg2) - 1;
                                address = (c >> params.head_size_lg2) *
                                              (X << params.head_size_lg2) +
                                          (x << params.head_size_lg2) +
                                          (c & mask);
                              } else if (params.input_transpose) {
                                address =
                                    (c + (x % rows)) * X + (x / rows) * rows;
                              }

                              send_packed_request<InputTypes...>(
                                  params.input_dtype, params.input_offset,
                                  address, params.input_burst_size, input_req);
                              fetcher_done.write(false);

                              if (!params.input_transpose) {
                                fetcher_done_2.write(false);
                              }
                            }

                            if (loop_counters[1][5] == loop_bounds[1][5] - 1)
                              break;
                          }
                          if (loop_counters[1][4] == loop_bounds[1][4] - 1)
                            break;
                        }
                        if (loop_counters[1][3] == loop_bounds[1][3] - 1) break;
                      }
                      if (loop_counters[1][2] == loop_bounds[1][2] - 1) break;
                    }
                    if (loop_counters[1][1] == loop_bounds[1][1] - 1) break;
                  }
                  if (loop_counters[1][0] == loop_bounds[1][0] - 1) break;
                }
                if (loop_counters[0][4] == loop_bounds[0][4] - 1) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3] - 1) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2] - 1) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1] - 1) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0] - 1) break;
      }

      fetcher_done.write(true);
      fetcher_done_2.write(true);
    }
  }

  void writer() {
    writer_params.ResetRead();
    transpose_out.ResetRead();
    write_request[0].Reset();
    write_request[1].Reset();

    bool bank_sel = 0;

    wait();

    while (true) {
      const MatrixParams params = writer_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weight_loop_idx[1]] = 1;
      loop_bounds[1][params.fx_loop_idx] = 1;
      loop_bounds[1][params.fy_loop_idx[1]] = 1;

      ac_int<LOOP_WIDTH, false> Y1 = params.loops[0][params.y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 = params.loops[0][params.x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reduction_loop_idx[1]];
      ac_int<16, false> Y0 = params.loops[1][params.y_loop_idx[1]];
      ac_int<16, false> X0 = params.loops[1][params.x_loop_idx[1]];
      ac_int<4, false> FX = params.loops[1][params.fx_loop_idx];
      ac_int<4, false> FY0 = params.loops[1][params.fy_loop_idx[1]];
      ac_int<8, false> STRIDE = params.stride;

      if (params.is_resnet_replication) {
        FX = 7;
      }

      ac_int<16, false> IX0 = X0;
      ac_int<16, false> IY0 = Y0;

      if (FX != 1) {
        IX0 = X0 * STRIDE;
      }

      if (FY0 != 1) {
        IY0 = Y0 * STRIDE;
      }

      loop_bounds[1][params.y_loop_idx[1]] = IY0 + FY0 - 1;

      if (params.is_resnet_replication) {
        loop_bounds[1][params.x_loop_idx[1]] =
            IX0 / packing_factor + 2 * boundary_words;
      } else if (params.is_generic_replication) {
        loop_bounds[1][params.x_loop_idx[1]] = IX0 >> params.fx_unrolling_lg2;
      } else {
        loop_bounds[1][params.x_loop_idx[1]] = IX0 + FX - 1;
      }

      // reduce the number of iterations by packing factor
      C1 = C1 >> params.input_pack_factor_lg2;
      loop_bounds[1][params.reduction_loop_idx[1]] = C1;
      ac_int<4, false> pack_offset_bound =
          (1 << params.input_pack_factor_lg2) - 1;

      ac_int<16, false> Y = params.input_y;
      ac_int<16, false> X = params.input_x;
      ac_int<16, false> y_stride = loop_bounds[1][params.x_loop_idx[1]] * C1;

      if (params.is_generic_replication) {
        Y = Y1 * IY0;
        X = X1 * IX0;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          for (loop_counters[1][5] = 0;;
                               loop_counters[1][5]++) {
                            for (ac_int<4, false> pack = 0;; pack++) {
                              ac_int<LOOP_WIDTH, true> y1 =
                                  loop_counters[0][params.y_loop_idx[0]];
                              ac_int<LOOP_WIDTH, true> x1 =
                                  loop_counters[0][params.x_loop_idx[0]];
                              ac_int<LOOP_WIDTH, true> y0 =
                                  loop_counters[1][params.y_loop_idx[1]];
                              ac_int<LOOP_WIDTH, true> x0 =
                                  loop_counters[1][params.x_loop_idx[1]];
                              ac_int<LOOP_WIDTH, true> c1 =
                                  loop_counters[1]
                                               [params.reduction_loop_idx[1]];

                              if (params.is_resnet_replication && x0 != 0) {
                                x0 = (x0 - boundary_words) * packing_factor +
                                     params.padding;
                              }

                              ac_int<16, true> x =
                                  x1 * IX0 + x0 - params.padding;
                              ac_int<16, true> y =
                                  y1 * IY0 + y0 - params.padding;

                              ac_int<buffer_width, false> data;
                              if (x < 0 || y < 0 || x >= X || y >= Y) {
                                set_zero<rows, buffer_width, InputTypes...>(
                                    params.input_dtype, data,
                                    params.use_input_codebook,
                                    params.input_code_zero_idx);
                              } else {
                                data = transpose_out.Pop();
                              }

                              ac_int<LOOP_WIDTH> orig_x0 =
                                  loop_counters[1][params.x_loop_idx[1]];
                              ac_int<16, false> address =
                                  y0 * y_stride + orig_x0 * C1 + c1;
                              address =
                                  (address << params.input_pack_factor_lg2) +
                                  pack;

                              bool is_last = loop_counters[1][5] ==
                                                 loop_bounds[1][5] - 1 &&
                                             loop_counters[1][4] ==
                                                 loop_bounds[1][4] - 1 &&
                                             loop_counters[1][3] ==
                                                 loop_bounds[1][3] - 1 &&
                                             loop_counters[1][2] ==
                                                 loop_bounds[1][2] - 1 &&
                                             loop_counters[1][1] ==
                                                 loop_bounds[1][1] - 1 &&
                                             loop_counters[1][0] ==
                                                 loop_bounds[1][0] - 1 &&
                                             pack == pack_offset_bound;

                              BufferWriteRequest<ac_int<buffer_width, false>>
                                  req;
                              req.address = address;
                              req.data = data;
                              req.last = is_last;
                              write_request[bank_sel].Push(req);

                              if (pack == pack_offset_bound) break;
                            }
                            if (loop_counters[1][5] == loop_bounds[1][5] - 1)
                              break;
                          }
                          if (loop_counters[1][4] == loop_bounds[1][4] - 1)
                            break;
                        }
                        if (loop_counters[1][3] == loop_bounds[1][3] - 1) break;
                      }
                      if (loop_counters[1][2] == loop_bounds[1][2] - 1) break;
                    }
                    if (loop_counters[1][1] == loop_bounds[1][1] - 1) break;
                  }
                  if (loop_counters[1][0] == loop_bounds[1][0] - 1) break;
                }
                bank_sel = !bank_sel;
                if (loop_counters[0][4] == loop_bounds[0][4] - 1) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3] - 1) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2] - 1) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1] - 1) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0] - 1) break;
      }
    }
  }

  void reader() {
    reader_params.ResetRead();

    read_request[0].Reset();
    read_request[1].Reset();

    bool bank_sel = 0;

    wait();

    while (true) {
      const MatrixParams params = reader_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      if (params.is_resnet_replication && rows >= 16) {
        loop_bounds[1][params.x_loop_idx[1]] =
            (loop_bounds[1][params.x_loop_idx[1]] * params.stride /
             packing_factor) +
            2;
      } else if (params.is_resnet_replication && rows == 8) {
        loop_bounds[1][params.x_loop_idx[1]] =
            (loop_bounds[1][params.x_loop_idx[1]] * params.stride /
             packing_factor) +
            1;
      }

      ac_int<LOOP_WIDTH, false> Y0 = params.loops[1][params.y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reduction_loop_idx[1]];
      ac_int<4, false> FX = params.loops[1][params.fx_loop_idx];
      ac_int<4, false> FY0 = params.loops[1][params.fy_loop_idx[1]];
      ac_int<8, false> STRIDE = params.stride;

      bool is_downsample = FX == 1 && FY0 == 1;

      if (is_downsample) {
        STRIDE = 1;
      }

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> x_bound;
      if (params.is_resnet_replication) {
        x_bound = IX0 / packing_factor + 2 * boundary_words;
      } else {
        x_bound = IX0 + FX - 1;
      }
      ac_int<16, false> y_stride = C1 * x_bound;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          for (loop_counters[1][5] = 0;;
                               loop_counters[1][5]++) {
                            ac_int<LOOP_WIDTH, false> x0 =
                                loop_counters[1][params.x_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> y0 =
                                loop_counters[1][params.y_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> fx =
                                loop_counters[1][params.fx_loop_idx];
                            ac_int<LOOP_WIDTH, false> fy0 =
                                loop_counters[1][params.fy_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> c1 =
                                loop_counters[1][params.reduction_loop_idx[1]];

                            ac_int<16, false> x = STRIDE * x0 + fx;
                            ac_int<16, false> y = STRIDE * y0 + fy0;
                            ac_int<16, false> address;
                            if (params.is_resnet_replication && rows >= 8) {
                              address = y * y_stride + (x0 + fx) * C1 + c1;
                            } else if (params.is_generic_replication) {
                              address =
                                  (y0 + fy0) *
                                      ((STRIDE * X0) >>
                                       params.fx_unrolling_lg2) *
                                      C1 +
                                  (((STRIDE * x0) >> params.fx_unrolling_lg2) +
                                   fx) *
                                      C1 +
                                  c1;
                            } else if (is_downsample) {
                              address = y0 * X0 * C1 + x0 * C1 + c1;
                            } else {
                              address = y * y_stride + x * C1 + c1;
                            }

                            bool is_last =
                                loop_counters[1][5] == loop_bounds[1][5] - 1 &&
                                loop_counters[1][4] == loop_bounds[1][4] - 1 &&
                                loop_counters[1][3] == loop_bounds[1][3] - 1 &&
                                loop_counters[1][2] == loop_bounds[1][2] - 1 &&
                                loop_counters[1][1] == loop_bounds[1][1] - 1 &&
                                loop_counters[1][0] == loop_bounds[1][0] - 1;

                            BufferReadRequest req = {
                                .address = address,
                                .last = is_last,
                            };
                            read_request[bank_sel].Push(req);

                            if (loop_counters[1][5] == loop_bounds[1][5] - 1)
                              break;
                          }
                          if (loop_counters[1][4] == loop_bounds[1][4] - 1)
                            break;
                        }
                        if (loop_counters[1][3] == loop_bounds[1][3] - 1) break;
                      }
                      if (loop_counters[1][2] == loop_bounds[1][2] - 1) break;
                    }
                    if (loop_counters[1][1] == loop_bounds[1][1] - 1) break;
                  }
                  if (loop_counters[1][0] == loop_bounds[1][0] - 1) break;
                }
                bank_sel = !bank_sel;
                if (loop_counters[0][4] == loop_bounds[0][4] - 1) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3] - 1) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2] - 1) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1] - 1) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0] - 1) break;
      }
    }
  }

  /*
   * Used for FX replication with stride=2
   * Window buffer cuts down on memory accesses
   */
  void window_buffer() {
    winder_buffer_params.ResetRead();
    window_buffer_in.Reset();
    window_buffer_out.Reset();

    wait();

    while (true) {
      const MatrixParams params = winder_buffer_params.Pop();
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // num x values packed in a word sent to the systolic array
      constexpr int unroll = (rows == 4)                  ? 1
                             : (rows == 8)                ? 2
                             : (rows == 16)               ? 4
                             : (rows == 32 || rows == 64) ? 7
                                                          : 0;

      // additional x values packed in a word sent to the systolic array, but
      // are unused by the systolic array
      constexpr int additional = (rows == 16) ? 1 : 0;

      if (params.is_resnet_replication &&
          (rows == 16 || rows == 32 || rows == 64)) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
              for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
                // inner memory
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          ac_int<buffer_width, false> data;
                          ac_int<buffer_width, false> buffer;

                          for (loop_counters[1][5] = 0;;
                               loop_counters[1][5]++) {
                            int buffer_pos;
                            if (loop_counters[1][5] == 1) {
#pragma hls_unroll yes
                              for (int x = 0; x < 3; x++) {
#pragma hls_unroll yes
                                for (int dim = 0; dim < 3; dim++) {
                                  int dst_idx = x * 3 + dim;
                                  int src_idx =
                                      (packing_factor - 3 + x) * 3 + dim;

                                  auto bits = buffer.template slc<DATA_WIDTH>(
                                      src_idx * DATA_WIDTH);
                                  data.set_slc(dst_idx * DATA_WIDTH, bits);
                                }
                              }
                            } else if (loop_counters[1][5] == 2) {
#pragma hls_unroll yes
                              for (int x = 3; x < unroll + additional; x++) {
#pragma hls_unroll yes
                                for (int dim = 0; dim < 3; dim++) {
                                  int dst_idx = x * 3 + dim;
                                  int src_idx = (x - 3) * 3 + dim;

                                  auto bits = buffer.template slc<DATA_WIDTH>(
                                      src_idx * DATA_WIDTH);
                                  data.set_slc(dst_idx * DATA_WIDTH, bits);
                                }
                              }
                            } else if (loop_counters[1][5] > 2) {
                              // shift two pixels over
#pragma hls_unroll yes
                              for (int x = 0;
                                   x < packing_factor + additional - 2; x++) {
#pragma hls_unroll yes
                                for (int dim = 0; dim < 3; dim++) {
                                  int dst_idx = x * 3 + dim;
                                  int src_idx = (x + 2) * 3 + dim;

                                  auto bits = data.template slc<DATA_WIDTH>(
                                      src_idx * DATA_WIDTH);
                                  data.set_slc(dst_idx * DATA_WIDTH, bits);
                                }
                              }

                              // grab 2 new pixels from buffer
                              buffer_pos = ((loop_counters[1][5] - 3) * 2 +
                                            unroll + additional - 3) %
                                           packing_factor;
#pragma hls_unroll yes
                              for (int x = unroll + additional - 2;
                                   x < unroll + additional; x++) {
#pragma hls_unroll yes
                                for (int dim = 0; dim < 3; dim++) {
                                  int dst_idx = x * 3 + dim;
                                  int src_idx = (buffer_pos + x -
                                                 (unroll + additional - 2)) *
                                                    3 +
                                                dim;

                                  auto bits = buffer.template slc<DATA_WIDTH>(
                                      src_idx * DATA_WIDTH);
                                  data.set_slc(dst_idx * DATA_WIDTH, bits);
                                }
                              }
                            }

                            if (loop_counters[1][5] > 1) {
                              window_buffer_out.Push(data);
                            }

                            if (loop_counters[1][5] < 2 ||
                                (loop_counters[1][5] > 2 &&
                                 buffer_pos == packing_factor - 2)) {
                              buffer = window_buffer_in.Pop();
                            }

                            if (loop_counters[1][5] == loop_bounds[1][5] + 1) {
                              break;
                            }
                          }
                          if (loop_counters[1][4] == loop_bounds[1][4] - 1)
                            break;
                        }
                        if (loop_counters[1][3] == loop_bounds[1][3] - 1) break;
                      }
                      if (loop_counters[1][2] == loop_bounds[1][2] - 1) break;
                    }
                    if (loop_counters[1][1] == loop_bounds[1][1] - 1) break;
                  }
                  if (loop_counters[1][0] == loop_bounds[1][0] - 1) break;
                }
                if (loop_counters[0][3] == loop_bounds[0][3] - 1) break;
              }
              if (loop_counters[0][2] == loop_bounds[0][2] - 1) break;
            }
            if (loop_counters[0][1] == loop_bounds[0][1] - 1) break;
          }
          if (loop_counters[0][0] == loop_bounds[0][0] - 1) break;
        }
      } else if (params.is_resnet_replication && rows == 8) {
        // no window buffer reuse, but need to combine
        // multiple words together
        for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
              for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
                // inner memory
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          ac_int<buffer_width, false> buffer =
                              window_buffer_in.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
                          for (loop_counters[1][5] = 0;;
                               loop_counters[1][5]++) {
                            ac_int<buffer_width, false> data;
#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              auto bits = buffer.template slc<DATA_WIDTH>(
                                  (dim + 3) * DATA_WIDTH);
                              data.set_slc(dim * DATA_WIDTH, bits);
                            }

                            buffer = window_buffer_in.Pop();

#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              auto bits = buffer.template slc<DATA_WIDTH>(
                                  dim * DATA_WIDTH);
                              data.set_slc((3 + dim) * DATA_WIDTH, bits);
                            }

                            window_buffer_out.Push(data);

                            if (loop_counters[1][5] == loop_bounds[1][5] - 1)
                              break;
                          }
                          if (loop_counters[1][4] == loop_bounds[1][4] - 1)
                            break;
                        }
                        if (loop_counters[1][3] == loop_bounds[1][3] - 1) break;
                      }
                      if (loop_counters[1][2] == loop_bounds[1][2] - 1) break;
                    }
                    if (loop_counters[1][1] == loop_bounds[1][1] - 1) break;
                  }
                  if (loop_counters[1][0] == loop_bounds[1][0] - 1) break;
                }
                if (loop_counters[0][3] == loop_bounds[0][3] - 1) break;
              }
              if (loop_counters[0][2] == loop_bounds[0][2] - 1) break;
            }
            if (loop_counters[0][1] == loop_bounds[0][1] - 1) break;
          }
          if (loop_counters[0][0] == loop_bounds[0][0] - 1) break;
        }
      } else {  // bypass
        ac_int<32, false> total_count =
            loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
            loop_bounds[0][3] * loop_bounds[0][4] * loop_bounds[1][0] *
            loop_bounds[1][1] * loop_bounds[1][2] * loop_bounds[1][3] *
            loop_bounds[1][4] * loop_bounds[1][5];
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < total_count; i++) {
          window_buffer_out.Push(window_buffer_in.Pop());
        }
      }
    }
  }

  void input_packer() {
    input_packer_params.ResetRead();
    input_resp.Reset();
    packed_bits.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = input_packer_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!fetcher_done.read()) {
        ac_int<MAX_FETCH_WIDTH, false> bits;

        for (ac_int<4, false> i = 0;; i++) {
          bits.set_slc(i * port_width, input_resp.Pop());
          if (i == params.input_num_beats - 1) break;
        }

        packed_bits.Push(bits);
      }
    }
  }

  void transposer() {
    transposer_params.ResetRead();
    packed_bits.ResetRead();
    transpose_out.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = transposer_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weight_loop_idx[1]] = 1;
      loop_bounds[1][params.fx_loop_idx] = 1;
      loop_bounds[1][params.fy_loop_idx[1]] = 1;

      if (params.input_transpose && rows <= 32) {
        ac_int<DATA_WIDTH> transpose_buffer[rows][rows];

        ac_int<32, false> total_count =
            loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
            loop_bounds[0][3] * loop_bounds[1][0] * loop_bounds[1][1] *
            loop_bounds[1][2] * loop_bounds[1][3] * loop_bounds[1][4] *
            loop_bounds[1][5] / rows;

        ac_int<32, false> count = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (count++ < total_count) {
          for (int c0 = 0; c0 < rows; c0++) {
            auto bits = packed_bits.Pop();

            ac_int<buffer_width, false> outputs = 0;
            bool handled = (unpack_bits<InputTypes, rows, buffer_width,
                                        MAX_FETCH_WIDTH, InputTypes...>(
                                params.input_dtype, bits, outputs, 0) ||
                            ...);

#ifndef __SYNTHESIS__
            if (!handled) {
              throw std::runtime_error("Unsupported dtype for matrix input: " +
                                       std::to_string(params.input_dtype));
            }
#endif

#pragma hls_unroll yes
            for (int dim = 0; dim < rows; dim++) {
              transpose_buffer[dim][c0] =
                  outputs.template slc<DATA_WIDTH>(dim * DATA_WIDTH);
            }
          }

          // Write out from tranpose buffer
          for (int c0 = 0; c0 < rows; c0++) {
            ac_int<buffer_width, false> transposed;
#pragma hls_unroll yes
            for (int dim = 0; dim < rows; dim++) {
              transposed.set_slc(dim * DATA_WIDTH, transpose_buffer[c0][dim]);
            }
            transpose_out.Push(transposed);
          }
        }
      } else {  // passthrough
        ac_int<4, false> pack_offset_bound =
            (1 << params.input_pack_factor_lg2) - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (!fetcher_done_2.read()) {
          ac_int<MAX_FETCH_WIDTH, false> bits = packed_bits.Pop();

          // Unpack bits into outputs based on dtype
          for (ac_int<4, false> i = 0;; i++) {
            ac_int<buffer_width, false> outputs = 0;
            bool handled = (unpack_bits<InputTypes, rows, buffer_width,
                                        MAX_FETCH_WIDTH, InputTypes...>(
                                params.input_dtype, bits, outputs, i) ||
                            ...);

#ifndef __SYNTHESIS__
            if (!handled) {
              throw std::runtime_error("Unsupported dtype for matrix input: " +
                                       std::to_string(params.input_dtype));
            }
#endif

            transpose_out.Push(outputs);

            if (i == pack_offset_bound) break;
          }
        }
      }
    }
  }

  void read_params() {
    params_in.Reset();
    fetcher_params.ResetWrite();
    writer_params.ResetWrite();
    reader_params.ResetWrite();
    winder_buffer_params.ResetWrite();
    input_packer_params.ResetWrite();
    transposer_params.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();

      fetcher_params.Push(params);
      writer_params.Push(params);
      reader_params.Push(params);
      winder_buffer_params.Push(params);
      input_packer_params.Push(params);
      transposer_params.Push(params);
    }
  }
};
