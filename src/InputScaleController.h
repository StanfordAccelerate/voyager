#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename Scale, int rows>
SC_MODULE(InputScaleController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(scale_req);
  Connections::In<ac_int<Scale::width, false>> CCS_INIT_S1(scale_resp);

  Connections::Out<BufferWriteRequest<ac_int<Scale::width, false>>>
      write_request[2];
  Connections::Out<BufferReadRequest> read_request[2];

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(reader_params);

  static constexpr int LOOP_WIDTH = 10;

  // num x values packed in a word
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

  SC_CTOR(InputScaleController) {
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
  }

  void fetcher() {
    fetcher_params.ResetRead();
    scale_req.Reset();

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

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> IY0 = Y0 * STRIDE;

      loop_bounds[1][params.y_loop_idx[1]] = (FY0 == 1 ? Y0 : IY0) + FY0 - 1;
      loop_bounds[1][params.x_loop_idx[1]] = (FX == 1 ? X0 : IX0) + FX - 1;

      ac_int<16, false> Y = params.input_y;
      ac_int<16, false> X = params.input_x;
      ac_int<16, false> C = C2 * C1;

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

                            // adjust address for stride
                            if (FX == 1) {
                              x0 = x0 * STRIDE;
                            }
                            if (FY0 == 1) {
                              y0 = y0 * STRIDE;
                            }

                            ac_int<16, false> y =
                                y1 * IY0 + y0 - params.padding;
                            ac_int<16, false> x =
                                x1 * IX0 + x0 - params.padding;
                            ac_int<16, false> c = c2 * C1 + c1;

                            if (x >= 0 && y >= 0 && x < X && y < Y) {
                              ac_int<32, false> address = y * X * C + x * C + c;

                              if (params.merge_heads) {
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

                              send_input_request<Scale, 1>(
                                  params.input_scale_offset, address,
                                  scale_req);
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
    }
  }

  void writer() {
    writer_params.ResetRead();

    scale_resp.Reset();
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

      ac_int<16, false> IX0 = X0;
      ac_int<16, false> IY0 = Y0;

      if (FX != 1) {
        IX0 = X0 * STRIDE;
      }

      if (FY0 != 1) {
        IY0 = Y0 * STRIDE;
      }

      loop_bounds[1][params.y_loop_idx[1]] = IY0 + FY0 - 1;
      loop_bounds[1][params.x_loop_idx[1]] = IX0 + FX - 1;

      ac_int<16, false> Y = params.input_y;
      ac_int<16, false> X = params.input_x;
      ac_int<16, false> y_stride = loop_bounds[1][params.x_loop_idx[1]] * C1;

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
                            ac_int<LOOP_WIDTH, true> x0 =
                                loop_counters[1][params.x_loop_idx[1]];
                            ac_int<LOOP_WIDTH, true> x1 =
                                loop_counters[0][params.x_loop_idx[0]];
                            ac_int<LOOP_WIDTH, true> y0 =
                                loop_counters[1][params.y_loop_idx[1]];
                            ac_int<LOOP_WIDTH, true> y1 =
                                loop_counters[0][params.y_loop_idx[0]];
                            ac_int<LOOP_WIDTH, true> c1 =
                                loop_counters[1][params.reduction_loop_idx[1]];

                            ac_int<16, true> x = x1 * IX0 + x0 - params.padding;
                            ac_int<16, true> y = y1 * IY0 + y0 - params.padding;

                            ac_int<Scale::width, false> scale;
                            if (x < 0 || y < 0 || x >= X || y >= Y) {
                              scale = Scale::one().bits_rep();
                            } else {
                              scale = scale_resp.Pop();
                            }

                            ac_int<LOOP_WIDTH> orig_x0 =
                                loop_counters[1][params.x_loop_idx[1]];
                            ac_int<16, false> address =
                                y0 * y_stride + orig_x0 * C1 + c1;

                            bool is_last =
                                loop_counters[1][5] == loop_bounds[1][5] - 1 &&
                                loop_counters[1][4] == loop_bounds[1][4] - 1 &&
                                loop_counters[1][3] == loop_bounds[1][3] - 1 &&
                                loop_counters[1][2] == loop_bounds[1][2] - 1 &&
                                loop_counters[1][1] == loop_bounds[1][1] - 1 &&
                                loop_counters[1][0] == loop_bounds[1][0] - 1;

                            BufferWriteRequest<ac_int<Scale::width, false>>
                                scale_req;
                            scale_req.address = address;
                            scale_req.data = scale;
                            scale_req.last = is_last;
                            write_request[bank_sel].Push(scale_req);

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

      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.x_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> Y0 = params.loops[1][params.y_loop_idx[1]];
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
      ac_int<16, false> y_stride = C1 * (IX0 + FX - 1);

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
                            ac_int<LOOP_WIDTH, false> fy =
                                loop_counters[1][params.fy_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> c1 =
                                loop_counters[1][params.reduction_loop_idx[1]];

                            ac_int<16, false> x = STRIDE * x0 + fx;
                            ac_int<16, false> y = STRIDE * y0 + fy;
                            ac_int<16, false> address;
                            if (is_downsample) {
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

  void read_params() {
    params_in.Reset();
    fetcher_params.ResetWrite();
    writer_params.ResetWrite();
    reader_params.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();

      if (params.is_mx_op) {
        fetcher_params.Push(params);
        writer_params.Push(params);
        reader_params.Push(params);
      }
    }
  }
};
