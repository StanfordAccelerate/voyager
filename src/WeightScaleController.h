#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

template <typename Scale, int rows, int cols, int port_width>
SC_MODULE(WeightScaleController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_scale_resp);

  Connections::Out<BufferWriteRequest<ac_int<Scale::width * cols, false>>>
      write_request[2];
  Connections::Out<BufferReadRequest> read_request[2];

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(reader_params);

  static constexpr int LOOP_WIDTH = 10;
  static constexpr int BLOCK_SIZE = rows > cols ? rows : cols;

  SC_CTOR(WeightScaleController) {
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
    weight_scale_req.Reset();
    fetcher_params.ResetRead();

    wait();

    while (true) {
      const MatrixParams params = fetcher_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][5];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][5];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 5; j++) {
          loop_bounds[i][j] = params.weight_addr_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> K2 =
          params.weight_addr_loops[0][params.weight_addr_weight_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> C2 =
          params.weight_addr_loops[0][params.weight_addr_reduction_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.weight_addr_loops[1][params.weight_addr_reduction_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> C0 =
          params.weight_addr_loops[1][params.weight_addr_reduction_loop_idx[2]];
      ac_int<LOOP_WIDTH, false> FX =
          params.weight_addr_loops[1][params.weight_addr_fx_idx];
      ac_int<LOOP_WIDTH, false> FY0 =
          params.weight_addr_loops[1][params.weight_addr_fy_idx[1]];
      ac_int<LOOP_WIDTH, false> FY1 =
          params.weight_addr_loops[0][params.weight_addr_fy_idx[0]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.weight_addr_loops[1][params.weight_addr_weight_loop_idx[1]];

      ac_int<24, false> C = C1 * C0 / BLOCK_SIZE;
      ac_int<24, false> c_stride = K2 * K1 * cols;
      ac_int<24, false> fx_stride = C2 * C * c_stride;
      ac_int<24, false> fy_stride = FX * fx_stride;

      loop_bounds[1][params.weight_addr_reduction_loop_idx[1]] = 0;
      loop_bounds[1][params.weight_addr_reduction_loop_idx[2]] = C - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                // inner memory
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          ac_int<LOOP_WIDTH, false> k2 = loop_counters
                              [0][params.weight_addr_weight_loop_idx[0]];
                          ac_int<LOOP_WIDTH, false> c2 = loop_counters
                              [0][params.weight_addr_reduction_loop_idx[0]];
                          ac_int<LOOP_WIDTH, false> c0 = loop_counters
                              [1][params.weight_addr_reduction_loop_idx[2]];
                          ac_int<LOOP_WIDTH, false> fx =
                              loop_counters[1][params.weight_addr_fx_idx];
                          ac_int<LOOP_WIDTH, false> fy0 =
                              loop_counters[1][params.weight_addr_fy_idx[1]];
                          ac_int<LOOP_WIDTH, false> fy1 =
                              loop_counters[0][params.weight_addr_fy_idx[0]];
                          ac_int<LOOP_WIDTH, false> k1 = loop_counters
                              [1][params.weight_addr_weight_loop_idx[1]];

                          ac_int<16, false> k = (k2 * K1 + k1) * cols;
                          ac_int<16, false> c = c2 * C + c0;
                          ac_int<16, false> fy = fy0 * FY1 + fy1;

                          ac_int<32, false> address = fy * fy_stride +
                                                      fx * fx_stride +
                                                      c * c_stride + k;

                          send_input_request<Scale, cols>(
                              params.weight_scale_offset, address,
                              weight_scale_req);

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
                if (loop_counters[0][4] == loop_bounds[0][4]) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3]) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void writer() {
    writer_params.ResetRead();
    weight_scale_resp.Reset();
    write_request[0].Reset();
    write_request[1].Reset();

    bool bank_sel = 0;

    wait();

    while (true) {
      const MatrixParams params = writer_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][5];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][5];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 5; j++) {
          loop_bounds[i][j] = params.weight_addr_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> K2 =
          params.weight_addr_loops[0][params.weight_addr_weight_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.weight_addr_loops[1][params.weight_addr_reduction_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> C0 =
          params.weight_addr_loops[1][params.weight_addr_reduction_loop_idx[2]];
      ac_int<LOOP_WIDTH, false> FX =
          params.weight_addr_loops[1][params.weight_addr_fx_idx];
      ac_int<LOOP_WIDTH, false> FY0 =
          params.weight_addr_loops[1][params.weight_addr_fy_idx[1]];
      ac_int<LOOP_WIDTH, false> FY1 =
          params.weight_addr_loops[0][params.weight_addr_fy_idx[0]];
      ac_int<LOOP_WIDTH, false> K1 =
          params.weight_addr_loops[1][params.weight_addr_weight_loop_idx[1]];

      ac_int<24, false> C = C1 * C0 / BLOCK_SIZE;
      ac_int<24, false> fx_stride = C * K1;
      ac_int<24, false> fy_stride = FX * fx_stride;

      loop_bounds[1][params.weight_addr_reduction_loop_idx[1]] = 0;
      loop_bounds[1][params.weight_addr_reduction_loop_idx[2]] = C - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                // inner memory
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          ac_int<LOOP_WIDTH, false> k2 = loop_counters
                              [0][params.weight_addr_weight_loop_idx[0]];
                          ac_int<LOOP_WIDTH, false> c0 = loop_counters
                              [1][params.weight_addr_reduction_loop_idx[2]];
                          ac_int<LOOP_WIDTH, false> fx =
                              loop_counters[1][params.weight_addr_fx_idx];
                          ac_int<LOOP_WIDTH, false> fy0 =
                              loop_counters[1][params.weight_addr_fy_idx[1]];
                          ac_int<LOOP_WIDTH, false> k1 = loop_counters
                              [1][params.weight_addr_weight_loop_idx[1]];

                          ac_int<16, false> address =
                              fy0 * fy_stride + fx * fx_stride + c0 * K1 + k1;

                          ac_int<Scale::width * cols, false> data;
                          process_matrix_input<Scale, cols, port_width,
                                               Scale::width * cols>(
                              weight_scale_resp, data);

                          BufferWriteRequest<ac_int<Scale::width * cols, false>>
                              req;
                          req.address = address;
                          req.data = data;
                          req.last = loop_counters[1][4] == loop_bounds[1][4] &&
                                     loop_counters[1][3] == loop_bounds[1][3] &&
                                     loop_counters[1][2] == loop_bounds[1][2] &&
                                     loop_counters[1][1] == loop_bounds[1][1] &&
                                     loop_counters[1][0] == loop_bounds[1][0];
                          write_request[bank_sel].Push(req);

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
                bank_sel = !bank_sel;
                if (loop_counters[0][4] == loop_bounds[0][4]) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3]) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
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
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      // set irrelevant loop bounds to 0
      loop_bounds[1][params.weight_reuse_idx[0]] = 0;
      loop_bounds[1][params.weight_reuse_idx[1]] = 0;

      ac_int<LOOP_WIDTH, false> K2 = params.loops[0][params.weight_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> FY1 = params.loops[0][params.fy_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> C2 =
          params.loops[0][params.reduction_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> FY0 = params.loops[1][params.fy_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> FX = params.loops[1][params.fx_loop_idx];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reduction_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> K1 = params.loops[1][params.weight_loop_idx[1]];

      // extra loop to control reuse which only occurs during transpose and
      // when cols > rows
      int transpose_reuse_bound = 0;
      constexpr int ratio = cols / rows;
      if (ratio > 1 && params.weight_transpose && C2 >= ratio) {
        // we can reuse the weights already in the buffer
        loop_bounds[0][params.reduction_loop_idx[0]] = C2 / ratio - 1;
        transpose_reuse_bound = ratio - 1;
      }

      bool reuse_weights =
          FY1 == 1 && C2 == 1 && FY0 == 1 && FX == 1 && C1 == 1 && K1 == 1;

      // extra loop for exploiting L1 buffer reuse.
      // this loop is used when OX and OY are the innermost L2 loops. when
      // this occurs, we can move OX and/or OY into the buffer reuse L1 loop
      ac_int<LOOP_WIDTH, false> spatial_reuse_bound = 1;
      if (C2 == 1) {
        // OX loop can be absorbed
        if (params.weight_loop_idx[0] < params.x_loop_idx[0]) {
          if (!reuse_weights) {
            spatial_reuse_bound = params.loops[0][params.x_loop_idx[0]];
          }
          loop_bounds[0][params.x_loop_idx[0]] = 0;
        }
        // OY loop can be absorbed
        if (params.weight_loop_idx[0] < params.y_loop_idx[0]) {
          if (!reuse_weights) {
            spatial_reuse_bound *= params.loops[0][params.y_loop_idx[0]];
          }
          loop_bounds[0][params.y_loop_idx[0]] = 0;
        }
      }

      spatial_reuse_bound = spatial_reuse_bound - 1;

      ac_int<16, false> fx_stride = C1 * K1;
      ac_int<16, false> fy_stride = FX * fx_stride;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                for (ac_int<LOOP_WIDTH, false> spatial_reuse_idx = 0;;
                     spatial_reuse_idx++) {
                  for (int transpose_reuse_idx = 0; transpose_reuse_idx < ratio;
                       transpose_reuse_idx++) {
                    for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                      for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                        for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                          for (loop_counters[1][3] = 0;;
                               loop_counters[1][3]++) {
                            for (loop_counters[1][4] = 0;;
                                 loop_counters[1][4]++) {
                              for (loop_counters[1][5] = 0;;
                                   loop_counters[1][5]++) {
                                ac_int<LOOP_WIDTH, false> c1 =
                                    loop_counters[1]
                                                 [params.reduction_loop_idx[1]];
                                ac_int<LOOP_WIDTH, false> fx =
                                    loop_counters[1][params.fx_loop_idx];
                                ac_int<LOOP_WIDTH, false> fy0 =
                                    loop_counters[1][params.fy_loop_idx[1]];
                                ac_int<LOOP_WIDTH, false> k1 =
                                    loop_counters[1][params.weight_loop_idx[1]];

                                ac_int<16, false> address = fy0 * fy_stride +
                                                            fx * fx_stride +
                                                            c1 * K1 + k1;

                                BufferReadRequest req;
                                req.address = address;
                                req.last =
                                    loop_counters[1][5] == loop_bounds[1][5] &&
                                    loop_counters[1][4] == loop_bounds[1][4] &&
                                    loop_counters[1][3] == loop_bounds[1][3] &&
                                    loop_counters[1][2] == loop_bounds[1][2] &&
                                    loop_counters[1][1] == loop_bounds[1][1] &&
                                    loop_counters[1][0] == loop_bounds[1][0] &&
                                    transpose_reuse_idx ==
                                        transpose_reuse_bound &&
                                    spatial_reuse_idx == spatial_reuse_bound;

                                read_request[bank_sel].Push(req);

                                if (loop_counters[1][5] == loop_bounds[1][5])
                                  break;
                              }
                              if (loop_counters[1][4] == loop_bounds[1][4])
                                break;
                            }
                            if (loop_counters[1][3] == loop_bounds[1][3]) break;
                          }
                          if (loop_counters[1][2] == loop_bounds[1][2]) break;
                        }
                        if (loop_counters[1][1] == loop_bounds[1][1]) break;
                      }
                      if (loop_counters[1][0] == loop_bounds[1][0]) break;
                    }
                    if (transpose_reuse_idx == transpose_reuse_bound) break;
                  }
                  if (spatial_reuse_idx == spatial_reuse_bound) break;
                }
                bank_sel = !bank_sel;
                if (loop_counters[0][4] == loop_bounds[0][4]) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3]) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
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
