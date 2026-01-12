#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

template <typename WeightTypeTuple, typename Bias, int rows, int cols,
          int port_width, int buffer_width>
struct WeightController;

template <typename... WeightTypes, typename Bias, int rows, int cols,
          int port_width, int buffer_width>
struct WeightController<std::tuple<WeightTypes...>, Bias, rows, cols,
                        port_width, buffer_width> : public sc_module {
  static constexpr int LOOP_WIDTH = 10;
  static constexpr int DATA_WIDTH = buffer_width / cols;
  static constexpr int MAX_FETCH_WIDTH = std::max(
      {dtype_fetch_config<WeightTypes, cols, port_width>::max_fetch_width...});

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_resp);

  Connections::Out<BufferWriteRequest<ac_int<buffer_width, false>>>
      write_request[2];
  Connections::Out<BufferReadRequest> read_request[2];

  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(bias_resp);
  Connections::Out<Pack1D<Bias, cols>> CCS_INIT_S1(bias_data);

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(reader_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(weight_packer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(bias_fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(bias_feeder_params);

  sc_fifo<bool> fetcher_done;
  sc_fifo<bool> fetcher_done_2;

  Connections::Combinational<ac_int<MAX_FETCH_WIDTH, false>> packed_bits;
  Connections::Combinational<ac_int<buffer_width, false>> transpose_out;

  SC_CTOR(WeightController) {
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

    SC_THREAD(transposer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(weight_packer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(bias_fetcher);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(bias_feeder);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetcher() {
    weight_req.Reset();
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

      // reduce the number of iterations by packing factor
      K1 = K1 >> params.weight_pack_factor_lg2;
      loop_bounds[1][params.weight_addr_weight_loop_idx[1]] = K1 - 1;

      ac_int<16, false> k_stride = cols << params.weight_pack_factor_lg2;
      ac_int<24, false> c_stride = K2 * K1 * k_stride;
      ac_int<24, false> fx_stride = C2 * C1 * C0 * c_stride;
      ac_int<24, false> fy_stride = FX * fx_stride;

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
                          ac_int<LOOP_WIDTH, false> c1 = loop_counters
                              [1][params.weight_addr_reduction_loop_idx[1]];
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

                          ac_int<16, false> k = (k2 * K1 + k1) * k_stride;
                          ac_int<16, false> c = (c2 * C1 + c1) * C0 + c0;
                          ac_int<16, false> fy = fy0 * FY1 + fy1;
                          ac_int<32, false> address = fy * fy_stride +
                                                      fx * fx_stride +
                                                      c * c_stride + k;

                          if (params.weight_transpose) {
                            address =
                                ((k + c0) * C2 * C1 + c2 * C1 + c1) * cols;
                          }

                          send_packed_request<WeightTypes...>(
                              params.weight_dtype, params.weight_offset,
                              address, params.weight_burst_size, weight_req);
                          fetcher_done.write(false);
                          if (!params.weight_transpose) {
                            fetcher_done_2.write(false);
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
      ac_int<LOOP_WIDTH, false> K1 =
          params.weight_addr_loops[1][params.weight_addr_weight_loop_idx[1]];

      // reduce the number of iterations by packing factor
      K1 = K1 >> params.weight_pack_factor_lg2;
      loop_bounds[1][params.weight_addr_weight_loop_idx[1]] = K1 - 1;
      ac_int<4, false> pack_offset_bound =
          (1 << params.weight_pack_factor_lg2) - 1;

      ac_int<24, false> fx_stride = C1 * C0 * K1;
      ac_int<24, false> fy_stride = FX * fx_stride;

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
                          for (ac_int<4, false> pack = 0;; pack++) {
                            ac_int<LOOP_WIDTH, false> k2 = loop_counters
                                [0][params.weight_addr_weight_loop_idx[0]];
                            ac_int<LOOP_WIDTH, false> c1 = loop_counters
                                [1][params.weight_addr_reduction_loop_idx[1]];
                            ac_int<LOOP_WIDTH, false> c0 = loop_counters
                                [1][params.weight_addr_reduction_loop_idx[2]];
                            ac_int<LOOP_WIDTH, false> fx =
                                loop_counters[1][params.weight_addr_fx_idx];
                            ac_int<LOOP_WIDTH, false> fy0 =
                                loop_counters[1][params.weight_addr_fy_idx[1]];
                            ac_int<LOOP_WIDTH, false> k1 = loop_counters
                                [1][params.weight_addr_weight_loop_idx[1]];

                            ac_int<buffer_width, false> data =
                                transpose_out.Pop();

                            ac_int<16, false> c = c1 * C0 + c0;
                            ac_int<16, false> address =
                                fy0 * fy_stride + fx * fx_stride + c * K1 + k1;
                            address =
                                (address << params.weight_pack_factor_lg2) +
                                pack;

                            BufferWriteRequest<ac_int<buffer_width, false>> req;
                            req.address = address;
                            req.data = data;
                            req.last =
                                loop_counters[1][4] == loop_bounds[1][4] &&
                                loop_counters[1][3] == loop_bounds[1][3] &&
                                loop_counters[1][2] == loop_bounds[1][2] &&
                                loop_counters[1][1] == loop_bounds[1][1] &&
                                loop_counters[1][0] == loop_bounds[1][0] &&
                                pack == pack_offset_bound;
                            write_request[bank_sel].Push(req);

                            if (pack == pack_offset_bound) break;
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
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weight_reuse_idx[0]] = 1;
      loop_bounds[1][params.weight_reuse_idx[1]] = 1;

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
      int transpose_reuse_bound = 1;
      constexpr int ratio = cols / rows;
      if (ratio > 1 && params.weight_transpose && C2 >= ratio) {
        // we can reuse the weights already in the buffer
        loop_bounds[0][params.reduction_loop_idx[0]] = C2 / ratio;
        transpose_reuse_bound = ratio;
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
            spatial_reuse_bound = loop_bounds[0][params.x_loop_idx[0]];
          }
          loop_bounds[0][params.x_loop_idx[0]] = 1;
        }
        // OY loop can be absorbed
        if (params.weight_loop_idx[0] < params.y_loop_idx[0]) {
          if (!reuse_weights) {
            spatial_reuse_bound *= loop_bounds[0][params.y_loop_idx[0]];
          }
          loop_bounds[0][params.y_loop_idx[0]] = 1;
        }
      }

      ac_int<16, false> fx_stride = rows * C1 * K1;
      ac_int<16, false> fy_stride = FX * fx_stride;
      ac_int<16, false> fx_stride_with_repl = fx_stride * transpose_reuse_bound;
      ac_int<16, false> fy_stride_with_repl = fy_stride * transpose_reuse_bound;

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
                                ac_int<LOOP_WIDTH, false> fy0 =
                                    loop_counters[1][params.fy_loop_idx[1]];
                                ac_int<LOOP_WIDTH, false> fx =
                                    loop_counters[1][params.fx_loop_idx];
                                ac_int<LOOP_WIDTH, false> k1 =
                                    loop_counters[1][params.weight_loop_idx[1]];

                                /*
                                 * If we have replication, then need to zero
                                 * pad the unused rows For 7x7 filter, we
                                 * split it into 4 filters and 3 filters
                                 */
                                ac_int<4, false> replication_bound = 1;
                                ac_int<8, false> c_end = rows;
                                if (params.is_resnet_replication) {
                                  c_end = 3;
                                  if (rows == 4) {
                                    replication_bound = 1;
                                  } else if (rows == 8) {
                                    // last iteration only unrolls 1 fx
                                    replication_bound = fx == 3 ? 1 : 2;
                                  } else if (rows == 16) {
                                    replication_bound = fx == 0 ? 4 : 3;
                                  } else if (rows == 32 || rows == 64) {
                                    replication_bound = 7;
                                  }
                                } else if (params.is_generic_replication) {
                                  c_end = params.num_channels;
                                  replication_bound =
                                      1 << params.fx_unrolling_lg2;
                                }

                                ac_int<LOOP_WIDTH, false> fx_repl = 0;
                                ac_int<LOOP_WIDTH, false> c = 0;
                                for (int row = 0; row < rows; row++) {
                                  ac_int<LOOP_WIDTH, false> C = rows * C1;
                                  ac_int<LOOP_WIDTH, false> C0 = rows;
                                  ac_int<16, false> address;

                                  if (params.is_resnet_replication ||
                                      params.is_generic_replication) {
                                    ac_int<LOOP_WIDTH, false> cur_fx = fx;
                                    if (params.is_resnet_replication) {
                                      C = 3;
                                      C0 = 3;
                                      FX = 7;
                                      if (rows == 4) {
                                        cur_fx = fx;
                                      } else if (rows == 8) {
                                        cur_fx = fx * 2 + fx_repl;
                                      } else if (rows == 16) {
                                        cur_fx = fx * 4 + fx_repl;
                                      } else if (rows == 32 || rows == 64) {
                                        cur_fx = fx_repl;
                                      }
                                    } else {
                                      C = params.num_channels;
                                      C0 = params.num_channels;
                                      FX = params.loops[1][params.fx_loop_idx] *
                                           replication_bound;
                                      cur_fx = fx * replication_bound + fx_repl;
                                    }

                                    if (fx_repl < replication_bound) {
                                      address = fy0 * FX * C * K1 +
                                                cur_fx * C * K1 + c * K1 + k1;
                                    } else {
                                      address = 0xFFFF;
                                    }

                                    // keep track of which C and FX we are on
                                    if (c < c_end - 1) {
                                      c++;
                                    } else {
                                      c = 0;
                                      fx_repl++;
                                    }
                                  } else {
                                    c = row;
                                    if (params.weight_transpose &&
                                        cols > rows) {
                                      address =
                                          fy0 * fy_stride_with_repl +
                                          fx * fx_stride_with_repl +
                                          ((c + transpose_reuse_idx * rows) +
                                           c1 * C0 * transpose_reuse_bound) *
                                              K1 +
                                          k1;
                                    } else {
                                      address = fy0 * fy_stride +
                                                fx * fx_stride +
                                                (c + c1 * C0) * K1 + k1;
                                    }
                                  }

                                  BufferReadRequest req;
                                  req.address = address;
                                  req.last = row == rows - 1 &&
                                             loop_counters[1][5] ==
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
                                             spatial_reuse_idx ==
                                                 spatial_reuse_bound - 1 &&
                                             transpose_reuse_idx ==
                                                 transpose_reuse_bound - 1;
                                  read_request[bank_sel].Push(req);
                                }

                                if (loop_counters[1][5] ==
                                    loop_bounds[1][5] - 1)
                                  break;
                              }
                              if (loop_counters[1][4] == loop_bounds[1][4] - 1)
                                break;
                            }
                            if (loop_counters[1][3] == loop_bounds[1][3] - 1)
                              break;
                          }
                          if (loop_counters[1][2] == loop_bounds[1][2] - 1)
                            break;
                        }
                        if (loop_counters[1][1] == loop_bounds[1][1] - 1) break;
                      }
                      if (loop_counters[1][0] == loop_bounds[1][0] - 1) break;
                    }
                    if (transpose_reuse_idx == transpose_reuse_bound - 1) break;
                  }
                  if (spatial_reuse_idx == spatial_reuse_bound - 1) break;
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

  void weight_packer() {
    weight_packer_params.ResetRead();
    weight_resp.Reset();
    packed_bits.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = weight_packer_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!fetcher_done.read()) {
        ac_int<MAX_FETCH_WIDTH, false> bits;

        for (ac_int<4, false> i = 0;; i++) {
          bits.set_slc(i * port_width, weight_resp.Pop());
          if (i == params.weight_num_beats - 1) break;
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

      ac_int<LOOP_WIDTH, false> loop_bounds[2][5];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 5; j++) {
          loop_bounds[i][j] = params.weight_addr_loops[i][j];
        }
      }

      ac_int<32, false> total_values =
          loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
          loop_bounds[0][3] * loop_bounds[0][4] * loop_bounds[1][0] *
          loop_bounds[1][1] * loop_bounds[1][2] * loop_bounds[1][3];
      ac_int<32, false> count = 0;

      // don't support transpose when systolic array is larger
      // than 32x32, as it will require a very large buffer
      if (params.weight_transpose && rows < 64 && cols < 64) {
        // we need a square buffer to store the transpose
        ac_int<DATA_WIDTH, false> transpose_buffer[rows > cols ? rows : cols]
                                                  [rows > cols ? rows : cols];

#ifndef __SYNTHESIS__
        // Assume that the innermost loop is the c0 loop
        // Must be true for the transpose case
        assert(loop_bounds[1][4] == cols);
#endif

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (count++ < total_values) {
          for (int c0 = 0; c0 < cols; c0++) {
            ac_int<buffer_width, false> bits = packed_bits.Pop();

            ac_int<buffer_width, false> outputs = 0;
            bool handled = (unpack_bits<WeightTypes, cols, buffer_width,
                                        MAX_FETCH_WIDTH, WeightTypes...>(
                                params.weight_dtype, bits, outputs, 0) ||
                            ...);

#ifndef __SYNTHESIS__
            if (!handled) {
              throw std::runtime_error("Unsupported dtype for matrix weight: " +
                                       std::to_string(params.weight_dtype));
            }
#endif

#pragma hls_unroll yes
            for (int dim = 0; dim < cols; dim++) {
              transpose_buffer[dim][c0] =
                  outputs.template slc<DATA_WIDTH>(dim * DATA_WIDTH);
            }
          }

          for (int c0 = 0; c0 < cols; c0++) {
            ac_int<buffer_width, false> transposed;

#pragma hls_unroll yes
            for (int dim = 0; dim < cols; dim++) {
              transposed.set_slc(dim * DATA_WIDTH, transpose_buffer[c0][dim]);
            }

            transpose_out.Push(transposed);
          }
        }
      } else {  // passthrough
        ac_int<4, false> pack_offset_bound =
            (1 << params.weight_pack_factor_lg2) - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (!fetcher_done_2.read()) {
          ac_int<MAX_FETCH_WIDTH, false> bits = packed_bits.Pop();

          // Unpack bits into outputs based on dtype
          for (ac_int<4, false> i = 0;; i++) {
            ac_int<buffer_width, false> outputs = 0;
            bool handled = (unpack_bits<WeightTypes, cols, buffer_width,
                                        MAX_FETCH_WIDTH, WeightTypes...>(
                                params.weight_dtype, bits, outputs, i) ||
                            ...);

#ifndef __SYNTHESIS__
            if (!handled) {
              throw std::runtime_error("Unsupported dtype for matrix weight: " +
                                       std::to_string(params.weight_dtype));
            }
#endif

            transpose_out.Push(outputs);

            if (i == pack_offset_bound) break;
          }
        }
      }
    }
  }

  void bias_fetcher() {
    bias_fetcher_params.ResetRead();
    bias_req.Reset();

    wait();

    while (true) {
      const MatrixParams params = bias_fetcher_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weight_reuse_idx[0]] = 0;
      loop_bounds[1][params.weight_reuse_idx[1]] = 0;
      loop_bounds[1][params.fx_loop_idx] = 0;
      loop_bounds[1][params.fy_loop_idx[1]] = 0;
      loop_bounds[1][params.reduction_loop_idx[1]] = 0;

      ac_int<LOOP_WIDTH, false> K2 = params.loops[0][params.weight_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K1 = params.loops[1][params.weight_loop_idx[1]];

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
                        ac_int<LOOP_WIDTH, false> k2 =
                            loop_counters[0][params.weight_loop_idx[0]];
                        ac_int<LOOP_WIDTH, false> k1 =
                            loop_counters[1][params.weight_loop_idx[1]];

                        ac_int<16, false> address = k2 * K1 * cols + k1 * cols;

                        MemoryRequest request = {
                            params.bias_offset + address * Bias::width / 8,
                            cols * Bias::width / 8};

                        bias_req.Push(request);

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

  void bias_feeder() {
    bias_feeder_params.ResetRead();
    bias_resp.Reset();
    bias_data.Reset();

    wait();

    while (true) {
      const MatrixParams params = bias_feeder_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weight_reuse_idx[0]] = 0;
      loop_bounds[1][params.weight_reuse_idx[1]] = 0;
      loop_bounds[1][params.fx_loop_idx] = 0;
      loop_bounds[1][params.fy_loop_idx[1]] = 0;
      loop_bounds[1][params.reduction_loop_idx[1]] = 0;

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
                        ac_int<Bias::width * cols, false> bits;

                        process_matrix_input<Bias, cols, port_width,
                                             Bias::width * cols>(bias_resp,
                                                                 bits);

                        Pack1D<Bias, cols> biases =
                            BitsToType<Pack1D<Bias, cols>>(TypeToBits(bits));

                        bias_data.Push(biases);
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

  void read_params() {
    params_in.Reset();
    fetcher_params.ResetWrite();
    writer_params.ResetWrite();
    reader_params.ResetWrite();
    transposer_params.ResetWrite();
    weight_packer_params.ResetWrite();
    bias_fetcher_params.ResetWrite();
    bias_feeder_params.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();

      fetcher_params.Push(params);
      writer_params.Push(params);
      reader_params.Push(params);
      transposer_params.Push(params);
      weight_packer_params.Push(params);

      if (params.has_bias) {
        bias_fetcher_params.Push(params);
        bias_feeder_params.Push(params);
      }
    }
  }
};
