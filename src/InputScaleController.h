#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ParamsDeserializer.h"

template <typename Scale, int NRows>
SC_MODULE(InputScaleController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(scale_req);
  Connections::In<ac_int<Scale::width, false>> CCS_INIT_S1(scale_resp);

  Connections::Out<BufferWriteRequest<ac_int<Scale::width, false>>>
      scale_write_request[2];
  Connections::Out<BufferReadRequest> scale_read_address[2];

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(reader_params);

  static constexpr int LOOP_WIDTH = 10;

  // num x values packed in a word
  static constexpr int packing_factor = (NRows == 4)    ? 1
                                        : (NRows == 8)  ? 2
                                        : (NRows == 16) ? 4
                                        : (NRows == 32) ? 8
                                        : (NRows == 64) ? 8
                                                        : 0;

  // num words needed to store the boundary pixels. essentially
  // ceil(3/packing_factor)
  static constexpr int boundary_words = (NRows == 4)    ? 3
                                        : (NRows == 8)  ? 2
                                        : (NRows == 16) ? 1
                                        : (NRows == 32) ? 1
                                        : (NRows == 64) ? 1
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
      loop_bounds[1][params.weightLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex[1]] = 1;

      ac_int<LOOP_WIDTH, false> Y1 = params.loops[0][params.inputYLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> X1 = params.loops[0][params.inputXLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C2 =
          params.loops[0][params.reductionLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reductionLoopIndex[1]];
      ac_int<16, false> Y0 = params.loops[1][params.inputYLoopIndex[1]];
      ac_int<16, false> X0 = params.loops[1][params.inputXLoopIndex[1]];
      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      ac_int<4, false> FY0 = params.loops[1][params.fyIndex[1]];
      ac_int<5, false> STRIDE = params.stride;

      if (params.is_resnet_replication) {
        FX = 7;
      }

      bool is_downsample = FX == 1 && FY0 == 1;

      ac_int<4, false> x_padding = params.padding;
      ac_int<4, false> y_padding = params.padding;

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> IY0 = Y0 * STRIDE;

      ac_int<16, false> x_bound = is_downsample ? X0 : IX0;
      ac_int<16, false> y_bound = is_downsample ? Y0 : IY0;

      if (params.is_resnet_replication) {
        x_bound /= packing_factor;
      }

      loop_bounds[1][params.inputXLoopIndex[1]] = x_bound;
      if (params.is_resnet_replication) {
        loop_bounds[1][params.inputXLoopIndex[1]] += 2 * boundary_words;
      } else {
        loop_bounds[1][params.inputXLoopIndex[1]] += 2 * params.padding;
      }

      loop_bounds[1][params.inputYLoopIndex[1]] = y_bound + 2 * params.padding;

      ac_int<16, false> Y = Y1 * IY0;
      ac_int<16, false> X = X1 * IX0;
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
                                loop_counters[0][params.inputYLoopIndex[0]];
                            ac_int<LOOP_WIDTH, false> x1 =
                                loop_counters[0][params.inputXLoopIndex[0]];
                            ac_int<LOOP_WIDTH, false> c2 =
                                loop_counters[0][params.reductionLoopIndex[0]];
                            ac_int<LOOP_WIDTH, false> y0 =
                                loop_counters[1][params.inputYLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> x0 =
                                loop_counters[1][params.inputXLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> c1 =
                                loop_counters[1][params.reductionLoopIndex[1]];

                            // adjust address for stride
                            if (FX == 1) {
                              x0 = x0 * STRIDE;
                            }
                            if (FY0 == 1) {
                              y0 = y0 * STRIDE;
                            }

                            if (params.is_resnet_replication) {
                              if (x0 != 0 && x1 != 0) {
                                x0 = (x0 - boundary_words) * packing_factor +
                                     x_padding;
                              } else {
                                x0 = x0 * packing_factor;
                              }
                            } else if (params.is_generic_replication) {
                              x0 = x0 << params.fx_unrolling_lg2;
                            }

                            ac_int<16, false> y = y1 * IY0 + y0 - y_padding;
                            ac_int<16, false> x = x1 * IX0 + x0 - x_padding;
                            ac_int<16, false> c = c2 * C1 + c1;

                            if ((x >= 0) && (y >= 0) && (x < X) && (y < Y)) {
                              ac_int<32, false> address = y * X * C + x * C + c;

                              if (params.is_resnet_replication) {
                                address = y * (X / packing_factor) * NRows +
                                          (x / packing_factor) * NRows + c;
                              } else if (params.is_generic_replication) {
                                address =
                                    y * (X >> params.fx_unrolling_lg2) * NRows +
                                    (x >> params.fx_unrolling_lg2) * NRows + c;
                              }

                              if (params.has_attn_output_permute) {
                                ac_int<8, false> head_size =
                                    params.head_size_power_of_two;
                                ac_int<16, false> mask = (1 << head_size) - 1;
                                address =
                                    ((c >> head_size) * (X << head_size)) +
                                    (x << head_size) + (c & mask);
                              }

                              if (params.has_input_transpose) {
                                address =
                                    (c + (x % NRows)) * X + (x / NRows) * NRows;
                              }

                              send_input_request<Scale, 1>(
                                  params.input_scale_offset, address,
                                  scale_req);
                            }

                            if (loop_counters[1][5] >= loop_bounds[1][5] - 1) {
                              break;
                            }
                          }
                          if (loop_counters[1][4] >= loop_bounds[1][4] - 1) {
                            break;
                          }
                        }
                        if (loop_counters[1][3] >= loop_bounds[1][3] - 1) {
                          break;
                        }
                      }
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
                if (loop_counters[0][4] >= loop_bounds[0][4] - 1) {
                  break;
                }
              }
              if (loop_counters[0][3] >= loop_bounds[0][3] - 1) {
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
    }
  }

  void writer() {
    writer_params.ResetRead();

    scale_resp.Reset();
    scale_write_request[0].Reset();
    scale_write_request[1].Reset();

    bool bankSel = 0;

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
      loop_bounds[1][params.weightLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex[1]] = 1;

      ac_int<LOOP_WIDTH, false> Y1 = params.loops[0][params.inputYLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> X1 = params.loops[0][params.inputXLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reductionLoopIndex[1]];
      ac_int<16, false> Y0 = params.loops[1][params.inputYLoopIndex[1]];
      ac_int<16, false> X0 = params.loops[1][params.inputXLoopIndex[1]];
      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      ac_int<4, false> FY0 = params.loops[1][params.fyIndex[1]];
      ac_int<5, false> STRIDE = params.stride;

      if (params.is_resnet_replication) {
        FX = 7;
      }

      ac_int<4, false> x_padding = params.padding;
      ac_int<4, false> y_padding = params.padding;

      bool is_downsample = FX == 1 && FY0 == 1;

      if (is_downsample) {
        STRIDE = 1;
      }

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> IY0 = Y0 * STRIDE;

      ac_int<16, false> x_bound = IX0;

      if (params.is_resnet_replication) {
        x_bound /= packing_factor;
      }

      ac_int<4, false> x_boundary = params.is_resnet_replication
                                        ? ac_int<4, false>(2 * boundary_words)
                                        : ac_int<4, false>(params.padding * 2);
      loop_bounds[1][params.inputXLoopIndex[1]] = x_bound + x_boundary;
      loop_bounds[1][params.inputYLoopIndex[1]] = IY0 + 2 * params.padding;

      ac_int<16, false> IX = X1 * IX0;
      ac_int<16, false> IY = Y1 * IY0;
      ac_int<16, false> y_stride =
          loop_bounds[1][params.inputXLoopIndex[1]] * C1;

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
                                loop_counters[1][params.inputXLoopIndex[1]];
                            ac_int<LOOP_WIDTH, true> x1 =
                                loop_counters[0][params.inputXLoopIndex[0]];
                            ac_int<LOOP_WIDTH, true> y0 =
                                loop_counters[1][params.inputYLoopIndex[1]];
                            ac_int<LOOP_WIDTH, true> y1 =
                                loop_counters[0][params.inputYLoopIndex[0]];
                            ac_int<LOOP_WIDTH, true> c1 =
                                loop_counters[1][params.reductionLoopIndex[1]];

                            if (params.is_resnet_replication && x0 != 0) {
                              x0 = (x0 - boundary_words) * packing_factor +
                                   x_padding;
                            }

                            ac_int<16, true> x = x1 * IX0 + x0 - x_padding;
                            ac_int<16, true> y = y1 * IY0 + y0 - y_padding;

                            ac_int<Scale::width, false> scale;

                            if ((x < 0) || (y < 0) || (x >= IX) || (y >= IY)) {
                              scale = Scale::one().bits_rep();
                            } else {
                              scale = scale_resp.Pop();
                            }

                            ac_int<LOOP_WIDTH> orig_x0 =
                                loop_counters[1][params.inputXLoopIndex[1]];
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
                            scale_write_request[bankSel].Push(scale_req);

                            if (loop_counters[1][5] >= loop_bounds[1][5] - 1) {
                              break;
                            }
                          }
                          if (loop_counters[1][4] >= loop_bounds[1][4] - 1) {
                            break;
                          }
                        }
                        if (loop_counters[1][3] >= loop_bounds[1][3] - 1) {
                          break;
                        }
                      }
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
                bankSel = !bankSel;
                if (loop_counters[0][4] >= loop_bounds[0][4] - 1) {
                  break;
                }
              }
              if (loop_counters[0][3] >= loop_bounds[0][3] - 1) {
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
    }
  }

  void reader() {
    reader_params.ResetRead();

    scale_read_address[0].Reset();
    scale_read_address[1].Reset();

    bool bankSel = 0;

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

      if (params.is_resnet_replication && NRows >= 16) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] * params.stride /
             packing_factor) +
            2;
      } else if (params.is_resnet_replication && NRows == 8) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] * params.stride /
             packing_factor) +
            1;
      }

      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.inputXLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> Y0 = params.loops[1][params.inputYLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reductionLoopIndex[1]];
      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      ac_int<4, false> FY0 = params.loops[1][params.fyIndex[1]];
      ac_int<5, false> STRIDE = params.stride;

      bool is_downsample = FX == 1 && FY0 == 1;

      if (is_downsample) {
        STRIDE = 1;
      }

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> y_stride = C1;
      if (params.is_resnet_replication) {
        y_stride *= IX0 / packing_factor + 2 * boundary_words;
      } else {
        y_stride *= IX0 + FX - 1;
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
                            ac_int<LOOP_WIDTH, false> x0 =
                                loop_counters[1][params.inputXLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> y0 =
                                loop_counters[1][params.inputYLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> fx =
                                loop_counters[1][params.fxIndex];
                            ac_int<LOOP_WIDTH, false> fy =
                                loop_counters[1][params.fyIndex[1]];
                            ac_int<LOOP_WIDTH, false> c1 =
                                loop_counters[1][params.reductionLoopIndex[1]];

                            ac_int<16, false> x = STRIDE * x0 + fx;
                            ac_int<16, false> y = STRIDE * y0 + fy;
                            ac_int<16, false> address;
                            if (params.is_resnet_replication && NRows >= 8) {
                              address = y * y_stride + (x0 + fx) * C1 + c1;
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
                            scale_read_address[bankSel].Push(req);

                            if (loop_counters[1][5] >= loop_bounds[1][5] - 1) {
                              break;
                            }
                          }
                          if (loop_counters[1][4] >= loop_bounds[1][4] - 1) {
                            break;
                          }
                        }
                        if (loop_counters[1][3] >= loop_bounds[1][3] - 1) {
                          break;
                        }
                      }
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
                bankSel = !bankSel;
                if (loop_counters[0][4] >= loop_bounds[0][4] - 1) {
                  break;
                }
              }
              if (loop_counters[0][3] >= loop_bounds[0][3] - 1) {
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
