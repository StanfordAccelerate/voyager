#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"
#include "Utils.h"

template <typename InputTypeTuple, int NRows, int PortWidth, int BufferWidth>
struct InputController;

template <typename... InputTypes, int NRows, int PortWidth, int BufferWidth>
struct InputController<std::tuple<InputTypes...>, NRows, PortWidth, BufferWidth>
    : public sc_module {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(input_resp);
  sc_fifo<bool> fetcher_done;

  Connections::Out<BufferWriteRequest<ac_int<BufferWidth, false>>>
      input_write_request[2];
  Connections::Out<BufferReadRequest> input_read_address[2];

  Connections::In<ac_int<BufferWidth, false>> CCS_INIT_S1(window_buffer_in);
  Connections::Out<ac_int<BufferWidth, false>> CCS_INIT_S1(window_buffer_out);

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(reader_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(winder_buffer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposer_params);

  Connections::Combinational<ac_int<BufferWidth, false>> transpose_out;

  static constexpr int LOOP_WIDTH = 10;
  static constexpr int DATA_WIDTH = BufferWidth / NRows;

  // num x values packed in a word
  static constexpr int packing_factor = (NRows == 4)    ? 1
                                        : (NRows == 8)  ? 2
                                        : (NRows == 16) ? 4
                                        : (NRows == 32) ? 8
                                                        : 0;

  // num words needed to store the boundary pixels. essentially
  // ceil(3/packing_factor)
  static constexpr int boundary_words = (NRows == 4)    ? 3
                                        : (NRows == 8)  ? 2
                                        : (NRows == 16) ? 1
                                        : (NRows == 32) ? 1
                                                        : 0;

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
      loop_bounds[1][params.weightLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;

      ac_int<LOOP_WIDTH, false> Y1 = params.loops[0][params.inputYLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> X1 = params.loops[0][params.inputXLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C2 =
          params.loops[0][params.reductionLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reductionLoopIndex[1]];
      ac_int<16, false> Y0 = params.loops[1][params.inputYLoopIndex[1]];
      ac_int<16, false> X0 = params.loops[1][params.inputXLoopIndex[1]];
      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      ac_int<4, false> FY = params.loops[1][params.fyIndex];
      ac_int<2, false> STRIDE = params.stride;

      if (params.is_replication) {
        FX = 7;
      }

      bool is_downsample = FX == 1 && FY == 1;

      ac_int<4, false> x_padding = (FX - 1) / 2;
      ac_int<4, false> y_padding = (FY - 1) / 2;

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> IY0 = Y0 * STRIDE;

      ac_int<16, false> x_bound = is_downsample ? X0 : IX0;
      ac_int<16, false> y_bound = is_downsample ? Y0 : IY0;

      if (params.is_replication) {
        x_bound /= packing_factor;
      }

      loop_bounds[1][params.inputXLoopIndex[1]] = x_bound;
      if (params.is_replication) {
        loop_bounds[1][params.inputXLoopIndex[1]] += 2 * boundary_words;
      } else {
        loop_bounds[1][params.inputXLoopIndex[1]] += FX - 1;
      }

      loop_bounds[1][params.inputYLoopIndex[1]] = y_bound + FY - 1;

      ac_int<16, false> Y = Y1 * IY0;
      ac_int<16, false> X = X1 * IX0;
      ac_int<16, false> C = C2 * C1 * NRows;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                  for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                    for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                      for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                        for (loop_counters[1][5] = 0;; loop_counters[1][5]++) {
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
                          if (is_downsample) {
                            x0 = x0 * STRIDE;
                            y0 = y0 * STRIDE;
                          }

                          if (params.is_replication) {
                            if (x0 != 0 && x1 != 0) {
                              x0 = (x0 - boundary_words) * packing_factor +
                                   x_padding;
                            } else {
                              x0 = x0 * packing_factor;
                            }
                          }

                          ac_int<16, false> y = y1 * IY0 + y0 - y_padding;
                          ac_int<16, false> x = x1 * IX0 + x0 - x_padding;
                          ac_int<16, false> c = c2 * C1 * NRows + c1 * NRows;

                          if ((x >= 0) && (y >= 0) && (x < X) && (y < Y)) {
                            ac_int<32, false> address = y * X * C + x * C + c;

                            if (params.is_replication) {
                              address = y * (X / packing_factor) * NRows +
                                        (x / packing_factor) * NRows + c;
                            }

                            if (params.has_attn_output_permute) {
                              ac_int<8, false> head_size =
                                  params.head_size_power_of_two;
                              ac_int<16, false> mask = (1 << head_size) - 1;
                              address = (((c >> head_size) * X) << head_size) +
                                        (x << head_size) + (c & mask);
                            }

                            if (params.has_input_transpose) {
                              address =
                                  (c + (x % NRows)) * X + (x / NRows) * NRows;
                            } else {
                              fetcher_done.write(false);
                            }

                            (fetch_matrix_input<InputTypes, NRows,
                                                InputTypes...>(
                                 params.input_dtype, params.INPUT_OFFSET,
                                 address, input_req),
                             ...);
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
      fetcher_done.write(true);
    }
  }

  void writer() {
    writer_params.ResetRead();
    transpose_out.ResetRead();

    input_write_request[0].Reset();
    input_write_request[1].Reset();

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
      loop_bounds[1][params.fyIndex] = 1;

      ac_int<LOOP_WIDTH, false> Y1 = params.loops[0][params.inputYLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> X1 = params.loops[0][params.inputXLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reductionLoopIndex[1]];
      ac_int<16, false> Y0 = params.loops[1][params.inputYLoopIndex[1]];
      ac_int<16, false> X0 = params.loops[1][params.inputXLoopIndex[1]];
      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      ac_int<4, false> FY = params.loops[1][params.fyIndex];
      ac_int<2, false> STRIDE = params.stride;

      if (params.is_replication) {
        FX = 7;
      }

      ac_int<4, false> x_padding = (FX - 1) / 2;
      ac_int<4, false> y_padding = (FY - 1) / 2;

      bool is_downsample = FX == 1 && FY == 1;

      if (is_downsample) {
        STRIDE = 1;
      }

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> IY0 = Y0 * STRIDE;

      ac_int<16, false> x_bound = IX0;

      if (params.is_replication) {
        x_bound /= packing_factor;
      }

      ac_int<4, false> x_boundary = params.is_replication
                                        ? ac_int<4, false>(2 * boundary_words)
                                        : ac_int<4, false>(FX - 1);
      loop_bounds[1][params.inputXLoopIndex[1]] = x_bound + x_boundary;
      loop_bounds[1][params.inputYLoopIndex[1]] = IY0 + FY - 1;

      ac_int<16, false> X = X1 * IX0;
      ac_int<16, false> Y = Y1 * IY0;
      ac_int<16, false> y_stride =
          loop_bounds[1][params.inputXLoopIndex[1]] * C1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                  for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                    for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                      for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                        for (loop_counters[1][5] = 0;; loop_counters[1][5]++) {
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

                          if (params.is_replication && x0 != 0) {
                            x0 = (x0 - boundary_words) * packing_factor +
                                 x_padding;
                          }

                          ac_int<16, true> x = x1 * IX0 + x0 - x_padding;
                          ac_int<16, true> y = y1 * IY0 + y0 - y_padding;

                          ac_int<BufferWidth, false> data;

                          if ((x < 0) || (y < 0) || (x >= X) || (y >= Y)) {
                            (set_zero<InputTypes, NRows, BufferWidth,
                                      InputTypes...>(params.input_dtype, data),
                             ...);
                          } else {
                            data = transpose_out.Pop();
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

                          BufferWriteRequest<ac_int<BufferWidth, false>> req;
                          req.address = address;
                          req.data = data;
                          req.last = is_last;
                          input_write_request[bankSel].Push(req);

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

    input_read_address[0].Reset();
    input_read_address[1].Reset();

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

      if (params.is_replication && NRows >= 16) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] * params.stride /
             packing_factor) +
            2;
      } else if (params.is_replication && NRows == 8) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] * params.stride /
             packing_factor) +
            1;
      }

      ac_int<LOOP_WIDTH, false> Y0 = params.loops[1][params.inputYLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.inputXLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reductionLoopIndex[1]];
      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      ac_int<4, false> FY = params.loops[1][params.fyIndex];
      ac_int<2, false> STRIDE = params.stride;

      bool is_downsample = FX == 1 && FY == 1;

      if (is_downsample) {
        STRIDE = 1;
      }

      ac_int<16, false> IX0 = X0 * STRIDE;
      ac_int<16, false> y_stride = C1;
      if (params.is_replication) {
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
              for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                  for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                    for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                      for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                        for (loop_counters[1][5] = 0;; loop_counters[1][5]++) {
                          ac_int<LOOP_WIDTH, false> x0 =
                              loop_counters[1][params.inputXLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> y0 =
                              loop_counters[1][params.inputYLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> fx =
                              loop_counters[1][params.fxIndex];
                          ac_int<LOOP_WIDTH, false> fy =
                              loop_counters[1][params.fyIndex];
                          ac_int<LOOP_WIDTH, false> c1 =
                              loop_counters[1][params.reductionLoopIndex[1]];

                          ac_int<16, false> x = STRIDE * x0 + fx;
                          ac_int<16, false> y = STRIDE * y0 + fy;
                          ac_int<16, false> address;
                          if (params.is_replication && NRows >= 8) {
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
                          input_read_address[bankSel].Push(req);

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

      int unrollingFactor;  // num x values packed in a word sent to the
      // systolic array
      int additionalUnrollingFactor;  // additional x values packed in a word
                                      // sent to the systolic array, but are
                                      // unused by the systolic array
      constexpr int initialReuseFactor = (NRows == 8) ? 1 : 3;
      int shiftFactor;

      if (NRows == 4) {
        unrollingFactor = 1;
        additionalUnrollingFactor = 0;
      } else if (NRows == 8) {
        unrollingFactor = 2;
        additionalUnrollingFactor = 0;
        shiftFactor = 1;
      } else if (NRows == 16) {
        unrollingFactor = 4;
        additionalUnrollingFactor = 1;
        shiftFactor = 2;
      } else if (NRows == 32) {
        unrollingFactor = 7;
        additionalUnrollingFactor = 0;
        shiftFactor = 2;
      }

      if (params.is_replication && (NRows == 16 || NRows == 32)) {
        // #pragma hls_pipeline_init_interval 1
        // #pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
              for (loop_counters[0][3] = 0;
                   loop_counters[0][3] < loop_bounds[0][3];
                   loop_counters[0][3]++) {
                // inner memory
                for (loop_counters[1][0] = 0;
                     loop_counters[1][0] < loop_bounds[1][0];
                     loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;
                       loop_counters[1][1] < loop_bounds[1][1];
                       loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;
                         loop_counters[1][2] < loop_bounds[1][2];
                         loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;
                           loop_counters[1][3] < loop_bounds[1][3];
                           loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;
                             loop_counters[1][4] < loop_bounds[1][4];
                             loop_counters[1][4]++) {
                          ac_int<BufferWidth, false> data;

                          ac_int<BufferWidth, false> buffer =
                              window_buffer_in.Pop();
#pragma hls_unroll yes
                          for (int x = 0; x < 3; x++) {
#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              int dst_idx = x * 3 + dim;
                              int src_idx = (packing_factor - 3 + x) * 3 + dim;

                              auto temp = buffer.template slc<DATA_WIDTH>(
                                  src_idx * DATA_WIDTH);
                              data.set_slc(dst_idx * DATA_WIDTH, temp);
                            }
                          }

                          buffer = window_buffer_in.Pop();
#pragma hls_unroll yes
                          for (int x = 3;
                               x < unrollingFactor + additionalUnrollingFactor;
                               x++) {
#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              int dst_idx = x * 3 + dim;
                              int src_idx = (x - 3) * 3 + dim;

                              auto temp = buffer.template slc<DATA_WIDTH>(
                                  src_idx * DATA_WIDTH);
                              data.set_slc(dst_idx * DATA_WIDTH, temp);
                            }
                          }

                          window_buffer_out.Push(data);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
                          for (loop_counters[1][5] = 0;
                               loop_counters[1][5] < loop_bounds[1][5] * 2 - 2;
                               loop_counters[1][5] += 2) {
                            // shift two pixels over
#pragma hls_unroll yes
                            for (int x = 0;
                                 x <
                                 packing_factor + additionalUnrollingFactor - 2;
                                 x++) {
#pragma hls_unroll yes
                              for (int dim = 0; dim < 3; dim++) {
                                // data[x * 3 + dim] = data[(x + 2) * 3 + dim];

                                int dst_idx = x * 3 + dim;
                                int src_idx = (x + 2) * 3 + dim;

                                auto temp = data.template slc<DATA_WIDTH>(
                                    src_idx * DATA_WIDTH);
                                data.set_slc(dst_idx * DATA_WIDTH, temp);
                              }
                            }

                            // grab 2 new pixels from buffer
                            int bufferPosition =
                                (loop_counters[1][5] + unrollingFactor +
                                 additionalUnrollingFactor - 3) %
                                packing_factor;
#pragma hls_unroll yes
                            for (int x = unrollingFactor +
                                         additionalUnrollingFactor - 2;
                                 x <
                                 unrollingFactor + additionalUnrollingFactor;
                                 x++) {
#pragma hls_unroll yes
                              for (int dim = 0; dim < 3; dim++) {
                                int dst_idx = x * 3 + dim;
                                int src_idx =
                                    (bufferPosition + x -
                                     (unrollingFactor +
                                      additionalUnrollingFactor - 2)) *
                                        3 +
                                    dim;

                                auto temp = buffer.template slc<DATA_WIDTH>(
                                    src_idx * DATA_WIDTH);
                                data.set_slc(dst_idx * DATA_WIDTH, temp);
                              }
                            }

                            window_buffer_out.Push(data);

                            if (bufferPosition == packing_factor - 2) {
                              buffer = window_buffer_in.Pop();
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else if (params.is_replication && NRows == 8) {
        // no window buffer reuse, but need to combine multiple words together
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
              for (loop_counters[0][3] = 0;
                   loop_counters[0][3] < loop_bounds[0][3];
                   loop_counters[0][3]++) {
                // inner memory
                for (loop_counters[1][0] = 0;
                     loop_counters[1][0] < loop_bounds[1][0];
                     loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;
                       loop_counters[1][1] < loop_bounds[1][1];
                       loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;
                         loop_counters[1][2] < loop_bounds[1][2];
                         loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;
                           loop_counters[1][3] < loop_bounds[1][3];
                           loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;
                             loop_counters[1][4] < loop_bounds[1][4];
                             loop_counters[1][4]++) {
                          ac_int<BufferWidth, false> buffer =
                              window_buffer_in.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
                          for (loop_counters[1][5] = 0;
                               loop_counters[1][5] < loop_bounds[1][5];
                               loop_counters[1][5]++) {
                            ac_int<BufferWidth, false> data;
#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              auto temp = buffer.template slc<DATA_WIDTH>(
                                  (dim + 3) * DATA_WIDTH);
                              data.set_slc(dim * DATA_WIDTH, temp);
                            }

                            buffer = window_buffer_in.Pop();

#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              auto temp = buffer.template slc<DATA_WIDTH>(
                                  dim * DATA_WIDTH);
                              data.set_slc((3 + dim) * DATA_WIDTH, temp);
                            }
                            window_buffer_out.Push(data);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {  // bypass

        ac_int<32, false> total_count = loop_bounds[0][0] * loop_bounds[0][1] *
                                        loop_bounds[0][2] * loop_bounds[0][3] *
                                        loop_bounds[1][0] * loop_bounds[1][1] *
                                        loop_bounds[1][2] * loop_bounds[1][3] *
                                        loop_bounds[1][4] * loop_bounds[1][5];
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < total_count; i++) {
          window_buffer_out.Push(window_buffer_in.Pop());
        }
      }
    }
  }

  void transposer() {
    transposer_params.ResetRead();
    input_resp.Reset();
    transpose_out.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = transposer_params.Pop();

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
      loop_bounds[1][params.fyIndex] = 1;

      if (params.has_input_transpose && NRows <= 32) {
        ac_int<DATA_WIDTH> transpose_buffer[NRows][NRows];

        ac_int<32, false> total_count =
            loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
            loop_bounds[0][3] * loop_bounds[1][0] * loop_bounds[1][1] *
            loop_bounds[1][2] * loop_bounds[1][3] * loop_bounds[1][4] *
            loop_bounds[1][5] / NRows;

        ac_int<32, false> count = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (count++ < total_count) {
          for (int c0 = 0; c0 < NRows; c0++) {
            ac_int<BufferWidth, false> bits = 0;

            bool success = (process_matrix_input<InputTypes, NRows, PortWidth,
                                                 BufferWidth, InputTypes...>(
                                params.input_dtype, input_resp, bits) ||
                            ...);

#ifndef __SYNTHESIS__
            if (!success) {
              std::cerr << "Error: matrix input dtype '" << params.input_dtype
                        << "' is not valid" << std::endl;
            }
#endif

#pragma hls_unroll yes
            for (int dim = 0; dim < NRows; dim++) {
              transpose_buffer[dim][c0] =
                  bits.template slc<DATA_WIDTH>(dim * DATA_WIDTH);
            }
          }

          // Write out from tranposeBuffer
          for (int c0 = 0; c0 < NRows; c0++) {
            ac_int<BufferWidth, false> transposed;

#pragma hls_unroll yes
            for (int dim = 0; dim < NRows; dim++) {
              transposed.set_slc(dim * DATA_WIDTH, transpose_buffer[c0][dim]);
            }
            transpose_out.Push(transposed);
          }
        }

      } else {  // passthrough
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (!fetcher_done.read()) {
          ac_int<BufferWidth, false> bits = 0;

          bool success = (process_matrix_input<InputTypes, NRows, PortWidth,
                                               BufferWidth, InputTypes...>(
                              params.input_dtype, input_resp, bits) ||
                          ...);
#ifndef __SYNTHESIS__
          if (!success) {
            std::cerr << "Error: matrix input dtype '" << params.input_dtype
                      << "' is not valid" << std::endl;
          }
#endif
          transpose_out.Push(bits);
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
    transposer_params.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();

      fetcher_params.Push(params);
      writer_params.Push(params);
      reader_params.Push(params);
      winder_buffer_params.Push(params);
      transposer_params.Push(params);
    }
  }
};
