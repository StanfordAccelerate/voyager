#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ParamsDeserializer.h"

template <typename Scale, int NRows>
SC_MODULE(InputScaleController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<ac_int<Scale::width, false>> CCS_INIT_S1(dataResponse);

  Connections::Out<BufferWriteRequest<ac_int<Scale::width, false>>>
      writeRequest[2];
  Connections::Out<BufferReadRequest> readAddress[2];

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(readerParams);

  MatrixParamsDeserializer<3> CCS_INIT_S1(paramsDeserializer);

  static constexpr int LOOP_WIDTH = 10;

  SC_CTOR(InputScaleController) {
    paramsDeserializer.clk(clk);
    paramsDeserializer.rstn(rstn);
    paramsDeserializer.serialParamsIn(serialParamsIn);
    paramsDeserializer.paramsOut(paramsIn);

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
    addressRequest.Reset();
    fetcherParams.ResetRead();

    wait();

    while (true) {
      const MatrixParams params = fetcherParams.Pop();

      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      if (params.is_replication) {
        FX = 7;
      }
      ac_int<4, false> FY = params.loops[1][params.fyIndex];
      ac_int<2, false> STRIDE = params.STRIDE;

      // replication packing factor
      int packingFactor;
      if (NRows == 16) {
        packingFactor = 4;
      } else if (NRows == 32) {
        packingFactor = 8;
      }

      bool isDownsample = FX == 1 && FY == 1;

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;

      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;
                 loop_counters[0][3] < loop_bounds[0][3];
                 loop_counters[0][3]++) {
              // fetching border pixels is a little tricky
              // for the outer tiles, we don't fetch borders (they are known to
              // be 0)

              // reset loop bounds
              if (isDownsample) {
                // don't include STRIDE for downsample
                loop_bounds[1][params.inputXLoopIndex[1]] =
                    params.loops[1][params.inputXLoopIndex[1]];
                loop_bounds[1][params.inputYLoopIndex[1]] =
                    params.loops[1][params.inputYLoopIndex[1]];
              } else {
                loop_bounds[1][params.inputXLoopIndex[1]] =
                    params.loops[1][params.inputXLoopIndex[1]] * STRIDE;
                loop_bounds[1][params.inputYLoopIndex[1]] =
                    params.loops[1][params.inputYLoopIndex[1]] * STRIDE;
              }

              if (params.is_replication) {
                loop_bounds[1][params.inputXLoopIndex[1]] /= packingFactor;
              }

              ac_int<4, false> x_min_offset = 0;
              ac_int<4, false> x_max_offset = 0;
              ac_int<4, false> y_min_offset = 0;
              ac_int<4, false> y_max_offset = 0;

              if (params.is_replication) {
                if (loop_counters[0][params.inputXLoopIndex[0]] != 0) {
                  x_min_offset = (FX - 1) / 2;
                  loop_bounds[1][params.inputXLoopIndex[1]] += 1;
                }
                if (loop_counters[0][params.inputXLoopIndex[0]] !=
                    loop_bounds[0][params.inputXLoopIndex[0]] - 1) {
                  x_max_offset = (FX - 1) / 2;
                  loop_bounds[1][params.inputXLoopIndex[1]] += 1;
                }
              } else {
                if (loop_counters[0][params.inputXLoopIndex[0]] != 0) {
                  x_min_offset = (FX - 1) / 2;
                  loop_bounds[1][params.inputXLoopIndex[1]] += (FX - 1) / 2;
                }

                if (loop_counters[0][params.inputXLoopIndex[0]] !=
                    loop_bounds[0][params.inputXLoopIndex[0]] - 1) {
                  x_max_offset = (FX - 1) / 2;
                  loop_bounds[1][params.inputXLoopIndex[1]] += (FX - 1) / 2;
                }
              }

              if (loop_counters[0][params.inputYLoopIndex[0]] != 0) {
                y_min_offset = (FY - 1) / 2;
                loop_bounds[1][params.inputYLoopIndex[1]] += (FY - 1) / 2;
              }

              if (loop_counters[0][params.inputYLoopIndex[0]] !=
                  loop_bounds[0][params.inputYLoopIndex[0]] - 1) {
                y_max_offset = (FY - 1) / 2;
                loop_bounds[1][params.inputYLoopIndex[1]] += (FY - 1) / 2;
              }

// inner memory
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
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
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5];
                             loop_counters[1][5]++) {
                          ac_int<LOOP_WIDTH, false> x0 =
                              loop_counters[1][params.inputXLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> x1 =
                              loop_counters[0][params.inputXLoopIndex[0]];
                          ac_int<16, false> X0 =
                              STRIDE *
                              params.loops[1][params.inputXLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> X1 =
                              params.loops[0][params.inputXLoopIndex[0]];
                          ac_int<LOOP_WIDTH, false> y0 =
                              loop_counters[1][params.inputYLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> y1 =
                              loop_counters[0][params.inputYLoopIndex[0]];
                          ac_int<16, false> Y0 =
                              STRIDE *
                              params.loops[1][params.inputYLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> Y1 =
                              params.loops[0][params.inputYLoopIndex[0]];
                          ac_int<LOOP_WIDTH, false> c1 =
                              loop_counters[1][params.reductionLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> C1 =
                              loop_bounds[1][params.reductionLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> c2 =
                              loop_counters[0][params.reductionLoopIndex[0]];
                          ac_int<LOOP_WIDTH, false> C2 =
                              params.loops[0][params.reductionLoopIndex[0]];

                          ac_int<16, false> c = c2 * C1 + c1;
                          ac_int<16, false> C = C2 * C1;

                          if (isDownsample) {
                            // adjust address for stride
                            x0 = x0 * STRIDE;
                            y0 = y0 * STRIDE;
                          }

                          if (params.is_replication) {
                            if (x0 != 0 && x_min_offset == 3) {
                              x0 = x_min_offset + (x0 - 1) * packingFactor;
                            } else {
                              x0 = x0 * packingFactor;
                            }
                          }

                          ac_int<16, false> x = (x0 - x_min_offset) + x1 * X0;
                          ac_int<16, false> X = X0 * X1;

                          ac_int<16, false> y = (y0 - y_min_offset) + y1 * Y0;
                          ac_int<16, false> Y = Y0 * Y1;

                          ac_int<32, false> address = y * X * C + x * C + c;

                          if (params.is_replication) {
                            address = y * (X / packingFactor) * NRows +
                                      (x / packingFactor) * NRows + c;
                          }

                          MemoryRequest request = {
                              params.INPUT_SCALE_OFFSET +
                                  address * Scale::width / 8,
                              Scale::width / 8};

                          addressRequest.Push(request);

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
    }
  }

  void writer() {
    writerParams.ResetRead();
    dataResponse.Reset();
    writeRequest[0].Reset();
    writeRequest[1].Reset();

    bool bankSel = 0;

    wait();

    while (true) {
      const MatrixParams params = writerParams.Pop();

      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      if (params.is_replication) {
        FX = 7;
      }

      // replication packing factor
      int packingFactor;
      if (NRows == 16) {
        packingFactor = 4;
      } else if (NRows == 32) {
        packingFactor = 8;
      }

      ac_int<4, false> FY = params.loops[1][params.fyIndex];
      ac_int<2, false> STRIDE = params.STRIDE;

      bool isDownsample = FX == 1 && FY == 1;

      ac_int<4, false> fx_bound = (FX - 1) / 2;
      ac_int<4, false> fy_bound = (FY - 1) / 2;

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;

      ac_int<LOOP_WIDTH, false> X1 = params.loops[0][params.inputXLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.inputXLoopIndex[1]];

      ac_int<LOOP_WIDTH, false> Y0 = params.loops[1][params.inputYLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> Y1 = params.loops[0][params.inputYLoopIndex[0]];

      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;
                 loop_counters[0][3] < loop_bounds[0][3];
                 loop_counters[0][3]++) {
              if (isDownsample) {
                // don't include STRIDE for downsample
                STRIDE = 1;
              }

              // reset loop bounds
              loop_bounds[1][params.inputXLoopIndex[1]] =
                  params.loops[1][params.inputXLoopIndex[1]] * STRIDE;
              loop_bounds[1][params.inputYLoopIndex[1]] =
                  params.loops[1][params.inputYLoopIndex[1]] * STRIDE;

              ac_int<4, false> x_min_offset = fx_bound;
              ac_int<4, false> y_min_offset = fy_bound;
              if (params.is_replication) {
                // make sure to grab border pixels
                loop_bounds[1][params.inputXLoopIndex[1]] =
                    STRIDE * X0 / packingFactor + 2;
                loop_bounds[1][params.inputYLoopIndex[1]] += FY - 1;
              } else {
                loop_bounds[1][params.inputXLoopIndex[1]] += FX - 1;
                loop_bounds[1][params.inputYLoopIndex[1]] += FY - 1;
              }

              // inner memory
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
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
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5];
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
                          ac_int<LOOP_WIDTH, true> C1 =
                              loop_bounds[1][params.reductionLoopIndex[1]];

                          if (params.is_replication) {
                            if (x0 != 0) {
                              x0 = x_min_offset + (x0 - 1) * packingFactor;
                            }
                          }

                          ac_int<16, true> full_x, full_y;
                          if (isDownsample) {
                            full_x =
                                (x0 * STRIDE - x_min_offset) + x1 * STRIDE * X0;
                            full_y =
                                (y0 * STRIDE - y_min_offset) + y1 * STRIDE * Y0;
                          } else {
                            full_x = (x0 - x_min_offset) + x1 * STRIDE * X0;
                            full_y = (y0 - y_min_offset) + y1 * STRIDE * Y0;
                          }

                          ac_int<Scale::width, false> data;

                          if ((full_x < 0) || (full_y < 0) ||
                              (full_x >= STRIDE * X0 * X1) ||
                              (full_y >= STRIDE * Y0 * Y1)) {
                            data = Scale::one().bits_rep();
                          } else {
                            data = dataResponse.Pop();
                          }

                          ac_int<32, false> address =
                              y0 * (X0 * STRIDE + FX - 1) * C1 + x0 * C1 + c1;

                          if (params.is_replication) {
                            address =
                                y0 * (X0 * STRIDE / packingFactor + 2) * C1 +
                                loop_counters[1][params.inputXLoopIndex[1]] *
                                    C1 +
                                c1;
                          }

                          BufferWriteRequest<ac_int<Scale::width, false>> req;
                          req.address = address;
                          req.data = data;
                          req.last =
                              loop_counters[1][5] == loop_bounds[1][5] - 1 &&
                              loop_counters[1][4] == loop_bounds[1][4] - 1 &&
                              loop_counters[1][3] == loop_bounds[1][3] - 1 &&
                              loop_counters[1][2] == loop_bounds[1][2] - 1 &&
                              loop_counters[1][1] == loop_bounds[1][1] - 1 &&
                              loop_counters[1][0] == loop_bounds[1][0] - 1;
                          writeRequest[bankSel].Push(req);

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
    readerParams.ResetRead();

    readAddress[0].Reset();
    readAddress[1].Reset();

    bool bankSel = 0;

    wait();

    while (true) {
      const MatrixParams params = readerParams.Pop();

      // replication packing factor
      int packingFactor;
      if (NRows == 16) {
        packingFactor = 4;
      } else if (NRows == 32) {
        packingFactor = 8;
      }

      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.inputXLoopIndex[1]];

      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      ac_int<4, false> FY = params.loops[1][params.fyIndex];
      bool isDownsample = FX == 1 && FY == 1;

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];
      ac_int<2, false> STRIDE = params.STRIDE;

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      if (params.is_replication) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] * STRIDE /
             packingFactor) +
            2;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;
                 loop_counters[0][3] < loop_bounds[0][3];
                 loop_counters[0][3]++) {
              // int total_reads = loop_bounds[1][0] * loop_bounds[1][1] *
              //                   loop_bounds[1][2] * loop_bounds[1][3] *
              //                   loop_bounds[1][4] * loop_bounds[1][5];
              // readControl[bankSel].Push(total_reads);
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
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5];
                             loop_counters[1][5]++) {
                          ac_int<LOOP_WIDTH, false> x0 =
                              loop_counters[1][params.inputXLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> X0 =
                              params.loops[1][params.inputXLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> y0 =
                              loop_counters[1][params.inputYLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> Y0 =
                              params.loops[1][params.inputYLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> fx =
                              loop_counters[1][params.fxIndex];
                          ac_int<LOOP_WIDTH, false> fy =
                              loop_counters[1][params.fyIndex];
                          ac_int<LOOP_WIDTH, false> c1 =
                              loop_counters[1][params.reductionLoopIndex[1]];
                          ac_int<LOOP_WIDTH, false> C1 =
                              params.loops[1][params.reductionLoopIndex[1]];

                          ac_int<16, false> x = STRIDE * x0 + fx;
                          ac_int<16, false> y = STRIDE * y0 + fy;
                          ac_int<32, false> address;
                          if (params.is_replication) {
                            address =
                                y * (((STRIDE * X0) / packingFactor) + 2) + x0 +
                                fx;
                          } else {
                            if (isDownsample) {
                              address = y0 * X0 * C1 + x0 * C1 + c1;
                            } else {
                              address =
                                  y * (STRIDE * X0 + FX - 1) * C1 + x * C1 + c1;
                            }
                          }

                          BufferReadRequest req;
                          req.address = address;
                          req.last =
                              loop_counters[1][5] == loop_bounds[1][5] - 1 &&
                              loop_counters[1][4] == loop_bounds[1][4] - 1 &&
                              loop_counters[1][3] == loop_bounds[1][3] - 1 &&
                              loop_counters[1][2] == loop_bounds[1][2] - 1 &&
                              loop_counters[1][1] == loop_bounds[1][1] - 1 &&
                              loop_counters[1][0] == loop_bounds[1][0] - 1;
                          readAddress[bankSel].Push(req);

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

  void read_params() {
    paramsIn.ResetRead();
    fetcherParams.ResetWrite();
    writerParams.ResetWrite();
    readerParams.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = paramsIn.Pop();

      if (params.is_mx_op) {
        fetcherParams.Push(params);
        writerParams.Push(params);
        readerParams.Push(params);
      }
    }
  }
};
