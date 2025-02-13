#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ParamsDeserializer.h"

template <typename Input, int NRows>
SC_MODULE(InputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<Pack1D<Input, NRows> > CCS_INIT_S1(dataResponse);

  Connections::Out<BufferWriteRequest<Input, NRows> > writeRequest[2];
  Connections::Out<ac_int<32, false> > writeControl[2];
  Connections::Out<ac_int<16, false> > readAddress[2];
  Connections::Out<ac_int<32, false> > readControl[2];

  Connections::In<Pack1D<Input, NRows> > CCS_INIT_S1(windowBufferIn);

#ifdef HYBRID_FP8
  Connections::Out<Pack1D<HYBRID_TYPE, NRows> > CCS_INIT_S1(windowBufferOut);
#else
  Connections::Out<Pack1D<Input, NRows> > CCS_INIT_S1(windowBufferOut);
#endif

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(readerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(windowBufferParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposerParams);

  Connections::Combinational<Pack1D<Input, NRows> > transposeOut;

  MatrixParamsDeserializer<0> CCS_INIT_S1(paramsDeserializer);

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int LOOP_WIDTH =
      (10 + int_log2(16 / (IC_DIMENSION < OC_DIMENSION ? IC_DIMENSION
                                                       : OC_DIMENSION)));

  SC_CTOR(InputController) {
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

    SC_THREAD(windowBuffer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(transposer);
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
      if (params.REPLICATION) {
        FX = 7;
      }
      ac_int<4, false> FY = params.loops[1][params.fyIndex];
      ac_int<2, false> STRIDE = params.STRIDE;

      int packingFactor;  // num x values packed in a word
      int boundaryWords;  // num words needed to store the boundary pixels.
                          // essentially ceil(3/packingFactor)
      if (IC_DIMENSION == 4) {
        packingFactor = 1;
        boundaryWords = 3;
      } else if (IC_DIMENSION == 8) {
        packingFactor = 2;
        boundaryWords = 2;
      } else if (IC_DIMENSION == 16) {
        packingFactor = 4;
        boundaryWords = 1;
      } else if (IC_DIMENSION == 32) {
        packingFactor = 8;
        boundaryWords = 1;
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
            // fetching border pixels is a little tricky
            // for the outer tiles, we don't fetch borders (they are known to be
            // 0)

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

            if (params.REPLICATION) {
              loop_bounds[1][params.inputXLoopIndex[1]] /= packingFactor;
            }

            ac_int<4, false> x_min_offset = 0;
            ac_int<4, false> x_max_offset = 0;
            ac_int<4, false> y_min_offset = 0;
            ac_int<4, false> y_max_offset = 0;

            if (params.REPLICATION) {
              if (loop_counters[0][params.inputXLoopIndex[0]] != 0) {
                x_min_offset = (FX - 1) / 2;
                loop_bounds[1][params.inputXLoopIndex[1]] += boundaryWords;
              }
              if (loop_counters[0][params.inputXLoopIndex[0]] !=
                  loop_bounds[0][params.inputXLoopIndex[0]] - 1) {
                x_max_offset = (FX - 1) / 2;
                loop_bounds[1][params.inputXLoopIndex[1]] += boundaryWords;
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
                            STRIDE * params.loops[1][params.inputXLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> X1 =
                            params.loops[0][params.inputXLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> y0 =
                            loop_counters[1][params.inputYLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> y1 =
                            loop_counters[0][params.inputYLoopIndex[0]];
                        ac_int<16, false> Y0 =
                            STRIDE * params.loops[1][params.inputYLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> Y1 =
                            params.loops[0][params.inputYLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> c1 =
                            loop_counters[1][params.reductionLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> C1 =
                            params.loops[1][params.reductionLoopIndex[1]];

                        ac_int<16, false> c = c1 * NRows;
                        ac_int<16, false> C = C1 * NRows;

                        if (isDownsample) {
                          // adjust address for stride
                          x0 = x0 * STRIDE;
                          y0 = y0 * STRIDE;
                        }

                        if (params.REPLICATION) {
                          if (x0 != 0 && x_min_offset == 3) {
                            x0 = x_min_offset +
                                 (x0 - boundaryWords) * packingFactor;
                          } else {
                            x0 = x0 * packingFactor;
                          }
                        }

                        ac_int<16, false> x = (x0 - x_min_offset) + x1 * X0;
                        ac_int<16, false> X = X0 * X1;

                        ac_int<16, false> y = (y0 - y_min_offset) + y1 * Y0;
                        ac_int<16, false> Y = Y0 * Y1;

                        ac_int<32, false> baseAddress = y * X * C + x * C + c;
                        int burstSize = NRows;

                        if (params.REPLICATION) {
                          baseAddress = y * (X / packingFactor) * IC_DIMENSION +
                                        (x / packingFactor) * IC_DIMENSION + c;
                        }

                        ac_int<8, false> headSize = params.headSizeInPowerOfTwo;
                        if (params.CONCAT_INPUT) {
                          ac_int<16, false> mask = (1 << headSize) - 1;
                          baseAddress = (((c >> headSize) * X) << headSize) +
                                        (x << headSize) + (c & mask);
                        }

                        if (params.TRANPOSE_INPUTS) {
                          baseAddress =
                              (c + (x % 16)) * X + (x / 16) * IC_DIMENSION;
                        }

                        MemoryRequest memRequest;
                        memRequest = {params.INPUT_OFFSET +
                                          baseAddress * (Input::width / 8),
                                      burstSize * (Input::width / 8)};

                        addressRequest.Push(memRequest);

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
    transposeOut.ResetRead();

    writeControl[0].Reset();
    writeControl[1].Reset();
    writeRequest[0].Reset();
    writeRequest[1].Reset();

    bool bankSel = 0;

    wait();

    while (true) {
      const MatrixParams params = writerParams.Pop();

      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      if (params.REPLICATION) {
        FX = 7;
      }

      // replication packing factor
      int packingFactor;
      int boundaryWords;
      if (IC_DIMENSION == 4) {
        packingFactor = 1;
        boundaryWords = 3;
      } else if (IC_DIMENSION == 8) {
        packingFactor = 2;
        boundaryWords = 2;
      } else if (IC_DIMENSION == 16) {
        packingFactor = 4;
        boundaryWords = 1;
      } else if (IC_DIMENSION == 32) {
        packingFactor = 8;
        boundaryWords = 1;
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
            if (params.REPLICATION) {
              // make sure to grab border pixels
              loop_bounds[1][params.inputXLoopIndex[1]] =
                  STRIDE * X0 / packingFactor + 2 * boundaryWords;
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
              // TODO: make this dynamic
              ac_int<32, false> total_writes;
              if (!params.REPLICATION) {
                total_writes = (loop_bounds[1][1] * loop_bounds[1][2] *
                                loop_bounds[1][3] * loop_bounds[1][4]) *
                               loop_bounds[1][5];
              } else {
                total_writes =
                    loop_bounds[1][1] * loop_bounds[1][2] * loop_bounds[1][3] *
                    loop_bounds[1][4] *
                    ((STRIDE)*X0 / packingFactor +
                     2 * boundaryWords);  // 2 extra writes for padding
              }

              writeControl[bankSel].Push(total_writes);
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

                        if (params.REPLICATION) {
                          if (x0 != 0) {
                            x0 = x_min_offset +
                                 (x0 - boundaryWords) * packingFactor;
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

                        Pack1D<Input, NRows> data;

                        if ((full_x < 0) || (full_y < 0) ||
                            (full_x >= STRIDE * X0 * X1) ||
                            (full_y >= STRIDE * Y0 * Y1)) {
#pragma hls_unroll yes
                          for (int dims = 0; dims < NRows; dims++) {
                            data[dims].set_zero();
                          }
                        } else {
                          data = transposeOut.Pop();
                        }

                        ac_int<32, false> address =
                            (y0) * (STRIDE * X0 + FX - 1) + (x0);

                        if (params.REPLICATION) {
                          address = y0 * (STRIDE * X0 / packingFactor +
                                          2 * boundaryWords) +
                                    loop_counters[1][params.inputXLoopIndex[1]];
                        }

                        int swapBank =
                            (loop_counters[1][1] == loop_bounds[1][1] - 1) &&
                            (loop_counters[1][2] == loop_bounds[1][2] - 1) &&
                            (loop_counters[1][3] == loop_bounds[1][3] - 1) &&
                            (loop_counters[1][4] == loop_bounds[1][4] - 1) &&
                            (loop_counters[1][5] == loop_bounds[1][5] - 1);
                        // writeControl[bankSel].Push(!swapBank);
                        BufferWriteRequest<Input, NRows> req;
                        req.address = address;
                        req.data = data;
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
              // writeControl[bankSel].Push(0);
              bankSel = !bankSel;
              if (loop_counters[1][0] >= loop_bounds[1][0] - 1) {
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

    readControl[0].Reset();
    readControl[1].Reset();
    readAddress[0].Reset();
    readAddress[1].Reset();

    bool bankSel = 0;

    wait();

    while (true) {
      const MatrixParams params = readerParams.Pop();

      // replication packing factor
      int packingFactor;
      int boundaryWords;
      if (IC_DIMENSION == 4) {
        packingFactor = 1;
        boundaryWords = 3;
      } else if (IC_DIMENSION == 8) {
        packingFactor = 2;
        boundaryWords = 2;
      } else if (IC_DIMENSION == 16) {
        packingFactor = 4;
        boundaryWords = 1;
      } else if (IC_DIMENSION == 32) {
        packingFactor = 8;
        boundaryWords = 1;
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

      if (params.REPLICATION && IC_DIMENSION >= 16) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] * STRIDE /
             packingFactor) +
            2;
      } else if (params.REPLICATION && IC_DIMENSION == 8) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] * STRIDE /
             packingFactor) +
            1;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            // inner memory
            for (loop_counters[1][0] = 0;
                 loop_counters[1][0] < loop_bounds[1][0];
                 loop_counters[1][0]++) {
              ac_int<32, false> total_reads =
                  loop_bounds[1][1] * loop_bounds[1][2] * loop_bounds[1][3] *
                  loop_bounds[1][4] * loop_bounds[1][5];

              readControl[bankSel].Push(total_reads);
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

                        ac_int<16, false> x = STRIDE * x0 + fx;
                        ac_int<16, false> y = STRIDE * y0 + fy;
                        ac_int<16, false> address;
                        if (params.REPLICATION && IC_DIMENSION >= 8) {
                          address = y * (((STRIDE * X0) / packingFactor) +
                                         2 * boundaryWords) +
                                    x0 + fx;
                        } else {
                          if (isDownsample) {
                            address = y0 * X0 + x0;
                          } else {
                            address = y * (STRIDE * X0 + FX - 1) + x;
                          }
                        }
                        // int swapBank =
                        //     (loop_counters[1][1] == loop_bounds[1][1] -
                        //     1)
                        //     && (loop_counters[1][2] == loop_bounds[1][2]
                        //     - 1) && (loop_counters[1][3] ==
                        //     loop_bounds[1][3]
                        //     - 1) && (loop_counters[1][4] ==
                        //     loop_bounds[1][4] - 1) &&
                        //     (loop_counters[1][5]
                        //     == loop_bounds[1][5] - 1);
                        // readControl[bankSel].Push(!swapBank);
                        readAddress[bankSel].Push(address);
                        // CCS_LOG("Reading from address " << address);

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
              bankSel = !bankSel;
              if (loop_counters[1][0] >= loop_bounds[1][0] - 1) {
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
  void windowBuffer() {
    windowBufferParams.ResetRead();
    windowBufferIn.Reset();
    windowBufferOut.Reset();

    wait();

    while (true) {
      const MatrixParams params = windowBufferParams.Pop();
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      int packingFactor;    // num x values packed in a word in the buffer
      int unrollingFactor;  // num x values packed in a word sent to the
      // systolic array
      int additionalUnrollingFactor;  // additional x values packed in a word
                                      // sent to the systolic array, but are
                                      // unused by the systolic array
      constexpr int initialReuseFactor = (IC_DIMENSION == 8) ? 1 : 3;
      int shiftFactor;

      if (IC_DIMENSION == 4) {
        packingFactor = 1;
        unrollingFactor = 1;
        additionalUnrollingFactor = 0;
      } else if (IC_DIMENSION == 8) {
        packingFactor = 2;
        unrollingFactor = 2;
        additionalUnrollingFactor = 0;
        shiftFactor = 1;
      } else if (IC_DIMENSION == 16) {
        packingFactor = 4;
        unrollingFactor = 4;
        additionalUnrollingFactor = 1;
        shiftFactor = 2;
      } else if (IC_DIMENSION == 32) {
        packingFactor = 8;
        unrollingFactor = 7;
        additionalUnrollingFactor = 0;
        shiftFactor = 2;
      }

      if (params.REPLICATION && (IC_DIMENSION == 16 || IC_DIMENSION == 32)) {
        // #pragma hls_pipeline_init_interval 1
        // #pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
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
                        Pack1D<Input, NRows> data;

                        Pack1D<Input, NRows> buffer = windowBufferIn.Pop();
#pragma hls_unroll yes
                        for (int x = 0; x < 3; x++) {
#pragma hls_unroll yes
                          for (int dim = 0; dim < 3; dim++) {
                            // grab last 3 x values from buffer
                            data.value[x * 3 + dim] =
                                buffer[(packingFactor - 3 + x) * 3 + dim];
                          }
                        }

                        buffer = windowBufferIn.Pop();
#pragma hls_unroll yes
                        for (int x = 3;
                             x < unrollingFactor + additionalUnrollingFactor;
                             x++) {
#pragma hls_unroll yes
                          for (int dim = 0; dim < 3; dim++) {
                            data[x * 3 + dim] = buffer[(x - 3) * 3 + dim];
                          }
                        }

                        windowBufferOut.Push(data);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5] * 2 - 2;
                             loop_counters[1][5] += 2) {
                          // shift two pixels over
#pragma hls_unroll yes
                          for (int x = 0; x < packingFactor +
                                                  additionalUnrollingFactor - 2;
                               x++) {
#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              data[x * 3 + dim] = data[(x + 2) * 3 + dim];
                            }
                          }

                          // grab 2 new pixels from buffer
                          int bufferPosition =
                              (loop_counters[1][5] + unrollingFactor +
                               additionalUnrollingFactor - 3) %
                              packingFactor;
#pragma hls_unroll yes
                          for (int x = unrollingFactor +
                                       additionalUnrollingFactor - 2;
                               x < unrollingFactor + additionalUnrollingFactor;
                               x++) {
#pragma hls_unroll yes
                            for (int dim = 0; dim < 3; dim++) {
                              data[x * 3 + dim] =
                                  buffer[(bufferPosition + x -
                                          (unrollingFactor +
                                           additionalUnrollingFactor - 2)) *
                                             3 +
                                         dim];
                            }
                          }

                          windowBufferOut.Push(data);

                          if (bufferPosition == packingFactor - 2) {
                            buffer = windowBufferIn.Pop();
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
      } else if (params.REPLICATION && IC_DIMENSION == 8) {
        // no window buffer reuse, but need to combine multiple words together
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
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
                        Pack1D<Input, NRows> buffer = windowBufferIn.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5];
                             loop_counters[1][5]++) {
                          Pack1D<Input, NRows> data;
#pragma hls_unroll yes
                          for (int dim = 0; dim < 3; dim++) {
                            data[dim] = buffer[3 + dim];
                          }

                          buffer = windowBufferIn.Pop();

#pragma hls_unroll yes
                          for (int dim = 0; dim < 3; dim++) {
                            data[3 + dim] = buffer[dim];
                          }
                          windowBufferOut.Push(data);
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

        ac_int<32, false> total_count =
            loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
            loop_bounds[1][0] * loop_bounds[1][1] * loop_bounds[1][2] *
            loop_bounds[1][3] * loop_bounds[1][4] * loop_bounds[1][5];
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < total_count; i++) {
          windowBufferOut.Push(windowBufferIn.Pop());
        }
      }
    }
  }

  void transposer() {
    transposerParams.ResetRead();
    dataResponse.Reset();
    transposeOut.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = transposerParams.Pop();

      ac_int<4, false> FX = params.loops[1][params.fxIndex];
      if (params.REPLICATION) {
        FX = 7;
      }

      int packingFactor;  // num x values packed in a word
      int boundaryWords;  // num words needed to store the boundary pixels.
                          // essentially ceil(3/packingFactor)
      if (IC_DIMENSION == 4) {
        packingFactor = 1;
        boundaryWords = 3;
      } else if (IC_DIMENSION == 8) {
        packingFactor = 2;
        boundaryWords = 2;
      } else if (IC_DIMENSION == 16) {
        packingFactor = 4;
        boundaryWords = 1;
      } else if (IC_DIMENSION == 32) {
        packingFactor = 8;
        boundaryWords = 1;
      }

      ac_int<4, false> FY = params.loops[1][params.fyIndex];

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

      if (params.TRANPOSE_INPUTS && IC_DIMENSION <= 32) {
        Input transposeBuffer[NRows][NRows];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
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
                        // innermost loop must be X0, and must be a multiple
                        // of NRows
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5] / NRows;
                             loop_counters[1][5]++) {
                          for (int c0 = 0; c0 < NRows; c0++) {
                            Pack1D<Input, NRows> originalValue =
                                dataResponse.Pop();
#pragma hls_unroll yes
                            for (int dim = 0; dim < NRows; dim++) {
                              transposeBuffer[dim][c0] = originalValue[dim];
                            }
                          }

                          // Write out from tranposeBuffer
                          for (int c0 = 0; c0 < NRows; c0++) {
                            Pack1D<Input, NRows> transposedValue;

#pragma hls_unroll yes
                            for (int dim = 0; dim < NRows; dim++) {
                              transposedValue[dim] = transposeBuffer[c0][dim];
                            }
                            transposeOut.Push(transposedValue);
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
      } else {  // passthrough
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
              // fetching border pixels is a little tricky
              // for the outer tiles, we don't fetch borders (they are known
              // to be 0)

              // reset loop bounds
              if (isDownsample) {
                // don't include STRIDE for downsample
                loop_bounds[1][params.inputXLoopIndex[1]] =
                    params.loops[1][params.inputXLoopIndex[1]];
                loop_bounds[1][params.inputYLoopIndex[1]] =
                    params.loops[1][params.inputYLoopIndex[1]];
              } else {
                loop_bounds[1][params.inputXLoopIndex[1]] =
                    params.loops[1][params.inputXLoopIndex[1]] * params.STRIDE;
                loop_bounds[1][params.inputYLoopIndex[1]] =
                    params.loops[1][params.inputYLoopIndex[1]] * params.STRIDE;
              }

              if (params.REPLICATION) {
                loop_bounds[1][params.inputXLoopIndex[1]] /= packingFactor;
              }

              ac_int<4, false> x_min_offset = 0;
              ac_int<4, false> x_max_offset = 0;
              ac_int<4, false> y_min_offset = 0;
              ac_int<4, false> y_max_offset = 0;

              if (params.REPLICATION) {
                if (loop_counters[0][params.inputXLoopIndex[0]] != 0) {
                  x_min_offset = (FX - 1) / 2;
                  loop_bounds[1][params.inputXLoopIndex[1]] += boundaryWords;
                }
                if (loop_counters[0][params.inputXLoopIndex[0]] !=
                    loop_bounds[0][params.inputXLoopIndex[0]] - 1) {
                  x_max_offset = (FX - 1) / 2;
                  loop_bounds[1][params.inputXLoopIndex[1]] += boundaryWords;
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
                          transposeOut.Push(dataResponse.Pop());

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
  }

  void read_params() {
    paramsIn.ResetRead();
    fetcherParams.ResetWrite();
    writerParams.ResetWrite();
    readerParams.ResetWrite();
    windowBufferParams.ResetWrite();
    transposerParams.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = paramsIn.Pop();

      fetcherParams.Push(params);
      writerParams.Push(params);
      readerParams.Push(params);
      windowBufferParams.Push(params);
      transposerParams.Push(params);
    }
  }
};
