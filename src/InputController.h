#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename DTYPE, int NROWS>
SC_MODULE(InputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> paramsIn;

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(dataResponse);

  Connections::Out<int> writeAddress[2];
  Connections::Out<Pack1D<DTYPE, NROWS> > writeData[2];
  Connections::Out<int> writeControl[2];
  Connections::Out<int> readAddress[2];
  Connections::Out<int> readControl[2];

  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(windowBufferIn);
  Connections::Out<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(windowBufferOut);

  Connections::Combinational<Params> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<Params> CCS_INIT_S1(writerParams);
  Connections::Combinational<Params> CCS_INIT_S1(readerParams);
  Connections::Combinational<Params> CCS_INIT_S1(windowBufferParams);

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

    SC_THREAD(windowBuffer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetcher() {
    addressRequest.Reset();
    fetcherParams.ResetRead();

    wait();

    while (true) {
      Params params = fetcherParams.Pop();

      int req_count = 0;

      int FX = params.loops[1][params.fxIndex];
      if (params.REPLICATION) {
        FX = 7;
      }
      int FY = params.loops[1][params.fyIndex];

      int loop_counters[2][6];
      int loop_bounds[2][6];

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

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
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
            loop_bounds[1][params.inputXLoopIndex[1]] =
                params.loops[1][params.inputXLoopIndex[1]] * params.STRIDE;
            loop_bounds[1][params.inputYLoopIndex[1]] =
                params.loops[1][params.inputYLoopIndex[1]] * params.STRIDE;

            int x_min_offset = 0;
            int x_max_offset = 0;
            int y_min_offset = 0;
            int y_max_offset = 0;

            if (loop_counters[0][params.inputXLoopIndex[0]] != 0) {
              x_min_offset = (FX - 1) / 2;
              loop_bounds[1][params.inputXLoopIndex[1]] += (FX - 1) / 2;
            }

            if (loop_counters[0][params.inputXLoopIndex[0]] !=
                loop_bounds[0][params.inputXLoopIndex[0]] - 1) {
              x_max_offset = (FX - 1) / 2;
              loop_bounds[1][params.inputXLoopIndex[1]] += (FX - 1) / 2;
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
                           params.REPLICATION ? loop_counters[1][5] += 4
                                              : loop_counters[1][5]++) {
                        MemoryRequest memRequest;
                        if (params.matMul) {
                          int m0 = loop_counters[1][params.inputXLoopIndex[1]];
                          int m1 = loop_counters[0][params.inputXLoopIndex[0]];
                          int M0 = params.loops[1][params.inputXLoopIndex[1]];
                          int N1 =
                              params.loops[1][params.reductionLoopIndex[1]];
                          int n1 =
                              loop_counters[1][params.reductionLoopIndex[1]];

                          // change addressing
                          int baseAddress =
                              (m0 + m1 * M0) * (N1 * NROWS) + n1 * NROWS;
                          int burstSize = NROWS;
                          memRequest = {params.INPUT_OFFSET + baseAddress,
                                        burstSize};
                        } else {
                          int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                          int x1 = loop_counters[0][params.inputXLoopIndex[0]];
                          int X0 = params.STRIDE *
                                   params.loops[1][params.inputXLoopIndex[1]];
                          int X1 = params.loops[0][params.inputXLoopIndex[0]];
                          int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                          int y1 = loop_counters[0][params.inputYLoopIndex[0]];
                          int Y0 = params.STRIDE *
                                   params.loops[1][params.inputYLoopIndex[1]];
                          int Y1 = params.loops[0][params.inputYLoopIndex[0]];
                          int c1 =
                              loop_counters[1][params.reductionLoopIndex[1]];
                          int C1 =
                              params.loops[1][params.reductionLoopIndex[1]];

                          int c = c1 * NROWS;
                          int C = C1 * NROWS;

                          int x = (x0 - x_min_offset) + x1 * X0;
                          int X = X0 * X1;

                          int y = (y0 - y_min_offset) + y1 * Y0;
                          int Y = Y0 * Y1;

                          int baseAddress = y * X * C + x * C + c;
                          int burstSize = NROWS;

                          if (params.REPLICATION) {
                            baseAddress = y * (X / 4) * 16 + (x / 4) * 16 + c;
                            // baseAddress = baseAddress >> 2;

                            CCS_LOG("y: " << y << " x: " << x << " c: " << c
                                          << " addr: " << baseAddress);
                          }

                          memRequest = {params.INPUT_OFFSET + baseAddress,
                                        burstSize};
                        }

                        // CCS_LOG("memory request: " << memRequest);
                        addressRequest.Push(memRequest);
                        req_count++;
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

  void writer() {
    writerParams.ResetRead();
    dataResponse.Reset();

    writeControl[0].Reset();
    writeControl[1].Reset();
    writeAddress[0].Reset();
    writeAddress[1].Reset();
    writeData[0].Reset();
    writeData[1].Reset();

    wait();

    while (true) {
      Params params = writerParams.Pop();

      bool bankSel = 0;

      int FX = params.loops[1][params.fxIndex];
      if (params.REPLICATION) {
        FX = 7;
      }
      int FY = params.loops[1][params.fyIndex];
      int fx_bound = (FX - 1) / 2;
      int fy_bound = (FY - 1) / 2;

      int loop_counters[2][6];
      int loop_bounds[2][6];

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

      int X1 = params.loops[0][params.inputXLoopIndex[0]];
      int X0 = params.loops[1][params.inputXLoopIndex[1]];

      int Y0 = params.loops[1][params.inputYLoopIndex[1]];
      int Y1 = params.loops[0][params.inputYLoopIndex[0]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            // reset loop bounds
            loop_bounds[1][params.inputXLoopIndex[1]] =
                params.loops[1][params.inputXLoopIndex[1]] * params.STRIDE;
            loop_bounds[1][params.inputYLoopIndex[1]] =
                params.loops[1][params.inputYLoopIndex[1]] * params.STRIDE;

            int x_min_offset = fx_bound;
            int y_min_offset = fy_bound;
            loop_bounds[1][params.inputXLoopIndex[1]] += FX - 1;
            loop_bounds[1][params.inputYLoopIndex[1]] += FY - 1;

            // inner memory
            for (loop_counters[1][0] = 0;
                 loop_counters[1][0] < loop_bounds[1][0];
                 loop_counters[1][0]++) {
              // TODO: make this dynamic

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
                      Pack1D<DTYPE, NROWS> data;
                      Pack1D<DTYPE, NROWS> temp;

                      if (params.REPLICATION) {
                        /*
                         * Start with left boundary
                         * First 3 words of the packed word need to be written
                         */
                        int x1 = loop_counters[0][params.inputXLoopIndex[0]];

                        int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                        int y1 = loop_counters[0][params.inputYLoopIndex[0]];

                        int full_y =
                            (y0 - y_min_offset) + y1 * params.STRIDE * Y0;

                        int starting_x = -fx_bound + x1 * params.STRIDE * X0;

                        // if outside boundary, write 0s
                        if ((starting_x < 0) || (full_y < 0) ||
                            (starting_x >= params.STRIDE * X0 * X1) ||
                            (full_y >= params.STRIDE * Y0 * Y1)) {
#pragma hls_unroll yes
                          for (int dims = 0; dims < 3 * 3; dims++) {
                            data[dims] = 0;
                          }
                        }

                        // if not outside boundary, write the words using
                        // offsets
                        else {
                          temp = dataResponse.Pop();
                          // CCS_LOG("left boundary read");

#pragma hls_unroll yes
                          for (int word = 0; word < 3; word++) {
                            int read_offset = 3 * ((starting_x + word) % 4);
                            int write_offset = 3 * (word % 4);

#pragma hls_unroll yes
                            for (int channel = 0; channel < 3; channel++) {
                              data[write_offset + channel] =
                                  temp[read_offset + channel];
                            }
                          }
                        }

                        /*
                         * Now go through the entire tile
                         */
                        for (int x = 0; x < params.STRIDE * X0; x += 4) {
                          int full_x = x + x1 * params.STRIDE * X0;
                          // CCS_LOG("x: " << x << " full_x: " << full_x);
                          // padding
                          if ((full_y < 0) ||
                              (full_y >= params.STRIDE * Y0 * Y1)) {
                            // CCS_LOG(x << " : zero pad");
#pragma hls_unroll yes
                            for (int dims = 0; dims < 3; dims++) {
                              int index = 3 * ((x + fx_bound) % 4) + dims;
                              data.value[index] = 0;
                            }
                          } else {
                            temp = dataResponse.Pop();
                            // CCS_LOG("tile read");

#pragma hls_unroll yes
                            for (int i = 0; i < 3; i++) {
                              int index = 3 * (full_x % 4) + i;
                              data.value[3 * ((x + fx_bound) % 4) + i] =
                                  temp.value[index];
                            }
                          }

                          int address =
                              (y0) * (((params.STRIDE * X0) >> 2) + 2) +
                              (x >> 2);

                          writeControl[bankSel].Push(1);
                          writeAddress[bankSel].Push(address);
                          writeData[bankSel].Push(data);
                          CCS_LOG(y0 << " " << x << "addr: " << address
                                     << " data: " << data);

                          if ((full_y < 0) ||
                              (full_y >= (params.STRIDE * Y0 * Y1))) {
#pragma hls_unroll yes
                            for (int next_x = x + 1; next_x < x + 4; next_x++) {
#pragma hls_unroll yes
                              for (int dims = 0; dims < 3; dims++) {
                                int index =
                                    3 * ((next_x + fx_bound) % 4) + dims;
                                data.value[index] = 0;
                              }
                            }
                          } else {
#pragma hls_unroll yes
                            for (int next_x = x + 1; next_x < x + 4; next_x++) {
                              full_x = next_x + x1 * params.STRIDE * X0;
#pragma hls_unroll yes
                              for (int i = 0; i < 3; i++) {
                                int index = 3 * (full_x % 4) + i;
                                data.value[3 * ((next_x + fx_bound) % 4) + i] =
                                    temp.value[index];
                              }
                            }
                          }
                        }

                        /*
                         * Now handle the right boundary
                         */
                        int x = params.STRIDE * X0;
                        int full_x = x + x1 * params.STRIDE * X0;

                        if ((full_x < 0) || (full_y < 0) ||
                            (full_x >= (X1 * params.STRIDE * X0)) ||
                            (full_y >= (Y1 * params.STRIDE * Y0))) {
#pragma hls_unroll yes
                          for (int dims = 0; dims < 3; dims++) {
                            int index = 3 * ((x + fx_bound) % 4) + dims;
                            data.value[index] = 0;
                          }
                        } else {
                          temp = dataResponse.Pop();
                          // CCS_LOG("right boundary read");

#pragma hls_unroll yes
                          for (int i = 0; i < 3; i++) {
                            int index = 3 * (full_x % 4) + i;
                            data.value[3 * ((x + fx_bound) % 4) + i] =
                                temp.value[index];
                          }
                        }

                        int address =
                            (y0) * (((params.STRIDE * X0) >> 2) + 2) + (x >> 2);

                        writeControl[bankSel].Push(1);
                        writeAddress[bankSel].Push(address);
                        writeData[bankSel].Push(data);
                        CCS_LOG("(1) addr: " << address << " data: " << data);

                        if ((full_x < 0) || (full_y < 0) ||
                            (full_x >= (X1 * params.STRIDE * X0)) ||
                            (full_y >= (Y1 * params.STRIDE * Y0))) {
#pragma hls_unroll yes
                          for (int next_x = x + 1; next_x < x + 3; next_x++) {
#pragma hls_unroll yes
                            for (int dims = 0; dims < 3; dims++) {
                              int index = 3 * ((next_x + fx_bound) % 4) + dims;
                              data.value[index] = 0;
                            }
                          }
                        } else {
#pragma hls_unroll yes
                          for (int next_x = x + 1; next_x < x + 3; next_x++) {
                            int next_full_x = next_x + x1 * params.STRIDE * X0;
#pragma hls_unroll yes
                            for (int i = 0; i < 3; i++) {
                              int index = 3 * (next_full_x % 4) + i;
                              data.value[3 * ((next_x + fx_bound) % 4) + i] =
                                  temp.value[index];
                            }
                          }
                        }

                        int next_address =
                            (y0) * (((params.STRIDE * X0) >> 2) + 2) +
                            (x >> 2) + 1;
                        writeControl[bankSel].Push(1);
                        writeAddress[bankSel].Push(next_address);
                        writeData[bankSel].Push(data);
                        CCS_LOG("(2)addr: " << next_address
                                            << " data: " << data);

                        // CCS_LOG("done " << loop_counters[1][4] << " out of "
                        //                 << loop_bounds[1][4]);
                      }

                      else {
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5];
                             loop_counters[1][5]++) {
                          int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                          int x1 = loop_counters[0][params.inputXLoopIndex[0]];

                          int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                          int y1 = loop_counters[0][params.inputYLoopIndex[0]];

                          int full_x =
                              (x0 - x_min_offset) + x1 * params.STRIDE * X0;
                          int full_y =
                              (y0 - y_min_offset) + y1 * params.STRIDE * Y0;

                          if ((full_x < 0) || (full_y < 0) ||
                              (full_x >= params.STRIDE * X0 * X1) ||
                              (full_y >= params.STRIDE * Y0 * Y1)) {
                            data = 0;
                            CCS_LOG(full_x << ", " << full_y);
                          } else {
                            data = dataResponse.Pop();
                            CCS_LOG(data);
                          }

                          int address =
                              (y0) * (params.STRIDE * X0 + FX - 1) + (x0);

                          writeControl[bankSel].Push(1);
                          writeAddress[bankSel].Push(address);
                          writeData[bankSel].Push(data);
                          CCS_LOG("addr: " << address << " data: " << data);
                        }
                      }

                      // CCS_LOG("address: " << address << " data: " << data);
                    }
                  }
                }
              }

              writeControl[bankSel].Push(0);
              bankSel = !bankSel;
            }
          }
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

    wait();

    while (true) {
      Params params = readerParams.Pop();

      int FX = params.loops[1][params.fxIndex];
      int FY = params.loops[1][params.fyIndex];

      bool bankSel = 0;

      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      if (params.REPLICATION) {
        loop_bounds[1][params.inputXLoopIndex[1]] =
            (loop_bounds[1][params.inputXLoopIndex[1]] / 2) + 1;
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
                        int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                        int X0 = params.loops[1][params.inputXLoopIndex[1]];
                        int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                        int Y0 = params.loops[1][params.inputYLoopIndex[1]];
                        int fx = loop_counters[1][params.fxIndex];
                        int fy = loop_counters[1][params.fyIndex];

                        int x = params.STRIDE * x0 + fx;
                        int y = params.STRIDE * y0 + fy;
                        int address;
                        if (params.REPLICATION) {
                          address =
                              y * (((params.STRIDE * X0) >> 2) + 2) + x0 + fx;
                        } else {
                          address = y * (params.STRIDE * X0 + FX - 1) + x;
                        }

                        readControl[bankSel].Push(1);
                        readAddress[bankSel].Push(address);

                        // CCS_LOG("x0: " << x0 << ", y0: " << y0 << ", fx: " <<
                        // fx
                        //                << ", fy: " << fy
                        //                << ", address: " << address);
                      }
                    }
                  }
                }
              }
              readControl[bankSel].Push(0);
              bankSel = !bankSel;
            }
          }
        }
      }
    }
  }

  /*
   * Used for FX replication with stride=2
   * Window buffer cuts down on memory accesses
   */
  void windowBuffer() {
    wait();

    windowBufferParams.ResetRead();
    windowBufferIn.Reset();
    windowBufferOut.Reset();

    while (true) {
      Params params = windowBufferParams.Pop();
      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      if (params.REPLICATION) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
              // pixels are packed together
              // loop_bounds[1][params.inputXLoopIndex[1]] =
              // (loop_bounds[1][params.inputXLoopIndex[1]] / 2) + 1;

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
                        Pack1D<DTYPE, NROWS> data;

                        Pack1D<DTYPE, NROWS> buffer = windowBufferIn.Pop();

                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5] / 2;
                             loop_counters[1][5]++) {
                          data = buffer;
                          windowBufferOut.Push(data);

                          buffer = windowBufferIn.Pop();

// shift 2 pixels over
#pragma hls_unroll yes
                          for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
                            for (int j = 0; j < 3; j++) {
                              data[i * 3 + j] = data[(i + 2) * 3 + j];
                            }
                          }

// fill remainder with new values
#pragma hls_unroll yes
                          for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
                            for (int j = 0; j < 3; j++) {
                              data[(i + 2) * 3 + j] = buffer[i * 3 + j];
                            }
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
                        for (loop_counters[1][5] = 0;
                             loop_counters[1][5] < loop_bounds[1][5];
                             loop_counters[1][5]++) {
                          windowBufferOut.Push(windowBufferIn.Pop());
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
  }

  void read_params() {
    paramsIn.Reset();
    fetcherParams.ResetWrite();
    writerParams.ResetWrite();
    readerParams.ResetWrite();
    windowBufferParams.ResetWrite();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      fetcherParams.Push(params);
      writerParams.Push(params);
      readerParams.Push(params);
      windowBufferParams.Push(params);
    }
  }
};
