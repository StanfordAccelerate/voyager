#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"
#define DIMENSION 16

template <typename DTYPE, int NROWS, int NCOLS>
SC_MODULE(WeightController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> serialParamsIn;

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(dataResponse);

  Connections::Out<BufferWriteRequest<DTYPE, NROWS> > writeRequest[2];
  Connections::Out<int> writeControl[2];
  Connections::Out<int> readAddress[2];
  Connections::Out<int> readControl[2];

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(readerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposerParams);

  Connections::Combinational<Pack1D<DTYPE, NCOLS> > transposeOut;

  MatrixParamsDeserializer<2> CCS_INIT_S1(paramsDeserializer);

  SC_CTOR(WeightController) {
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

    SC_THREAD(transposer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetcher() {
    addressRequest.Reset();
    fetcherParams.ResetRead();

    wait();

    while (true) {
      MatrixParams params = fetcherParams.Pop();

      int loop_counters[2][5];
      int loop_bounds[2][5];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
          loop_bounds[i][j] = params.weightAddressGenLoops[i][j];
        }
      }

      // int c0_bound = NROWS;
      // if (params.REPLICATION) {
      //   c0_bound = 3;
      //   loop_bounds[1][params.fxIndex] = 7;
      // }

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
                      int k2 = loop_counters
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      int K2 = loop_bounds
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      int k1 = loop_counters
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      int K1 = loop_bounds
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      int C1 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      int c1 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      int fx = loop_counters[1][params.weightAddressGenFxIndex];
                      int FX = loop_bounds[1][params.weightAddressGenFxIndex];
                      int fy = loop_counters[1][params.weightAddressGenFyIndex];
                      int FY = loop_bounds[1][params.weightAddressGenFyIndex];
                      int c0 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      int C0 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[1]];

                      int c = c1 * C0 + c0;
                      int C = C1 * C0;
                      int k = k2 * K1 * DIMENSION + k1 * DIMENSION;
                      int K = K2 * K1 * DIMENSION;

                      int baseAddress =
                          (fy * FX * C * K) + (fx * C * K) + (c * K) + k;
                      if (params.TRANSPOSE) {
                        baseAddress = (k + c0) * C + c1 * DIMENSION;
                      } else if (params.CONCAT_HEAD_WEIGHTS) {
                        baseAddress = ((k / 32) * C * 32) + (c * 32) + (k % 32);
                      }
                      int burstSize = NCOLS;

                      MemoryRequest memRequest = {
                          params.WEIGHT_OFFSET + baseAddress, burstSize};
                      addressRequest.Push(memRequest);

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

    wait();

    while (true) {
      MatrixParams params = writerParams.Pop();

      bool bankSel = 0;

      int loop_counters[2][5];
      int loop_bounds[2][5];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
          loop_bounds[i][j] = params.weightAddressGenLoops[i][j];
        }
      }

      // int c0_bound = NROWS;
      // if (params.REPLICATION) {
      //   c0_bound = 3;
      //   loop_bounds[1][params.fxIndex] = 7;
      // }

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
              writeControl[bankSel].Push(loop_bounds[1][1] * loop_bounds[1][2] *
                                         loop_bounds[1][3] * loop_bounds[1][4]);
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
                      int k2 = loop_counters
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      int K2 = loop_bounds
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      int k1 = loop_counters
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      int K1 = loop_bounds
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      int C1 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      int c1 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      int fx = loop_counters[1][params.weightAddressGenFxIndex];
                      int FX = loop_bounds[1][params.weightAddressGenFxIndex];
                      int fy = loop_counters[1][params.weightAddressGenFyIndex];
                      int FY = loop_bounds[1][params.weightAddressGenFyIndex];
                      int c0 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      int C0 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[1]];

                      int C = C0;
                      int k = k2 * K1 * DIMENSION + k1 * DIMENSION;
                      int K = K2 * K1 * DIMENSION;

                      Pack1D<DTYPE, NROWS> data = transposeOut.Pop();

                      int address =
                          (fy * FX * C * K1) + (fx * C * K1) + (c0 * K1) + k1;

                      // int swapBank =
                      //     (loop_counters[1][1] == loop_bounds[1][1] - 1)
                      //     && (loop_counters[1][2] == loop_bounds[1][2] -
                      //     1) && (loop_counters[1][3] == loop_bounds[1][3]
                      //     - 1) && (loop_counters[1][4] ==
                      //     loop_bounds[1][4] - 1) && (loop_counters[1][5]
                      //     == loop_bounds[1][5] - 1);

                      // writeControl[bankSel].Push(!swapBank);
                      BufferWriteRequest<DTYPE, NROWS> req;
                      req.address = address;
                      req.data = data;
                      writeRequest[bankSel].Push(req);

                      // CCS_LOG("c: " << c);
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

    wait();

    while (true) {
      MatrixParams params = readerParams.Pop();

      bool bankSel = 0;

      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightReuseIndex[0]] = 1;
      loop_bounds[1][params.weightReuseIndex[1]] = 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            for (loop_counters[1][0] = 0;
                 loop_counters[1][0] < loop_bounds[1][0];
                 loop_counters[1][0]++) {
              readControl[bankSel].Push(loop_bounds[1][1] * loop_bounds[1][2] *
                                        loop_bounds[1][3] * loop_bounds[1][4] *
                                        loop_bounds[1][5] * NROWS);
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
                        /*
                         * If we have replication, then need to zero pad the
                         * unused rows For 7x7 filter, we split it into 4
                         * filters and 3 filters
                         */
                        int numPadding = 0;
                        int replicationBound = 1;
                        int startingC = NROWS - 1;
                        if (params.REPLICATION) {
                          startingC = 3 - 1;
                          if (loop_counters[1][params.fxIndex] == 0) {
                            numPadding = NROWS - 12;
                            replicationBound = 4;
                          } else {
                            numPadding = NROWS - 9;
                            replicationBound = 3;
                          }
                          // zero_padding
                          for (int i = 0; i < numPadding; i++) {
                            // readControl[bankSel].Push(1);
                            readAddress[bankSel].Push(-1);

                            if (i >= numPadding - 1) {
                              break;
                            }
                          }
                        }

                        for (int fx_repl = replicationBound - 1; fx_repl >= 0;
                             fx_repl--) {
                          for (int c = startingC; c >= 0;
                               c--) {  // reverse order
                            int k2 =
                                loop_counters[0][params.weightLoopIndex[0]];
                            int K2 = params.loops[0][params.weightLoopIndex[0]];
                            int k1 =
                                loop_counters[1][params.weightLoopIndex[1]];
                            int K1 = params.loops[1][params.weightLoopIndex[1]];
                            int C1 =
                                params.loops[1][params.reductionLoopIndex[1]];
                            int c1 =
                                loop_counters[1][params.reductionLoopIndex[1]];
                            int fx = loop_counters[1][params.fxIndex];
                            int FX = params.loops[1][params.fxIndex];
                            int fy = loop_counters[1][params.fyIndex];
                            int FY = params.loops[1][params.fyIndex];

                            int C = DIMENSION;
                            if (params.REPLICATION) {
                              C = 3;
                              fx = loop_counters[1][params.fxIndex] * 4 +
                                   fx_repl;
                              FX = 7;
                            }

                            int k = k2 * K1 * DIMENSION + k1 * DIMENSION;
                            int K = K2 * K1 * DIMENSION;
                            int address = (fy * FX * C * K1) + (fx * C * K1) +
                                          (c * K1) + k1;

                            int swapBank = (loop_counters[1][1] ==
                                            loop_bounds[1][1] - 1) &&
                                           (loop_counters[1][2] ==
                                            loop_bounds[1][2] - 1) &&
                                           (loop_counters[1][3] ==
                                            loop_bounds[1][3] - 1) &&
                                           (loop_counters[1][4] ==
                                            loop_bounds[1][4] - 1) &&
                                           (loop_counters[1][5] ==
                                            loop_bounds[1][5] - 1) &&
                                           (c == 0) && (fx_repl == 0);

                            // readControl[bankSel].Push(!swapBank);
                            readAddress[bankSel].Push(address);

                            if (c == 0) {
                              break;
                            }
                          }
                          if (fx_repl == 0) {
                            break;
                          }
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

  void transposer() {
    transposerParams.ResetRead();
    dataResponse.Reset();
    transposeOut.ResetWrite();

    wait();

    while (true) {
      MatrixParams params = transposerParams.Pop();

      int loop_counters[2][5];
      int loop_bounds[2][5];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
          loop_bounds[i][j] = params.weightAddressGenLoops[i][j];
        }
      }

      // int c0_bound = NROWS;
      // if (params.REPLICATION) {
      //   c0_bound = 3;
      //   loop_bounds[1][params.fxIndex] = 7;
      // }

      if (params.TRANSPOSE) {
        INPUT_DATATYPE transposeBuffer[NROWS][NCOLS];

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
                      // Assume that the innermost loop is the c0 loop
                      // Must be true for the transpose case

                      // Fill up transposeBuffer
                      for (int c0 = 0; c0 < NROWS; c0++) {
                        Pack1D<DTYPE, NCOLS> originalValue = dataResponse.Pop();
#pragma hls_unroll yes
                        for (int dim = 0; dim < NCOLS; dim++) {
                          transposeBuffer[dim][c0] = originalValue[dim];
                        }
                      }

                      // Write out from tranposeBuffer
                      for (int c0 = 0; c0 < NROWS; c0++) {
                        Pack1D<DTYPE, NCOLS> transposedValue;

#pragma hls_unroll yes
                        for (int dim = 0; dim < NCOLS; dim++) {
                          transposedValue[dim] = transposeBuffer[c0][dim];
                        }
                        transposeOut.Push(transposedValue);
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
      } else {  // passthrough
        int total_values = loop_bounds[0][0] * loop_bounds[0][1] *
                           loop_bounds[0][2] * loop_bounds[1][0] *
                           loop_bounds[1][1] * loop_bounds[1][2] *
                           loop_bounds[1][3] * loop_bounds[1][4];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < total_values; i++) {
          transposeOut.Push(dataResponse.Pop());
        }
      }
    }
  }

  void read_params() {
    paramsIn.ResetRead();
    fetcherParams.ResetWrite();
    writerParams.ResetWrite();
    readerParams.ResetWrite();
    transposerParams.ResetWrite();

    wait();

    while (true) {
      MatrixParams params = paramsIn.Pop();

      fetcherParams.Push(params);
      writerParams.Push(params);
      readerParams.Push(params);
      transposerParams.Push(params);
    }
  }
};
