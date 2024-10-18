#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"

template <typename DTYPE, typename ACC_DTYPE, int NROWS, int NCOLS>
SC_MODULE(WeightController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> serialParamsIn;

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<Pack1D<DTYPE, NCOLS> > CCS_INIT_S1(dataResponse);

  Connections::Out<BufferWriteRequest<DTYPE, NCOLS> > writeRequest[2];
  Connections::Out<ac_int<32, false> > writeControl[2];
  Connections::Out<ac_int<16, false> > readAddress[2];
  Connections::Out<ac_int<32, false> > readControl[2];

  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<Pack1D<DTYPE, NCOLS> > CCS_INIT_S1(biasDataResponse);

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int LOOP_WIDTH =
      (8 + int_log2(16 / (IC_DIMENSION < OC_DIMENSION ? IC_DIMENSION
                                                      : OC_DIMENSION)));

#ifdef HYBRID_FP8
  Connections::Out<Pack1D<HYBRID_TYPE, NCOLS> > CCS_INIT_S1(
      weightsToSystolicArray);
#else
  Connections::Out<Pack1D<typename DTYPE::AccumulationDatatype, NCOLS> >
      CCS_INIT_S1(weightsToSystolicArray);
#endif

  Connections::Out<Pack1D<ACC_DTYPE, NCOLS> > CCS_INIT_S1(biasToSystolicArray);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(readerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(biasFetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(biasCombinerParams);

  Connections::Combinational<Pack1D<DTYPE, NCOLS> > transposeOut;
  Connections::Combinational<Pack1D<DTYPE, NCOLS> > gradTransposeOut;

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

    SC_THREAD(biasFetcher);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(biasCombiner);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetcher() {
    addressRequest.Reset();
    fetcherParams.ResetRead();

    wait();

    while (true) {
      const MatrixParams params = fetcherParams.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][5];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][5];

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
                      ac_int<LOOP_WIDTH, false> k2 = loop_counters
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> K2 = loop_bounds
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> k1 = loop_counters
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> K1 = loop_bounds
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> C1 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> c1 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> fx =
                          loop_counters[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> FX =
                          loop_bounds[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> fy =
                          loop_counters[1][params.weightAddressGenFyIndex];
                      ac_int<LOOP_WIDTH, false> FY =
                          loop_bounds[1][params.weightAddressGenFyIndex];
                      ac_int<LOOP_WIDTH, false> c0 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> C0 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[1]];

                      ac_int<16, false> c = c1 * C0 + c0;
                      ac_int<16, false> C = C1 * C0;
                      ac_int<16, false> k =
                          k2 * K1 * OC_DIMENSION + k1 * OC_DIMENSION;
                      ac_int<16, false> K = K2 * K1 * OC_DIMENSION;

                      ac_int<32, false> baseAddress =
                          (fy * FX * C * K) + (fx * C * K) + (c * K) + k;
                      if (params.WEIGHT_TRANSPOSE) {
                        C = C1 * NCOLS;
                        baseAddress = (k + c0) * C + c1 * OC_DIMENSION;
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

    bool bankSel = 0;

    wait();

    while (true) {
      const MatrixParams params = writerParams.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][5];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][5];

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
                      ac_int<LOOP_WIDTH, false> k2 = loop_counters
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> K2 = loop_bounds
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> k1 = loop_counters
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> K1 = loop_bounds
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> C1 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> c1 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> fx =
                          loop_counters[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> FX =
                          loop_bounds[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> fy =
                          loop_counters[1][params.weightAddressGenFyIndex];
                      ac_int<LOOP_WIDTH, false> FY =
                          loop_bounds[1][params.weightAddressGenFyIndex];
                      ac_int<LOOP_WIDTH, false> c0 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> C0 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[1]];

                      ac_int<16, false> C = C0;
                      ac_int<16, false> k =
                          k2 * K1 * OC_DIMENSION + k1 * OC_DIMENSION;
                      ac_int<16, false> K = K2 * K1 * OC_DIMENSION;

                      Pack1D<DTYPE, NCOLS> data = transposeOut.Pop();

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
                      BufferWriteRequest<DTYPE, NCOLS> req;
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

    bool bankSel = 0;

    wait();

    while (true) {
      const MatrixParams params = readerParams.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightReuseIndex[0]] = 1;
      loop_bounds[1][params.weightReuseIndex[1]] = 1;

      // extra loop to control reuse which only occurs during transpose and when
      // OC_DIMENSION > IC_DIMENSION
      int rep_bound = 1;
      if (OC_DIMENSION > IC_DIMENSION) {
        if (params.WEIGHT_TRANSPOSE) {
          // we are able to reuse the weights already in the buffer
          loop_bounds[1][0] /= (OC_DIMENSION / IC_DIMENSION);
          rep_bound = (OC_DIMENSION / IC_DIMENSION);
        }
      }

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
              readControl[bankSel].Push(
                  (loop_bounds[1][1] * loop_bounds[1][2]) *
                  (loop_bounds[1][3] * loop_bounds[1][4]) *
                  (loop_bounds[1][5] * NROWS * rep_bound));
              for (int rep = 0; rep < rep_bound; rep++) {
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
                          ac_int<8, false> numPadding = 0;
                          ac_int<4, false> replicationBound = 1;
                          ac_int<8, false> startingC = NROWS - 1;
                          ac_int<8, false> endingC = NROWS;
                          if (params.REPLICATION) {
                            startingC = 3 - 1;
                            endingC = 3;
                            if (IC_DIMENSION == 4) {
                              numPadding = NROWS - 3;
                              replicationBound = 1;
                            } else if (IC_DIMENSION == 8) {
                              if (loop_counters[1][params.fxIndex] ==
                                  3) {  // last iteration only unrolls 1 fx
                                numPadding = NROWS - 3;
                                replicationBound = 1;
                              } else {
                                numPadding = NROWS - 6;
                                replicationBound = 2;
                              }
                            } else if (IC_DIMENSION == 16) {
                              if (loop_counters[1][params.fxIndex] == 0) {
                                numPadding = NROWS - 12;
                                replicationBound = 4;
                              } else {
                                numPadding = NROWS - 9;
                                replicationBound = 3;
                              }
                            } else if (IC_DIMENSION == 32) {
                              replicationBound = 7;
                              numPadding = NROWS - replicationBound * 3;
                            }
                          }

                          ac_int<LOOP_WIDTH, false> fx_repl = 0;
                          ac_int<LOOP_WIDTH, false> c = 0;
                          for (int row = 0; row < NROWS; row++) {
                            ac_int<LOOP_WIDTH, false> k2 =
                                loop_counters[0][params.weightLoopIndex[0]];
                            ac_int<LOOP_WIDTH, false> K2 =
                                params.loops[0][params.weightLoopIndex[0]];
                            ac_int<LOOP_WIDTH, false> k1 =
                                loop_counters[1][params.weightLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> K1 =
                                params.loops[1][params.weightLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> C1 =
                                params.loops[1][params.reductionLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> c1 =
                                loop_counters[1][params.reductionLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> fx =
                                loop_counters[1][params.fxIndex];
                            ac_int<LOOP_WIDTH, false> FX =
                                params.loops[1][params.fxIndex];
                            ac_int<LOOP_WIDTH, false> fy =
                                loop_counters[1][params.fyIndex];
                            ac_int<LOOP_WIDTH, false> FY =
                                params.loops[1][params.fyIndex];

                            ac_int<16, false> k =
                                k2 * K1 * OC_DIMENSION + k1 * OC_DIMENSION;
                            ac_int<16, false> K = K2 * K1 * OC_DIMENSION;

                            ac_int<LOOP_WIDTH, false> C = IC_DIMENSION;

                            if (params.REPLICATION) {
                              C = 3;
                              if (IC_DIMENSION == 4) {
                                fx = loop_counters[1][params.fxIndex];
                              } else if (IC_DIMENSION == 8) {
                                fx = loop_counters[1][params.fxIndex] * 2 +
                                     fx_repl;
                              } else if (IC_DIMENSION == 16) {
                                fx = loop_counters[1][params.fxIndex] * 4 +
                                     fx_repl;
                              } else if (IC_DIMENSION == 32) {
                                fx = fx_repl;
                              }
                              FX = 7;
                              if (fx_repl < replicationBound) {
                                ac_int<16, false> address = (fy * FX * C * K1) +
                                                            (fx * C * K1) +
                                                            (c * K1) + k1;
                                readAddress[bankSel].Push(address);
                              } else {
                                readAddress[bankSel].Push(0xFFFF);
                              }

                              // keep track of which C and FX we are on
                              if (c < endingC - 1) {
                                c++;
                              } else {
                                c = 0;
                                fx_repl++;
                              }
                            } else {
                              c = row;

                              ac_int<16, false> address = (fy * FX * C * K1) +
                                                          (fx * C * K1) +
                                                          (c * K1) + k1;

                              if (params.WEIGHT_TRANSPOSE &&
                                  OC_DIMENSION > IC_DIMENSION) {
                                address = (fy * FX * C * 2 * K1) + (fx * C * 2 * K1) +
                                    (c + rep * NROWS) * K1 + k1;
                              }
                              readAddress[bankSel].Push(address);
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
      const MatrixParams params = transposerParams.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][5];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][5];

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

      if (params.WEIGHT_TRANSPOSE && NROWS < 64 &&
          NCOLS <
              64) {  // don't support transpose when systolic array is larger
                     // than 32x32, as it will require a very large buffer
        // we need a square buffer to store the transpose
        INPUT_DATATYPE transposeBuffer[NROWS > NCOLS ? NROWS : NCOLS]
                                      [NROWS > NCOLS ? NROWS : NCOLS];

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
                      for (int c0 = 0; c0 < NCOLS; c0++) {
                        Pack1D<DTYPE, NCOLS> originalValue = dataResponse.Pop();
#pragma hls_unroll yes
                        for (int dim = 0; dim < NCOLS; dim++) {
                          transposeBuffer[dim][c0] = originalValue[dim];
                        }
                      }

                      // Write out from tranposeBuffer
                      for (int c0 = 0; c0 < NCOLS; c0++) {
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
        ac_int<32, false> total_values = loop_bounds[0][0] * loop_bounds[0][1] *
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

  void biasFetcher() {
    biasFetcherParams.ResetRead();
    biasAddressRequest.Reset();

    wait();

    while (true) {
      const MatrixParams params = biasFetcherParams.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightReuseIndex[0]] = 1;
      loop_bounds[1][params.weightReuseIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;
      loop_bounds[1][params.reductionLoopIndex[1]] = 1;

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
                        ac_int<LOOP_WIDTH, false> k2 =
                            loop_counters[0][params.weightLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> K2 =
                            loop_bounds[0][params.weightLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> k1 =
                            loop_counters[1][params.weightLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> K1 =
                            loop_bounds[1][params.weightLoopIndex[1]];

                        ac_int<16, false> k =
                            k2 * K1 * OC_DIMENSION + k1 * OC_DIMENSION;
                        ac_int<16, false> K = K2 * K1 * OC_DIMENSION;

                        constexpr int num_words =
                            ACC_DTYPE::width / DTYPE::width;

                        int baseAddress = params.BIAS_OFFSET + k * num_words;
                        MemoryRequest memRequest = {baseAddress,
                                                    OC_DIMENSION * num_words};

                        biasAddressRequest.Push(memRequest);

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

  void biasCombiner() {
    biasCombinerParams.ResetRead();
    biasDataResponse.Reset();
    biasToSystolicArray.Reset();

    wait();

    while (true) {
      const MatrixParams params = biasCombinerParams.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightReuseIndex[0]] = 1;
      loop_bounds[1][params.weightReuseIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;
      loop_bounds[1][params.reductionLoopIndex[1]] = 1;

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
                        constexpr int num_words =
                            ACC_DTYPE::width / DTYPE::width;
                        Pack1D<DTYPE, NCOLS> response[num_words];
                        for (int word = 0; word < num_words; word++) {
                          response[word] = biasDataResponse.Pop();
                        }

                        Pack1D<ACC_DTYPE, NCOLS> fullPrecisionDataResponse;
                        convertPack1D<DTYPE, ACC_DTYPE, NCOLS>(
                            response, fullPrecisionDataResponse);

                        biasToSystolicArray.Push(fullPrecisionDataResponse);

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

  void read_params() {
    paramsIn.ResetRead();
    fetcherParams.ResetWrite();
    writerParams.ResetWrite();
    readerParams.ResetWrite();
    transposerParams.ResetWrite();
    biasFetcherParams.ResetWrite();
    biasCombinerParams.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = paramsIn.Pop();

      fetcherParams.Push(params);
      writerParams.Push(params);
      readerParams.Push(params);
      transposerParams.Push(params);

      if (params.BIAS) {
        biasFetcherParams.Push(params);
        biasCombinerParams.Push(params);
      }
    }
  }
};
