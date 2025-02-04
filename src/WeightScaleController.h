#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"

template <typename Weight, typename Scale, int NRows, int NCols>
SC_MODULE(WeightScaleController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> serialParamsIn;

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<Pack1D<Weight, NCols>> CCS_INIT_S1(dataResponse);

  Connections::Out<BufferWriteRequest<Scale, NCols>> writeRequest[2];
  Connections::Out<ac_int<32, false>> writeControl[2];
  Connections::Out<ac_int<16, false>> readAddress[2];
  Connections::Out<ac_int<32, false>> readControl[2];

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(readerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposerParams);

  Connections::Combinational<Pack1D<Scale, NCols>> transposeOut;

  MatrixParamsDeserializer<4> CCS_INIT_S1(paramsDeserializer);

  static constexpr int LOOP_WIDTH = 10;

  SC_CTOR(WeightScaleController) {
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
      const MatrixParams params = fetcherParams.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][5];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][5];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
          loop_bounds[i][j] = params.weightAddressGenLoops[i][j];
        }
      }

      // int c0_bound = NRows;
      // if (params.is_replication) {
      //   c0_bound = 3;
      //   loop_bounds[1][params.fxIndex] = 7;
      // }

      // microscaling batch size of 32 along C dimension
      loop_bounds[1][params.weightAddressGenReductionLoopIndex[0]] =
          loop_bounds[1][params.weightAddressGenReductionLoopIndex[0]] /
          (32 / NRows);
      loop_bounds[1][params.weightAddressGenReductionLoopIndex[1]] = 1;

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
                      ac_int<LOOP_WIDTH, false> C0 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> C1 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> K1 = loop_bounds
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> K2 = loop_bounds
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> FX =
                          loop_bounds[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> FY =
                          loop_bounds[1][params.weightAddressGenFyIndex];

                      ac_int<LOOP_WIDTH, false> c0 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> c1 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> k1 = loop_counters
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> k2 = loop_counters
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> fx =
                          loop_counters[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> fy =
                          loop_counters[1][params.weightAddressGenFyIndex];

                      ac_int<16, false> c = c1;
                      ac_int<16, false> C = C1;
                      ac_int<16, false> k =
                          k2 * K1 * OC_DIMENSION + k1 * OC_DIMENSION;
                      ac_int<16, false> K = K2 * K1 * OC_DIMENSION;

                      int address =
                          (fy * FX * C * K) + (fx * C * K) + (c * K) + k;
                      if (params.has_weight_tranpose) {
                        C = C1 * NCols;
                        address = (k + c0) * C + c1 * OC_DIMENSION;
                      }

                      // CCS_LOG("Fetching from address " << address);

                      address = address * Scale::width / 8;

                      constexpr int num_words = Scale::width / Weight::width;

                      MemoryRequest memRequest = {
                          params.WEIGHT_SCALE_OFFSET + address,
                          NCols * num_words};
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

      // microscaling batch size of 32 along C dimension
      loop_bounds[1][params.weightAddressGenReductionLoopIndex[0]] =
          loop_bounds[1][params.weightAddressGenReductionLoopIndex[0]] /
          (32 / NRows);
      loop_bounds[1][params.weightAddressGenReductionLoopIndex[1]] = 1;

      // int c0_bound = NRows;
      // if (params.is_replication) {
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
                      ac_int<LOOP_WIDTH, false> C0 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> C1 = loop_bounds
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> K1 = loop_bounds
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> K2 = loop_bounds
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> FX =
                          loop_bounds[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> FY =
                          loop_bounds[1][params.weightAddressGenFyIndex];

                      ac_int<LOOP_WIDTH, false> c0 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> c1 = loop_counters
                          [1][params.weightAddressGenReductionLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> k1 = loop_counters
                          [1][params.weightAddressGenWeightLoopIndex[1]];
                      ac_int<LOOP_WIDTH, false> k2 = loop_counters
                          [0][params.weightAddressGenWeightLoopIndex[0]];
                      ac_int<LOOP_WIDTH, false> fx =
                          loop_counters[1][params.weightAddressGenFxIndex];
                      ac_int<LOOP_WIDTH, false> fy =
                          loop_counters[1][params.weightAddressGenFyIndex];

                      ac_int<16, false> C = C0;
                      ac_int<16, false> k =
                          k2 * K1 * OC_DIMENSION + k1 * OC_DIMENSION;
                      ac_int<16, false> K = K2 * K1 * OC_DIMENSION;

                      Pack1D<Scale, NCols> data = transposeOut.Pop();

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
                      BufferWriteRequest<Scale, NCols> req;
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
      // microscaling batch size of 32 along C dimension
      loop_bounds[1][params.reductionLoopIndex[1]] =
          loop_bounds[1][params.reductionLoopIndex[1]] / (32 / NRows);
      ac_int<8, false> mxscale_reuse = (32 / NRows);

      // extra loop to control reuse which only occurs during transpose and when
      // OC_DIMENSION > IC_DIMENSION
      int rep_bound = 1;
      if (OC_DIMENSION > IC_DIMENSION) {
        if (params.has_weight_tranpose) {
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
                  static_cast<ac_int<16, false>>(loop_bounds[1][1] *
                                                 loop_bounds[1][2]) *
                  static_cast<ac_int<16, false>>(loop_bounds[1][3] *
                                                 loop_bounds[1][4]) *
                  static_cast<ac_int<16, false>>(loop_bounds[1][5] * rep_bound *
                                                 mxscale_reuse));
              for (int reuse = 0; reuse < mxscale_reuse; reuse++) {
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
                            ac_int<8, false> startingC = NRows - 1;
                            ac_int<8, false> endingC = NRows;

                            if (params.is_replication) {
                              startingC = 3 - 1;
                              endingC = 3;
                              if (IC_DIMENSION == 16) {
                                if (loop_counters[1][params.fxIndex] == 0) {
                                  numPadding = NRows - 12;
                                  replicationBound = 4;
                                } else {
                                  numPadding = NRows - 9;
                                  replicationBound = 3;
                                }
                              } else if (IC_DIMENSION == 32) {
                                replicationBound = 7;
                                numPadding = NRows - replicationBound * 3;
                              }
                            }

                            ac_int<LOOP_WIDTH, false> c1 =
                                loop_counters[1][params.reductionLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> k1 =
                                loop_counters[1][params.weightLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> k2 =
                                loop_counters[0][params.weightLoopIndex[0]];
                            ac_int<LOOP_WIDTH, false> fx =
                                loop_counters[1][params.fxIndex];
                            ac_int<LOOP_WIDTH, false> fy =
                                loop_counters[1][params.fyIndex];

                            ac_int<LOOP_WIDTH, false> C1 =
                                params.loops[1][params.reductionLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> K1 =
                                params.loops[1][params.weightLoopIndex[1]];
                            ac_int<LOOP_WIDTH, false> K2 =
                                params.loops[0][params.weightLoopIndex[0]];
                            ac_int<LOOP_WIDTH, false> FX =
                                params.loops[1][params.fxIndex];
                            ac_int<LOOP_WIDTH, false> FY =
                                params.loops[1][params.fyIndex];

                            ac_int<16, false> k =
                                k2 * K1 * OC_DIMENSION + k1 * OC_DIMENSION;
                            ac_int<16, false> K = K2 * K1 * OC_DIMENSION;

                            int address =
                                static_cast<ac_int<16, false>>((fy * FX * K1)) +
                                static_cast<ac_int<16, false>>((fx * K1)) + k1;

                            if (params.has_weight_tranpose &&
                                OC_DIMENSION > IC_DIMENSION) {
                              address = static_cast<ac_int<16, false>>(
                                            (fy * FX * 2 * K1)) +
                                        static_cast<ac_int<16, false>>(
                                            (fx * 2 * K1)) +
                                        static_cast<ac_int<16, false>>(
                                            (rep * NRows) * K1) +
                                        k1;
                            }

                            // readControl[bankSel].Push(!swapBank);
                            readAddress[bankSel].Push(address);
                            // CCS_LOG("pushing read address: " << address);

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

      // microscaling batch size of 32 along C dimension
      loop_bounds[1][params.weightAddressGenReductionLoopIndex[0]] =
          loop_bounds[1][params.weightAddressGenReductionLoopIndex[0]] /
          (32 / NRows);
      loop_bounds[1][params.weightAddressGenReductionLoopIndex[1]] = 1;

      // int c0_bound = NRows;
      // if (params.is_replication) {
      //   c0_bound = 3;
      //   loop_bounds[1][params.fxIndex] = 7;
      // }

      if (params.has_weight_tranpose && NRows < 64 &&
          NCols <
              64) {  // don't support transpose when systolic array is larger
                     // than 32x32, as it will require a very large buffer
        // we need a square buffer to store the transpose
        Scale transposeBuffer[NRows > NCols ? NRows : NCols]
                             [NRows > NCols ? NRows : NCols];

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
                      for (int c0 = 0; c0 < NCols; c0++) {
                        constexpr int num_words = Scale::width / Weight::width;

                        Pack1D<Scale, NCols> converted_data;

                        if constexpr (num_words == 1) {
                          Pack1D<Weight, NCols> response = dataResponse.Pop();
#pragma hls_unroll yes
                          for (int dim = 0; dim < NCols; dim++) {
                            converted_data[dim].set_bits(
                                response[dim].bits_rep());
                          }
                        } else {
                          Pack1D<Weight, NCols> response[num_words];
                          for (int word = 0; word < num_words; word++) {
                            response[word] = dataResponse.Pop();
                          }

                          convertPack1D<Weight, Scale, NCols>(response,
                                                              converted_data);
                        }

#pragma hls_unroll yes
                        for (int dim = 0; dim < NCols; dim++) {
                          transposeBuffer[dim][c0] = converted_data[dim];
                        }
                      }

                      // Write out from tranposeBuffer
                      for (int c0 = 0; c0 < NCols; c0++) {
                        Pack1D<Scale, NCols> transposedValue;

#pragma hls_unroll yes
                        for (int dim = 0; dim < NCols; dim++) {
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
          constexpr int num_words = Scale::width / Weight::width;

          Pack1D<Scale, NCols> converted_data;

          if constexpr (num_words == 1) {
            Pack1D<Weight, NCols> response = dataResponse.Pop();
#pragma hls_unroll yes
            for (int dim = 0; dim < NCols; dim++) {
              converted_data[dim].set_bits(response[dim].bits_rep());
            }
          } else {
            Pack1D<Weight, NCols> response[num_words];
            for (int word = 0; word < num_words; word++) {
              response[word] = dataResponse.Pop();
            }

            convertPack1D<Weight, Scale, NCols>(response, converted_data);
          }
          transposeOut.Push(converted_data);
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
      const MatrixParams params = paramsIn.Pop();

      if (params.is_mx_op) {
        fetcherParams.Push(params);
        writerParams.Push(params);
        readerParams.Push(params);
        transposerParams.Push(params);
      }
    }
  }
};
