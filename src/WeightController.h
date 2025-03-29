#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"

template <typename Weight, typename Bias, int NRows, int NCols>
SC_MODULE(WeightController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> serialParamsIn;

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(dataResponse);

  Connections::Out<BufferWriteRequest<OC_PORT_WIDTH>> writeRequest[2];
  Connections::Out<ac_int<32, false>> writeControl[2];
  Connections::Out<ac_int<16, false>> readAddress[2];
  Connections::Out<ac_int<32, false>> readControl[2];

  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(biasDataResponse);

  Connections::Out<Pack1D<Bias, NCols>> CCS_INIT_S1(biasToSystolicArray);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(readerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposerParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(biasFetcherParams);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(biasCombinerParams);

  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> transposeOut;

  MatrixParamsDeserializer<2> CCS_INIT_S1(paramsDeserializer);

  static constexpr int LOOP_WIDTH = 10;

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

      ac_int<LOOP_WIDTH, false> K2 =
          loop_bounds[0][params.weightAddressGenWeightLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C2 =
          loop_bounds[0][params.weightAddressGenReductionLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> C1 =
          loop_bounds[1][params.weightAddressGenReductionLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> C0 =
          loop_bounds[1][params.weightAddressGenReductionLoopIndex[2]];
      ac_int<LOOP_WIDTH, false> FX =
          loop_bounds[1][params.weightAddressGenFxIndex];
      ac_int<LOOP_WIDTH, false> FY =
          loop_bounds[1][params.weightAddressGenFyIndex];
      ac_int<LOOP_WIDTH, false> K1 =
          loop_bounds[1][params.weightAddressGenWeightLoopIndex[1]];

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
                        ac_int<LOOP_WIDTH, false> c2 = loop_counters
                            [0][params.weightAddressGenReductionLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> c1 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> c0 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[2]];
                        ac_int<LOOP_WIDTH, false> fx =
                            loop_counters[1][params.weightAddressGenFxIndex];
                        ac_int<LOOP_WIDTH, false> fy =
                            loop_counters[1][params.weightAddressGenFyIndex];
                        ac_int<LOOP_WIDTH, false> k1 = loop_counters
                            [1][params.weightAddressGenWeightLoopIndex[1]];

                        ac_int<16, false> c = c2 * C1 * C0 + c1 * C0 + c0;
                        ac_int<16, false> C = C2 * C1 * C0;
                        ac_int<16, false> k = k2 * K1 * NCols + k1 * NCols;
                        ac_int<16, false> K = K2 * K1 * NCols;

                        ac_int<32, false> address =
                            (fy * FX * C * K) + (fx * C * K) + (c * K) + k;

                        if (params.has_weight_transpose) {
                          address = ((k + c0) * C2 * C1 + c2 * C1 + c1) * NCols;
                        }

                        MemoryRequest request = {
                            params.WEIGHT_OFFSET + address * Weight::width / 8,
                            NCols * Weight::width / 8};
                        addressRequest.Push(request);

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

      ac_int<LOOP_WIDTH, false> K2 =
          loop_bounds[0][params.weightAddressGenWeightLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> K1 =
          loop_bounds[1][params.weightAddressGenWeightLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> C1 =
          loop_bounds[1][params.weightAddressGenReductionLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> C0 =
          loop_bounds[1][params.weightAddressGenReductionLoopIndex[2]];
      ac_int<LOOP_WIDTH, false> FX =
          loop_bounds[1][params.weightAddressGenFxIndex];
      ac_int<LOOP_WIDTH, false> FY =
          loop_bounds[1][params.weightAddressGenFyIndex];

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
              // inner memory
              writeControl[bankSel].Push(loop_bounds[1][0] * loop_bounds[1][1] *
                                         loop_bounds[1][2] * loop_bounds[1][3] *
                                         loop_bounds[1][4]);
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
                        ac_int<LOOP_WIDTH, false> k1 = loop_counters
                            [1][params.weightAddressGenWeightLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> c1 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> fx =
                            loop_counters[1][params.weightAddressGenFxIndex];
                        ac_int<LOOP_WIDTH, false> fy =
                            loop_counters[1][params.weightAddressGenFyIndex];
                        ac_int<LOOP_WIDTH, false> c0 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[2]];

                        ac_int<16, false> C = C0 * C1;
                        ac_int<16, false> k = k2 * K1 * NCols + k1 * NCols;
                        ac_int<16, false> K = K2 * K1 * NCols;

                        ac_int<OC_PORT_WIDTH, false> data = transposeOut.Pop();

                        int address = (fy * FX * C * K1) + (fx * C * K1) +
                                      ((c0 + c1 * C0) * K1) + k1;

                        BufferWriteRequest<OC_PORT_WIDTH> req;
                        req.address = address;
                        req.data = data;
                        writeRequest[bankSel].Push(req);

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

    readControl[0].Reset();
    readControl[1].Reset();
    readAddress[0].Reset();
    readAddress[1].Reset();

    bool bankSel = 0;

    wait();

    while (true) {
      const MatrixParams params = readerParams.Pop();

      ac_int<LOOP_WIDTH, false> K2 = params.loops[0][params.weightLoopIndex[0]];
      ac_int<LOOP_WIDTH, false> K1 = params.loops[1][params.weightLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> C1 =
          params.loops[1][params.reductionLoopIndex[1]];
      ac_int<LOOP_WIDTH, false> FY = params.loops[1][params.fyIndex];
      ac_int<LOOP_WIDTH, false> FX = params.loops[1][params.fxIndex];

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
      // NCols > NRows
      int rep_bound = 1;

      if (params.has_weight_transpose && NCols > NRows) {
        if (loop_bounds[0][params.reductionLoopIndex[0]] >= (NCols / NRows)) {
          // we are able to reuse the weights already in the buffer
          loop_bounds[0][params.reductionLoopIndex[0]] /= (NCols / NRows);
          rep_bound = (NCols / NRows);
        }
      }

      // extra loop for exploiting L1 buffer reuse.
      // this loop is used when OX and OY are the innermost L2 loops. when this
      // occurs, we can move OX and/or OY into the buffer reuse L1 loop
      int buffer_reuse = 1;
      if (params.loops[0][params.reductionLoopIndex[0]] == 1) {
        // OX loop can be absorbed
        if (params.weightLoopIndex[0] < params.inputXLoopIndex[0]) {
          buffer_reuse *= loop_bounds[0][params.inputXLoopIndex[0]];
          loop_bounds[0][params.inputXLoopIndex[0]] = 1;
        }
        // OY loop can be absorbed
        if (params.weightLoopIndex[0] < params.inputYLoopIndex[0]) {
          buffer_reuse *= loop_bounds[0][params.inputYLoopIndex[0]];
          loop_bounds[0][params.inputYLoopIndex[0]] = 1;
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
            for (loop_counters[0][3] = 0;
                 loop_counters[0][3] < loop_bounds[0][3];
                 loop_counters[0][3]++) {
              readControl[bankSel].Push(loop_bounds[1][0] * loop_bounds[1][1] *
                                        loop_bounds[1][2] * loop_bounds[1][3] *
                                        loop_bounds[1][4] * loop_bounds[1][5] *
                                        NRows * rep_bound * buffer_reuse);
              for (int reuse = 0; reuse < buffer_reuse; reuse++) {
                for (int rep = 0; rep < rep_bound; rep++) {
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
                              /*
                               * If we have replication, then need to zero pad
                               * the unused rows For 7x7 filter, we split it
                               * into 4 filters and 3 filters
                               */
                              ac_int<8, false> numPadding = 0;
                              ac_int<4, false> replicationBound = 1;
                              ac_int<8, false> startingC = NRows - 1;
                              ac_int<8, false> endingC = NRows;
                              if (params.is_replication) {
                                startingC = 3 - 1;
                                endingC = 3;
                                if (NRows == 4) {
                                  numPadding = NRows - 3;
                                  replicationBound = 1;
                                } else if (NRows == 8) {
                                  if (loop_counters[1][params.fxIndex] ==
                                      3) {  // last iteration only unrolls 1 fx
                                    numPadding = NRows - 3;
                                    replicationBound = 1;
                                  } else {
                                    numPadding = NRows - 6;
                                    replicationBound = 2;
                                  }
                                } else if (NRows == 16) {
                                  if (loop_counters[1][params.fxIndex] == 0) {
                                    numPadding = NRows - 12;
                                    replicationBound = 4;
                                  } else {
                                    numPadding = NRows - 9;
                                    replicationBound = 3;
                                  }
                                } else if (NRows == 32) {
                                  replicationBound = 7;
                                  numPadding = NRows - replicationBound * 3;
                                }
                              }

                              ac_int<LOOP_WIDTH, false> fx_repl = 0;
                              ac_int<LOOP_WIDTH, false> c = 0;
                              for (int row = 0; row < NRows; row++) {
                                ac_int<LOOP_WIDTH, false> k2 =
                                    loop_counters[0][params.weightLoopIndex[0]];
                                ac_int<LOOP_WIDTH, false> k1 =
                                    loop_counters[1][params.weightLoopIndex[1]];
                                ac_int<LOOP_WIDTH, false> c1 =
                                    loop_counters[1]
                                                 [params.reductionLoopIndex[1]];
                                ac_int<LOOP_WIDTH, false> fy =
                                    loop_counters[1][params.fyIndex];
                                ac_int<LOOP_WIDTH, false> fx =
                                    loop_counters[1][params.fxIndex];

                                ac_int<16, false> k =
                                    k2 * K1 * NCols + k1 * NCols;
                                ac_int<16, false> K = K2 * K1 * NCols;

                                ac_int<LOOP_WIDTH, false> C = NRows * C1;
                                ac_int<LOOP_WIDTH, false> C0 = NRows;

                                if (params.is_replication) {
                                  C = 3;
                                  C0 = 3;
                                  if (NRows == 4) {
                                    fx = loop_counters[1][params.fxIndex];
                                  } else if (NRows == 8) {
                                    fx = loop_counters[1][params.fxIndex] * 2 +
                                         fx_repl;
                                  } else if (NRows == 16) {
                                    fx = loop_counters[1][params.fxIndex] * 4 +
                                         fx_repl;
                                  } else if (NRows == 32) {
                                    fx = fx_repl;
                                  }
                                  FX = 7;
                                  if (fx_repl < replicationBound) {
                                    ac_int<16, false> address =
                                        fy * FX * C * K1 + fx * C * K1 +
                                        c * K1 + k1;
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

                                  ac_int<16, false> address =
                                      (fy * FX * C * K1) + (fx * C * K1) +
                                      ((c + c1 * C0) * K1) + k1;

                                  if (params.has_weight_transpose &&
                                      NCols > NRows) {
                                    address = (fy * FX * C * rep_bound * K1) +
                                              (fx * C * rep_bound * K1) +
                                              ((c + rep * NRows) +
                                               c1 * C0 * rep_bound) *
                                                  K1 +
                                              k1;
                                  }
                                  readAddress[bankSel].Push(address);
                                }
                              }

                              if (loop_counters[1][5] >=
                                  loop_bounds[1][5] - 1) {
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
                }
                if (reuse == buffer_reuse - 1) {
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

      // don't support transpose when systolic array is larger
      // than 32x32, as it will require a very large buffer
      if (params.has_weight_transpose && NRows < 64 && NCols < 64) {
        // we need a square buffer to store the transpose
        ac_int<Weight::width, false>
            transposeBuffer[NRows > NCols ? NRows : NCols]
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
                        // Assume that the innermost loop is the c0 loop
                        // Must be true for the transpose case

                        // Fill up transposeBuffer
                        for (int c0 = 0; c0 < NCols; c0++) {
                          ac_int<OC_PORT_WIDTH, false> response =
                              dataResponse.Pop();
#pragma hls_unroll yes
                          for (int dim = 0; dim < NCols; dim++) {
                            transposeBuffer[dim][c0] =
                                response.template slc<Weight::width>(
                                    dim * Weight::width);
                          }
                        }

                        // Write out from tranposeBuffer
                        for (int c0 = 0; c0 < NCols; c0++) {
                          ac_int<OC_PORT_WIDTH, false> transposed;

#pragma hls_unroll yes
                          for (int dim = 0; dim < NCols; dim++) {
                            transposed.set_slc(dim * Weight::width,
                                               transposeBuffer[c0][dim]);
                          }

                          transposeOut.Push(transposed);
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
      } else {  // passthrough
        ac_int<32, false> total_values =
            loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
            loop_bounds[0][3] * loop_bounds[1][0] * loop_bounds[1][1] *
            loop_bounds[1][2] * loop_bounds[1][3] * loop_bounds[1][4];

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
                        ac_int<LOOP_WIDTH, false> K2 =
                            loop_bounds[0][params.weightLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> K1 =
                            loop_bounds[1][params.weightLoopIndex[1]];

                        ac_int<LOOP_WIDTH, false> k2 =
                            loop_counters[0][params.weightLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> k1 =
                            loop_counters[1][params.weightLoopIndex[1]];

                        ac_int<16, false> address =
                            k2 * K1 * NCols + k1 * NCols;

                        MemoryRequest request = {
                            params.BIAS_OFFSET + address * Bias::width / 8,
                            NCols * Bias::width / 8};

                        biasAddressRequest.Push(request);

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
                        constexpr int num_words = Bias::width / Weight::width;

                        ac_int<Bias::width * NCols, false> bits;

                        for (int i = 0; i < num_words; i++) {
                          bits.set_slc(i * OC_PORT_WIDTH,
                                       biasDataResponse.Pop());
                        }

                        Pack1D<Bias, NCols> biases =
                            BitsToType<Pack1D<Bias, NCols>>(TypeToBits(bits));

                        biasToSystolicArray.Push(biases);
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

      if (params.has_bias) {
        biasFetcherParams.Push(params);
        biasCombinerParams.Push(params);
      }
    }
  }
};
