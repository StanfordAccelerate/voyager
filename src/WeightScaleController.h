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

  MatrixParamsDeserializer<4> CCS_INIT_S1(paramsDeserializer);

  static constexpr int LOOP_WIDTH = 10;
  static constexpr int BLOCK_SIZE = NRows > NCols ? NRows : NCols;

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
                        ac_int<LOOP_WIDTH, false> K2 = loop_bounds
                            [0][params.weightAddressGenWeightLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> k1 = loop_counters
                            [1][params.weightAddressGenWeightLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> K1 = loop_bounds
                            [1][params.weightAddressGenWeightLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> C1 = loop_bounds
                            [1][params.weightAddressGenReductionLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> c1 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> fx =
                            loop_counters[1][params.weightAddressGenFxIndex];
                        ac_int<LOOP_WIDTH, false> FX =
                            loop_bounds[1][params.weightAddressGenFxIndex];
                        ac_int<LOOP_WIDTH, false> fy =
                            loop_counters[1][params.weightAddressGenFyIndex];
                        ac_int<LOOP_WIDTH, false> FY =
                            loop_bounds[1][params.weightAddressGenFyIndex];
                        ac_int<LOOP_WIDTH, false> c0 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[2]];
                        ac_int<LOOP_WIDTH, false> C0 = loop_bounds
                            [1][params.weightAddressGenReductionLoopIndex[2]];
                        ac_int<LOOP_WIDTH, false> c2 = loop_counters
                            [0][params.weightAddressGenReductionLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> C2 = loop_bounds
                            [0][params.weightAddressGenReductionLoopIndex[0]];

                        ac_int<16, false> c = (c2 * C1 * C0 + c1 * C0 + c0);
                        ac_int<16, false> C = (C2 * C1 * C0);

                        ac_int<16, false> k = k2 * K1 * NCols + k1 * NCols;
                        ac_int<16, false> K = K2 * K1 * NCols;

                        if (c % BLOCK_SIZE == 0) {
                          c = c / BLOCK_SIZE;
                          C = C / BLOCK_SIZE;
                          ac_int<32, false> address =
                              (fy * FX * C * K) + (fx * C * K) + (c * K) + k;

                          MemoryRequest request = {
                              params.WEIGHT_SCALE_OFFSET +
                                  address * Scale::width / 8,
                              NCols * Scale::width / 8};
                          addressRequest.Push(request);
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
              writeControl[bankSel].Push(loop_bounds[1][0] * loop_bounds[1][1] *
                                         loop_bounds[1][2] * loop_bounds[1][3] *
                                         loop_bounds[1][4] / NCols);
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
                            [1][params.weightAddressGenReductionLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> c1 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[1]];
                        ac_int<LOOP_WIDTH, false> fx =
                            loop_counters[1][params.weightAddressGenFxIndex];
                        ac_int<LOOP_WIDTH, false> FX =
                            loop_bounds[1][params.weightAddressGenFxIndex];
                        ac_int<LOOP_WIDTH, false> fy =
                            loop_counters[1][params.weightAddressGenFyIndex];
                        ac_int<LOOP_WIDTH, false> FY =
                            loop_bounds[1][params.weightAddressGenFyIndex];
                        ac_int<LOOP_WIDTH, false> c0 = loop_counters
                            [1][params.weightAddressGenReductionLoopIndex[2]];
                        ac_int<LOOP_WIDTH, false> C0 = loop_bounds
                            [1][params.weightAddressGenReductionLoopIndex[2]];
                        ac_int<LOOP_WIDTH, false> c2 = loop_counters
                            [0][params.weightAddressGenReductionLoopIndex[0]];
                        ac_int<LOOP_WIDTH, false> C2 = loop_bounds
                            [0][params.weightAddressGenReductionLoopIndex[0]];

                        ac_int<16, false> C = C0 * C1;
                        ac_int<16, false> c = c1 * C0 + c0;
                        ac_int<16, false> k = k2 * K1 * NCols + k1 * NCols;
                        ac_int<16, false> K = K2 * K1 * NCols;

                        if (c % BLOCK_SIZE == 0) {
                          c = c / BLOCK_SIZE;
                          C = C > BLOCK_SIZE ? static_cast<int>(C / BLOCK_SIZE)
                                             : 1;

                          int address = (fy * FX * C * K1) + (fx * C * K1) +
                                        (c * K1) + k1;

                          Pack1D<Scale, NCols> data;

                          constexpr int num_words =
                              Scale::width / Weight::width;
                          if constexpr (num_words == 1) {
                            Pack1D<Weight, NCols> response = dataResponse.Pop();
#pragma hls_unroll yes
                            for (int dim = 0; dim < NCols; dim++) {
                              data[dim].set_bits(response[dim].bits_rep());
                            }
                          } else {
                            Pack1D<Weight, NCols> response[num_words];
                            for (int word = 0; word < num_words; word++) {
                              response[word] = dataResponse.Pop();
                            }

                            convertPack1D<Weight, Scale, NCols>(response, data);
                          }

                          BufferWriteRequest<Scale, NCols> req;
                          req.address = address;
                          req.data = data;
                          writeRequest[bankSel].Push(req);
                        }

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
        if (loop_bounds[1][0] >= (NCols / NRows)) {
          // we are able to reuse the weights already in the buffer
          loop_bounds[1][0] /= (NCols / NRows);
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
              readControl[bankSel].Push(
                  static_cast<ac_int<16, false>>(loop_bounds[1][0] *
                                                 loop_bounds[1][1] *
                                                 loop_bounds[1][2]) *
                  static_cast<ac_int<16, false>>(loop_bounds[1][3] *
                                                 loop_bounds[1][4]) *
                  static_cast<ac_int<16, false>>(loop_bounds[1][5] * rep_bound *
                                                 buffer_reuse));
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
                                  loop_counters[1]
                                               [params.reductionLoopIndex[1]];
                              ac_int<LOOP_WIDTH, false> fx =
                                  loop_counters[1][params.fxIndex];
                              ac_int<LOOP_WIDTH, false> FX =
                                  params.loops[1][params.fxIndex];
                              ac_int<LOOP_WIDTH, false> fy =
                                  loop_counters[1][params.fyIndex];
                              ac_int<LOOP_WIDTH, false> FY =
                                  params.loops[1][params.fyIndex];
                              ac_int<16, false> k =
                                  k2 * K1 * NCols + k1 * NCols;
                              ac_int<16, false> K = K2 * K1 * NCols;

                              ac_int<32, false> address = fy * FX * C1 * K1 +
                                                          fx * C1 * K1 +
                                                          c1 * K1 + k1;

                              readAddress[bankSel].Push(address);

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
                  }

                  if (loop_counters[1][0] >= loop_bounds[1][0] - 1) {
                    break;
                  }
                  if (loop_counters[1][1] >= loop_bounds[1][1] - 1) {
                    break;
                  }
                }
              }
              // std::cout << "Reading ---- " << std::endl;
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
