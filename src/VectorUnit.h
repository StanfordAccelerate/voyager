#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <int NROWS>
SC_MODULE(FetchUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::Out<int> CCS_INIT_S1(vectorFetchAddressRequest);
  Connections::Out<int> CCS_INIT_S1(scalarAddressRequest);
  Connections::Out<int> CCS_INIT_S1(varianceAddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(residualAddressRequest);

  Connections::Combinational<Params> CCS_INIT_S1(vectorFetchParams);
  Connections::Combinational<Params> CCS_INIT_S1(subtractionFetchParams);
  Connections::Combinational<Params> CCS_INIT_S1(varianceFetchParams);
  Connections::Combinational<Params> CCS_INIT_S1(biasFetchParams);
  Connections::Combinational<Params> CCS_INIT_S1(residualFetchParams);

  SC_CTOR(FetchUnit) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_vector);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_subtraction);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_variance);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_bias);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_residual);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetch_vector() {
    vectorFetchParams.ResetRead();
    vectorFetchAddressRequest.Reset();

    wait();

    while (true) {
      Params params = vectorFetchParams.Pop();

      int rows = params.loops[0][params.inputXLoopIndex[0]] *
                 params.loops[1][params.inputXLoopIndex[1]];
      int cols = NROWS * params.loops[0][params.weightLoopIndex[0]] *
                 params.loops[1][params.weightLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          int address = i * cols + j;
          address = params.VECTOR_OFFSET + address;
          vectorFetchAddressRequest.Push(address);
        }
      }
    }
  }

  void fetch_subtraction() {
    subtractionFetchParams.ResetRead();
    scalarAddressRequest.Reset();

    wait();

    while (true) {
      Params params = subtractionFetchParams.Pop();

      int rows = params.loops[0][params.inputXLoopIndex[0]] *
                 params.loops[1][params.inputXLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
      for (int i = 0; i < rows; i++) {
        int address = params.VEC_SUB_OFFSET + i;
        scalarAddressRequest.Push(address);
      }
    }
  }

  void fetch_variance() {
    varianceFetchParams.ResetRead();
    varianceAddressRequest.Reset();

    wait();

    while (true) {
      Params params = varianceFetchParams.Pop();

      int rows = params.loops[0][params.inputXLoopIndex[0]] *
                 params.loops[1][params.inputXLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
      for (int i = 0; i < rows; i++) {
        int address = params.VEC_SCALE_OFFSET + i;
        varianceAddressRequest.Push(address);
      }
    }
  }

  void fetch_bias() {
    biasFetchParams.ResetRead();
    biasAddressRequest.Reset();

    wait();

    while (true) {
      Params params = biasFetchParams.Pop();
      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.reductionLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;
      loop_bounds[1][params.inputXLoopIndex[1]] = 1;
      loop_bounds[1][params.inputYLoopIndex[1]] = 1;
#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            for (int k1 = 0; k1 < params.loops[1][params.weightLoopIndex[1]];
                 k1++) {
              int k2 = loop_counters[0][params.weightLoopIndex[0]];
              int K2 = params.loops[0][params.weightLoopIndex[0]];

              int K1 = params.loops[1][params.weightLoopIndex[1]];
              int k = k2 * K1 * NROWS + k1 * NROWS;
              int K = K2 * K1 * NROWS;
              int address = params.BIAS_OFFSET + k;
              MemoryRequest memRequest = {address, NROWS};
              biasAddressRequest.Push(memRequest);
            }
          }
        }
      }
    }
  }

  void fetch_residual() {
    residualFetchParams.ResetRead();
    residualAddressRequest.Reset();

    wait();

    while (true) {
      Params params = residualFetchParams.Pop();

      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.reductionLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;

#pragma hls_pipeline_init_interval 1
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
                        int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                        int x1 = loop_counters[0][params.inputXLoopIndex[0]];
                        int X0 = params.loops[1][params.inputXLoopIndex[1]];
                        int X1 = params.loops[0][params.inputXLoopIndex[0]];
                        int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                        int y1 = loop_counters[0][params.inputYLoopIndex[0]];
                        int Y0 = params.loops[1][params.inputYLoopIndex[1]];
                        int Y1 = params.loops[0][params.inputYLoopIndex[0]];
                        int k2 = loop_counters[0][params.weightLoopIndex[0]];
                        int K2 = params.loops[0][params.weightLoopIndex[0]];
                        int k1 = loop_counters[1][params.weightLoopIndex[1]];
                        int K1 = params.loops[1][params.weightLoopIndex[1]];
                        int k = k2 * K1 * NROWS + k1 * NROWS;
                        int K = K2 * K1 * NROWS;

                        int x = x0 + x1 * X0;
                        int X = X0 * X1;

                        int y = y0 + y1 * Y0;
                        int Y = Y0 * Y1;

                        int address = y * X * K + x * K + k;
                        MemoryRequest memRequest = {
                            params.RESIDUAL_OFFSET + address, NROWS};
                        residualAddressRequest.Push(memRequest);
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

    vectorFetchParams.ResetWrite();
    subtractionFetchParams.ResetWrite();
    varianceFetchParams.ResetWrite();
    biasFetchParams.ResetWrite();
    residualFetchParams.ResetWrite();
    wait();

    while (true) {
      Params params = paramsIn.Pop();

      if (params.VEC_OP) {
        vectorFetchParams.Push(params);

        if (params.VEC_SUB) {
          subtractionFetchParams.Push(params);
        }

        if (!params.CONST_SCALE) {
          varianceFetchParams.Push(params);
        }
      }

      if (params.BIAS) {
        biasFetchParams.Push(params);
      }

      if (params.RESIDUAL) {
        residualFetchParams.Push(params);
      }
    }
  }
};

template <typename DTYPE, int WIDTH, int NROWS>
SC_MODULE(VectorOpUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(vectorIn);
  Connections::Out<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(vectorOut);
  Connections::In<DTYPE> CCS_INIT_S1(scalarSubtraction);

  SC_CTOR(VectorOpUnit) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    paramsIn.Reset();
    vectorIn.Reset();
    vectorOut.Reset();
    scalarSubtraction.Reset();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      int rows = params.loops[0][params.inputXLoopIndex[0]] *
                 params.loops[1][params.inputXLoopIndex[1]];
      int cols = NROWS * params.loops[0][params.weightLoopIndex[0]] *
                 params.loops[1][params.weightLoopIndex[1]];

      if (params.VEC_SUB) {
#pragma hls_pipeline_init_interval 1
        for (int m = 0; m < rows; m++) {
          DTYPE subtract = scalarSubtraction.Pop();

          for (int chunk = 0; chunk < cols / WIDTH; chunk++) {
            Pack1D<DTYPE, WIDTH> vector = vectorIn.Pop();
#pragma hls_unroll yes
            for (int i = 0; i < WIDTH; i++) {
              vector.value[i] -= subtract;
            }

            if (params.VEC_SQUARE) {
#pragma hls_unroll yes
              for (int i = 0; i < WIDTH; i++) {
                vector.value[i] *= vector.value[i];
              }
            }

            vectorOut.Push(vector);
          }
        }
      } else {  // bypass
#pragma hls_pipeline_init_interval 1
        for (int i = 0; i < rows * cols / WIDTH; i++) {
          vectorOut.Push(vectorIn.Pop());
        }
      }
    }
  }
};

template <typename DTYPE, int WIDTH, int NROWS>
SC_MODULE(ReduceUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(vectorIn);
  Connections::Out<DTYPE> CCS_INIT_S1(scalarOut);

  SC_CTOR(ReduceUnit) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    paramsIn.Reset();
    vectorIn.Reset();
    scalarOut.Reset();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      int rows = params.loops[0][params.inputXLoopIndex[0]] *
                 params.loops[1][params.inputXLoopIndex[1]];
      int cols = NROWS * params.loops[0][params.weightLoopIndex[0]] *
                 params.loops[1][params.weightLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
      for (int m = 0; m < rows; m++) {
        DTYPE sum = 0;

        for (int chunk = 0; chunk < cols / WIDTH; chunk++) {
          Pack1D<DTYPE, WIDTH> vector = vectorIn.Pop();

#pragma hls_unroll yes
#pragma cluster addtree
#pragma cluster_type both
          for (int i = 0; i < WIDTH; i++) {
            sum += vector.value[i];
          }
        }

        scalarOut.Push(sum);
      }
    }
  }
};

template <typename DTYPE, int WIDTH, int NROWS>
SC_MODULE(ScaleUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(vectorIn);
  Connections::Out<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(vectorOut);
  Connections::In<DTYPE> CCS_INIT_S1(scaleChannel);

  SC_CTOR(ScaleUnit) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    paramsIn.Reset();
    scaleChannel.Reset();
    vectorIn.Reset();
    vectorOut.Reset();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      int rows = params.loops[0][params.inputXLoopIndex[0]] *
                 params.loops[1][params.inputXLoopIndex[1]] *
                 params.loops[0][params.inputYLoopIndex[0]] *
                 params.loops[1][params.inputYLoopIndex[1]];
      int cols = params.loops[0][params.weightLoopIndex[0]] *
                 params.loops[1][params.weightLoopIndex[1]];

      int p = 0;
      DTYPE scale = params.SCALE;
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int count = 0; count < rows * cols; count++) {
        Pack1D<DTYPE, NROWS> vec = vectorIn.Pop();

        // TODO: fix scale
        // #pragma hls_unroll yes
        // for(int i = 0; i < NROWS; i++){
        //   vec[i] /= scale;
        // }

        vectorOut.Push(vec);
        p++;
        if (p == cols) {
          p = 0;
          if (!params.CONST_SCALE) {
            scale = scaleChannel.Pop();
          }
        }
      }
    }
  }
};

/*
 * Performs bias, residual, maxpool and avgpool operations
 */
template <typename DTYPE, int WIDTH, int NROWS>
SC_MODULE(ArithmeticUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(tensorIn);
  Connections::Out<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(tensorOut);

  Connections::In<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(biasIn);
  Connections::In<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(residualIn);

  SC_CTOR(ArithmeticUnit) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    paramsIn.Reset();
    tensorIn.Reset();
    tensorOut.Reset();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.reductionLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;

      Pack1D<DTYPE, NROWS> bias;

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
                        int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                        int x1 = loop_counters[0][params.inputXLoopIndex[0]];
                        int X0 = params.loops[1][params.inputXLoopIndex[1]];
                        int X1 = params.loops[0][params.inputXLoopIndex[0]];
                        int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                        int y1 = loop_counters[0][params.inputYLoopIndex[0]];
                        int Y0 = params.loops[1][params.inputYLoopIndex[1]];
                        int Y1 = params.loops[0][params.inputYLoopIndex[0]];
                        int k2 = loop_counters[0][params.weightLoopIndex[0]];
                        int K2 = params.loops[0][params.weightLoopIndex[0]];
                        int k1 = loop_counters[1][params.weightLoopIndex[1]];
                        int K1 = params.loops[1][params.weightLoopIndex[1]];
                        int k = k2 * K1 * NROWS + k1 * NROWS;
                        int K = K2 * K1 * NROWS;

                        int x = x0 + x1 * X0;
                        int X = X0 * X1;

                        int y = y0 + y1 * Y0;
                        int Y = Y0 * Y1;

                        if (x0 == 0 && y0 == 0) {
                          if (params.BIAS) {
                            bias = biasIn.Pop();
                          }
                        }

                        Pack1D<DTYPE, NROWS> outputPixel = tensorIn.Pop();

                        if (params.BIAS) {
#pragma hls_unroll yes
                          for (int i = 0; i < NROWS; i++) {
                            outputPixel.value[i] += bias.value[i];
                          }
                        }

                        if (params.RESIDUAL) {
                          Pack1D<DTYPE, NROWS> residualPixel = residualIn.Pop();
#pragma hls_unroll yes
                          for (int i = 0; i < NROWS; i++) {
                            outputPixel.value[i] += residualPixel.value[i];
                          }
                        }

                        if (params.MAXPOOL) {
                          if (x0 % 2 == 0 && y0 % 2 == 0) {
#pragma hls_unroll yes
                            for (int i = 0; i < NROWS; i++) {
                              // update maxpool comparator
                              maxpool_comparator[(x0 / 2)].value[i] =
                                  outputPixel.value[i];
                            }
                          } else if (x0 % 2 == 1 && y0 % 2 == 0) {
#pragma hls_unroll yes
                            for (int i = 0; i < NROWS; i++) {
                              // update maxpool comparator
                              if (maxpool_comparator[(x0 - 1) / 2].value[i] <
                                  outputPixel.value[i]) {
                                maxpool_comparator[(x0 - 1) / 2].value[i] =
                                    outputPixel.value[i];
                              }
                            }
                          } else if (x0 % 2 == 0 && y0 % 2 == 1) {
#pragma hls_unroll yes
                            for (int i = 0; i < NROWS; i++) {
                              // update maxpool comparator
                              if (maxpool_comparator[(x0) / 2].value[i] <
                                  outputPixel.value[i]) {
                                maxpool_comparator[(x0) / 2].value[i] =
                                    outputPixel.value[i];
                              }
                            }
                          } else {
#pragma hls_unroll yes
                            for (int i = 0; i < NROWS; i++) {
                              if (maxpool_comparator[(x0 - 1) / 2].value[i] <
                                  outputPixel.value[i]) {
                                maxpool_comparator[(x0 - 1) / 2].value[i] =
                                    outputPixel.value[i];
                              }
                            }
                            tensorOut.Push(maxpool_comparator[(x0 - 1) / 2]);
                          }
                        } else if (params.AVGPOOL) {
                          if (x0 == 0 && y0 == 0) {
                            avgpool = outputPixel;
                          } else {
#pragma hls_unroll yes
                            for (int i = 0; i < NROWS; i++) {
                              avgpool.value[i] += outputPixel.value[i];
                            }
                          }

                          if (x0 == X0 - 1 && y0 == Y0 - 1) {
                            tensorOut.Push(avgpool);
                          }
                        } else {
                          tensorOut.Push(outputPixel);
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

 private:
  Pack1D<DTYPE, WIDTH> maxpool_comparator[16];  // row buffer for maxpool
  Pack1D<DTYPE, WIDTH> avgpool;
};

template <int NROWS, int WIDTH>
SC_MODULE(OutputAddressGenerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::Out<int> CCS_INIT_S1(outputAddress);

  SC_CTOR(OutputAddressGenerator) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    paramsIn.Reset();
    outputAddress.Reset();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      int rows = params.loops[0][params.inputXLoopIndex[0]] *
                 params.loops[1][params.inputXLoopIndex[1]];
      int cols = params.loops[0][params.weightLoopIndex[0]] *
                 params.loops[1][params.weightLoopIndex[1]];

      if (params.VEC_OP) {
        if (params.VEC_REDUCE) {
#pragma hls_pipeline_init_interval 1
          for (int m = 0; m < rows / WIDTH; m++) {
            int address = params.OUTPUT_OFFSET + m;
            outputAddress.Push(address);
          }
        } else {
#pragma hls_pipeline_init_interval 1
          for (int m = 0; m < rows; m++) {
            for (int p = 0; p < NROWS * cols; p++) {
              int address = m * (NROWS * cols) + p;
              address = params.OUTPUT_OFFSET + address;
              outputAddress.Push(address);
            }
          }
        }
      } else {
        int loop_counters[2][6];
        int loop_bounds[2][6];

#pragma hls_unroll yes
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 6; j++) {
            loop_bounds[i][j] = params.loops[i][j];
          }
        }

        // set irrelevant loop bounds to 1
        loop_bounds[1][params.reductionLoopIndex[1]] = 1;
        loop_bounds[1][params.fxIndex] = 1;
        loop_bounds[1][params.fyIndex] = 1;

        if (params.MAXPOOL) {
          loop_bounds[1][params.inputXLoopIndex[1]] =
              loop_bounds[1][params.inputXLoopIndex[1]] / 2;
          loop_bounds[1][params.inputYLoopIndex[1]] =
              loop_bounds[1][params.inputYLoopIndex[1]] / 2;
        }

        if (params.AVGPOOL) {
          loop_bounds[1][params.inputXLoopIndex[1]] = 1;
          loop_bounds[1][params.inputYLoopIndex[1]] = 1;
        }

#pragma hls_pipeline_init_interval 1
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
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
                          int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                          int x1 = loop_counters[0][params.inputXLoopIndex[0]];
                          int X0 = loop_bounds[1][params.inputXLoopIndex[1]];
                          int X1 = params.loops[0][params.inputXLoopIndex[0]];
                          int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                          int y1 = loop_counters[0][params.inputYLoopIndex[0]];
                          int Y0 = loop_bounds[1][params.inputYLoopIndex[1]];
                          int Y1 = params.loops[0][params.inputYLoopIndex[0]];
                          int k2 = loop_counters[0][params.weightLoopIndex[0]];
                          int K2 = params.loops[0][params.weightLoopIndex[0]];
                          int k1 = loop_counters[1][params.weightLoopIndex[1]];
                          int K1 = params.loops[1][params.weightLoopIndex[1]];
                          int k = k2 * K1 * NROWS + k1 * NROWS;
                          int K = K2 * K1 * NROWS;

                          int x = x0 + x1 * X0;
                          int X = X0 * X1;

                          int y = y0 + y1 * Y0;
                          int Y = Y0 * Y1;

                          int baseAddress = y * X * K + x * K + k;
                          int address = params.OUTPUT_OFFSET + baseAddress;
                          outputAddress.Push(address);

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
};

template <typename DTYPE, int WIDTH, int NROWS>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(systolicArrayOutput);

  Connections::Out<int> CCS_INIT_S1(vectorFetchAddressRequest);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(vectorFetchDataResponse);
  Connections::Out<int> CCS_INIT_S1(scalarAddressRequest);
  Connections::In<DTYPE> CCS_INIT_S1(scalarDataResponse);
  Connections::Out<int> CCS_INIT_S1(varianceAddressRequest);
  Connections::In<DTYPE> CCS_INIT_S1(varianceDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(biasDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(residualAddressRequest);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(residualDataResponse);
  Connections::Out<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(vectorUnitOutput);
  Connections::Out<int> CCS_INIT_S1(outputAddress);
  Connections::SyncOut CCS_INIT_S1(done);

  FetchUnit<NROWS> CCS_INIT_S1(fetchUnit);
  Connections::Combinational<Params> CCS_INIT_S1(fetchUnitParams);

  VectorOpUnit<DTYPE, NROWS, NROWS> CCS_INIT_S1(vectorOpUnit);
  Connections::Combinational<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(
      vectorOpUnitOutput);
  Connections::Combinational<Params> CCS_INIT_S1(vectorOpUnitParams);

  ReduceUnit<DTYPE, NROWS, NROWS> CCS_INIT_S1(reduceUnit);
  Connections::Combinational<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(
      reduceUnitInput);
  Connections::Combinational<DTYPE> CCS_INIT_S1(reduceUnitOutput);
  Connections::Combinational<Params> CCS_INIT_S1(reduceUnitParams);

  ScaleUnit<DTYPE, WIDTH, NROWS> CCS_INIT_S1(scaleUnit);
  Connections::Combinational<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(scaleUnitInput);
  Connections::Combinational<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(
      scaleUnitOutput);
  Connections::Combinational<Params> CCS_INIT_S1(scaleUnitParams);

  ArithmeticUnit<DTYPE, WIDTH, NROWS> CCS_INIT_S1(arithmeticUnit);
  Connections::Combinational<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(
      arithmeticUnitInput);
  Connections::Combinational<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(
      arithmeticUnitOutput);
  Connections::Combinational<Params> CCS_INIT_S1(arithmeticUnitParams);

  OutputAddressGenerator<NROWS, NROWS> CCS_INIT_S1(outputAddressGenerator);
  Connections::Combinational<Params> CCS_INIT_S1(outputAddressGenParams);

  Connections::Combinational<Params> CCS_INIT_S1(inputConnectionParams);
  Connections::Combinational<Params> CCS_INIT_S1(outputConnectionParams);

  SC_CTOR(VectorUnit) {
    fetchUnit.clk(clk);
    fetchUnit.rstn(rstn);
    fetchUnit.vectorFetchAddressRequest(vectorFetchAddressRequest);
    fetchUnit.scalarAddressRequest(scalarAddressRequest);
    fetchUnit.varianceAddressRequest(varianceAddressRequest);
    fetchUnit.biasAddressRequest(biasAddressRequest);
    fetchUnit.residualAddressRequest(residualAddressRequest);
    fetchUnit.paramsIn(fetchUnitParams);

    vectorOpUnit.clk(clk);
    vectorOpUnit.rstn(rstn);
    vectorOpUnit.vectorIn(vectorFetchDataResponse);
    vectorOpUnit.vectorOut(vectorOpUnitOutput);
    vectorOpUnit.scalarSubtraction(scalarDataResponse);
    vectorOpUnit.paramsIn(vectorOpUnitParams);

    reduceUnit.clk(clk);
    reduceUnit.rstn(rstn);
    reduceUnit.vectorIn(reduceUnitInput);
    reduceUnit.scalarOut(reduceUnitOutput);
    reduceUnit.paramsIn(reduceUnitParams);

    scaleUnit.clk(clk);
    scaleUnit.rstn(rstn);
    scaleUnit.paramsIn(scaleUnitParams);
    scaleUnit.vectorIn(scaleUnitInput);
    scaleUnit.vectorOut(scaleUnitOutput);
    scaleUnit.scaleChannel(varianceDataResponse);

    arithmeticUnit.clk(clk);
    arithmeticUnit.rstn(rstn);
    arithmeticUnit.paramsIn(arithmeticUnitParams);
    arithmeticUnit.tensorIn(arithmeticUnitInput);
    arithmeticUnit.tensorOut(arithmeticUnitOutput);
    arithmeticUnit.biasIn(biasDataResponse);
    arithmeticUnit.residualIn(residualDataResponse);

    outputAddressGenerator.clk(clk);
    outputAddressGenerator.rstn(rstn);
    outputAddressGenerator.paramsIn(outputAddressGenParams);
    outputAddressGenerator.outputAddress(outputAddress);

    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(connect_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(connect_outputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    paramsIn.Reset();
    fetchUnitParams.ResetWrite();
    vectorOpUnitParams.ResetWrite();
    reduceUnitParams.ResetWrite();
    arithmeticUnitParams.ResetWrite();
    outputAddressGenParams.ResetWrite();
    inputConnectionParams.ResetWrite();
    outputConnectionParams.ResetWrite();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      fetchUnitParams.Push(params);
      if (params.VEC_OP) {
        vectorOpUnitParams.Push(params);
        if (params.VEC_REDUCE) {
          reduceUnitParams.Push(params);
        }
      }

      if (!params.VEC_REDUCE) {
        // scaleUnitParams.Push(params);
        arithmeticUnitParams.Push(params);
      }

      inputConnectionParams.Push(params);
      outputConnectionParams.Push(params);
      outputAddressGenParams.Push(params);
    }
  }

  void connect_inputs() {
    inputConnectionParams.ResetRead();
    vectorOpUnitOutput.ResetRead();
    systolicArrayOutput.Reset();

    reduceUnitInput.ResetWrite();
    arithmeticUnitInput.ResetWrite();

    wait();

    while (true) {
      Params params = inputConnectionParams.Pop();

      int total_outputs = params.loops[0][params.inputXLoopIndex[0]] *
                          params.loops[1][params.inputXLoopIndex[1]] *
                          params.loops[0][params.inputYLoopIndex[0]] *
                          params.loops[1][params.inputYLoopIndex[1]] *
                          params.loops[0][params.weightLoopIndex[0]] *
                          params.loops[1][params.weightLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < total_outputs; i++) {
        Pack1D<DTYPE, NROWS> data;

        if (params.VEC_OP) {
          data = vectorOpUnitOutput.Pop();

        } else {
          data = systolicArrayOutput.Pop();
        }

        if (params.VEC_REDUCE) {
          reduceUnitInput.Push(data);
        } else {
          arithmeticUnitInput.Push(data);
        }
      }
    }
  }

  void connect_outputs() {
    outputConnectionParams.ResetRead();
    arithmeticUnitOutput.ResetRead();
    reduceUnitOutput.ResetRead();
    vectorUnitOutput.Reset();

    done.Reset();

    wait();

    while (true) {
      Params params = outputConnectionParams.Pop();

      if (params.VEC_REDUCE) {
        int inputSize = params.loops[0][params.inputXLoopIndex[0]] *
                        params.loops[1][params.inputXLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < inputSize / WIDTH; i++) {
          Pack1D<DTYPE, WIDTH> reduceUnitOutputVector;
          for (int i = 0; i < WIDTH; i++) {
            reduceUnitOutputVector[i] = reduceUnitOutput.Pop();
          }
          vectorUnitOutput.Push(reduceUnitOutputVector);
        }
      } else {
        int total_outputs = params.loops[0][params.inputXLoopIndex[0]] *
                            params.loops[1][params.inputXLoopIndex[1]] *
                            params.loops[0][params.inputYLoopIndex[0]] *
                            params.loops[1][params.inputYLoopIndex[1]] *
                            params.loops[0][params.weightLoopIndex[0]] *
                            params.loops[1][params.weightLoopIndex[1]];
        if (params.MAXPOOL) {
          total_outputs = total_outputs / 4;
        }
        if (params.AVGPOOL) {
          total_outputs = params.loops[0][params.weightLoopIndex[0]] *
                          params.loops[1][params.weightLoopIndex[1]];
        }
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < total_outputs; i++) {
          vectorUnitOutput.Push(arithmeticUnitOutput.Pop());
        }
      }

      done.SyncPush();
    }
  }
};
