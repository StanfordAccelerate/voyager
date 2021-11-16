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

  Connections::Combinational<Params> CCS_INIT_S1(vectorFetchParams);
  Connections::Combinational<Params> CCS_INIT_S1(subtractionFetchParams);
  Connections::Combinational<Params> CCS_INIT_S1(varianceFetchParams);

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
  }

  void fetch_vector() {
    vectorFetchParams.ResetRead();
    vectorFetchAddressRequest.Reset();

    wait();

    while (true) {
      Params params = vectorFetchParams.Pop();

      int rows = params.loops[0][params.inputLoopIndex[0]] *
                 params.loops[1][params.inputLoopIndex[1]];
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

      int rows = params.loops[0][params.inputLoopIndex[0]] *
                 params.loops[1][params.inputLoopIndex[1]];

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

      int rows = params.loops[0][params.inputLoopIndex[0]] *
                 params.loops[1][params.inputLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
      for (int i = 0; i < rows; i++) {
        int address = params.VEC_SCALE_OFFSET + i;
        varianceAddressRequest.Push(address);
      }
    }
  }

  void read_params() {
    paramsIn.Reset();

    vectorFetchParams.ResetWrite();
    subtractionFetchParams.ResetWrite();
    varianceFetchParams.ResetWrite();

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
    }
  }
};

template <typename DTYPE, int WIDTH, int NROWS>
SC_MODULE(ArithmeticUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(vectorIn);
  Connections::Out<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(vectorOut);
  Connections::In<DTYPE> CCS_INIT_S1(scalarSubtraction);

  SC_CTOR(ArithmeticUnit) {
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

      int rows = params.loops[0][params.inputLoopIndex[0]] *
                 params.loops[1][params.inputLoopIndex[1]];
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

      int rows = params.loops[0][params.inputLoopIndex[0]] *
                 params.loops[1][params.inputLoopIndex[1]];
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

      int rows = params.loops[0][params.inputLoopIndex[0]] *
                 params.loops[1][params.inputLoopIndex[1]];
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

      int rows = params.loops[0][params.inputLoopIndex[0]] *
                 params.loops[1][params.inputLoopIndex[1]];
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
        int loop_counters[2][3];

#pragma hls_pipeline_init_interval 1
        for (loop_counters[0][0] = 0; loop_counters[0][0] < params.loops[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0;
               loop_counters[0][1] < params.loops[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < params.loops[0][2];
                 loop_counters[0][2]++) {
              // inner memory
              for (loop_counters[1][params.weightLoopIndex[1]] = 0;
                   loop_counters[1][params.weightLoopIndex[1]] <
                   params.loops[1][params.weightLoopIndex[1]];
                   loop_counters[1][params.weightLoopIndex[1]]++) {
                for (loop_counters[1][params.inputLoopIndex[1]] = 0;
                     loop_counters[1][params.inputLoopIndex[1]] <
                     params.loops[1][params.inputLoopIndex[1]];
                     loop_counters[1][params.inputLoopIndex[1]]++) {
                  int m0 = loop_counters[1][params.inputLoopIndex[1]];
                  int m1 = loop_counters[0][params.inputLoopIndex[0]];
                  int M0 = params.loops[1][params.inputLoopIndex[1]];
                  int P1 = params.loops[1][params.weightLoopIndex[1]];
                  int p1 = loop_counters[1][params.weightLoopIndex[1]];
                  int p2 = loop_counters[0][params.weightLoopIndex[0]];
                  int P2 = params.loops[0][params.weightLoopIndex[0]];

                  int m = m1 * M0 + m0;
                  int p = p2 * P1 * NROWS + p1 * NROWS;

                  int address =
                      params.OUTPUT_OFFSET + (m * (P1 * P2 * NROWS) + p);
                  outputAddress.Push(address);
                }
              }
            }
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
  Connections::Out<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(vectorUnitOutput);
  Connections::Out<int> CCS_INIT_S1(outputAddress);
  Connections::SyncOut CCS_INIT_S1(done);

  FetchUnit<NROWS> CCS_INIT_S1(fetchUnit);
  Connections::Combinational<Params> CCS_INIT_S1(fetchUnitParams);

  ArithmeticUnit<DTYPE, NROWS, NROWS> CCS_INIT_S1(arithmeticUnit);
  Connections::Combinational<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(
      arithmeticUnitOutput);
  Connections::Combinational<Params> CCS_INIT_S1(arithmeticUnitParams);

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
    fetchUnit.paramsIn(fetchUnitParams);

    arithmeticUnit.clk(clk);
    arithmeticUnit.rstn(rstn);
    arithmeticUnit.vectorIn(vectorFetchDataResponse);
    arithmeticUnit.vectorOut(arithmeticUnitOutput);
    arithmeticUnit.scalarSubtraction(scalarDataResponse);
    arithmeticUnit.paramsIn(arithmeticUnitParams);

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
    arithmeticUnitParams.ResetWrite();
    reduceUnitParams.ResetWrite();
    scaleUnitParams.ResetWrite();
    outputAddressGenParams.ResetWrite();
    inputConnectionParams.ResetWrite();
    outputConnectionParams.ResetWrite();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      if (params.VEC_OP) {
        fetchUnitParams.Push(params);
        arithmeticUnitParams.Push(params);
        if (params.VEC_REDUCE) {
          reduceUnitParams.Push(params);
        }
      }

      if (!params.VEC_REDUCE) {
        scaleUnitParams.Push(params);
      }

      inputConnectionParams.Push(params);
      outputConnectionParams.Push(params);
      outputAddressGenParams.Push(params);
    }
  }

  void connect_inputs() {
    inputConnectionParams.ResetRead();
    arithmeticUnitOutput.ResetRead();
    systolicArrayOutput.Reset();

    reduceUnitInput.ResetWrite();
    scaleUnitInput.ResetWrite();

    wait();

    while (true) {
      Params params = inputConnectionParams.Pop();

      int total_outputs = params.loops[0][params.inputLoopIndex[0]] *
                          params.loops[1][params.inputLoopIndex[1]] *
                          params.loops[0][params.weightLoopIndex[0]] *
                          params.loops[1][params.weightLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < total_outputs; i++) {
        Pack1D<DTYPE, NROWS> data;

        if (params.VEC_OP) {
          data = arithmeticUnitOutput.Pop();

        } else {
          data = systolicArrayOutput.Pop();
        }

        if (params.VEC_REDUCE) {
          reduceUnitInput.Push(data);
        } else {
          scaleUnitInput.Push(data);
        }
      }
    }
  }

  void connect_outputs() {
    outputConnectionParams.ResetRead();
    scaleUnitOutput.ResetRead();
    reduceUnitOutput.ResetRead();
    vectorUnitOutput.Reset();

    done.Reset();

    wait();

    while (true) {
      Params params = outputConnectionParams.Pop();

      if (params.VEC_REDUCE) {
        int inputSize = params.loops[0][params.inputLoopIndex[0]] *
                        params.loops[1][params.inputLoopIndex[1]];

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
        int total_outputs = params.loops[0][params.inputLoopIndex[0]] *
                            params.loops[1][params.inputLoopIndex[1]] *
                            params.loops[0][params.weightLoopIndex[0]] *
                            params.loops[1][params.weightLoopIndex[1]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < total_outputs; i++) {
          vectorUnitOutput.Push(scaleUnitOutput.Pop());
        }
      }

      done.SyncPush();
    }
  }
};
