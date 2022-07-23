#pragma once

template <int WIDTH>
SC_MODULE(OutputAddressGenerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::Out<int> CCS_INIT_S1(vectorOutputAddress);
  Connections::Out<int> CCS_INIT_S1(scalarOutputAddress);

  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vectorOutputAddressParams);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      scalarOutputAddressParams);

  SC_CTOR(OutputAddressGenerator) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(vectorAddressGen);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(scalarAddressGen);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    paramsIn.Reset();
    vectorOutputAddressParams.ResetWrite();
    scalarOutputAddressParams.ResetWrite();

    wait();

    while (true) {
      VectorParams params = paramsIn.Pop();

      vectorOutputAddressParams.Push(params);

      if (params.scalarOutputCount != 0) {
        scalarOutputAddressParams.Push(params);
      }
    }
  }

  void scalarAddressGen() {
    scalarOutputAddress.Reset();
    scalarOutputAddressParams.ResetRead();

    wait();

    while (true) {
      VectorParams params = scalarOutputAddressParams.Pop();

      for (int i = 0; i < params.scalarOutputCount; i++) {
        scalarOutputAddress.Push(params.SCALAR_OUTPUT_OFFSET + i * WIDTH);
      }
    }
  }

  void vectorAddressGen() {
    vectorOutputAddress.Reset();
    vectorOutputAddressParams.ResetRead();

    wait();

    while (true) {
      VectorParams params = vectorOutputAddressParams.Pop();

      int loop_counters[2][3];
      int loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.outputLoops[i][j];
        }
      }

      if (params.MAXPOOL) {
        loop_bounds[1][params.outputXLoopIndex[1]] =
            loop_bounds[1][params.outputXLoopIndex[1]] / 2;
        loop_bounds[1][params.outputYLoopIndex[1]] =
            loop_bounds[1][params.outputYLoopIndex[1]] / 2;
      }

      if (params.AVGPOOL) {
        loop_bounds[1][params.outputXLoopIndex[1]] = 1;
        loop_bounds[1][params.outputYLoopIndex[1]] = 1;
      }

      if (params.SPLIT_HEAD) DLOG("splitting heads");

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
                  int x0 = loop_counters[1][params.outputXLoopIndex[1]];
                  int x1 = loop_counters[0][params.outputXLoopIndex[0]];
                  int X0 = loop_bounds[1][params.outputXLoopIndex[1]];
                  int X1 = params.outputLoops[0][params.outputXLoopIndex[0]];
                  int y0 = loop_counters[1][params.outputYLoopIndex[1]];
                  int y1 = loop_counters[0][params.outputYLoopIndex[0]];
                  int Y0 = loop_bounds[1][params.outputYLoopIndex[1]];
                  int Y1 = params.outputLoops[0][params.outputYLoopIndex[0]];
                  int k2 = loop_counters[0][params.outputWeightLoopIndex[0]];
                  int K2 =
                      params.outputLoops[0][params.outputWeightLoopIndex[0]];
                  int k1 = loop_counters[1][params.outputWeightLoopIndex[1]];
                  int K1 =
                      params.outputLoops[1][params.outputWeightLoopIndex[1]];
                  int k = k2 * K1 * WIDTH + k1 * WIDTH;
                  int K = K2 * K1 * WIDTH;
                  if (params.DP_OUTPUT) {
                    K = K * 2;
                  }

                  int x = x0 + x1 * X0;
                  int X = X0 * X1;

                  int y = y0 + y1 * Y0;
                  int Y = Y0 * Y1;

                  int baseAddress = y * X * K + x * K + k;
                  if (params.SPLIT_HEAD) {
                    baseAddress = ((k / 32) * X * 32) + (x * 32) + (k % 32);
                  }

                  int address = params.VECTOR_OUTPUT_OFFSET + baseAddress;
                  if (params.DP_OUTPUT) {
                    for (int precision = 0; precision < 2; precision++) {
                      vectorOutputAddress.Push(address + precision);
                    }
                  } else {
                    vectorOutputAddress.Push(address);
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
};
