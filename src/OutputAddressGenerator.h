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

      // if (params.scalarOutputCount != 0) {
      //   scalarOutputAddressParams.Push(params);
      // }
    }
  }

  // TODO(fpedd): This SC_THREAD seems dead since I removed the for loop in the
  // while loop...
  void scalarAddressGen() {
    scalarOutputAddress.Reset();
    scalarOutputAddressParams.ResetRead();

    wait();

    while (true) {
      VectorParams params = scalarOutputAddressParams.Pop();

      // for (int i = 0; i < params.scalarOutputCount; i++) {
      //   scalarOutputAddress.Push(params.SCALAR_OUTPUT_OFFSET + i * WIDTH);
      // }
    }
  }

  void vectorAddressGen() {
    vectorOutputAddress.Reset();
    vectorOutputAddressParams.ResetRead();

    wait();

    while (true) {
      VectorParams params = vectorOutputAddressParams.Pop();

      ac_int<10, false> loop_counters[2][3];
      ac_int<10, false> loop_bounds[2][3];

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

      if (params.SPLIT_OUTPUT) DLOG("splitting heads");

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
                  ac_int<10, false> x0 =
                      loop_counters[1][params.outputXLoopIndex[1]];
                  ac_int<10, false> x1 =
                      loop_counters[0][params.outputXLoopIndex[0]];
                  ac_int<10, false> X0 =
                      loop_bounds[1][params.outputXLoopIndex[1]];
                  ac_int<10, false> X1 =
                      params.outputLoops[0][params.outputXLoopIndex[0]];
                  ac_int<10, false> y0 =
                      loop_counters[1][params.outputYLoopIndex[1]];
                  ac_int<10, false> y1 =
                      loop_counters[0][params.outputYLoopIndex[0]];
                  ac_int<10, false> Y0 =
                      loop_bounds[1][params.outputYLoopIndex[1]];
                  ac_int<10, false> Y1 =
                      params.outputLoops[0][params.outputYLoopIndex[0]];
                  ac_int<10, false> k2 =
                      loop_counters[0][params.outputWeightLoopIndex[0]];
                  ac_int<10, false> K2 =
                      params.outputLoops[0][params.outputWeightLoopIndex[0]];
                  ac_int<10, false> k1 =
                      loop_counters[1][params.outputWeightLoopIndex[1]];
                  ac_int<10, false> K1 =
                      params.outputLoops[1][params.outputWeightLoopIndex[1]];
                  ac_int<16, false> k = k2 * K1 * WIDTH + k1 * WIDTH;
                  ac_int<16, false> K = K2 * K1 * WIDTH;

                  ac_int<16, false> x = x0 + x1 * X0;
                  ac_int<16, false> X = X0 * X1;

                  ac_int<16, false> y = y0 + y1 * Y0;
                  ac_int<16, false> Y = Y0 * Y1;

                  int baseAddress = y * X * K + x * K + k;
                  if (params.SPLIT_OUTPUT) {
                    baseAddress =
                        static_cast<ac_int<32, false> >((k / 32) * X * 32) +
                        static_cast<ac_int<32, false> >(x * 32) +
                        static_cast<ac_int<32, false> >(k % 32);
                  }

                  int address = params.VECTOR_OUTPUT_OFFSET + baseAddress;
                  if (params.DP_OUTPUT) {
                    for (int precision = 0; precision < 2; precision++) {
                      vectorOutputAddress.Push(params.VECTOR_OUTPUT_OFFSET +
                                               baseAddress * 2 +
                                               precision * WIDTH);
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
