#pragma once

template <typename DTYPE, int WIDTH>
SC_MODULE(OutputAddressGenerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::Out<ac_int<64, false> > CCS_INIT_S1(vectorOutputAddress);

  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vectorOutputAddressParams);

  SC_CTOR(OutputAddressGenerator) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(vectorAddressGen);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    paramsIn.Reset();
    vectorOutputAddressParams.ResetWrite();

    wait();

    while (true) {
      VectorParams params = paramsIn.Pop();

      vectorOutputAddressParams.Push(params);
    }
  }

  void vectorAddressGen() {
    vectorOutputAddress.Reset();
    vectorOutputAddressParams.ResetRead();

    wait();

    while (true) {
      VectorParams params = vectorOutputAddressParams.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.outputLoops[i][j];
        }
      }

      // if (params.MAXPOOL) {
      //   loop_bounds[1][params.outputXLoopIndex[1]] =
      //       loop_bounds[1][params.outputXLoopIndex[1]] / 2;
      //   loop_bounds[1][params.outputYLoopIndex[1]] =
      //       loop_bounds[1][params.outputYLoopIndex[1]] / 2;
      // }

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
                  ac_int<32, false> baseAddress;
                  if (params.outputAddressMode == 1) {
                    ac_int<11, false> x0 =
                        loop_counters[1][params.outputXLoopIndex[1]];
                    ac_int<11, false> x1 =
                        loop_counters[0][params.outputXLoopIndex[0]];
                    ac_int<11, false> y0 =
                        loop_counters[1][params.outputYLoopIndex[1]];
                    ac_int<11, false> y1 =
                        loop_counters[0][params.outputYLoopIndex[0]];
                    ac_int<11, false> k1 =
                        loop_counters[1][params.outputWeightLoopIndex[1]];
                    ac_int<11, false> k2 =
                        loop_counters[0][params.outputWeightLoopIndex[0]];

                    ac_int<11, false> X0 =
                        loop_bounds[1][params.outputXLoopIndex[1]];
                    ac_int<11, false> X1 =
                        loop_bounds[0][params.outputXLoopIndex[0]];
                    ac_int<11, false> Y0 =
                        loop_bounds[1][params.outputYLoopIndex[1]];
                    ac_int<11, false> Y1 =
                        loop_bounds[0][params.outputYLoopIndex[0]];
                    ac_int<11, false> K2 =
                        loop_bounds[0][params.outputWeightLoopIndex[0]];
                    ac_int<11, false> K1 =
                        loop_bounds[1][params.outputWeightLoopIndex[1]];

                    ac_int<16, false> k = k2 * K1 * WIDTH + k1 * WIDTH;
                    ac_int<16, false> K = K2 * K1 * WIDTH;

                    ac_int<16, false> x = x0 + x1 * X0;
                    ac_int<16, false> X = X0 * X1;

                    ac_int<16, false> y = y0 + y1 * Y0;
                    ac_int<16, false> Y = Y0 * Y1;

                    ac_int<8, false> headSize = params.headSizeInPowerOfTwo;
                    ac_int<16, false> mask = (1 << headSize) - 1;

                    if (params.SPLIT_OUTPUT) {
                      baseAddress = (((k >> headSize) * X) << headSize) +
                                    (x << headSize) + (k & mask);
                    } else if (params.CONCAT_OUTPUT) {
                      baseAddress = ((k >> headSize) * K) +
                                    ((y & mask) * K * 4) + (k & mask) +
                                    (y >> headSize * K / 4);
                    } else {
                      baseAddress = y * X * K + x * K + k;
                    }
                  } else if (params.outputAddressMode == 2) {
                    ac_int<11, false> loop_0 = loop_counters[0][0];
                    ac_int<11, false> loop_1 = loop_counters[0][1];
                    ac_int<11, false> loop_2 = loop_counters[0][2];
                    ac_int<11, false> loop_3 = loop_counters[1][0];
                    ac_int<11, false> loop_4 = loop_counters[1][1];
                    ac_int<11, false> loop_5 = loop_counters[1][2];

                    ac_int<11, false> loop_bound_0 = loop_bounds[0][0];
                    ac_int<11, false> loop_bound_1 = loop_bounds[0][1];
                    ac_int<11, false> loop_bound_2 = loop_bounds[0][2];
                    ac_int<11, false> loop_bound_3 = loop_bounds[1][0];
                    ac_int<11, false> loop_bound_4 = loop_bounds[1][1];
                    ac_int<11, false> loop_bound_5 = loop_bounds[1][2];

                    baseAddress =
                        (loop_0 * loop_bound_1 * loop_bound_2 * loop_bound_3 *
                             loop_bound_4 * loop_bound_5 +
                         loop_1 * loop_bound_2 * loop_bound_3 * loop_bound_4 *
                             loop_bound_5 +
                         loop_2 * loop_bound_3 * loop_bound_4 * loop_bound_5 +
                         loop_3 * loop_bound_4 * loop_bound_5 +
                         loop_4 * loop_bound_5 + loop_5) *
                        WIDTH;
                  }

                  if (params.DP_OUTPUT) {
                    for (int precision = 0; precision < 2; precision++) {
                      vectorOutputAddress.Push(
                          params.VECTOR_OUTPUT_OFFSET +
                          baseAddress * (DTYPE::width / 8) * 2 +
                          precision * (DTYPE::width / 8) * WIDTH);
                    }
                  } else {
                    vectorOutputAddress.Push(params.VECTOR_OUTPUT_OFFSET +
                                             baseAddress);
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
