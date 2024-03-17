#pragma once

/*
 * Performs bias, residual, maxpool and avgpool operations
 */
template <typename ACC_DTYPE, typename DTYPE, int WIDTH>
SC_MODULE(MaxpoolUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<typename ACC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(tensorIn);
  Connections::Out<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(tensorOut);

  Connections::SyncOut CCS_INIT_S1(doneSignal);

  SC_CTOR(MaxpoolUnit) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    paramsIn.Reset();
    tensorIn.Reset();
    tensorOut.Reset();
    doneSignal.Reset();

    wait();

    while (true) {
      VectorParams params = paramsIn.Pop();

      ac_int<10, false> loop_counters[2][3];
      ac_int<10, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.outputLoops[i][j];
          DLOG(i << " " << j << " " << loop_bounds[i][j]);
        }
      }

      if (params.AVGPOOL) {
        loop_bounds[1][params.outputXLoopIndex[1]] = 1;
        loop_bounds[1][params.outputYLoopIndex[1]] = 1;
      }

      Pack1D<DTYPE, WIDTH> maxpool_comparator[16];  // row buffer for maxpool

      if (params.DP_OUTPUT) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
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
                    ac_int<10, false> x0 =
                        loop_counters[1][params.outputXLoopIndex[1]];
                    ac_int<10, false> x1 =
                        loop_counters[0][params.outputXLoopIndex[0]];
                    ac_int<10, false> X0 =
                        params.outputLoops[1][params.outputXLoopIndex[1]];
                    ac_int<10, false> X1 =
                        params.outputLoops[0][params.outputXLoopIndex[0]];
                    ac_int<10, false> y0 =
                        loop_counters[1][params.outputYLoopIndex[1]];
                    ac_int<10, false> y1 =
                        loop_counters[0][params.outputYLoopIndex[0]];
                    ac_int<10, false> Y0 =
                        params.outputLoops[1][params.outputYLoopIndex[1]];
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

                    Pack1D<typename ACC_DTYPE::AccumulationDatatype, WIDTH>
                        uncastedOutputPixel = tensorIn.Pop();
                    Pack1D<DTYPE, WIDTH> outputPixel;

                    Pack1D<ACC_DTYPE, WIDTH> dpOutputPixel;
#pragma hls_unroll yes
                    for (int i = 0; i < WIDTH; i++) {
                      dpOutputPixel[i] =
                          static_cast<ACC_DTYPE>(uncastedOutputPixel[i]);
                    }

                    for (int vecSlice = 0; vecSlice < 2; vecSlice++) {
                      Pack1D<DTYPE, WIDTH> dpHalfVec;
#pragma hls_unroll yes
                      for (int i = 0; i < WIDTH / 2; i++) {
#pragma hls_unroll yes
                        for (int byte = 0; byte < 2; byte++) {
                          dpHalfVec[i * 2 + byte].setbits(
                              dpOutputPixel[vecSlice * (WIDTH / 2) + i]
                                  .bits_rep().template slc<8>(byte * 8));
                        }
                      }
                      tensorOut.Push(dpHalfVec);
                    }

                    

                    // sc_lv<ACC_DTYPE::width * WIDTH> dpOutputPixelBits =
                    //   TypeToBits<Pack1D<ACC_DTYPE, WIDTH> > (dpOutputPixel);

                    // for (int vecSlice = 0; vecSlice < 2; vecSlice++) {
                    //   Pack1D<DTYPE, WIDTH> dpHalfVec;

                    //   dpHalfVec = 
                    //       BitsToType<Pack1D<DTYPE, WIDTH> >(
                    //       static_cast<sc_lv<DTYPE::width * WIDTH> > 
                    //       (dpOutputPixelBits[(vecSlice * (WIDTH / 2) * ACC_DTYPE::width), 
                    //       ((vecSlice + 1) * (WIDTH / 2) * ACC_DTYPE::width)]));

                    //   tensorOut.Push(dpHalfVec);
                    // }

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

      } else {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
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
                    ac_int<10, false> x0 =
                        loop_counters[1][params.outputXLoopIndex[1]];
                    ac_int<10, false> x1 =
                        loop_counters[0][params.outputXLoopIndex[0]];
                    ac_int<10, false> X0 =
                        params.outputLoops[1][params.outputXLoopIndex[1]];
                    ac_int<10, false> X1 =
                        params.outputLoops[0][params.outputXLoopIndex[0]];
                    ac_int<10, false> y0 =
                        loop_counters[1][params.outputYLoopIndex[1]];
                    ac_int<10, false> y1 =
                        loop_counters[0][params.outputYLoopIndex[0]];
                    ac_int<10, false> Y0 =
                        params.outputLoops[1][params.outputYLoopIndex[1]];
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

                    Pack1D<typename ACC_DTYPE::AccumulationDatatype, WIDTH>
                        uncastedOutputPixel = tensorIn.Pop();
                    Pack1D<DTYPE, WIDTH> outputPixel;

#pragma hls_unroll yes
                    for (int i = 0; i < WIDTH; i++) {
                      outputPixel[i] =
                          static_cast<DTYPE>(uncastedOutputPixel[i]);
                    }

                    if (params.MAXPOOL) {
                      // Don't support maxpool for now
//                       if (x0 % 2 == 0 && y0 % 2 == 0) {
// #pragma hls_unroll yes
//                         for (int i = 0; i < WIDTH; i++) {
//                           // update maxpool comparator
//                           maxpool_comparator[(x0 / 2)].value[i] =
//                               outputPixel.value[i];
//                         }
//                       } else if (x0 % 2 == 1 && y0 % 2 == 0) {
// #pragma hls_unroll yes
//                         for (int i = 0; i < WIDTH; i++) {
//                           // update maxpool comparator
//                           if (maxpool_comparator[(x0 - 1) / 2].value[i] <
//                               outputPixel.value[i]) {
//                             maxpool_comparator[(x0 - 1) / 2].value[i] =
//                                 outputPixel.value[i];
//                           }
//                         }
//                       } else if (x0 % 2 == 0 && y0 % 2 == 1) {
// #pragma hls_unroll yes
//                         for (int i = 0; i < WIDTH; i++) {
//                           // update maxpool comparator
//                           if (maxpool_comparator[(x0) / 2].value[i] <
//                               outputPixel.value[i]) {
//                             maxpool_comparator[(x0) / 2].value[i] =
//                                 outputPixel.value[i];
//                           }
//                         }
//                       } else {
// #pragma hls_unroll yes
//                         for (int i = 0; i < WIDTH; i++) {
//                           if (maxpool_comparator[(x0 - 1) / 2].value[i] <
//                               outputPixel.value[i]) {
//                             maxpool_comparator[(x0 - 1) / 2].value[i] =
//                                 outputPixel.value[i];
//                           }
//                         }
//                         tensorOut.Push(maxpool_comparator[(x0 - 1) / 2]);
//                       }
                    } else {
                      tensorOut.Push(outputPixel);
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

      doneSignal.SyncPush();
    }
  }
};
