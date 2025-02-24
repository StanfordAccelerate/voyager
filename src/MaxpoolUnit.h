#pragma once

/*
 * Performs bias, residual, maxpool and avgpool operations
 */
template <typename VEC_DTYPE, typename IO_DTYPE, typename Scale, int WIDTH>
SC_MODULE(MaxpoolUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::In<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(tensorIn);
  Connections::Out<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(tensorOut);

  Connections::In<Scale> CCS_INIT_S1(mxScaleIn);

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
    mxScaleIn.Reset();

    wait();

    while (true) {
      VectorParams params = paramsIn.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

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
                      params.outputLoops[1][params.outputXLoopIndex[1]];
                  ac_int<11, false> X1 =
                      params.outputLoops[0][params.outputXLoopIndex[0]];
                  ac_int<11, false> Y0 =
                      params.outputLoops[1][params.outputYLoopIndex[1]];
                  ac_int<11, false> Y1 =
                      params.outputLoops[0][params.outputYLoopIndex[0]];
                  ac_int<11, false> K1 =
                      params.outputLoops[1][params.outputWeightLoopIndex[1]];
                  ac_int<11, false> K2 =
                      params.outputLoops[0][params.outputWeightLoopIndex[0]];

                  ac_int<16, false> k = k2 * K1 * WIDTH + k1 * WIDTH;
                  ac_int<16, false> K = K2 * K1 * WIDTH;

                  ac_int<16, false> x = x0 + x1 * X0;
                  ac_int<16, false> X = X0 * X1;

                  ac_int<16, false> y = y0 + y1 * Y0;
                  ac_int<16, false> Y = Y0 * Y1;

                  Pack1D<VEC_DTYPE, WIDTH> uncastedOutputPixel = tensorIn.Pop();

                  if (params.DP_OUTPUT) {
                    constexpr int num_words =
                        VEC_DTYPE::width / IO_DTYPE::width;
                    Pack1D<IO_DTYPE, WIDTH> outputPixel[num_words];

                    convertPack1D<IO_DTYPE, VEC_DTYPE, WIDTH>(
                        uncastedOutputPixel, outputPixel);

                    for (int word = 0; word < num_words; word++) {
                      tensorOut.Push(outputPixel[word]);
                    }
                  } else {
                    Pack1D<IO_DTYPE, WIDTH> outputPixel;

                    if (params.OUTPUT_QUANTIZE_MX) {
                      Scale scale = mxScaleIn.Pop();
                      vquantize<VEC_DTYPE, IO_DTYPE, Scale, WIDTH>(
                          uncastedOutputPixel, outputPixel, scale);
                    } else if (params.OUTPUT_QUANTIZE) {
                      VEC_DTYPE scale;
                      scale.set_bits(params.outputQuantizeScale);
                      vquantize<VEC_DTYPE, IO_DTYPE, VEC_DTYPE, WIDTH>(
                          uncastedOutputPixel, outputPixel, scale);
                    } else {
#pragma hls_unroll yes
                      for (int i = 0; i < WIDTH; i++) {
                        outputPixel[i].set_bits(
                            uncastedOutputPixel[i].bits_rep());
                      }
                    }

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

      doneSignal.SyncPush();
    }
  }
};
