#pragma once

template <int WIDTH>
SC_MODULE(VectorFetchUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);

  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen0Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen1Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen2Params);

  SC_CTOR(VectorFetchUnit) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_vector);
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
    addressGen0Params.ResetRead();
    vectorFetch0AddressRequest.Reset();

    wait();

    while (true) {
      VectorParams params = addressGen0Params.Pop();

#pragma hls_pipeline_init_interval 1
      for (int i = 0; i < params.addressGen0Loop[2]; i++) {
        for (int j = 0; j < params.addressGen0Loop[1]; j++) {
          for (int k = 0; k < params.addressGen0Loop[0]; k++) {
            int address = k * params.addressGen0Loop[0] * WIDTH + k * WIDTH;
            address = params.VECTOR_OFFSET + address;
            MemoryRequest memRequest = {address, WIDTH};
            vectorFetch0AddressRequest.Push(memRequest);
          }
        }
      }
    }
  }

  void fetch_residual() {
    addressGen1Params.ResetRead();
    vectorFetch1AddressRequest.Reset();

    wait();

    while (true) {
      VectorParams params = addressGen1Params.Pop();

      int loop_counters[2][3];
      int loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.addressGen1Loops[i][j];
        }
      }

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
                  int x0 =
                      loop_counters[1][params.addressGen1InputXLoopIndex[1]];
                  int x1 =
                      loop_counters[0][params.addressGen1InputXLoopIndex[0]];
                  int X0 = params.addressGen1Loops
                               [1][params.addressGen1InputXLoopIndex[1]];
                  int X1 = params.addressGen1Loops
                               [0][params.addressGen1InputXLoopIndex[0]];
                  int y0 =
                      loop_counters[1][params.addressGen1InputYLoopIndex[1]];
                  int y1 =
                      loop_counters[0][params.addressGen1InputYLoopIndex[0]];
                  int Y0 = params.addressGen1Loops
                               [1][params.addressGen1InputYLoopIndex[1]];
                  int Y1 = params.addressGen1Loops
                               [0][params.addressGen1InputYLoopIndex[0]];
                  int k2 =
                      loop_counters[0][params.addressGen1WeightLoopIndex[0]];
                  int K2 = params.addressGen1Loops
                               [0][params.addressGen1WeightLoopIndex[0]];
                  int k1 =
                      loop_counters[1][params.addressGen1WeightLoopIndex[1]];
                  int K1 = params.addressGen1Loops
                               [1][params.addressGen1WeightLoopIndex[1]];
                  int k = k2 * K1 * WIDTH + k1 * WIDTH;
                  int K = K2 * K1 * WIDTH;

                  int x = x0 + x1 * X0;
                  int X = X0 * X1;

                  int y = y0 + y1 * Y0;
                  int Y = Y0 * Y1;

                  int address = y * X * K + x * K + k;
                  MemoryRequest memRequest = {
                      params.ADDRESS_GEN1_OFFSET + address, WIDTH};
                  vectorFetch2AddressRequest.Push(memRequest);
                }
              }
            }
          }
        }
      }
    }
  }

  void fetch_bias() {
    addressGen2Params.ResetRead();
    vectorFetch2AddressRequest.Reset();

    wait();

    while (true) {
      VectorParams params = addressGen2Params.Pop();

      if (params.addressGen2Mode == 1) {
        int loop_counters[2][3];
        int loop_bounds[2][3];

#pragma hls_unroll yes
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 3; j++) {
            loop_bounds[i][j] = params.addressGen2Loops[i][j];
          }
        }

#pragma hls_pipeline_init_interval 1
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
              for (int k1 = 0;
                   k1 <
                   params
                       .addressGen2Loops[1]
                                        [params.addressGen2WeightLoopIndex[1]];
                   k1++) {
                int k2 = loop_counters[0][params.addressGen2WeightLoopIndex[0]];
                int K2 =
                    params
                        .addressGen2Loops[0]
                                         [params.addressGen2WeightLoopIndex[0]];

                int K1 =
                    params
                        .addressGen2Loops[1]
                                         [params.addressGen2WeightLoopIndex[1]];
                int k = k2 * K1 * WIDTH + k1 * WIDTH;
                int K = K2 * K1 * WIDTH;
                int address = params.ADDRESS_GEN2_OFFSET + k;
                MemoryRequest memRequest = {address, WIDTH};
                vectorFetch1AddressRequest.Push(memRequest);
              }
            }
          }
        }
      } else {
      }
    }
  }

  //   void replicate_bias() {
  //     wait();

  //     while (true) {
  //       if (params.addressGen2Mode == 1) {
  //         int loop_counters[2][6];
  //         int loop_bounds[2][6];

  // #pragma hls_unroll yes
  //         for (int i = 0; i < 2; i++) {
  //           for (int j = 0; j < 6; j++) {
  //             loop_bounds[i][j] = params.loops[i][j];
  //           }
  //         }

  //         // set irrelevant loop bounds to 1
  //         loop_bounds[1][params.reductionLoopIndex[1]] = 1;
  //         loop_bounds[1][params.fxIndex] = 1;
  //         loop_bounds[1][params.fyIndex] = 1;
  //         loop_bounds[1][params.inputXLoopIndex[1]] = 1;
  //         loop_bounds[1][params.inputYLoopIndex[1]] = 1;
  // #pragma hls_pipeline_init_interval 1
  //         for (loop_counters[0][0] = 0; loop_counters[0][0] <
  //         loop_bounds[0][0];
  //              loop_counters[0][0]++) {
  //           for (loop_counters[0][1] = 0; loop_counters[0][1] <
  //           loop_bounds[0][1];
  //                loop_counters[0][1]++) {
  //             for (loop_counters[0][2] = 0;
  //                  loop_counters[0][2] < loop_bounds[0][2];
  //                  loop_counters[0][2]++) {
  //               for (int k1 = 0; k1 <
  //               params.loops[1][params.weightLoopIndex[1]];
  //                    k1++) {
  //                 int k2 = loop_counters[0][params.weightLoopIndex[0]];
  //                 int K2 = params.loops[0][params.weightLoopIndex[0]];

  //                 int K1 = params.loops[1][params.weightLoopIndex[1]];
  //                 int k = k2 * K1 * WIDTH + k1 * WIDTH;
  //                 int K = K2 * K1 * WIDTH;
  //                 int address = params.BIAS_OFFSET + k;
  //                 MemoryRequest memRequest = {address, WIDTH};
  //                 vectorFetch2AddressRequest.Push(memRequest);
  //               }
  //             }
  //           }
  //         }
  //       } else {
  //         // passthrough
  //       }
  //     }
  //   }

  void read_params() {
    paramsIn.Reset();

    addressGen0Params.ResetWrite();
    addressGen1Params.ResetWrite();
    addressGen2Params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = paramsIn.Pop();

      if (params.addressGen0Enable) {
        addressGen0Params.Push(params);
      }

      if (params.addressGen1Mode != 0) {  // residual
        addressGen1Params.Push(params);
      }

      if (params.addressGen2Mode != 0) {  // bias
        addressGen2Params.Push(params);
      }
    }
  }
};
