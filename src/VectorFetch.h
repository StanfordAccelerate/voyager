#pragma once

template <typename ODTYPE, typename ACC_DTYPE, int WIDTH>
SC_MODULE(VectorFetchUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);

  Connections::In<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(vectorFetch0DataResponse);
  Connections::Out<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch0DataResponseBroadcasted);

  Connections::In<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(vectorFetch2DataResponse);
  Connections::Out<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch2DataResponseReplicated);

  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen0Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vector0BroadcastParams);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen1Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen2Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(replicateBiasParams);

  SC_CTOR(VectorFetchUnit) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_vector);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(vector0_broadcast);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_bias);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(replicate_bias);
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
      for (ac_int<8, false> i0 = 0; i0 < params.addressGen0Loop[0][0]; i0++) {
        for (ac_int<8, false> j0 = 0; j0 < params.addressGen0Loop[0][1]; j0++) {
          for (ac_int<8, false> k0 = 0; k0 < params.addressGen0Loop[0][2];
               k0++) {
            for (ac_int<8, false> i1 = 0; i1 < params.addressGen0Loop[1][0];
                 i1++) {
              for (ac_int<8, false> j1 = 0; j1 < params.addressGen0Loop[1][1];
                   j1++) {
                for (ac_int<8, false> k1 = 0; k1 < params.addressGen0Loop[1][2];
                     k1++) {
                  ac_int<16, false> j = j0 * params.addressGen0Loop[1][1] + j1;
                  ac_int<16, false> k =
                      k0 * params.addressGen0Loop[1][2] * WIDTH + k1 * WIDTH;
                  ac_int<16, false> K = params.addressGen0Loop[0][2] *
                                        params.addressGen0Loop[1][2] * WIDTH;

                  if (params.DP_VEC0) {
                    K = K * 2;
                    for (int precision = 0; precision < 2; precision++) {
                      int address =
                          static_cast<ac_int<32, false> >((j * K + k) * 2) +
                          precision * WIDTH;
                      // DLOG("addressgen0 " << j << " " << k << " " <<
                      // address);
                      MemoryRequest memRequest = {
                          params.VECTOR_OFFSET + address, WIDTH};
                      vectorFetch0AddressRequest.Push(memRequest);
                    }

                  } else {
                    int address = j * K + k;

                    // DLOG("addressgen0 " << j << " " << k << " " << address);
                    MemoryRequest memRequest = {params.VECTOR_OFFSET + address,
                                                WIDTH};
                    vectorFetch0AddressRequest.Push(memRequest);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void vector0_broadcast() {
    vectorFetch0DataResponse.Reset();
    vectorFetch0DataResponseBroadcasted.Reset();
    vector0BroadcastParams.ResetRead();

    wait();

    while (true) {
      VectorParams params = vector0BroadcastParams.Pop();

      if (params.addressGen0Broadcast) {
        ac_int<16, false> broadcastCount = params.addressGen0BroadcastCount;

#pragma hls_pipeline_init_interval 1
        for (ac_int<8, false> i0 = 0; i0 < params.addressGen0Loop[0][0]; i0++) {
          for (ac_int<8, false> j0 = 0; j0 < params.addressGen0Loop[0][1];
               j0++) {
            for (ac_int<8, false> k0 = 0; k0 < params.addressGen0Loop[0][2];
                 k0++) {
              for (ac_int<8, false> i1 = 0; i1 < params.addressGen0Loop[1][0];
                   i1++) {
                for (ac_int<8, false> j1 = 0; j1 < params.addressGen0Loop[1][1];
                     j1++) {
                  for (ac_int<8, false> k1 = 0;
                       k1 < params.addressGen0Loop[1][2]; k1++) {
                    ac_int<16, false> K = params.addressGen0Loop[0][2] *
                                          params.addressGen0Loop[1][2] * WIDTH;

                    Pack1D<ODTYPE, WIDTH> data = vectorFetch0DataResponse.Pop();
                    for (int dim = 0; dim < WIDTH; dim++) {
                      ac_int<16, false> K_unpacked = K + dim;

                      // negate
                      ACC_DTYPE singleVal = data[dim];

                      ACC_DTYPE negated = singleVal;
                      ACC_DTYPE negatedPlusOne = singleVal;

                      if (params.SOFTMAX_GRAD_NEGATE) {
                        typename ACC_DTYPE::DecomposedPosit one;
                        one.sign = 0;
                        one.fraction = 0;
                        one.scale = 0;
                        one._zero = false;

                        singleVal.negate();
                        negatedPlusOne =
                            static_cast<typename ACC_DTYPE::DecomposedPosit>(
                                singleVal) +
                            one;
                      }

                      for (int broadcast = 0;
                           broadcast < broadcastCount / WIDTH; broadcast++) {
                        Pack1D<ACC_DTYPE, WIDTH> broadcastVec;

#pragma hls_unroll yes
                        for (int broadcastDim = 0; broadcastDim < WIDTH;
                             broadcastDim++) {
                          if (broadcast * WIDTH + broadcastDim == K_unpacked) {
                            broadcastVec[broadcastDim] = negatedPlusOne;
                          } else {
                            broadcastVec[broadcastDim] = negated;
                          }
                        }

                        vectorFetch0DataResponseBroadcasted.Push(broadcastVec);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {  // passthrough
#pragma hls_pipeline_init_interval 1
        for (ac_int<8, false> i0 = 0; i0 < params.addressGen0Loop[0][0]; i0++) {
          for (ac_int<8, false> j0 = 0; j0 < params.addressGen0Loop[0][1];
               j0++) {
            for (ac_int<8, false> k0 = 0; k0 < params.addressGen0Loop[0][2];
                 k0++) {
              for (ac_int<8, false> i1 = 0; i1 < params.addressGen0Loop[1][0];
                   i1++) {
                for (ac_int<8, false> j1 = 0; j1 < params.addressGen0Loop[1][1];
                     j1++) {
                  for (ac_int<8, false> k1 = 0;
                       k1 < params.addressGen0Loop[1][2]; k1++) {
                    if (params.DP_VEC0) {
                      // convert 2 8b bias into 1 16b bias
                      Pack1D<ACC_DTYPE, WIDTH> fullPrecisionVec;

                      for (int precision = 0; precision < 2; precision++) {
                        Pack1D<ODTYPE, WIDTH> bias =
                            vectorFetch0DataResponse.Pop();

#pragma hls_unroll yes
                        for (int i = 0; i < WIDTH / 2; i++) {
                          ac_int<16, false> temp;
#pragma hls_unroll yes
                          for (int byte = 0; byte < 2; byte++) {
                            temp.set_slc(byte * 8, bias[i * 2 + byte].bits);
                          }
                          fullPrecisionVec[precision * (WIDTH / 2) + i].setbits(
                              temp);
                        }
                      }

                      vectorFetch0DataResponseBroadcasted.Push(
                          fullPrecisionVec);
                    } else {
                      // cast up to 16b
                      Pack1D<ODTYPE, WIDTH> originalVec =
                          vectorFetch0DataResponse.Pop();
                      Pack1D<ACC_DTYPE, WIDTH> castedVec;
#pragma hls_unroll yes
                      for (int dim = 0; dim < WIDTH; dim++) {
                        castedVec[dim] =
                            static_cast<ACC_DTYPE>(originalVec[dim]);
                      }

                      vectorFetch0DataResponseBroadcasted.Push(castedVec);
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

  void fetch_residual() {
    addressGen1Params.ResetRead();
    vectorFetch1AddressRequest.Reset();

    wait();

    while (true) {
      VectorParams params = addressGen1Params.Pop();

      if (params.addressGen1Mode == 1) {
        ac_int<8, false> loop_counters[2][3];
        ac_int<8, false> loop_bounds[2][3];

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
                    ac_int<8, false> x0 =
                        loop_counters[1][params.addressGen1InputXLoopIndex[1]];
                    ac_int<8, false> x1 =
                        loop_counters[0][params.addressGen1InputXLoopIndex[0]];
                    ac_int<8, false> X0 =
                        params.addressGen1Loops
                            [1][params.addressGen1InputXLoopIndex[1]];
                    ac_int<8, false> X1 =
                        params.addressGen1Loops
                            [0][params.addressGen1InputXLoopIndex[0]];
                    ac_int<8, false> y0 =
                        loop_counters[1][params.addressGen1InputYLoopIndex[1]];
                    ac_int<8, false> y1 =
                        loop_counters[0][params.addressGen1InputYLoopIndex[0]];
                    ac_int<8, false> Y0 =
                        params.addressGen1Loops
                            [1][params.addressGen1InputYLoopIndex[1]];
                    ac_int<8, false> Y1 =
                        params.addressGen1Loops
                            [0][params.addressGen1InputYLoopIndex[0]];
                    ac_int<8, false> k2 =
                        loop_counters[0][params.addressGen1WeightLoopIndex[0]];
                    ac_int<8, false> K2 =
                        params.addressGen1Loops
                            [0][params.addressGen1WeightLoopIndex[0]];
                    ac_int<8, false> k1 =
                        loop_counters[1][params.addressGen1WeightLoopIndex[1]];
                    ac_int<8, false> K1 =
                        params.addressGen1Loops
                            [1][params.addressGen1WeightLoopIndex[1]];
                    ac_int<16, false> k = k2 * K1 * WIDTH + k1 * WIDTH;
                    ac_int<16, false> K = K2 * K1 * WIDTH;

                    ac_int<16, false> x = x0 + x1 * X0;
                    ac_int<16, false> X = X0 * X1;

                    ac_int<16, false> y = y0 + y1 * Y0;
                    ac_int<16, false> Y = Y0 * Y1;

                    int address = y * X * K + x * K + k;
                    MemoryRequest memRequest = {
                        params.ADDRESS_GEN1_OFFSET + address, WIDTH};
                    vectorFetch1AddressRequest.Push(memRequest);
                  }
                }
              }
            }
          }
        }
      } else {  // 2d tensor
#pragma hls_pipeline_init_interval 1
        for (ac_int<8, false> i0 = 0; i0 < params.addressGen1Loops[0][0];
             i0++) {
          for (ac_int<8, false> j0 = 0; j0 < params.addressGen1Loops[0][1];
               j0++) {
            for (ac_int<8, false> k0 = 0; k0 < params.addressGen1Loops[0][2];
                 k0++) {
              for (ac_int<8, false> i1 = 0; i1 < params.addressGen1Loops[1][0];
                   i1++) {
                for (ac_int<8, false> j1 = 0;
                     j1 < params.addressGen1Loops[1][1]; j1++) {
                  for (ac_int<8, false> k1 = 0;
                       k1 < params.addressGen1Loops[1][2]; k1++) {
                    ac_int<16, false> j =
                        j0 * params.addressGen1Loops[1][1] + j1;
                    ac_int<16, false> k =
                        k0 * params.addressGen1Loops[1][2] * WIDTH + k1 * WIDTH;
                    ac_int<16, false> K = params.addressGen1Loops[0][2] *
                                          params.addressGen1Loops[1][2] * WIDTH;

                    int address = j * K + k;

                    DLOG("addressgen1 " << j << " " << k << " " << address);
                    MemoryRequest memRequest = {
                        params.ADDRESS_GEN1_OFFSET + address, WIDTH};
                    vectorFetch1AddressRequest.Push(memRequest);
                  }
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
        ac_int<8, false> loop_counters[2][3];
        ac_int<8, false> loop_bounds[2][3];

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
              for (ac_int<8, false> k1 = 0;
                   k1 <
                   params
                       .addressGen2Loops[1]
                                        [params.addressGen2WeightLoopIndex[1]];
                   k1++) {
                ac_int<8, false> k2 =
                    loop_counters[0][params.addressGen2WeightLoopIndex[0]];
                ac_int<8, false> K2 =
                    params
                        .addressGen2Loops[0]
                                         [params.addressGen2WeightLoopIndex[0]];

                ac_int<8, false> K1 =
                    params
                        .addressGen2Loops[1]
                                         [params.addressGen2WeightLoopIndex[1]];
                // double precision bias
                ac_int<16, false> k = k2 * K1 * WIDTH * 2 + k1 * WIDTH * 2;
                ac_int<16, false> K = K2 * K1 * WIDTH * 2;
                int address = params.ADDRESS_GEN2_OFFSET + k;
                MemoryRequest memRequest = {address, WIDTH * 2};
                vectorFetch2AddressRequest.Push(memRequest);
              }
            }
          }
        }
      } else {  // 2d tensor
        DLOG("2d tensor for bias");
#pragma hls_pipeline_init_interval 1
        for (ac_int<8, false> i = 0; i < params.addressGen2Loops[0][0]; i++) {
          for (ac_int<8, false> j = 0; j < params.addressGen2Loops[0][1]; j++) {
            for (ac_int<8, false> k = 0; k < params.addressGen2Loops[0][2];
                 k++) {
              int address = static_cast<ac_int<32, false> >(
                                j * params.addressGen2Loops[0][2] * WIDTH) +
                            static_cast<ac_int<32, false> >(k * WIDTH);
              DLOG("addressgen2 " << j << " " << k << " " << address);
              MemoryRequest memRequest = {params.ADDRESS_GEN2_OFFSET + address,
                                          WIDTH};
              vectorFetch2AddressRequest.Push(memRequest);
            }
          }
        }
      }
    }
  }

  void replicate_bias() {
    replicateBiasParams.ResetRead();
    vectorFetch2DataResponse.Reset();
    vectorFetch2DataResponseReplicated.Reset();

    wait();

    while (true) {
      VectorParams params = replicateBiasParams.Pop();

      if (params.addressGen2Mode == 1) {
        ac_int<8, false> loop_counters[2][3];
        ac_int<8, false> loop_bounds[2][3];

        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 3; j++) {
            loop_bounds[i][j] = params.addressGen2Loops[i][j];
          }
        }

        ac_int<16, false> replicationCount =
            params.addressGen2Loops[1][params.addressGen2InputXLoopIndex[1]] *
            params.addressGen2Loops[1][params.addressGen2InputYLoopIndex[1]];
        DLOG("bias repl count " << replicationCount);

#pragma hls_pipeline_init_interval 1
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;
                 loop_counters[0][2] < loop_bounds[0][2];
                 loop_counters[0][2]++) {
              for (ac_int<8, false> k1 = 0;
                   k1 <
                   params
                       .addressGen2Loops[1]
                                        [params.addressGen2WeightLoopIndex[1]];
                   k1++) {
                // convert 2 8b bias into 1 16b bias
                Pack1D<ACC_DTYPE, WIDTH> fullPrecisionBias;

                for (int pack = 0; pack < 2; pack++) {
                  Pack1D<ODTYPE, WIDTH> bias = vectorFetch2DataResponse.Pop();

#pragma hls_unroll yes
                  for (int i = 0; i < WIDTH / 2; i++) {
                    ac_int<16, false> temp;
#pragma hls_unroll yes
                    for (int byte = 0; byte < 2; byte++) {
                      temp.set_slc(byte * 8, bias[i * 2 + byte].bits);
                    }
                    fullPrecisionBias[pack * (WIDTH / 2) + i].setbits(temp);
                  }
                }

                for (int repl = 0; repl < replicationCount; repl++) {
                  vectorFetch2DataResponseReplicated.Push(fullPrecisionBias);
                }
              }
            }
          }
        }
      } else {  // pasthrough for a standard 2d tensor
        for (ac_int<8, false> i = 0; i < params.addressGen2Loops[0][0]; i++) {
          for (ac_int<8, false> j = 0; j < params.addressGen2Loops[0][1]; j++) {
            for (ac_int<8, false> k = 0; k < params.addressGen2Loops[0][2];
                 k++) {
              // cast up to 16b
              Pack1D<ODTYPE, WIDTH> originalVec =
                  vectorFetch2DataResponse.Pop();
              Pack1D<ACC_DTYPE, WIDTH> castedVec;
#pragma hls_unroll yes
              for (int dim = 0; dim < WIDTH; dim++) {
                castedVec[dim] = static_cast<ACC_DTYPE>(originalVec[dim]);
              }

              vectorFetch2DataResponseReplicated.Push(castedVec);
            }
          }
        }
      }
    }
  }

  void read_params() {
    paramsIn.Reset();

    addressGen0Params.ResetWrite();
    addressGen1Params.ResetWrite();
    addressGen2Params.ResetWrite();
    replicateBiasParams.ResetWrite();
    vector0BroadcastParams.ResetWrite();

    wait();

    while (true) {
      VectorParams params = paramsIn.Pop();

      if (params.addressGen0Enable) {
        addressGen0Params.Push(params);
        vector0BroadcastParams.Push(params);
      }

      if (params.addressGen1Mode != 0) {  // residual
        addressGen1Params.Push(params);
      }

      if (params.addressGen2Mode != 0) {  // bias
        addressGen2Params.Push(params);
        replicateBiasParams.Push(params);
      }
    }
  }
};
