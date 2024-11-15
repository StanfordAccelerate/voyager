#pragma once

template <typename IO_DTYPE, typename VEC_DTYPE, typename MX_DTYPE, int WIDTH>
SC_MODULE(VectorFetchUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);

  Connections::In<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch0DataResponse);
  Connections::Out<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch0DataResponseBroadcasted);

  Connections::In<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch1DataResponse);
  Connections::Out<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch1DataResponseConverted);

  Connections::In<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch2DataResponse);
  Connections::Out<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch2DataResponseConverted);

  Connections::Out<MX_DTYPE> CCS_INIT_S1(vectorFetch1Scale);

  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen0Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(dataResponse0Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen1Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen2Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(dataResponse2Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(dataResponse1Params);

  SC_CTOR(VectorFetchUnit) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_address_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_response_0);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_address_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_response_1);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_address_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(feed_data_response_2);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetch_address_0() {
    addressGen0Params.ResetRead();
    vectorFetch0AddressRequest.Reset();

    wait();

    while (true) {
      VectorParams params = addressGen0Params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.addressGen0Loop[i][j];
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
                  ac_int<11, false> x0 =
                      loop_counters[1][params.addressGen0InputXLoopIndex[1]];
                  ac_int<11, false> x1 =
                      loop_counters[0][params.addressGen0InputXLoopIndex[0]];
                  ac_int<11, false> y0 =
                      loop_counters[1][params.addressGen0InputYLoopIndex[1]];
                  ac_int<11, false> y1 =
                      loop_counters[0][params.addressGen0InputYLoopIndex[0]];
                  ac_int<11, false> k0 =
                      loop_counters[1][params.addressGen0WeightLoopIndex[1]];
                  ac_int<11, false> k1 =
                      loop_counters[0][params.addressGen0WeightLoopIndex[0]];

                  ac_int<11, false> X0 =
                      loop_bounds[1][params.addressGen0InputXLoopIndex[1]];
                  ac_int<11, false> X1 =
                      loop_bounds[0][params.addressGen0InputXLoopIndex[0]];
                  ac_int<11, false> Y0 =
                      loop_bounds[1][params.addressGen0InputYLoopIndex[1]];
                  ac_int<11, false> Y1 =
                      loop_bounds[0][params.addressGen0InputYLoopIndex[0]];
                  ac_int<11, false> K0 =
                      loop_bounds[1][params.addressGen0WeightLoopIndex[1]];
                  ac_int<11, false> K1 =
                      loop_bounds[0][params.addressGen0WeightLoopIndex[0]];

                  ac_int<16, false> k = k1 * K0 * WIDTH + k0 * WIDTH;
                  ac_int<16, false> K = K1 * K0 * WIDTH;

                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> X = X1 * X0;

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> Y = Y1 * Y0;

                  ac_int<32, false> address;
                  if (params.addressGen0Mode == 1) {
                    address = y * X * K + x * K + k;
                  } else if (params.addressGen0Mode == 2) {
                    address = x * K + k;
                  } else if (params.addressGen0Mode == 3) {
                    address = k;
                  }

                  if (params.DP_VEC0) {
                    MemoryRequest memRequest = {
                        params.VECTOR_OFFSET + address * (VEC_DTYPE::width / 8),
                        WIDTH * (VEC_DTYPE::width / 8)};
                    vectorFetch0AddressRequest.Push(memRequest);
                  } else {
                    MemoryRequest memRequest = {
                        params.VECTOR_OFFSET + address * (IO_DTYPE::width / 8),
                        WIDTH * (IO_DTYPE::width / 8)};
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

  void feed_data_response_0() {
    vectorFetch0DataResponse.Reset();
    vectorFetch0DataResponseBroadcasted.Reset();
    dataResponse0Params.ResetRead();

    wait();

    while (true) {
      VectorParams params = dataResponse0Params.Pop();

#pragma hls_pipeline_init_interval 1
      for (ac_int<11, false> i0 = 0; i0 < params.addressGen0Loop[0][0]; i0++) {
        for (ac_int<11, false> j0 = 0; j0 < params.addressGen0Loop[0][1];
             j0++) {
          for (ac_int<11, false> k0 = 0; k0 < params.addressGen0Loop[0][2];
               k0++) {
            for (ac_int<11, false> i1 = 0; i1 < params.addressGen0Loop[1][0];
                 i1++) {
              for (ac_int<11, false> j1 = 0; j1 < params.addressGen0Loop[1][1];
                   j1++) {
                for (ac_int<11, false> k1 = 0;
                     k1 < params.addressGen0Loop[1][2]; k1++) {
                  Pack1D<VEC_DTYPE, WIDTH> fullPrecisionDataResponse;
                  if (params.DP_VEC0) {
                    // combine multiple IO_DTYPE into one VEC_DTYPE
                    constexpr int num_words =
                        VEC_DTYPE::width / IO_DTYPE::width;

                    Pack1D<IO_DTYPE, WIDTH> response[num_words];

                    for (int i = 0; i < num_words; i++) {
                      response[i] = vectorFetch0DataResponse.Pop();
                    }
                    convertPack1D<IO_DTYPE, VEC_DTYPE, WIDTH>(
                        response, fullPrecisionDataResponse);
                  } else {
                    Pack1D<IO_DTYPE, WIDTH> response =
                        vectorFetch0DataResponse.Pop();

                    if constexpr (VEC_DTYPE::is_floating_point &&
                                  IO_DTYPE::is_floating_point) {
                      // static cast to VEC_DTYPE

                    UNROLL:
                      for (int dim = 0; dim < WIDTH; dim++) {
                        fullPrecisionDataResponse[dim] =
                            static_cast<VEC_DTYPE>(response[dim]);
                      }
                    } else {
                      // dequantize IO_DTYPE to VEC_DTYPE
                      vdequantize<VEC_DTYPE, IO_DTYPE, WIDTH>(
                          response, fullPrecisionDataResponse,
                          params.vec0DequantizeScale);
                    }
                  }
                  vectorFetch0DataResponseBroadcasted.Push(
                      fullPrecisionDataResponse);
                }
              }
            }
          }
        }
      }
    }
  }

  void fetch_address_1() {
    addressGen1Params.ResetRead();
    vectorFetch1AddressRequest.Reset();

    wait();

    while (true) {
      VectorParams params = addressGen1Params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
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
                  ac_int<11, false> x0 =
                      loop_counters[1][params.addressGen1InputXLoopIndex[1]];
                  ac_int<11, false> x1 =
                      loop_counters[0][params.addressGen1InputXLoopIndex[0]];
                  ac_int<11, false> y0 =
                      loop_counters[1][params.addressGen1InputYLoopIndex[1]];
                  ac_int<11, false> y1 =
                      loop_counters[0][params.addressGen1InputYLoopIndex[0]];
                  ac_int<11, false> k0 =
                      loop_counters[1][params.addressGen1WeightLoopIndex[1]];
                  ac_int<11, false> k1 =
                      loop_counters[0][params.addressGen1WeightLoopIndex[0]];

                  ac_int<11, false> X0 =
                      loop_bounds[1][params.addressGen1InputXLoopIndex[1]];
                  ac_int<11, false> X1 =
                      loop_bounds[0][params.addressGen1InputXLoopIndex[0]];
                  ac_int<11, false> Y0 =
                      loop_bounds[1][params.addressGen1InputYLoopIndex[1]];
                  ac_int<11, false> Y1 =
                      loop_bounds[0][params.addressGen1InputYLoopIndex[0]];
                  ac_int<11, false> K0 =
                      loop_bounds[1][params.addressGen1WeightLoopIndex[1]];
                  ac_int<11, false> K1 =
                      loop_bounds[0][params.addressGen1WeightLoopIndex[0]];

                  ac_int<16, false> k = k1 * K0 * WIDTH + k0 * WIDTH;
                  ac_int<16, false> K = K1 * K0 * WIDTH;

                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> X = X1 * X0;

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> Y = Y1 * Y0;

                  ac_int<32, false> address;
                  if (params.addressGen1Mode == 1) {
                    address = y * X * K + x * K + k;
                  } else if (params.addressGen1Mode == 2) {
                    address = x * K + k;
                  } else if (params.addressGen1Mode == 3) {
                    address = k;
                  }

                  if (params.DP_VEC1) {
                    MemoryRequest memRequest = {
                        params.ADDRESS_GEN1_OFFSET +
                            address * (VEC_DTYPE::width / 8),
                        WIDTH * (VEC_DTYPE::width / 8)};
                    vectorFetch1AddressRequest.Push(memRequest);
                  } else {
                    MemoryRequest memRequest = {
                        params.ADDRESS_GEN1_OFFSET +
                            address * (IO_DTYPE::width / 8),
                        WIDTH * (IO_DTYPE::width / 8)};
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

  void feed_data_response_1() {
    dataResponse1Params.ResetRead();
    vectorFetch1DataResponse.Reset();
    vectorFetch1DataResponseConverted.Reset();
    vectorFetch1Scale.Reset();

    wait();

    while (true) {
      VectorParams params = dataResponse1Params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
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
                  Pack1D<VEC_DTYPE, WIDTH> fullPrecisionDataResponse;
                  if (params.DP_VEC1) {
                    // combine multiple IO_DTYPE into one VEC_DTYPE
                    constexpr int num_words =
                        VEC_DTYPE::width / IO_DTYPE::width;

                    Pack1D<IO_DTYPE, WIDTH> response[num_words];

                    for (int i = 0; i < num_words; i++) {
                      response[i] = vectorFetch1DataResponse.Pop();
                    }
                    convertPack1D<IO_DTYPE, VEC_DTYPE, WIDTH>(
                        response, fullPrecisionDataResponse);
                  } else {
                    Pack1D<IO_DTYPE, WIDTH> response =
                        vectorFetch1DataResponse.Pop();

                    if constexpr (VEC_DTYPE::is_floating_point &&
                                  IO_DTYPE::is_floating_point) {
                      // static cast to VEC_DTYPE

                    UNROLL:
                      for (int dim = 0; dim < WIDTH; dim++) {
                        fullPrecisionDataResponse[dim] =
                            static_cast<VEC_DTYPE>(response[dim]);
                      }
                    } else if constexpr (!std::is_same<IO_DTYPE,
                                                       MX_DTYPE>::value) {
#pragma hls_unroll yes
                      for (int i = 0; i < WIDTH; i++) {
                        fullPrecisionDataResponse[i].setbits(
                            response[i].bits_rep());
                      }
                    } else {
                      // dequantize IO_DTYPE to VEC_DTYPE
                      vdequantize<VEC_DTYPE, IO_DTYPE, WIDTH>(
                          response, fullPrecisionDataResponse,
                          params.vec1DequantizeScale);
                    }
                  }

                  if (params.BROADCAST_VEC1_SCALE) {
                    for (int i = 0; i < WIDTH; i++) {
                      for (int count = 0; count < params.vec1BroadcastCount;
                           count++) {
                        vectorFetch1Scale.Push(
                            fullPrecisionDataResponse[i].bits_rep());
                      }
                    }
                  } else {
                    vectorFetch1DataResponseConverted.Push(
                        fullPrecisionDataResponse);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void fetch_address_2() {
    addressGen2Params.ResetRead();
    vectorFetch2AddressRequest.Reset();

    wait();

    while (true) {
      VectorParams params = addressGen2Params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.addressGen2Loops[i][j];
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
                  ac_int<11, false> x0 =
                      loop_counters[1][params.addressGen2InputXLoopIndex[1]];
                  ac_int<11, false> x1 =
                      loop_counters[0][params.addressGen2InputXLoopIndex[0]];
                  ac_int<11, false> y0 =
                      loop_counters[1][params.addressGen2InputYLoopIndex[1]];
                  ac_int<11, false> y1 =
                      loop_counters[0][params.addressGen2InputYLoopIndex[0]];
                  ac_int<11, false> k0 =
                      loop_counters[1][params.addressGen2WeightLoopIndex[1]];
                  ac_int<11, false> k1 =
                      loop_counters[0][params.addressGen2WeightLoopIndex[0]];

                  ac_int<11, false> X0 =
                      loop_bounds[1][params.addressGen2InputXLoopIndex[1]];
                  ac_int<11, false> X1 =
                      loop_bounds[0][params.addressGen2InputXLoopIndex[0]];
                  ac_int<11, false> Y0 =
                      loop_bounds[1][params.addressGen2InputYLoopIndex[1]];
                  ac_int<11, false> Y1 =
                      loop_bounds[0][params.addressGen2InputYLoopIndex[0]];
                  ac_int<11, false> K0 =
                      loop_bounds[1][params.addressGen2WeightLoopIndex[1]];
                  ac_int<11, false> K1 =
                      loop_bounds[0][params.addressGen2WeightLoopIndex[0]];

                  ac_int<16, false> k = k1 * K0 * WIDTH + k0 * WIDTH;
                  ac_int<16, false> K = K1 * K0 * WIDTH;

                  ac_int<16, false> x = x1 * X0 + x0;
                  ac_int<16, false> X = X1 * X0;

                  ac_int<16, false> y = y1 * Y0 + y0;
                  ac_int<16, false> Y = Y1 * Y0;

                  ac_int<32, false> address;
                  if (params.addressGen2Mode == 1) {
                    address = y * X * K + x * K + k;
                  } else if (params.addressGen2Mode == 2) {
                    address = x * K + k;
                  } else if (params.addressGen2Mode == 3) {
                    address = k;
                  }

                  if (params.DP_VEC2) {
                    MemoryRequest memRequest = {
                        params.ADDRESS_GEN2_OFFSET +
                            address * (VEC_DTYPE::width / 8),
                        WIDTH * (VEC_DTYPE::width / 8)};
                    vectorFetch2AddressRequest.Push(memRequest);
                  } else {
                    MemoryRequest memRequest = {
                        params.ADDRESS_GEN2_OFFSET +
                            address * (IO_DTYPE::width / 8),
                        WIDTH * (IO_DTYPE::width / 8)};
                    vectorFetch2AddressRequest.Push(memRequest);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void feed_data_response_2() {
    dataResponse2Params.ResetRead();
    vectorFetch2DataResponse.Reset();
    vectorFetch2DataResponseConverted.Reset();

    wait();

    while (true) {
      VectorParams params = dataResponse2Params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.addressGen2Loops[i][j];
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
                  Pack1D<VEC_DTYPE, WIDTH> fullPrecisionDataResponse;
                  if (params.DP_VEC2) {
                    constexpr int num_words =
                        VEC_DTYPE::width / IO_DTYPE::width;
                    Pack1D<IO_DTYPE, WIDTH> response[num_words];

                    for (int i = 0; i < num_words; i++) {
                      response[i] = vectorFetch2DataResponse.Pop();
                    }
                    convertPack1D<IO_DTYPE, VEC_DTYPE, WIDTH>(
                        response, fullPrecisionDataResponse);
                  } else {
                    Pack1D<IO_DTYPE, WIDTH> response =
                        vectorFetch2DataResponse.Pop();

                    if constexpr (VEC_DTYPE::is_floating_point &&
                                  IO_DTYPE::is_floating_point) {
                      // static cast to VEC_DTYPE

                    UNROLL:
                      for (int dim = 0; dim < WIDTH; dim++) {
                        fullPrecisionDataResponse[dim] =
                            static_cast<VEC_DTYPE>(response[dim]);
                      }
                    } else {
                      // dequantize IO_DTYPE to VEC_DTYPE
                      vdequantize<VEC_DTYPE, IO_DTYPE, WIDTH>(
                          response, fullPrecisionDataResponse,
                          params.vec2DequantizeScale);
                    }
                  }
                  vectorFetch2DataResponseConverted.Push(
                      fullPrecisionDataResponse);
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

    addressGen0Params.ResetWrite();
    addressGen1Params.ResetWrite();
    addressGen2Params.ResetWrite();

    dataResponse0Params.ResetWrite();
    dataResponse1Params.ResetWrite();
    dataResponse2Params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = paramsIn.Pop();

      if (params.addressGen0Mode != 0) {
        addressGen0Params.Push(params);
        dataResponse0Params.Push(params);
      }

      if (params.addressGen1Mode != 0) {
        addressGen1Params.Push(params);
        dataResponse1Params.Push(params);
      }

      if (params.addressGen2Mode != 0) {
        addressGen2Params.Push(params);
        dataResponse2Params.Push(params);
      }
    }
  }
};
