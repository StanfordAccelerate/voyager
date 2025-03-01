#pragma once

template <typename Input, typename Vector, typename Scale, int Width>
SC_MODULE(VectorFetchUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(paramsIn);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);

  Connections::In<Pack1D<Input, Width>> CCS_INIT_S1(vectorFetch0DataResponse);
  Connections::Out<Pack1D<Vector, Width>> CCS_INIT_S1(
      vectorFetch0DataResponseConverted);

  Connections::In<Pack1D<Input, Width>> CCS_INIT_S1(vectorFetch1DataResponse);
  Connections::Out<Pack1D<Vector, Width>> CCS_INIT_S1(
      vectorFetch1DataResponseConverted);

  Connections::In<Pack1D<Input, Width>> CCS_INIT_S1(vectorFetch2DataResponse);
  Connections::Out<Pack1D<Vector, Width>> CCS_INIT_S1(
      vectorFetch2DataResponseConverted);

  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen0Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen1Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(addressGen2Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(dataResponse0Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(dataResponse2Params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(dataResponse1Params);

  static constexpr int BUFSIZE = Width < 32 ? Width : 32;

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

      ac_int<11, false> Y1 =
          params.addressGen0Loop[0][params.addressGen0InputYLoopIndex[0]];
      ac_int<11, false> X1 =
          params.addressGen0Loop[0][params.addressGen0InputXLoopIndex[0]];
      ac_int<11, false> K1 =
          params.addressGen0Loop[0][params.addressGen0WeightLoopIndex[0]];
      ac_int<11, false> Y0 =
          params.addressGen0Loop[1][params.addressGen0InputYLoopIndex[1]];
      ac_int<11, false> X0 =
          params.addressGen0Loop[1][params.addressGen0InputXLoopIndex[1]];
      ac_int<11, false> K0 =
          params.addressGen0Loop[1][params.addressGen0WeightLoopIndex[1]];

      if (params.has_transpose && BUFSIZE != Width) {
        X1 = X1 * BUFSIZE / Width;
      }

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addressGen0Loop[i][j];
          loop_steps[i][j] = 1;
        }
      }

      if (params.has_slicing) {
        int slice_dim = params.vec0_dim;
        int i = slice_dim >= 3 ? 1 : 0;
        int j = slice_dim >= 3 ? slice_dim - 3 : slice_dim;
        loop_starts[i][j] = params.vec0_start;
        loop_ends[i][j] = params.vec0_end;
        loop_steps[i][j] = params.vec0_step;
      } else if (params.has_permute) {
#pragma hls_unroll yes
        for (int dim = 0; dim < 6; dim++) {
          int i = dim >= 3 ? 1 : 0;
          int j = dim >= 3 ? dim - 3 : dim;
          int new_dim = params.vec0_dim_order[dim];
          int new_i = new_dim >= 3 ? 1 : 0;
          int new_j = new_dim >= 3 ? new_dim - 3 : new_dim;
          loop_ends[i][j] = params.addressGen0Loop[new_i][new_j];
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  ac_int<32, false> address;
                  if (params.addressGen0Mode == 1) {
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

                    ac_int<16, false> k = k1 * K0 * Width + k0 * Width;
                    ac_int<16, false> K = K1 * K0 * Width;

                    ac_int<16, false> x = x1 * X0 + x0;
                    ac_int<16, false> X = X1 * X0;

                    ac_int<16, false> y = y1 * Y0 + y0;
                    ac_int<16, false> Y = Y1 * Y0;

                    if (params.has_transpose) {
                      address = y * K * X + (k + x0) * X + x1 * BUFSIZE;
                    } else {
                      address = y * X * K + x * K + k;
                    }
                  } else if (params.addressGen0Mode == 2) {
                    ac_int<11, false> indices[6] = {
                        loop_counters[0][0], loop_counters[0][1],
                        loop_counters[0][2], loop_counters[1][0],
                        loop_counters[1][1], loop_counters[1][2]};

                    ac_int<11, false> loop_bounds[6] = {
                        params.addressGen0Loop[0][0],
                        params.addressGen0Loop[0][1],
                        params.addressGen0Loop[0][2],
                        params.addressGen0Loop[1][0],
                        params.addressGen0Loop[1][1],
                        params.addressGen0Loop[1][2]};

#pragma hls_unroll yes
                    for (int i = 0; i < 6; i++) {
                      if (params.vec0_broadcast[i]) {
                        indices[i] = 0;
                        loop_bounds[i] = 1;
                      }
                    }

                    if (params.has_permute) {
                      ac_int<11, false> permuted_indices[6];
#pragma hls_unroll yes
                      for (int i = 0; i < 6; i++) {
                        permuted_indices[params.vec0_dim_order[i]] = indices[i];
                      }

#pragma hls_unroll yes
                      for (int i = 0; i < 6; i++) {
                        indices[i] = permuted_indices[i];
                      }
                    }

                    address =
                        (indices[0] * loop_bounds[1] * loop_bounds[2] *
                             loop_bounds[3] * loop_bounds[4] * loop_bounds[5] +
                         indices[1] * loop_bounds[2] * loop_bounds[3] *
                             loop_bounds[4] * loop_bounds[5] +
                         indices[2] * loop_bounds[3] * loop_bounds[4] *
                             loop_bounds[5] +
                         indices[3] * loop_bounds[4] * loop_bounds[5] +
                         indices[4] * loop_bounds[5] + indices[5]) *
                        Width;
                  }

                  MemoryRequest request;
                  if (params.fetch_vector_type_0) {
                    address = address * Vector::width / 8;
                    request = {params.VECTOR_OFFSET + address,
                               Width * Vector::width / 8};
                  } else {
                    address = address * Input::width / 8;
                    request = {params.VECTOR_OFFSET + address,
                               Width * Input::width / 8};
                  }

                  vectorFetch0AddressRequest.Push(request);

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
        }
      }
    }
  }

  void feed_data_response_0() {
    vectorFetch0DataResponse.Reset();
    vectorFetch0DataResponseConverted.Reset();
    dataResponse0Params.ResetRead();

    wait();

    while (true) {
      VectorParams params = dataResponse0Params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addressGen0Loop[i][j];
          loop_steps[i][j] = 1;
        }
      }

      if (params.has_slicing) {
        int slice_dim = params.vec0_dim;
        int i = slice_dim >= 3 ? 1 : 0;
        int j = slice_dim >= 3 ? slice_dim - 3 : slice_dim;
        loop_starts[i][j] = params.vec0_start;
        loop_ends[i][j] = params.vec0_end;
        loop_steps[i][j] = params.vec0_step;
      }

      if (params.has_transpose) {
        Vector buffer[BUFSIZE][Width];

        assert(loop_ends[1][2] == Width);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_ends[0][0];
             loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_ends[0][1];
               loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_ends[0][2];
                 loop_counters[0][2]++) {
              for (loop_counters[1][0] = 0;
                   loop_counters[1][0] < loop_ends[1][0];
                   loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;
                     loop_counters[1][1] < loop_ends[1][1];
                     loop_counters[1][1]++) {
                  for (int col = 0; col < Width; col++) {
                    Pack1D<Vector, Width> converted_response;
                    if (params.fetch_vector_type_0) {
                      constexpr int num_words = Vector::width / Input::width;

                      Pack1D<Input, Width> response[num_words];

                      for (int i = 0; i < num_words; i++) {
                        response[i] = vectorFetch0DataResponse.Pop();
                      }
                      convertPack1D<Input, Vector, Width>(response,
                                                          converted_response);
                    } else {
                      Pack1D<Input, Width> response =
                          vectorFetch0DataResponse.Pop();
                      vdequantize<Input, Vector, Width>(
                          response, converted_response, params.vec0_dq_scale);
                    }

                    // We may not use all the data in the response
#pragma hls_unroll yes
                    for (int row = 0; row < BUFSIZE; row++) {
                      buffer[row][col] = converted_response[row];
                    }
                  }

                  // Write out from transpose buffer
                  for (int row = 0; row < BUFSIZE; row++) {
                    Pack1D<Vector, Width> transposed;
#pragma hls_unroll yes
                    for (int col = 0; col < Width; col++) {
                      transposed[col] = buffer[row][col];
                    }

                    vectorFetch0DataResponseConverted.Push(transposed);
                  }
                }
              }
            }
          }
        }
      } else {
        // passthrough
        ac_int<32, false> num_writes = loop_ends[0][0] * loop_ends[0][1] *
                                       loop_ends[0][2] * loop_ends[1][0] *
                                       loop_ends[1][1] * loop_ends[1][2];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int i = 0; i < num_writes; i++) {
          Pack1D<Vector, Width> converted_response;
          if (params.fetch_vector_type_0) {
            constexpr int num_words = Vector::width / Input::width;

            Pack1D<Input, Width> response[num_words];

            for (int i = 0; i < num_words; i++) {
              response[i] = vectorFetch0DataResponse.Pop();
            }
            convertPack1D<Input, Vector, Width>(response, converted_response);
          } else {
            Pack1D<Input, Width> response = vectorFetch0DataResponse.Pop();
            vdequantize<Input, Vector, Width>(response, converted_response,
                                              params.vec0_dq_scale);
          }

          vectorFetch0DataResponseConverted.Push(converted_response);
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

      ac_int<11, false> Y1 =
          params.addressGen1Loops[0][params.addressGen1InputYLoopIndex[0]];
      ac_int<11, false> X1 =
          params.addressGen1Loops[0][params.addressGen1InputXLoopIndex[0]];
      ac_int<11, false> K1 =
          params.addressGen1Loops[0][params.addressGen1WeightLoopIndex[0]];
      ac_int<11, false> Y0 =
          params.addressGen1Loops[1][params.addressGen1InputYLoopIndex[1]];
      ac_int<11, false> X0 =
          params.addressGen1Loops[1][params.addressGen1InputXLoopIndex[1]];
      ac_int<11, false> K0 =
          params.addressGen1Loops[1][params.addressGen1WeightLoopIndex[1]];

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addressGen1Loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
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

                  ac_int<16, false> k = k1 * K0 * Width + k0 * Width;
                  ac_int<16, false> K = K1 * K0 * Width;

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

                  MemoryRequest request;
                  if (params.fetch_vector_type_1) {
                    address = address * Vector::width / 8;
                    request = {params.ADDRESS_GEN1_OFFSET + address,
                               Width * Vector::width / 8};
                  } else {
                    address = address * Input::width / 8;
                    request = {params.ADDRESS_GEN1_OFFSET + address,
                               Width * Input::width / 8};
                  }

                  vectorFetch1AddressRequest.Push(request);

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
        }
      }
    }
  }

  void feed_data_response_1() {
    dataResponse1Params.ResetRead();
    vectorFetch1DataResponse.Reset();
    vectorFetch1DataResponseConverted.Reset();

    wait();

    while (true) {
      VectorParams params = dataResponse1Params.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addressGen1Loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  Pack1D<Vector, Width> converted_response;

                  if (params.fetch_vector_type_1) {
                    constexpr int num_words = Vector::width / Input::width;

                    Pack1D<Input, Width> response[num_words];
                    for (int i = 0; i < num_words; i++) {
                      response[i] = vectorFetch1DataResponse.Pop();
                    }

                    convertPack1D<Input, Vector, Width>(response,
                                                        converted_response);

                    vectorFetch1DataResponseConverted.Push(converted_response);
                  } else {
                    Pack1D<Input, Width> response =
                        vectorFetch1DataResponse.Pop();

                    vdequantize<Input, Vector, Width>(
                        response, converted_response, params.vec1_dq_scale);

                    vectorFetch1DataResponseConverted.Push(converted_response);
                  }

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
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

      ac_int<11, false> Y1 =
          params.addressGen2Loops[0][params.addressGen2InputYLoopIndex[0]];
      ac_int<11, false> X1 =
          params.addressGen2Loops[0][params.addressGen2InputXLoopIndex[0]];
      ac_int<11, false> K1 =
          params.addressGen2Loops[0][params.addressGen2WeightLoopIndex[0]];
      ac_int<11, false> Y0 =
          params.addressGen2Loops[1][params.addressGen2InputYLoopIndex[1]];
      ac_int<11, false> X0 =
          params.addressGen2Loops[1][params.addressGen2InputXLoopIndex[1]];
      ac_int<11, false> K0 =
          params.addressGen2Loops[1][params.addressGen2WeightLoopIndex[1]];

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addressGen2Loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
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

                  ac_int<16, false> k = k1 * K0 * Width + k0 * Width;
                  ac_int<16, false> K = K1 * K0 * Width;

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

                  MemoryRequest request;
                  if (params.fetch_vector_type_2) {
                    address = address * Vector::width / 8;
                    request = {params.ADDRESS_GEN2_OFFSET + address,
                               Width * Vector::width / 8};
                  } else {
                    address = address * Input::width / 8;
                    request = {params.ADDRESS_GEN2_OFFSET + address,
                               Width * Input::width / 8};
                  }

                  vectorFetch2AddressRequest.Push(request);

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
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
      ac_int<11, false> loop_starts[2][3];
      ac_int<11, false> loop_ends[2][3];
      ac_int<11, false> loop_steps[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_starts[i][j] = 0;
          loop_ends[i][j] = params.addressGen2Loops[i][j];
          loop_steps[i][j] = 1;
        }
      }

#pragma hls_pipeline_init_interval 1
      for (loop_counters[0][0] = loop_starts[0][0];
           loop_counters[0][0] < loop_ends[0][0];
           loop_counters[0][0] += loop_steps[0][0]) {
        for (loop_counters[0][1] = loop_starts[0][1];
             loop_counters[0][1] < loop_ends[0][1];
             loop_counters[0][1] += loop_steps[0][1]) {
          for (loop_counters[0][2] = loop_starts[0][2];
               loop_counters[0][2] < loop_ends[0][2];
               loop_counters[0][2] += loop_steps[0][2]) {
            for (loop_counters[1][0] = loop_starts[1][0];
                 loop_counters[1][0] < loop_ends[1][0];
                 loop_counters[1][0] += loop_steps[1][0]) {
              for (loop_counters[1][1] = loop_starts[1][1];
                   loop_counters[1][1] < loop_ends[1][1];
                   loop_counters[1][1] += loop_steps[1][1]) {
                for (loop_counters[1][2] = loop_starts[1][2];
                     loop_counters[1][2] < loop_ends[1][2];
                     loop_counters[1][2] += loop_steps[1][2]) {
                  Pack1D<Vector, Width> converted_response;

                  if (params.fetch_vector_type_2) {
                    constexpr int num_words = Vector::width / Input::width;
                    Pack1D<Input, Width> response[num_words];

                    for (int i = 0; i < num_words; i++) {
                      response[i] = vectorFetch2DataResponse.Pop();
                    }
                    convertPack1D<Input, Vector, Width>(response,
                                                        converted_response);
                  } else {
                    Pack1D<Input, Width> response =
                        vectorFetch2DataResponse.Pop();

                    vdequantize<Input, Vector, Width>(
                        response, converted_response, params.vec2_dq_scale);
                  }
                  vectorFetch2DataResponseConverted.Push(converted_response);

                  if (loop_counters[1][2] >=
                      loop_ends[1][2] - loop_steps[1][2]) {
                    break;
                  }
                }
                if (loop_counters[1][1] >= loop_ends[1][1] - loop_steps[1][1]) {
                  break;
                }
              }
              if (loop_counters[1][0] >= loop_ends[1][0] - loop_steps[1][0]) {
                break;
              }
            }
            if (loop_counters[0][2] >= loop_ends[0][2] - loop_steps[0][2]) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_ends[0][1] - loop_steps[0][1]) {
            break;
          }
        }
        if (loop_counters[0][0] >= loop_ends[0][0] - loop_steps[0][0]) {
          break;
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
