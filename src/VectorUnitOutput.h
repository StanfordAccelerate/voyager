#pragma once

template <typename VectorType, typename ScaleType, typename IOType, int Width>
SC_MODULE(VectorUnitOutput) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(tensor_in);
  Connections::Out<Pack1D<IOType, Width>> CCS_INIT_S1(vector_output);
  Connections::Out<ac_int<64, false>> CCS_INIT_S1(vector_output_address);
  Connections::Out<Pack1D<INT8_, 1>> CCS_INIT_S1(scalar_output);
  Connections::Out<ac_int<64, false>> CCS_INIT_S1(scalar_output_address);

  Connections::SyncOut CCS_INIT_S1(done);

  SC_CTOR(VectorUnitOutput) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    params_in.Reset();
    tensor_in.Reset();
    vector_output.Reset();
    vector_output_address.Reset();
    scalar_output.Reset();
    scalar_output_address.Reset();
    done.Reset();

    wait();

    while (true) {
      VectorParams params = params_in.Pop();

      ac_int<11, false> loop_counters[2][3];
      ac_int<11, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.outputLoops[i][j];
        }
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
                  ac_int<32, false> address;
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

                    ac_int<16, false> k = k2 * K1 * Width + k1 * Width;
                    ac_int<16, false> K = K2 * K1 * Width;

                    ac_int<16, false> x = x0 + x1 * X0;
                    ac_int<16, false> X = X0 * X1;

                    ac_int<16, false> y = y0 + y1 * Y0;
                    ac_int<16, false> Y = Y0 * Y1;

                    ac_int<8, false> head_size = params.head_size_power_of_two;
                    ac_int<16, false> mask = (1 << head_size) - 1;

                    if (params.has_attn_head_permute) {
                      address = (((k >> head_size) * X) << head_size) +
                                (x << head_size) + (k & mask);
                    } else if (params.CONCAT_OUTPUT) {
                      address = ((k >> head_size) * K) + ((y & mask) * K * 4) +
                                (k & mask) + (y >> head_size * K / 4);
                    } else {
                      address = y * X * K + x * K + k;
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

                    address =
                        (loop_0 * loop_bound_1 * loop_bound_2 * loop_bound_3 *
                             loop_bound_4 * loop_bound_5 +
                         loop_1 * loop_bound_2 * loop_bound_3 * loop_bound_4 *
                             loop_bound_5 +
                         loop_2 * loop_bound_3 * loop_bound_4 * loop_bound_5 +
                         loop_3 * loop_bound_4 * loop_bound_5 +
                         loop_4 * loop_bound_5 + loop_5) *
                        Width;
                  }

                  Pack1D<VectorType, Width> outputs = tensor_in.Pop();

                  if (params.output_vector_type) {
                    constexpr int num_words = VectorType::width / IOType::width;
                    Pack1D<IOType, Width> converted_outputs[num_words];

                    convertPack1D<IOType, VectorType, Width>(outputs,
                                                             converted_outputs);

                    for (int i = 0; i < num_words; i++) {
                      vector_output.Push(converted_outputs[i]);
                      vector_output_address.Push(params.VECTOR_OUTPUT_OFFSET +
                                                 address * VectorType::width /
                                                     8 +
                                                 i * Width * IOType::width / 8);
                    }
                  } else {
                    Pack1D<IOType, Width> converted_outputs;
                    if (params.quantize_output) {
                      VectorType scale;
                      scale.set_bits(params.output_scale);
                      vquantize<VectorType, IOType, VectorType, Width>(
                          outputs, converted_outputs, scale);
#if SUPPORT_MX
                    } else if (params.quantize_output_mx) {
                      Pack1D<ScaleType, 1> scale;
                      vquantize_mx<VectorType, IOType, ScaleType, Width>(
                          outputs, converted_outputs, scale[0]);

                      constexpr int num_words = ScaleType::width / INT8_::width;
                      Pack1D<INT8_, 1> converted_scale[num_words];

                      convertPack1D<INT8_, ScaleType, 1>(scale,
                                                         converted_scale);

                      ac_int<64, false> scale_address = address / Width;

                      for (int i = 0; i < num_words; i++) {
                        scalar_output.Push(converted_scale[i]);
                        scalar_output_address.Push(params.SCALE_OFFSET +
                                                   scale_address *
                                                       ScaleType::width / 8 +
                                                   i * INT8_::width / 8);
                      }
#endif
                    } else {
#pragma hls_unroll yes
                      for (int i = 0; i < Width; i++) {
                        converted_outputs[i] = outputs[i];
                      }
                    }

                    vector_output.Push(converted_outputs);
                    vector_output_address.Push(params.VECTOR_OUTPUT_OFFSET +
                                               address * IOType::width / 8);
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

      done.SyncPush();
    }
  }
};
