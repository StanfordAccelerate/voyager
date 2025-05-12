#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"

template <typename InputTypeTuple, typename WeightTypeTuple, typename Input,
          typename Weight, typename Psum, typename Output, typename Scale,
          int PortWidth, int W, int BS>
struct MatrixVectorUnit;

template <typename... InputTypes, typename... WeightTypes, typename Input,
          typename Weight, typename Psum, typename Output, typename Scale,
          int PortWidth, int W, int BS>
struct MatrixVectorUnit<std::tuple<InputTypes...>, std::tuple<WeightTypes...>,
                        Input, Weight, Psum, Output, Scale, PortWidth, W, BS>
    : public sc_module {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixParamsDeserializer<1> CCS_INIT_S1(params_deserializer);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_inputs_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_inputs_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_weights_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_weights_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_weight_scales_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(run_accumulation_param);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(input_resp);
  Connections::Combinational<Input> CCS_INIT_S1(decoded_input);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(weight_resp);
  Connections::Combinational<Pack1D<Weight, W>> CCS_INIT_S1(decoded_weights);

  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(bias_resp);
  Connections::Combinational<Pack1D<Output, W>> CCS_INIT_S1(decoded_biases);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_scale_req);
  Connections::In<ac_int<8, false>> CCS_INIT_S1(input_scale_resp);
  Connections::Combinational<Scale> CCS_INIT_S1(decoded_input_scales);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(weight_scale_resp);
  Connections::Combinational<Pack1D<Scale, W>> CCS_INIT_S1(
      decoded_weight_scales);
#endif

  Connections::Out<Pack1D<Output, BS>> CCS_INIT_S1(matrix_out);

  Connections::SyncOut CCS_INIT_S1(start_signal);
  Connections::SyncOut CCS_INIT_S1(done_signal);

  static constexpr int LOOP_WIDTH = 16;
  using loop_t = ac_int<LOOP_WIDTH, false>;

  SC_CTOR(MatrixVectorUnit) {
    params_deserializer.clk(clk);
    params_deserializer.rstn(rstn);
    params_deserializer.serialParamsIn(serial_params_in);
    params_deserializer.paramsOut[0](params_in);

    SC_THREAD(fetch_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#if SUPPORT_MX
    SC_THREAD(process_weight_scales);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
    SC_THREAD(run_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetch_inputs() {
    fetch_inputs_param.ResetRead();
    input_req.Reset();
#if SUPPORT_MX
    input_scale_req.Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = fetch_inputs_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> k_bound = (K2 * K1 + W - 1) / W;
      ac_int<32, false> c_bound = (C2 * C1 + BS - 1) / BS;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k = 0; k < k_bound; k++) {
        ac_int<32, false> address = 0;
        for (int c = 0; c < c_bound; c++) {
          (fetch_matrix_input<InputTypes, BS, InputTypes...>(
               params.input_dtype, params.INPUT_OFFSET, address, input_req),
           ...);

#if SUPPORT_MX
          if (params.is_mx_op) {
            send_input_request<Scale, 1>(params.INPUT_SCALE_OFFSET,
                                         address / BS, input_scale_req);
          }
#endif
          address += BS;
        }
      }
      std::cerr << "fetch inputs done" << std::endl;
    }
  }

  void process_inputs() {
    process_inputs_param.ResetRead();
    input_resp.Reset();
    decoded_input.ResetWrite();
#if SUPPORT_MX
    input_scale_resp.Reset();
    decoded_input_scales.ResetWrite();
#endif

    wait();

    while (true) {
      const MatrixParams params = process_inputs_param.Pop();

      // Fetch BS number of inputs at a time
      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> C = C2 * C1;
      ac_int<32, false> k_bound = (K2 * K1 + W - 1) / W;
      ac_int<32, false> c_bound = (C + BS - 1) / BS;

      constexpr int buffer_width = Input::width * BS;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k = 0; k < k_bound; k++) {
        ac_int<32, false> address = 0;
        for (int c = 0; c < c_bound; c++) {
          ac_int<buffer_width, false> bits = 0;
          bool success = (process_matrix_input<InputTypes, BS, PortWidth,
                                               buffer_width, InputTypes...>(
                              params.input_dtype, input_resp, bits) ||
                          ...);

          for (int i = 0; i < BS; i++) {
            if (address++ < C) {
              auto data = bits.template slc<Input::width>(i * Input::width);

              Input input;
#if SUPPORT_CODEBOOK_QUANT
              if (params.use_input_codebook) {
                auto value = params.input_code[data];
                input.set_bits(value);
              } else
#endif
              {
                bool success = (decode_type<InputTypes, Input, Input::width,
                                            InputTypes...>(params.input_dtype,
                                                           data, input) ||
                                ...);
#ifndef __SYNTHESIS__
                if (!success) {
                  std::cerr << "Error: matrix input dtype '"
                            << params.input_dtype << "' is not valid"
                            << std::endl;
                }
#endif
              }

              decoded_input.Push(input);
            }
          }

#if SUPPORT_MX
          if (params.is_mx_op) {
            ac_int<Scale::width, false> data;
            process_matrix_input<Scale, 1, 8, Scale::width>(input_scale_resp,
                                                            data);

            Scale scale;
            scale.set_bits(data);
            decoded_input_scales.Push(scale);
          }
#endif
        }
      }
      std::cerr << "process inputs done" << std::endl;
    }
  }

  void fetch_weights() {
    fetch_weights_param.ResetRead();
    weight_req.Reset();
#if SUPPORT_MX
    weight_scale_req.Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = fetch_weights_param.Pop();

      int K2 = params.loops[0][params.weightLoopIndex[0]];
      int K1 = params.loops[1][params.weightLoopIndex[1]];
      int C2 = params.loops[0][params.reductionLoopIndex[0]];
      int C1 = params.loops[1][params.reductionLoopIndex[1]];
      int K = K2 * K1;
      int k_bound = (K + W - 1) / W;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k = 0; k < k_bound; k++) {
        ac_int<32, false> offset = k * W;

        if (params.has_bias) {
          send_input_request<Output, W>(params.BIAS_OFFSET, offset, bias_req);
        }

        for (int c2 = 0; c2 < C2; c2++) {
          for (int c1 = 0; c1 < C1; c1++) {
            ac_int<32, false> address = (c2 * C1 + c1) * K + offset;
            (fetch_matrix_input<WeightTypes, W, WeightTypes...>(
                 params.weight_dtype, params.WEIGHT_OFFSET, address,
                 weight_req),
             ...);

            // TODO: if MX is applied on non-reduction dimension, C1 will be 1.
            // In this case, fetch weight scales like fetching input scales
            // every cycle.
          }

#if SUPPORT_MX
          if (params.is_mx_op) {
            ac_int<32, false> address = c2 * K + offset;
            send_input_request<Scale, W>(params.WEIGHT_SCALE_OFFSET, address,
                                         weight_scale_req);
          }
#endif
        }
      }
      std::cerr << "fetch weights done" << std::endl;
    }
  }

  void process_weights() {
    process_weights_param.ResetRead();
    weight_resp.Reset();
    decoded_weights.ResetWrite();
    decoded_biases.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weights_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> C = C2 * C1;
      ac_int<32, false> k_bound = (K2 * K1 + W - 1) / W;

      constexpr int buffer_width = Weight::width * W;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k = 0; k < k_bound; k++) {
        if (params.has_bias) {
          constexpr int bias_buf_width = Output::width * W;
          ac_int<bias_buf_width, false> bits = 0;
          process_matrix_input<Output, W, PortWidth, bias_buf_width>(bias_resp,
                                                                     bits);

          Pack1D<Output, W> biases;
#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            auto data = bits.template slc<Output::width>(i * Output::width);
            biases[i].set_bits(data);
          }

          decoded_biases.Push(biases);
        }

        for (int c = 0; c < C; c++) {
          ac_int<buffer_width, false> bits = 0;
          bool success = (process_matrix_input<WeightTypes, W, PortWidth,
                                               buffer_width, WeightTypes...>(
                              params.weight_dtype, weight_resp, bits) ||
                          ...);

          Pack1D<Weight, W> weights;

#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            auto data = bits.template slc<Weight::width>(i * Weight::width);

#if SUPPORT_CODEBOOK_QUANT
            if (params.use_weight_codebook) {
              auto value = params.weight_code[data];
              weights[i].set_bits(value);
            } else
#endif
            {
              bool success = (decode_type<WeightTypes, Weight, Weight::width,
                                          WeightTypes...>(params.weight_dtype,
                                                          data, weights[i]) ||
                              ...);
#ifndef __SYNTHESIS__
              if (!success) {
                std::cerr << "Error: matrix weight dtype '"
                          << params.weight_dtype << "' is not valid"
                          << std::endl;
              }
#endif
            }
          }

          decoded_weights.Push(weights);
        }
      }
      std::cerr << "process weights done" << std::endl;
    }
  }
#if SUPPORT_MX
  void process_weight_scales() {
    process_weight_scales_param.ResetRead();
    weight_scale_resp.Reset();
    decoded_weight_scales.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weight_scales_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> loop_bound = (K2 * K1 + W - 1) / W * C2;

      if (params.is_mx_op) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (int k = 0; k < loop_bound; k++) {
          ac_int<Scale::width * W, false> data;
          process_matrix_input<Scale, W, PortWidth, Scale::width * W>(
              weight_scale_resp, data);

          Pack1D<Scale, W> scales;
#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            scales[i].set_bits(
                data.template slc<Scale::width>(i * Scale::width));
          }

          decoded_weight_scales.Push(scales);
        }
      }
    }
  }
#endif
  void run_accumulation() {
    run_accumulation_param.ResetRead();
    decoded_input.ResetRead();
    decoded_weights.ResetRead();
#if SUPPORT_MX
    decoded_input_scales.ResetRead();
    decoded_weight_scales.ResetRead();
#endif
    matrix_out.Reset();
    start_signal.Reset();
    done_signal.Reset();

    wait();

    while (true) {
      MatrixParams params = run_accumulation_param.Pop();

      std::cerr << "MatrixVectorUnit params: " << std::endl
                << params << std::endl;

      start_signal.SyncPush();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> K = K2 * K1;

      constexpr int num_blocks = W / BS;
      static_assert(W % BS == 0 && num_blocks > 0, "W must be divisible by BS");

      loop_t k_init = (K + W - 1) / W - 1;
      loop_t c2_init = C2 - 1;
      loop_t c1_init = C1 - 1;

      loop_t k_bound = k_init > 0 ? loop_t(k_init - 1) : loop_t(0);
      loop_t c2_bound = c2_init > 0 ? loop_t(c2_init - 1) : loop_t(0);
      loop_t c1_bound = c1_init > 0 ? loop_t(c1_init - 1) : loop_t(0);

      loop_t k_old = k_init;
      loop_t address = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    K_LOOP:
      for (int k = 0; k < 65536; k++) {
        // Set the outputs to biases if they exist
        Pack1D<Output, W> outputs;
        if (params.has_bias) {
          auto biases = decoded_biases.Pop();
#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            outputs[i] = biases[i];
          }
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            outputs[i] = Output::zero();
          }
        }

        loop_t c2_old = c2_init;

      C2_LOOP:
        for (int c2 = 0; c2 < 65536; c2++) {
          // Initialize the partial sums
          Pack1D<Psum, W> psums;
#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            psums[i] = Psum::zero();
          }

          loop_t c1_old = c1_init;

        // accumulate for block_size number of times
        C1_LOOP:
          for (int c1 = 0; c1 < 65536; c1++) {
            Input input = decoded_input.Pop();
            Pack1D<Weight, W> weights = decoded_weights.Pop();
#pragma hls_unroll yes
            for (int k0 = 0; k0 < W; k0++) {
              psums[k0] = input.fma(weights[k0], psums[k0]);
            }

            if (c1_old == c1_bound) {
              break;
            }
            c1_old = c1;
          }

#if SUPPORT_MX
          // If MX, rescale the partial sums
          Scale input_scale = decoded_input_scales.Pop();
          Pack1D<Scale, W> weight_scales = decoded_weight_scales.Pop();
#pragma hls_unroll yes
          for (int k0 = 0; k0 < W; k0++) {
            outputs[k0] += static_cast<Output>(input_scale) *
                           static_cast<Output>(weight_scales[k0]) *
                           static_cast<Output>(psums[k0]);
          }
#else
          // If not MX, directly add the partial sums
#pragma hls_unroll yes
          for (int k0 = 0; k0 < W; k0++) {
            outputs[k0] += static_cast<Output>(psums[k0]);
          }
#endif

          if (c2_old == c2_bound) {
            break;
          }
          c2_old = c2;
        }

        ac_int<10, false> start = 0;

      OUTPUT_LOOP:
        for (int i = 0; i < num_blocks; i++) {
          Pack1D<Output, BS> output_block;
#pragma hls_unroll yes
          for (int j = 0; j < BS; j++) {
            output_block[j] = outputs[start + j];
          }
          start += BS;

          if (address < K) {
            matrix_out.Push(output_block);
          }
          address += BS;
        }

        if (k_old == k_bound) {
          break;
        }
        k_old = k;
      }

      done_signal.SyncPush();
    }
  }

  void send_params() {
    params_in.ResetRead();
    fetch_inputs_param.ResetWrite();
    process_inputs_param.ResetWrite();
    fetch_weights_param.ResetWrite();
    process_weights_param.ResetWrite();
    process_weight_scales_param.ResetWrite();
    run_accumulation_param.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();
      fetch_inputs_param.Push(params);
      process_inputs_param.Push(params);
      fetch_weights_param.Push(params);
      process_weights_param.Push(params);
#if SUPPORT_MX
      process_weight_scales_param.Push(params);
#endif
      run_accumulation_param.Push(params);
    }
  }
};
