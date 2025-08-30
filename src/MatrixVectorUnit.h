#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"
#include "Utils.h"

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
  Connections::Combinational<MatrixParams> CCS_INIT_S1(send_outputs_param);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(input_resp);
  Connections::Combinational<Input> CCS_INIT_S1(input_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(weight_resp);
  Connections::Combinational<Pack1D<Weight, W>> CCS_INIT_S1(weight_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(bias_resp);
  Connections::Combinational<Pack1D<Output, W>> CCS_INIT_S1(bias_data);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_scale_req);
  Connections::In<ac_int<8, false>> CCS_INIT_S1(input_scale_resp);
  Connections::Combinational<Scale> CCS_INIT_S1(input_scale_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(weight_scale_resp);
  Connections::Combinational<Pack1D<Scale, W>> CCS_INIT_S1(weight_scales_data);
#endif

  Connections::Combinational<Pack1D<Output, W>> CCS_INIT_S1(accumulation_out);
  Connections::Out<Pack1D<Output, BS>> CCS_INIT_S1(matrix_out);

  Connections::SyncOut CCS_INIT_S1(start_signal);
  Connections::SyncOut CCS_INIT_S1(done_signal);

  static constexpr int LOOP_WIDTH = 16;
  using loop_t = ac_int<LOOP_WIDTH, false>;

  static constexpr int FEEDBACK_DEPTH = 4;
  static constexpr int FEEDBACK_LAST = FEEDBACK_DEPTH - 1;

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

    SC_THREAD(send_outputs);
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
      ac_int<32, false> k_bound = (K2 * K1 + W - 1) / W - 1;
      ac_int<32, false> c_bound = (C2 * C1 + BS - 1) / BS - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        ac_int<32, false> address = 0;

        for (ac_int<32, false> c = 0;; c++) {
          (fetch_matrix_input<InputTypes, BS, InputTypes...>(
               params.input_dtype, params.input_offset, address, input_req),
           ...);

#if SUPPORT_MX
          if (params.is_mx_op) {
            send_input_request<Scale, 1>(params.input_scale_offset,
                                         address / BS, input_scale_req);
          }
#endif
          address += BS;

          if (c == c_bound) {
            break;
          }
        }

        if (k == k_bound) {
          break;
        }
      }
      std::cerr << "fetch inputs done" << std::endl;
    }
  }

  void process_inputs() {
    process_inputs_param.ResetRead();
    input_resp.Reset();
    input_data.ResetWrite();
#if SUPPORT_MX
    input_scale_resp.Reset();
    input_scale_data.ResetWrite();
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
      ac_int<32, false> k_bound = (K2 * K1 + W - 1) / W - 1;
      ac_int<32, false> c_bound = (C + BS - 1) / BS - 1;

      constexpr int buffer_width = Input::width * BS;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k = 0;; k++) {
        ac_int<32, false> address = 0;
        for (int c = 0;; c++) {
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

              input_data.Push(input);
            }
          }

#if SUPPORT_MX
          if (params.is_mx_op) {
            ac_int<Scale::width, false> data;
            process_matrix_input<Scale, 1, 8, Scale::width>(input_scale_resp,
                                                            data);

            Scale scale;
            scale.set_bits(data);
            input_scale_data.Push(scale);
          }
#endif
          if (c == c_bound) {
            break;
          }
        }

        if (k == k_bound) {
          break;
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

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> K = K2 * K1;

      loop_t c1_bound = C1 - 1;
      loop_t c2_bound = C2 - 1;
      ac_int<32, false> k_bound = (K + W - 1) / W - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        ac_int<32, false> offset = k * W;

        if (params.has_bias) {
          send_input_request<Output, W>(params.bias_offset, offset, bias_req);
        }

        for (loop_t c2 = 0;; c2++) {
          for (loop_t c1 = 0;; c1++) {
            ac_int<32, false> address = (c2 * C1 + c1) * K + offset;
            (fetch_matrix_input<WeightTypes, W, WeightTypes...>(
                 params.weight_dtype, params.weight_offset, address,
                 weight_req),
             ...);

            // TODO: if MX is applied on non-reduction dimension, C1 will be 1.
            // In this case, fetch weight scales like fetching input scales
            // every cycle.
            if (c1 == c1_bound) {
              break;
            }
          }

#if SUPPORT_MX
          if (params.is_mx_op) {
            ac_int<32, false> address = c2 * K + offset;
            send_input_request<Scale, W>(params.weight_scale_offset, address,
                                         weight_scale_req);
          }
#endif

          if (c2 == c2_bound) {
            break;
          }
        }

        if (k == k_bound) {
          break;
        }
      }
      std::cerr << "fetch weights done" << std::endl;
    }
  }

  void process_weights() {
    process_weights_param.ResetRead();
    weight_resp.Reset();
    weight_data.ResetWrite();
    bias_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weights_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> c_bound = C2 * C1 - 1;
      ac_int<32, false> k_bound = (K2 * K1 + W - 1) / W - 1;

      constexpr int buffer_width = Weight::width * W;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
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

          bias_data.Push(biases);
        }

        for (ac_int<32, false> c = 0;; c++) {
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

          weight_data.Push(weights);

          if (c == c_bound) {
            break;
          }
        }

        if (k == k_bound) {
          break;
        }
      }
      std::cerr << "process weights done" << std::endl;
    }
  }
#if SUPPORT_MX
  void process_weight_scales() {
    process_weight_scales_param.ResetRead();
    weight_scale_resp.Reset();
    weight_scales_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weight_scales_param.Pop();

      if (!params.is_mx_op) {
        continue;
      }

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> loop_bound = (K2 * K1 + W - 1) / W * C2 - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        ac_int<Scale::width * W, false> data;
        process_matrix_input<Scale, W, PortWidth, Scale::width * W>(
            weight_scale_resp, data);

        Pack1D<Scale, W> scales;
#pragma hls_unroll yes
        for (int i = 0; i < W; i++) {
          scales[i].set_bits(data.template slc<Scale::width>(i * Scale::width));
        }

        weight_scales_data.Push(scales);

        if (k == loop_bound) {
          break;
        }
      }

      std::cerr << "process weight scales done" << std::endl;
    }
  }
#endif
  void run_accumulation() {
    run_accumulation_param.ResetRead();
    input_data.ResetRead();
    weight_data.ResetRead();
#if SUPPORT_MX
    input_scale_data.ResetRead();
    weight_scales_data.ResetRead();
#endif
    accumulation_out.ResetWrite();
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
      ac_int<32, false> k_bound = (K + W - 1) / W - 1;
      loop_t c2_bound = C2 - 1;
      loop_t c1_bound = C1 - 1;

      Scale input_scale;
      Pack1D<Scale, W> weight_scales;
      Pack1D<Output, W> acc_old[FEEDBACK_DEPTH];

      loop_t address = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    K_LOOP:
      for (ac_int<32, false> k = 0;; k++) {
        // Reset the outputs
#pragma hls_unroll yes
        for (int i = 0; i < FEEDBACK_DEPTH; i++) {
#pragma hls_unroll yes
          for (int j = 0; j < W; j++) {
            acc_old[i][j] = Output::zero();
          }
        }

        // Set the outputs to biases if they exist
        if (params.has_bias) {
          auto biases = bias_data.Pop();
#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            acc_old[0][i] = biases[i];
          }
        }

      C2_LOOP:
        for (loop_t c2 = 0;; c2++) {
          // Initialize the partial sums
          Pack1D<Psum, W> psums;
#pragma hls_unroll yes
          for (int i = 0; i < W; i++) {
            psums[i] = Psum::zero();
          }

        // accumulate for block_size number of times
        C1_LOOP:
          for (loop_t c1 = 0;; c1++) {
            Input input = input_data.Pop();
            Pack1D<Weight, W> weights = weight_data.Pop();
#pragma hls_unroll yes
            for (int k0 = 0; k0 < W; k0++) {
              psums[k0] = input.fma(weights[k0], psums[k0]);
            }

            if (c1 == c1_bound) {
              break;
            }
          }

#if SUPPORT_MX
          if (params.is_mx_op) {
            input_scale = input_scale_data.Pop();
            weight_scales = weight_scales_data.Pop();
          }
          auto acc = dequantize_mx<Psum, Scale, Output, W>(
              psums, input_scale, weight_scales, acc_old[FEEDBACK_LAST],
              params.is_mx_op);
#else
          Pack1D<Output, W> acc;
#pragma hls_unroll yes
          for (int k0 = 0; k0 < W; k0++) {
            acc[k0] =
                static_cast<Output>(psums[k0]) + acc_old[FEEDBACK_LAST][k0];
          }
#endif

#pragma hls_unroll yes
          for (int i = FEEDBACK_LAST; i > 0; i--) {
            acc_old[i] = acc_old[i - 1];
          }

          acc_old[0] = acc;

          if (c2 == c2_bound) {
            break;
          }
        }

        Pack1D<Output, W> outputs;
#pragma hls_unroll yes
        for (int i = 0; i < W; i++) {
          Pack1D<Output, FEEDBACK_DEPTH> col;
#pragma hls_unroll yes
          for (int j = 0; j < FEEDBACK_DEPTH; j++) {
            col[j] = acc_old[j][i];
          }
          outputs[i] = tree_sum(col);
        }

        accumulation_out.Push(outputs);

        if (k == k_bound) {
          break;
        }
      }

      std::cerr << "run accumulation done" << std::endl;
      done_signal.SyncPush();
    }
  }

  void send_outputs() {
    matrix_out.Reset();

    wait();

    while (true) {
      MatrixParams params = send_outputs_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      ac_int<32, false> K = K2 * K1;
      ac_int<32, false> k_bound = (K + W - 1) / W - 1;
      ac_int<4, false> num_blocks = W / BS - 1;

      loop_t address = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        auto outputs = accumulation_out.Pop();

        for (int i = 0;; i++) {
          Pack1D<Output, BS> output_block;
#pragma hls_unroll yes
          for (int j = 0; j < BS; j++) {
            output_block[j] = outputs[i * BS + j];
          }
          matrix_out.Push(output_block);

          address += BS;
          if (i == num_blocks || address >= K) {
            break;
          }
        }

        if (k == k_bound) {
          break;
        }
      }

      std::cerr << "send outputs done" << std::endl;
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
    send_outputs_param.ResetWrite();

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
      send_outputs_param.Push(params);
    }
  }
};
