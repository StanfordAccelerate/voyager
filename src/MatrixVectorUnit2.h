#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"
#include "Utils.h"

template <typename InputTypeTuple, typename WeightTypeTuple, typename Input,
          typename Weight, typename Psum, typename Output, typename Scale,
          int PortWidth, int Width, int BS, int VectorUnitWidth>
struct MatrixVectorUnit;

template <typename... InputTypes, typename... WeightTypes, typename Input,
          typename Weight, typename Psum, typename Output, typename Scale,
          int PortWidth, int Width, int BS, int VectorUnitWidth>
struct MatrixVectorUnit<std::tuple<InputTypes...>, std::tuple<WeightTypes...>,
                        Input, Weight, Psum, Output, Scale, PortWidth, Width,
                        BS, VectorUnitWidth> : public sc_module {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  static constexpr int FEEDBACK_DELAY = 4;
  static constexpr int LAST = FEEDBACK_DELAY - 1;

  static constexpr int LOOP_WIDTH = 16;
  using loop_t = ac_int<LOOP_WIDTH, false>;

  MatrixParamsDeserializer<1> CCS_INIT_S1(params_deserializer);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_input_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_input_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(input_scale_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_weight_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_weight_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(weight_scale_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_bias_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(run_accumulation_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(send_outputs_param);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(input_resp);
  Connections::Combinational<Pack1D<Weight, Width>> CCS_INIT_S1(input_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(weight_resp);
  Connections::Combinational<Pack1D<Weight, Width>> CCS_INIT_S1(weight_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<PortWidth, false>> CCS_INIT_S1(bias_resp);
  Connections::Combinational<Pack1D<Output, VectorUnitWidth>> CCS_INIT_S1(
      bias_data);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_scale_req);
  Connections::In<ac_int<Scale::width * Width / BS, false>> CCS_INIT_S1(
      input_scale_resp);
  Connections::Combinational<Pack1D<Scale, Width / BS>> CCS_INIT_S1(
      input_scale_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<Scale::width * Width / BS, false>> CCS_INIT_S1(
      weight_scale_resp);
  Connections::Combinational<Pack1D<Scale, Width / BS>> CCS_INIT_S1(
      weight_scale_data);
#endif

  Connections::Combinational<Output> CCS_INIT_S1(accumulation_out);
  Connections::Out<Pack1D<Output, VectorUnitWidth>> CCS_INIT_S1(matrix_out);

  Connections::SyncOut CCS_INIT_S1(start_signal);
  Connections::SyncOut CCS_INIT_S1(done_signal);

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
    SC_THREAD(process_input_scales);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_weight_scales);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
    SC_THREAD(fetch_biases);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

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
    fetch_input_param.ResetRead();
    input_req.Reset();
#if SUPPORT_MX
    input_scale_req.Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = fetch_input_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> k_bound = K2 * K1 - 1;
      ac_int<32, false> c_bound = (C2 * C1 + Width - 1) / Width - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        ac_int<32, false> address = 0;

        for (ac_int<32, false> c = 0;; c++) {
          (fetch_matrix_input<InputTypes, Width, InputTypes...>(
               params.input_dtype, params.input_offset, address, input_req),
           ...);

#if SUPPORT_MX
          if (params.is_mx_op) {
            send_input_request<Scale, Width / BS>(
                params.input_scale_offset, address / BS, input_scale_req);
          }
#endif
          address += Width;

          if (c == c_bound) {
            break;
          }
        }
        if (k == k_bound) {
          break;
        }
      }
    }
  }

  void process_inputs() {
    process_input_param.ResetRead();
    input_resp.Reset();
    input_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_input_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> k_bound = K2 * K1 - 1;
      ac_int<32, false> c_bound = (C2 * C1 + Width - 1) / Width - 1;

      constexpr int buffer_width = Input::width * Width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k = 0;; k++) {
        for (int c = 0;; c++) {
          ac_int<buffer_width, false> bits = 0;
          bool success = (process_matrix_input<InputTypes, Width, PortWidth,
                                               buffer_width, InputTypes...>(
                              params.input_dtype, input_resp, bits) ||
                          ...);

          Pack1D<Input, Width> inputs;

#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            auto data = bits.template slc<Input::width>(i * Input::width);

#if SUPPORT_CODEBOOK_QUANT
            if (params.use_input_codebook) {
              auto value = params.input_code[data];
              inputs[i].set_bits(value);
            } else
#endif
            {
              bool success =
                  (decode_type<InputTypes, Input, Input::width, InputTypes...>(
                       params.input_dtype, data, inputs[i]) ||
                   ...);
#ifndef __SYNTHESIS__
              if (!success) {
                std::cerr << "Error: matrix input dtype '" << params.input_dtype
                          << "' is not valid" << std::endl;
              }
#endif
            }
          }

          input_data.Push(inputs);

          if (c == c_bound) {
            break;
          }
        }
        if (k == k_bound) {
          break;
        }
      }
    }
  }
#if SUPPORT_MX
  void process_input_scales() {
    input_scale_param.ResetRead();
    input_scale_resp.Reset();
    input_scale_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = input_scale_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> k_bound = K2 * K1 - 1;
      ac_int<32, false> c_bound = (C2 * C1 + Width - 1) / Width - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int k = 0;; k++) {
        for (int c = 0;; c++) {
          ac_int<Scale::width * Width / BS, false> data;
          process_matrix_input<Scale, Width / BS, Scale::width * Width / BS,
                               Scale::width * Width / BS>(input_scale_resp,
                                                          data);

          Pack1D<Scale, Width / BS> scales;
#pragma hls_unroll yes
          for (int i = 0; i < Width / BS; i++) {
            scales[i].set_bits(
                data.template slc<Scale::width>(i * Scale::width));
          }
          input_scale_data.Push(scales);

          if (c == c_bound) {
            break;
          }
        }
        if (k == k_bound) {
          break;
        }
      }
    }
  }
#endif
  void fetch_weights() {
    fetch_weight_param.ResetRead();
    weight_req.Reset();
#if SUPPORT_MX
    weight_scale_req.Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = fetch_weight_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> C = C2 * C1;
      ac_int<32, false> c_bound = (C + Width - 1) / Width - 1;
      ac_int<32, false> k_bound = K2 * K1 - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        for (ac_int<32, false> c = 0;; c++) {
          ac_int<32, false> address = k * C + c * Width;
          (fetch_matrix_input<WeightTypes, Width, WeightTypes...>(
               params.weight_dtype, params.weight_offset, address, weight_req),
           ...);

#if SUPPORT_MX
          if (params.is_mx_op) {
            ac_int<32, false> address = k * (C / BS) + c * (Width / BS);
            send_input_request<Scale, Width / BS>(params.weight_scale_offset,
                                                  address, weight_scale_req);
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
    }
  }

  void process_weights() {
    process_weight_param.ResetRead();
    weight_resp.Reset();
    weight_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weight_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> k_bound = K2 * K1 - 1;
      ac_int<32, false> c_bound = (C2 * C1 + Width - 1) / Width - 1;

      constexpr int buffer_width = Weight::width * Width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        for (ac_int<32, false> c = 0;; c++) {
          ac_int<buffer_width, false> bits = 0;
          bool success = (process_matrix_input<WeightTypes, Width, PortWidth,
                                               buffer_width, WeightTypes...>(
                              params.weight_dtype, weight_resp, bits) ||
                          ...);

          Pack1D<Weight, Width> weights;

#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
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
    }
  }
#if SUPPORT_MX
  void process_weight_scales() {
    weight_scale_param.ResetRead();
    weight_scale_resp.Reset();
    weight_scale_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = weight_scale_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      loop_t c_bound = (C2 * C1 + Width - 1) / Width;
      ac_int<32, false> loop_bound = K2 * K1 * c_bound - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        ac_int<Scale::width * Width / BS, false> data;
        process_matrix_input<Scale, Width / BS, Scale::width * Width / BS,
                             Scale::width * Width / BS>(weight_scale_resp,
                                                        data);

        Pack1D<Scale, Width / BS> scales;
#pragma hls_unroll yes
        for (int i = 0; i < Width / BS; i++) {
          scales[i].set_bits(data.template slc<Scale::width>(i * Scale::width));
        }
        weight_scale_data.Push(scales);

        if (k == loop_bound) {
          break;
        }
      }
    }
  }
#endif

  void fetch_biases() {
    fetch_bias_param.ResetRead();
    bias_req.Reset();
    bias_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = fetch_bias_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      ac_int<32, false> k_bound =
          (K2 * K1 + VectorUnitWidth - 1) / VectorUnitWidth - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        send_input_request<Output, VectorUnitWidth>(
            params.bias_offset, k * VectorUnitWidth, bias_req);

        ac_int<Output::width * VectorUnitWidth, false> bits = 0;
        process_matrix_input<Output, VectorUnitWidth, PortWidth,
                             Output::width * VectorUnitWidth>(bias_resp, bits);

        Pack1D<Output, VectorUnitWidth> biases;
#pragma hls_unroll yes
        for (int i = 0; i < VectorUnitWidth; i++) {
          auto data = bits.template slc<Output::width>(i * Output::width);
          biases[i].set_bits(data);
        }

        bias_data.Push(biases);

        if (k == k_bound) {
          break;
        }
      }
    }
  }

  void run_accumulation() {
    run_accumulation_param.ResetRead();
    input_data.ResetRead();
    weight_data.ResetRead();
#if SUPPORT_MX
    input_scale_data.ResetRead();
    weight_scale_data.ResetRead();
#endif
    accumulation_out.ResetWrite();
    start_signal.Reset();
    done_signal.Reset();

    wait();

    while (true) {
      MatrixParams params = run_accumulation_param.Pop();

      start_signal.SyncPush();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      loop_t C2 = params.loops[0][params.reductionLoopIndex[0]];
      loop_t C1 = params.loops[1][params.reductionLoopIndex[1]];
      ac_int<32, false> C = C2 * C1;
      loop_t c_bound = (C + Width - 1) / Width - 1;
      ac_int<32, false> k_bound = K2 * K1 - 1;

      Pack1D<Scale, Width / BS> input_scales;
      Pack1D<Scale, Width / BS> weight_scales;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    K_LOOP:
      for (ac_int<32, false> k = 0;; k++) {
        Pack1D<Output, FEEDBACK_DELAY> acc_old;
#pragma hls_unroll yes
        for (int i = 0; i < FEEDBACK_DELAY; i++) {
          acc_old[i] = Output::zero();
        }

      C2_LOOP:
        for (loop_t c = 0;; c++) {
          // Initialize the partial sums
          Pack1D<Input, Width> inputs = input_data.Pop();
          Pack1D<Weight, Width> weights = weight_data.Pop();
#if SUPPORT_MX
          if (params.is_mx_op) {
            input_scales = input_scale_data.Pop();
            weight_scales = weight_scale_data.Pop();
          }
#endif

          Pack1D<Psum, Width> psums;
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            psums[i] = (Psum)inputs[i] * (Psum)weights[i];
            if (c * Width + i >= C) {
              psums[i] = Psum::zero();  // Zero out elements out of bounds
            }
          }

          Pack1D<Output, Width / BS> outputs;

        // Perform reduction on each block
        C1_LOOP:
#pragma hls_unroll yes
          for (int c1 = 0; c1 < Width / BS; c1++) {
            Pack1D<Psum, BS> psum_block;
#pragma hls_unroll yes
            for (int i = 0; i < BS; i++) {
              psum_block[i] = psums[c1 * BS + i];
            }
            outputs[c1] = tree_sum(psum_block);
#if SUPPORT_MX
            if (params.is_mx_op) {
              outputs[c1] *=
                  (Output)input_scales[c1] * (Output)weight_scales[c1];
            }
#endif
          }

          Output acc = tree_sum(outputs) + acc_old[LAST];

#pragma hls_unroll yes
          for (int i = LAST; i > 0; i--) {
            acc_old[i] = acc_old[i - 1];
          }

          acc_old[0] = acc;

          if (c == c_bound) {
            break;
          }
        }

        Output acc = tree_sum(acc_old);
        accumulation_out.Push(acc);

        if (k == k_bound) {
          break;
        }
      }

      done_signal.SyncPush();
    }
  }

  void send_outputs() {
    bias_data.ResetRead();
    accumulation_out.ResetRead();
    matrix_out.Reset();

    wait();

    while (true) {
      MatrixParams params = send_outputs_param.Pop();

      loop_t K2 = params.loops[0][params.weightLoopIndex[0]];
      loop_t K1 = params.loops[1][params.weightLoopIndex[1]];
      ac_int<32, false> k_bound =
          (K2 * K1 + VectorUnitWidth - 1) / VectorUnitWidth - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> k = 0;; k++) {
        Pack1D<Output, VectorUnitWidth> biases;
        if (params.has_bias) {
          biases = bias_data.Pop();
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < VectorUnitWidth; i++) {
            biases[i] = Output::zero();
          }
        }

        Pack1D<Output, VectorUnitWidth> outputs;
        for (int i = 0; i < VectorUnitWidth; i++) {
          Output acc = accumulation_out.Pop();
          outputs[i] = acc + biases[i];
        }

        matrix_out.Push(outputs);

        if (k == k_bound) {
          break;
        }
      }
    }
  }

  void send_params() {
    params_in.ResetRead();
    fetch_input_param.ResetWrite();
    process_input_param.ResetWrite();
    fetch_weight_param.ResetWrite();
    process_weight_param.ResetWrite();
#if SUPPORT_MX
    input_scale_param.ResetWrite();
    weight_scale_param.ResetWrite();
#endif
    fetch_bias_param.ResetWrite();
    run_accumulation_param.ResetWrite();
    send_outputs_param.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();
      fetch_input_param.Push(params);
      process_input_param.Push(params);
      fetch_weight_param.Push(params);
      process_weight_param.Push(params);
#if SUPPORT_MX
      if (params.is_mx_op) {
        input_scale_param.Push(params);
        weight_scale_param.Push(params);
      }
#endif
      if (params.has_bias) {
        fetch_bias_param.Push(params);
      }
      run_accumulation_param.Push(params);
      send_outputs_param.Push(params);
    }
  }
};
