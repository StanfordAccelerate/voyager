#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "../ParamsDeserializer.h"
#include "../Utils.h"
#include "VectorMacUnit.h"

template <typename InputTypeTuple, typename WeightTypeTuple, typename Input,
          typename Weight, typename Psum, typename Output, typename Scale,
          int port_width, int width, int bs, int vu_width>
struct MatrixVectorUnit;

template <typename... InputTypes, typename... WeightTypes, typename Input,
          typename Weight, typename Psum, typename Output, typename Scale,
          int port_width, int width, int bs, int vu_width>
struct MatrixVectorUnit<std::tuple<InputTypes...>, std::tuple<WeightTypes...>,
                        Input, Weight, Psum, Output, Scale, port_width, width,
                        bs, vu_width> : public sc_module {
  static constexpr int LOOP_WIDTH = 16;

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixParamsDeserializer<1, 1> CCS_INIT_S1(params_deserializer);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_input_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_input_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_input_scale_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      fetch_weight_dq_qparams_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_weight_dq_qparams_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_weight_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_weight_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_weight_scale_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_bias_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(send_inputs_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(run_accumulation_param);

  VectorMacUnit<Input, Weight, Psum, Scale, Output, width, bs> CCS_INIT_S1(
      vector_mac_unit);
  Connections::Combinational<Pack1D<Input, width>> CCS_INIT_S1(mac_inputs);
#if SUPPORT_MX
  Connections::Combinational<Pack1D<Scale, width / bs>> CCS_INIT_S1(
      mac_input_scales);
#endif
  Connections::Combinational<Output> CCS_INIT_S1(psum_out);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_resp);
  Connections::Combinational<Pack1D<Input, width>> CCS_INIT_S1(input_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_resp);
  Connections::Combinational<Pack1D<Weight, width>> CCS_INIT_S1(weight_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(bias_resp);
  Connections::Combinational<Pack1D<Output, bs>> CCS_INIT_S1(bias_data);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_scale_req);
  Connections::In<ac_int<Scale::width * width / bs, false>> CCS_INIT_S1(
      input_scale_resp);
  Connections::Combinational<Pack1D<Scale, width / bs>> CCS_INIT_S1(
      input_scale_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<Scale::width * width / bs, false>> CCS_INIT_S1(
      weight_scale_resp);
  Connections::Combinational<Pack1D<Scale, width / bs>> CCS_INIT_S1(
      weight_scale_data);
#endif

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_dq_scale_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_dq_scale_resp);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_dq_zero_point_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(
      weight_dq_zero_point_resp);
  Connections::Combinational<Pack1D<Scale, 2 * width>> CCS_INIT_S1(
      weight_dq_qparams_data);

  Connections::Combinational<Pack1D<Output, bs>> CCS_INIT_S1(accumulation_out);
  Connections::Out<Pack1D<Output, vu_width>> CCS_INIT_S1(matrix_out);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  using loop_t = ac_int<LOOP_WIDTH, false>;

  SC_CTOR(MatrixVectorUnit) {
    params_deserializer.clk(clk);
    params_deserializer.rstn(rstn);
    params_deserializer.serial_params_in(serial_params_in);
    params_deserializer.params_out[0](params_in);

    vector_mac_unit.clk(clk);
    vector_mac_unit.rstn(rstn);
    vector_mac_unit.inputs_in(mac_inputs);
    vector_mac_unit.weights_in(weight_data);
#if SUPPORT_MX
    vector_mac_unit.input_scales_in(mac_input_scales);
    vector_mac_unit.weight_scales_in(weight_scale_data);
#endif
    vector_mac_unit.psum_out(psum_out);

    SC_THREAD(fetch_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#if SUPPORT_MX
    SC_THREAD(process_input_scales);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
    SC_THREAD(send_inputs)
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_weight_dq_qparams);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_weight_dq_qparams);
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
    SC_THREAD(fetch_biases);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(unpack_outputs);
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

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      loop_t c_bound = (C2 * C1 + width - 1) / width - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          ac_int<32, false> address = c * width;
          (fetch_matrix_input<InputTypes, width, InputTypes...>(
               params.input_dtype, params.input_offset, address, input_req),
           ...);
#if SUPPORT_MX
          if (params.is_mx_op) {
            send_input_request<Scale, width / bs>(
                params.input_scale_offset, address / bs, input_scale_req);
          }
#endif
          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
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

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      ac_int<32, false> C = C2 * C1;
      loop_t c_bound = (C + width - 1) / width - 1;

      constexpr int buffer_width = Input::width * width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          ac_int<buffer_width, false> bits = 0;
          bool success = (process_matrix_input<InputTypes, width, port_width,
                                               buffer_width, InputTypes...>(
                              params.input_dtype, input_resp, bits) ||
                          ...);

          Pack1D<Input, width> inputs;
          ac_int<32, false> address = c * width;

#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            auto data = bits.template slc<Input::width>(i * Input::width);
            inputs[i] =
                decode_input<Input, NUM_CODEBOOK_ENTRIES, InputTypes...>(
                    data, params.input_dtype, params.use_input_codebook,
                    params.input_code);

            // Set any out-of-bounds inputs to zero
            if (address + i >= C) {
              inputs[i] = Input::zero();
            }
          }

          input_data.Push(inputs);

          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }

#if SUPPORT_MX
  void process_input_scales() {
    process_input_scale_param.ResetRead();
    input_scale_resp.Reset();
    input_scale_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_input_scale_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      loop_t c_bound = (C2 * C1 + width - 1) / width - 1;

      constexpr int buffer_width = Scale::width * width / bs;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          ac_int<buffer_width, false> bits;
          process_matrix_input<Scale, width / bs, buffer_width, buffer_width>(
              input_scale_resp, bits);

          Pack1D<Scale, width / bs> scales;
#pragma hls_unroll yes
          for (int i = 0; i < width / bs; i++) {
            scales[i].set_bits(
                bits.template slc<Scale::width>(i * Scale::width));
          }
          input_scale_data.Push(scales);

          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }
#endif

  void send_inputs() {
    send_inputs_param.ResetRead();
    input_data.ResetRead();
    mac_inputs.ResetWrite();
    start.Reset();
#if SUPPORT_MX
    input_scale_data.ResetRead();
    mac_input_scales.ResetWrite();
#endif

    wait();

    while (true) {
      MatrixParams params = send_inputs_param.Pop();
      start.SyncPush();

      loop_t K1 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K0 = params.loops[1][params.weight_loop_idx[1]];
      loop_t k_bound = (K1 * K0 + bs - 1) / bs - 1;

      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      ac_int<32, false> C = C2 * C1;
      loop_t c_bound = (C + width - 1) / width - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          Pack1D<Input, width> inputs = input_data.Pop();
#if SUPPORT_MX
          Pack1D<Scale, width / bs> input_scales;
          if (params.is_mx_op) {
            input_scales = input_scale_data.Pop();
          }
#endif

          for (loop_t k0 = 0;; k0++) {
            mac_inputs.Push(inputs);
#if SUPPORT_MX
            if (params.is_mx_op) {
              mac_input_scales.Push(input_scales);
            }
#endif
            if (k0 == bs - 1) break;
          }
          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }

  void fetch_weight_dq_qparams() {
    fetch_weight_dq_qparams_param.ResetRead();
    weight_dq_scale_req.Reset();
    weight_dq_zero_point_req.Reset();

    wait();

    while (true) {
      const MatrixParams params = fetch_weight_dq_qparams_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      ac_int<32, false> C = C2 * C1;
      loop_t c_bound = (C + width - 1) / width - 1;
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          ac_int<32, false> address = k1 * C + c * width;
          send_input_request<Scale, width>(params.dq_scale_offset, address,
                                           weight_dq_scale_req);
          send_input_request<Scale, width>(params.dq_zero_point_offset, address,
                                           weight_dq_zero_point_req);
          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }

  void process_weight_dq_qparams() {
    process_weight_dq_qparams_param.ResetRead();
    weight_dq_scale_resp.Reset();
    weight_dq_zero_point_resp.Reset();
    weight_dq_qparams_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weight_dq_qparams_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      ac_int<32, false> C = C2 * C1;
      loop_t c_bound = (C + width - 1) / width - 1;
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

      constexpr int num_words =
          (Scale::width * width + port_width - 1) / port_width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          ac_int<num_words * port_width, false> scale_bits;
          ac_int<num_words * port_width, false> zero_point_bits;

          for (int i = 0; i < num_words; i++) {
            scale_bits.set_slc(i * port_width, weight_dq_scale_resp.Pop());
            zero_point_bits.set_slc(i * port_width,
                                    weight_dq_zero_point_resp.Pop());
          }

          Pack1D<Scale, 2 * width> qparams;
#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            qparams[i].set_bits(
                scale_bits.template slc<Scale::width>(i * Scale::width));
            qparams[width + i].set_bits(
                zero_point_bits.template slc<Scale::width>(i * Scale::width));
          }

          weight_dq_qparams_data.Push(qparams);

          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }

  void fetch_weights() {
    fetch_weight_param.ResetRead();
    weight_req.Reset();
#if SUPPORT_MX
    weight_scale_req.Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = fetch_weight_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      ac_int<32, false> C = C2 * C1;
      loop_t c_bound = (C + width - 1) / width - 1;
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          for (loop_t k0 = 0;; k0++) {
            ac_int<32, false> address = (k1 * bs + k0) * C + c * width;
            (fetch_matrix_input<WeightTypes, width, WeightTypes...>(
                 params.weight_dtype, params.weight_offset, address,
                 weight_req),
             ...);
#if SUPPORT_MX
            if (params.is_mx_op) {
              send_input_request<Scale, width / bs>(
                  params.weight_scale_offset, address / bs, weight_scale_req);
            }
#endif
            if (k0 == bs - 1) break;
          }
          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }

  void process_weights() {
    process_weight_param.ResetRead();
    weight_resp.Reset();
    weight_dq_qparams_data.ResetRead();
    weight_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weight_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;
      loop_t c_bound = (C2 * C1 + width - 1) / width - 1;

      constexpr int buffer_width = Weight::width * width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          Pack1D<Scale, width> dq_scales;
          Pack1D<Scale, width> dq_zero_points;
          if (params.weight_dequant) {
            Pack1D<Scale, 2 * width> qparams = weight_dq_qparams_data.Pop();
#pragma hls_unroll yes
            for (int i = 0; i < width; i++) {
              dq_scales[i] = qparams[i];
              dq_zero_points[i] = qparams[width + i];
            }
          }

          for (loop_t k0 = 0;; k0++) {
            ac_int<buffer_width, false> bits = 0;
            bool success = (process_matrix_input<WeightTypes, width, port_width,
                                                 buffer_width, WeightTypes...>(
                                params.weight_dtype, weight_resp, bits) ||
                            ...);

            Pack1D<Weight, width> weights;

#pragma hls_unroll yes
            for (int i = 0; i < width; i++) {
              auto data = bits.template slc<Weight::width>(i * Weight::width);
              weights[i] =
                  decode_and_dequantize<Weight, Scale, Output,
                                        NUM_CODEBOOK_ENTRIES, WeightTypes...>(
                      data, params.weight_dtype, params.use_weight_codebook,
                      params.weight_code, params.weight_dequant, dq_scales[i],
                      dq_zero_points[i]);
            }

            weight_data.Push(weights);

            if (k0 == bs - 1) break;
          }
          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }

#if SUPPORT_MX
  void process_weight_scales() {
    process_weight_scale_param.ResetRead();
    weight_scale_resp.Reset();
    weight_scale_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weight_scale_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      loop_t c_bound = (C2 * C1 + width - 1) / width - 1;
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

      constexpr int buffer_width = Scale::width * width / bs;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        for (loop_t c = 0;; c++) {
          for (loop_t k0 = 0;; k0++) {
            ac_int<buffer_width, false> bits;
            process_matrix_input<Scale, width / bs, buffer_width, buffer_width>(
                weight_scale_resp, bits);

            Pack1D<Scale, width / bs> scales;
#pragma hls_unroll yes
            for (int i = 0; i < width / bs; i++) {
              scales[i].set_bits(
                  bits.template slc<Scale::width>(i * Scale::width));
            }

            weight_scale_data.Push(scales);

            if (k0 == bs - 1) break;
          }
          if (c == c_bound) break;
        }
        if (k1 == k_bound) break;
      }
    }
  }
#endif

  void fetch_biases() {
    fetch_bias_param.ResetRead();
    bias_req.Reset();
    bias_resp.Reset();
    bias_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = fetch_bias_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        send_input_request<Output, bs>(params.bias_offset, k * bs, bias_req);

        ac_int<Output::width * bs, false> bits;
        process_matrix_input<Output, bs, port_width, Output::width * bs>(
            bias_resp, bits);

        Pack1D<Output, bs> biases;
#pragma hls_unroll yes
        for (int i = 0; i < bs; i++) {
          biases[i].set_bits(
              bits.template slc<Output::width>(i * Output::width));
        }
        bias_data.Push(biases);

        if (k == k_bound) break;
      }
    }
  }

  void run_accumulation() {
    run_accumulation_param.ResetRead();
    bias_data.ResetRead();
    psum_out.ResetRead();
    accumulation_out.ResetWrite();
    done.Reset();

    wait();

    while (true) {
      MatrixParams params = run_accumulation_param.Pop();

      loop_t K2 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K1 = params.loops[1][params.weight_loop_idx[1]];
      loop_t C2 = params.loops[0][params.reduction_loop_idx[0]];
      loop_t C1 = params.loops[1][params.reduction_loop_idx[1]];
      loop_t c_bound = (C2 * C1 + width - 1) / width - 1;
      loop_t k_bound = (K2 * K1 + bs - 1) / bs - 1;

      constexpr int DELAY = 4;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k1 = 0;; k1++) {
        Pack1D<Output, bs> outputs = Pack1D<Output, bs>::zero();

        if (params.has_bias) {
          outputs = bias_data.Pop();
        }

        Output outputs_old[DELAY];

#pragma hls_unroll yes
        for (int i = 0; i < DELAY; i++) {
          int k0 = bs - DELAY + i;
          outputs_old[i] = outputs[k0];
        }

        for (loop_t c = 0;; c++) {
          for (loop_t k0 = 0; k0 < bs; k0++) {
            int k0_delayed = (k0 + bs - DELAY) % bs;
            outputs[k0_delayed] = outputs_old[0];
#pragma hls_unroll yes
            for (int i = 0; i < DELAY - 1; i++) {
              outputs_old[i] = outputs_old[i + 1];
            }
            outputs_old[DELAY - 1] = outputs[k0] + psum_out.Pop();
          }
          if (c == c_bound) break;
        }

#pragma hls_unroll yes
        for (int i = 0; i < DELAY; i++) {
          int k0 = bs - DELAY + i;
          outputs[k0] = outputs_old[i];
        }

        accumulation_out.Push(outputs);
        if (k1 == k_bound) break;
      }
      done.SyncPush();
    }
  }

  void unpack_outputs() {
    accumulation_out.ResetRead();
    matrix_out.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<Output, bs> outputs = accumulation_out.Pop();
      for (int i = 0; i < bs / vu_width; i++) {
        Pack1D<Output, vu_width> output_block;
#pragma hls_unroll yes
        for (int j = 0; j < vu_width; j++) {
          output_block[j] = outputs[i * vu_width + j];
        }
        matrix_out.Push(output_block);
      }
    }
  }

  void send_params() {
    params_in.ResetRead();
    fetch_input_param.ResetWrite();
    process_input_param.ResetWrite();
    fetch_weight_param.ResetWrite();
    process_weight_param.ResetWrite();
    fetch_weight_dq_qparams_param.ResetWrite();
    process_weight_dq_qparams_param.ResetWrite();
    fetch_bias_param.ResetWrite();
    send_inputs_param.ResetWrite();
    run_accumulation_param.ResetWrite();
#if SUPPORT_MX
    process_input_scale_param.ResetWrite();
    process_weight_scale_param.ResetWrite();
#endif

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();
      fetch_input_param.Push(params);
      process_input_param.Push(params);
      fetch_weight_param.Push(params);
      process_weight_param.Push(params);
      if (params.weight_dequant) {
        fetch_weight_dq_qparams_param.Push(params);
        process_weight_dq_qparams_param.Push(params);
      }
#if SUPPORT_MX
      if (params.is_mx_op) {
        process_input_scale_param.Push(params);
        process_weight_scale_param.Push(params);
      }
#endif
      if (params.has_bias) {
        fetch_bias_param.Push(params);
      }
      send_inputs_param.Push(params);
      run_accumulation_param.Push(params);
    }
  }
};
