#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "Params.h"
#include "ParamsDeserializer.h"
#include "Utils.h"
#include "connections/connections.h"

template <typename WeightTypeTuple, typename Vector, typename Weight,
          typename Meta, typename Output, typename Scale, int port_width,
          int width, int bs, int vu_width>
struct SpMMUnit;

template <typename... WeightTypes, typename Vector, typename Weight,
          typename Meta, typename Output, typename Scale, int port_width,
          int width, int bs, int vu_width>
struct SpMMUnit<std::tuple<WeightTypes...>, Vector, Weight, Meta, Output, Scale,
                port_width, width, bs, vu_width> : public sc_module {
  static constexpr int LOOP_WIDTH = 16;
  using loop_t = ac_int<LOOP_WIDTH, false>;

  static constexpr int FEEDBACK_DEPTH = 4;
  static constexpr int FEEDBACK_LAST = FEEDBACK_DEPTH - 1;

  static constexpr int NUM_META = port_width / Meta::width;
  static constexpr int NUM_DATA = port_width / Vector::width;

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixParamsDeserializer<1> CCS_INIT_S1(params_deserializer);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      fetch_input_indptr_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_input_indptr_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      fetch_input_indices_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_input_indices_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_input_data_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_input_data_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_weights_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_weights_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_weight_scales_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(run_accumulation_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(send_outputs_param);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_indptr_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_indptr_resp);
  Connections::Combinational<Pack1D<Meta, NUM_META>> CCS_INIT_S1(
      input_indptr_pack);
  Connections::Combinational<Meta> CCS_INIT_S1(input_indptr);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_indices_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_indices_resp);
  Connections::Combinational<Pack1D<Meta, NUM_META>> CCS_INIT_S1(input_indices);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_data_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_data_resp);
  Connections::Combinational<Pack1D<Vector, NUM_DATA>> CCS_INIT_S1(
      input_data_packed);
  Connections::Combinational<Vector> CCS_INIT_S1(input_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<Weight::width * width, false>> CCS_INIT_S1(
      weight_resp);
  Connections::Combinational<Pack1D<Weight, width>> CCS_INIT_S1(weight_data);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<Scale::width * width, false>> CCS_INIT_S1(
      weight_scale_resp);
  Connections::Combinational<Pack1D<Scale, width>> CCS_INIT_S1(
      weight_scale_data);
#endif

  Connections::Combinational<Pack1D<Output, width>> CCS_INIT_S1(
      accumulation_out);
  Connections::Out<Pack1D<Output, bs>> CCS_INIT_S1(matrix_out);

  Connections::SyncOut CCS_INIT_S1(start_signal);
  Connections::SyncOut CCS_INIT_S1(done_signal);

  SC_CTOR(SpMMUnit) {
    params_deserializer.clk(clk);
    params_deserializer.rstn(rstn);
    params_deserializer.serial_params_in(serial_params_in);
    params_deserializer.params_out[0](params_in);

    SC_THREAD(fetch_input_indptr);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_input_indptr);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_input_indices);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_inputs_indices);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_input_data);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_input_data);
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

  void fetch_input_indptr() {
    fetch_input_indptr_param.ResetRead();
    input_indptr_req.Reset();
    input_indptr_resp.Reset();
    input_indptr_pack.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = fetch_input_indptr_param.Pop();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;
      loop_t indptr_len = params.loops[0][params.y_loop_idx[0]];
      loop_t indptr_bound = (indptr_len + NUM_META - 1) / NUM_META - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          send_input_request<Meta, NUM_META>(params.spmm_indptr_offset,
                                             i * NUM_META, input_indptr_req);
          ac_int<Meta::width * NUM_META, false> bits = 0;
          process_matrix_input<Meta, NUM_META, port_width,
                               Meta::width * NUM_META>(input_indptr_resp, bits);
          Pack1D<Meta, NUM_META> indptrs;
#pragma hls_unroll yes
          for (int j = 0; j < NUM_META; j++) {
            indptrs[j].set_bits(
                bits.template slc<Meta::width>(j * Meta::width));
          }
          input_indptr_pack.Push(indptrs);
          if (i == indptr_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

  void process_input_indptr() {
    process_input_indptr_param.ResetRead();
    input_indptr_pack.ResetRead();
    input_indptr.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_input_indptr_param.Pop();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;
      loop_t indptr_len = params.loops[0][params.y_loop_idx[0]];
      loop_t indptr_bound = (indptr_len + NUM_META - 1) / NUM_META - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          auto indptrs = input_indptr_pack.Pop();
          for (int j = 0; j < NUM_META; j++) {
            input_indptr.Push(indptrs[j]);
            if (i * NUM_META + j == indptr_len - 1) break;
          }
          if (i == indptr_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

  void fetch_input_indices() {
    fetch_input_indices_param.ResetRead();
    input_indices_req.Reset();

    wait();

    while (true) {
      const MatrixParams params = fetch_input_indices_param.Pop();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;
      loop_t indices_len = params.loops[0][params.x_loop_idx[0]];
      loop_t indices_bound = (indices_len + NUM_META - 1) / NUM_META - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          send_input_request<Meta, NUM_META>(params.spmm_indices_offset,
                                             i * NUM_META, input_indices_req);
          if (i == indices_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

  void process_inputs_indices() {
    process_input_indices_param.ResetRead();
    input_indices_resp.Reset();
    input_indices.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_input_indices_param.Pop();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;
      loop_t indices_len = params.loops[0][params.x_loop_idx[0]];
      loop_t indices_bound = (indices_len + NUM_META - 1) / NUM_META - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          ac_int<Meta::width * NUM_META, false> bits = 0;
          process_matrix_input<Meta, NUM_META, port_width,
                               Meta::width * NUM_META>(input_indices_resp,
                                                       bits);
          Pack1D<Meta, NUM_META> indices;
#pragma hls_unroll yes
          for (int j = 0; j < NUM_META; j++) {
            indices[j].set_bits(
                bits.template slc<Meta::width>(j * Meta::width));
          }
          input_indices.Push(indices);
          if (i == indices_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

  void fetch_input_data() {
    fetch_input_data_param.ResetRead();
    input_data_req.Reset();
    input_data_resp.Reset();
    input_data_packed.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = fetch_input_data_param.Pop();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;
      // the length of the nonzero is the same as the length of indices
      loop_t input_len = params.loops[0][params.x_loop_idx[0]];
      loop_t input_bound = (input_len + NUM_DATA - 1) / NUM_DATA - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          send_input_request<Vector, NUM_DATA>(params.spmm_data_offset,
                                               i * NUM_DATA, input_data_req);
          ac_int<Vector::width * NUM_DATA, false> bits = 0;
          process_matrix_input<Vector, NUM_DATA, port_width,
                               Vector::width * NUM_DATA>(input_data_resp, bits);
          Pack1D<Vector, NUM_DATA> data;
#pragma hls_unroll yes
          for (int j = 0; j < NUM_DATA; j++) {
            data[j].set_bits(
                bits.template slc<Vector::width>(j * Vector::width));
          }
          input_data_packed.Push(data);
          if (i == input_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

  void process_input_data() {
    process_input_data_param.ResetRead();
    input_data_packed.ResetRead();
    input_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_input_data_param.Pop();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;
      loop_t input_len = params.loops[0][params.x_loop_idx[0]];
      loop_t input_bound = (input_len + NUM_DATA - 1) / NUM_DATA - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          auto input_pack = input_data_packed.Pop();
          for (int j = 0; j < NUM_DATA; j++) {
            input_data.Push(input_pack[j]);
            if (i * NUM_DATA + j == input_len - 1) break;
          }
          if (i == input_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

  void fetch_weights() {
    fetch_weights_param.ResetRead();
    input_indices.ResetRead();
    weight_req.Reset();
#if SUPPORT_MX
    weight_scale_req.Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = fetch_weights_param.Pop();

      loop_t K = params.loops[0][params.weight_loop_idx[0]];
      loop_t indices_len = params.loops[0][params.x_loop_idx[0]];

      loop_t k_bound = K - 1;
      loop_t indices_bound = (indices_len + NUM_META - 1) / NUM_META - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          // index here is row index (y dimension)
          Pack1D<Meta, NUM_META> indices = input_indices.Pop();
          for (int j = 0; j < NUM_META; j++) {
            ac_int<32, false> address =
                (loop_t)indices[j] * K * width + k * width;
            (fetch_matrix_input<WeightTypes, width, WeightTypes...>(
                 params.weight_dtype, params.weight_offset, address,
                 weight_req),
             ...);

#if SUPPORT_MX
            if (params.is_mx_op) {
              ac_int<32, false> address =
                  (loop_t)indices[j] / bs * K * width + k * width;
              send_input_request<Scale, width>(params.weight_scale_offset,
                                               address, weight_scale_req);
            }
#endif

            if (i * NUM_META + j == indices_len - 1) break;
          }
          if (i == indices_bound) break;
        }
        if (k == k_bound) break;
      }
      std::cerr << "fetch weights done" << std::endl;
    }
  }

  void process_weights() {
    process_weights_param.ResetRead();
    weight_resp.Reset();
    weight_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weights_param.Pop();

      loop_t K = params.loops[0][params.weight_loop_idx[0]];
      loop_t indices_len = params.loops[0][params.x_loop_idx[0]];

      loop_t k_bound = K - 1;
      loop_t indices_bound = indices_len - 1;

      constexpr int buffer_width = Weight::width * width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          ac_int<buffer_width, false> bits = 0;
          bool success =
              (process_matrix_input<WeightTypes, width, Weight::width * width,
                                    buffer_width, WeightTypes...>(
                   params.weight_dtype, weight_resp, bits) ||
               ...);
          Pack1D<Weight, width> weights;

#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
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

          if (i == indices_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

#if SUPPORT_MX
  void process_weight_scales() {
    process_weight_scales_param.ResetRead();
    weight_scale_resp.Reset();
    weight_scale_data.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_weight_scales_param.Pop();

      loop_t K = params.loops[0][params.weight_loop_idx[0]];
      loop_t indices_len = params.loops[0][params.x_loop_idx[0]];

      loop_t k_bound = K - 1;
      loop_t indices_bound = indices_len - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          ac_int<Scale::width * width, false> data;
          process_matrix_input<Scale, width, Scale::width * width,
                               Scale::width * width>(weight_scale_resp, data);

          Pack1D<Scale, width> scales;
#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            scales[i].set_bits(
                data.template slc<Scale::width>(i * Scale::width));
          }
          weight_scale_data.Push(scales);
          if (i == indices_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }
#endif
  void run_accumulation() {
    run_accumulation_param.ResetRead();
    input_indptr.ResetRead();
    input_data.ResetRead();
    weight_data.ResetRead();
#if SUPPORT_MX
    weight_scale_data.ResetRead();
#endif
    accumulation_out.ResetWrite();
    start_signal.Reset();
    done_signal.Reset();

    wait();

    while (true) {
      MatrixParams params = run_accumulation_param.Pop();

      std::cerr << "SpMM Unit params: " << std::endl << params << std::endl;

      start_signal.SyncPush();

      loop_t indptr_len = params.loops[0][params.y_loop_idx[0]];
      loop_t K = params.loops[0][params.weight_loop_idx[0]];

      // -1 because the first indptr is popped outside of the loop
      loop_t indptr_bound = indptr_len - 2;
      loop_t k_bound = K - 1;

      Pack1D<Scale, width> weight_scales;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        Meta start_index = input_indptr.Pop();
        for (loop_t i = 0;; i++) {
          Meta end_index = input_indptr.Pop();
          Meta num_nonzero = end_index - start_index;

          Pack1D<Output, width> psums;
#pragma hls_unroll yes
          for (int j = 0; j < width; j++) {
            psums[j] = Output::zero();
          }

          for (loop_t nnz = 0;; nnz++) {
            if (nnz == (loop_t)num_nonzero) break;
#if SUPPORT_MX
            Pack1D<Scale, width> weight_scale;
            if (params.is_mx_op) {
              weight_scale = weight_scale_data.Pop();
            }
#endif
            Vector input = input_data.Pop();
            Pack1D<Weight, width> weights = weight_data.Pop();

#pragma hls_unroll yes
            for (int w = 0; w < width; w++) {
              Output product = (Output)weights[w] * (Output)input;
#if SUPPORT_MX
              if (params.is_mx_op) {
                product *= (Output)weight_scale[w];
              }
#endif
              psums[w] += product;
            }
          }

          accumulation_out.Push(psums);
          start_index = end_index;

          if (i == indptr_bound) break;
        }
        if (k == k_bound) break;
      }

      done_signal.SyncPush();
    }
  }

  void send_outputs() {
    send_outputs_param.ResetRead();
    accumulation_out.ResetRead();
    matrix_out.Reset();

    wait();

    while (true) {
      MatrixParams params = send_outputs_param.Pop();

      loop_t indptr_len = params.loops[0][params.y_loop_idx[0]];
      loop_t K = params.loops[0][params.weight_loop_idx[0]];
      loop_t num_blocks = (width + bs - 1) / bs;

      // the length of the indptr is the number of rows + an extra 0
      loop_t indptr_bound = indptr_len - 2;
      loop_t k_bound = K - 1;
      loop_t num_blocks_bound = num_blocks - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t i = 0;; i++) {
          Pack1D<Output, width> acc = accumulation_out.Pop();
          for (loop_t b = 0;; b++) {
            Pack1D<Output, bs> outputs;
#pragma hls_unroll yes
            for (int j = 0; j < bs; j++) {
              int index = b * bs + j;
              outputs[j] = acc[index];
            }
            matrix_out.Push(outputs);
            if (b == num_blocks_bound) break;
          }
          if (i == indptr_bound) break;
        }
        if (k == k_bound) break;
      }
    }
  }

  void send_params() {
    params_in.ResetRead();
    fetch_input_indptr_param.ResetWrite();
    process_input_indptr_param.ResetWrite();
    fetch_input_indices_param.ResetWrite();
    process_input_indices_param.ResetWrite();
    fetch_input_data_param.ResetWrite();
    process_input_data_param.ResetWrite();
    fetch_weights_param.ResetWrite();
    process_weights_param.ResetWrite();
    process_weight_scales_param.ResetWrite();
    run_accumulation_param.ResetWrite();
    send_outputs_param.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();
      fetch_input_indptr_param.Push(params);
      process_input_indptr_param.Push(params);
      fetch_input_indices_param.Push(params);
      process_input_indices_param.Push(params);
      fetch_input_data_param.Push(params);
      process_input_data_param.Push(params);
      fetch_weights_param.Push(params);
      process_weights_param.Push(params);
#if SUPPORT_MX
      if (params.is_mx_op) {
        process_weight_scales_param.Push(params);
      }
#endif
      run_accumulation_param.Push(params);
      send_outputs_param.Push(params);
    }
  }
};
