#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "Params.h"
#include "ParamsDeserializer.h"
#include "Utils.h"
#include "connections/connections.h"

#pragma hls_design ccore
template <typename Input, typename Weight, typename Scale, typename Output>
Output multiply_accumulate(const Input input, const Weight weight,
                           const Scale weight_scale, const Output psum) {
  return psum + (Output)input * (Output)weight * (Output)weight_scale;
}

template <typename WeightTypeTuple, typename Input, typename Weight,
          typename Meta, typename Output, typename Scale, int port_width,
          int width, int bs, int vu_width>
struct SpMMUnit;

template <typename... WeightTypes, typename Input, typename Weight,
          typename Meta, typename Output, typename Scale, int port_width,
          int width, int bs, int vu_width>
struct SpMMUnit<std::tuple<WeightTypes...>, Input, Weight, Meta, Output, Scale,
                port_width, width, bs, vu_width> : public sc_module {
  static constexpr int LOOP_WIDTH = 16;
  using loop_t = ac_int<LOOP_WIDTH, false>;

#ifdef CLOCK_PERIOD
  static constexpr double clock_period = CLOCK_PERIOD;
#else
  static constexpr double clock_period = 5.0;  // Default to 5 ns if not defined
#endif

  static constexpr int FEEDBACK_DEPTH = (clock_period < 5) ? 6 : 4;
  static constexpr int FEEDBACK_LAST = FEEDBACK_DEPTH - 1;

  static constexpr int NUM_META = port_width / Meta::width;
  static constexpr int NUM_DATA = port_width / Input::width;

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixParamsDeserializer<2, 1> CCS_INIT_S1(params_deserializer);
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
  Connections::Combinational<Pack1D<Input, NUM_DATA>> CCS_INIT_S1(
      input_data_packed);
  Connections::Combinational<Input> CCS_INIT_S1(input_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_resp);
  Connections::Combinational<Pack1D<Weight, width>> CCS_INIT_S1(weight_data);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_scale_resp);
  Connections::Combinational<Pack1D<Scale, width>> CCS_INIT_S1(
      weight_scale_data);
#endif

  Connections::Combinational<Pack1D<Output, width>> CCS_INIT_S1(
      accumulation_out);
  Connections::Out<Pack1D<Output, vu_width>> CCS_INIT_S1(spmm_unit_output);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

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
        for (loop_t x = 0;; x++) {
          send_input_request<Meta, NUM_META>(params.spmm_indptr_offset,
                                             x * NUM_META, input_indptr_req);
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
          if (x == indptr_bound) break;
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
        Meta last_indptr;

        for (loop_t x = 0;; x++) {
          auto indptrs = input_indptr_pack.Pop();

          for (int i = 0; i < NUM_META; i++) {
            if (x != 0 || i != 0) {
              Meta start_index = i == 0 ? last_indptr : indptrs[i - 1];
              Meta end_index = indptrs[i];
              Meta nnz = end_index - start_index;

              input_indptr.Push(nnz);

              if (x * NUM_META + i == indptr_len - 1) break;
            }
          }

          last_indptr = indptrs[NUM_META - 1];

          if (x == indptr_bound) break;
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

      if (indices_len == 0) {
        continue;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          send_input_request<Meta, NUM_META>(params.spmm_indices_offset,
                                             x * NUM_META, input_indices_req);
          if (x == indices_bound) break;
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

      if (indices_len == 0) {
        continue;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
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
          if (x == indices_bound) break;
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
      loop_t indices_len = params.loops[0][params.x_loop_idx[0]];
      loop_t indices_bound = (indices_len + NUM_DATA - 1) / NUM_DATA - 1;

      if (indices_len == 0) {
        continue;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          send_input_request<Input, NUM_DATA>(params.spmm_data_offset,
                                              x * NUM_DATA, input_data_req);
          ac_int<Input::width * NUM_DATA, false> bits = 0;
          process_matrix_input<Input, NUM_DATA, port_width,
                               Input::width * NUM_DATA>(input_data_resp, bits);
          Pack1D<Input, NUM_DATA> data;
#pragma hls_unroll yes
          for (int j = 0; j < NUM_DATA; j++) {
            data[j].set_bits(bits.template slc<Input::width>(j * Input::width));
          }
          input_data_packed.Push(data);
          if (x == indices_bound) break;
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
      loop_t indices_len = params.loops[0][params.x_loop_idx[0]];
      loop_t indices_bound = (indices_len + NUM_DATA - 1) / NUM_DATA - 1;

      if (indices_len == 0) {
        continue;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          auto input_pack = input_data_packed.Pop();
          for (int j = 0; j < NUM_DATA; j++) {
            input_data.Push(input_pack[j]);
            if (x * NUM_DATA + j == indices_len - 1) break;
          }
          if (x == indices_bound) break;
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

      if (indices_len == 0) {
        continue;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          // index here is row index (y dimension)
          Pack1D<Meta, NUM_META> indices = input_indices.Pop();
          for (int j = 0; j < NUM_META; j++) {
            ac_int<32, false> address =
                indices[j].int_val * K * width + k * width;
            (fetch_matrix_input<WeightTypes, width, WeightTypes...>(
                 params.weight_dtype, params.weight_offset, address,
                 weight_req),
             ...);

#if SUPPORT_MX
            if (params.is_mx_op) {
              ac_int<32, false> address =
                  indices[j].int_val / bs * K * width + k * width;
              send_input_request<Scale, width>(params.weight_scale_offset,
                                               address, weight_scale_req);
            }
#endif

            if (x * NUM_META + j == indices_len - 1) break;
          }
          if (x == indices_bound) break;
        }
        if (k == k_bound) break;
      }
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

      if (indices_len == 0) {
        continue;
      }

      constexpr int buffer_width = Weight::width * width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          ac_int<buffer_width, false> bits = 0;
          bool success = (process_matrix_input<WeightTypes, width, port_width,
                                               buffer_width, WeightTypes...>(
                              params.weight_dtype, weight_resp, bits) ||
                          ...);

          Pack1D<Weight, width> weights;
#pragma hls_unroll yes
          for (int j = 0; j < width; j++) {
            auto data = bits.template slc<Weight::width>(j * Weight::width);
            weights[j] =
                decode_input<Weight, NUM_CODEBOOK_ENTRIES, WeightTypes...>(
                    data, params.weight_dtype, params.use_weight_codebook,
                    params.weight_code);
          }

          weight_data.Push(weights);

          if (x == indices_bound) break;
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

      if (indices_len == 0) {
        continue;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          ac_int<Scale::width * width, false> data;
          process_matrix_input<Scale, width, port_width, Scale::width * width>(
              weight_scale_resp, data);

          Pack1D<Scale, width> scales;
#pragma hls_unroll yes
          for (int j = 0; j < width; j++) {
            scales[j].set_bits(
                data.template slc<Scale::width>(j * Scale::width));
          }
          weight_scale_data.Push(scales);
          if (x == indices_bound) break;
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
    start.Reset();
    done.Reset();

    wait();

    while (true) {
      MatrixParams params = run_accumulation_param.Pop();

      start.SyncPush();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;

      // -1 because the first indptr is popped outside of the loop
      loop_t indptr_len = params.loops[0][params.y_loop_idx[0]];
      loop_t indptr_bound = indptr_len - 2;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          Meta nnz_bound = input_indptr.Pop();

          Pack1D<Output, width> acc_old[FEEDBACK_DEPTH];
#pragma hls_unroll yes
          for (int i = 0; i < FEEDBACK_DEPTH; i++) {
            acc_old[i] = Pack1D<Output, width>::zero();
          }

          for (loop_t nnz = 0;; nnz++) {
            if (nnz == nnz_bound.int_val) break;

            Input input = input_data.Pop();
            Pack1D<Weight, width> weights = weight_data.Pop();
#if SUPPORT_MX
            auto weight_scale = Pack1D<Scale, width>::one();
            if (params.is_mx_op) {
              weight_scale = weight_scale_data.Pop();
            }
#endif
            Pack1D<Output, width> psums;
#pragma hls_unroll yes
            for (int i = 0; i < width; i++) {
              psums[i] = multiply_accumulate(input, weights[i], weight_scale[i],
                                             acc_old[FEEDBACK_LAST][i]);
            }

#pragma hls_unroll yes
            for (int k = FEEDBACK_LAST; k > 0; k--) {
              acc_old[k] = acc_old[k - 1];
            }
            acc_old[0] = psums;
          }

          Pack1D<Output, width> outputs;
#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            Pack1D<Output, FEEDBACK_DEPTH> col;
#pragma hls_unroll yes
            for (int j = 0; j < FEEDBACK_DEPTH; j++) {
              col[j] = acc_old[j][i];
            }
            outputs[i] = tree_sum(col);
          }

          accumulation_out.Push(outputs);

          if (x == indptr_bound) break;
        }
        if (k == k_bound) break;
      }

      done.SyncPush();
    }
  }

  void send_outputs() {
    send_outputs_param.ResetRead();
    accumulation_out.ResetRead();
    spmm_unit_output.Reset();

    wait();

    while (true) {
      MatrixParams params = send_outputs_param.Pop();

      loop_t k_bound = params.loops[0][params.weight_loop_idx[0]] - 1;

      // the length of the indptr is the number of rows + an extra 0
      loop_t indptr_len = params.loops[0][params.y_loop_idx[0]];
      loop_t indptr_bound = indptr_len - 2;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t k = 0;; k++) {
        for (loop_t x = 0;; x++) {
          auto acc = accumulation_out.Pop();
          for (int i = 0; i < width / vu_width; i++) {
            Pack1D<Output, vu_width> outputs;
#pragma hls_unroll yes
            for (int j = 0; j < vu_width; j++) {
              outputs[j] = acc[i * vu_width + j];
            }
            spmm_unit_output.Push(outputs);
          }
          if (x == indptr_bound) break;
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
