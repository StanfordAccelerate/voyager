#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"
#include "Skewer.h"
#include "SystolicArray.h"
#include "Utils.h"

template <typename InputTypeTuple, typename WeightTypeTuple, typename Input,
          typename Weight, typename Psum, typename Buffer, typename Scale,
          int rows, int cols, int buffer_size>
struct MatrixProcessor;

template <typename... InputTypes, typename... WeightTypes, typename Input,
          typename Weight, typename Psum, typename Buffer, typename Scale,
          int rows, int cols, int buffer_size>
struct MatrixProcessor<std::tuple<InputTypes...>, std::tuple<WeightTypes...>,
                       Input, Weight, Psum, Buffer, Scale, rows, cols,
                       buffer_size> : public sc_module {
  static constexpr int LOOP_WIDTH = 10;
  static constexpr int FIFO_DEPTH = SUPPORT_MX ? 8 : 1;

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  static constexpr int ACCUM_BUFFER_BANKS = 2;
#else
  static constexpr int ACCUM_BUFFER_BANKS = 1;
#endif

 private:
  InputSerializedSkewer<Input, rows> CCS_INIT_S1(input_skewer);
  Connections::Combinational<Pack1D<PEInput<Input>, rows>> CCS_INIT_S1(
      input_skewer_din);

  WeightSerializedSkewer<Weight, cols> CCS_INIT_S1(weight_skewer);
  Connections::Combinational<Pack1D<PEWeight<Weight>, cols>> CCS_INIT_S1(
      weight_skewer_din);

  DeserializedSkewer<Psum, cols> CCS_INIT_S1(psum_out_skewer);
  Connections::Combinational<Pack1D<Psum, cols>> CCS_INIT_S1(
      psum_out_skewer_dout);

  SystolicArray<Input, Weight, Psum, rows, cols> CCS_INIT_S1(systolic_array);

  Connections::Fifo<Pack1D<Buffer, cols>, FIFO_DEPTH> CCS_INIT_S1(
      accum_to_wb_fifo);
  Connections::Combinational<Pack1D<Buffer, cols>> CCS_INIT_S1(accum_to_wb_enq);
  Connections::Combinational<Pack1D<Buffer, cols>> CCS_INIT_S1(accum_to_wb_deq);

  Connections::Fifo<Pack1D<Buffer, cols>, 8> CCS_INIT_S1(accum_output_fifo);
  Connections::Combinational<Pack1D<Buffer, cols>> CCS_INIT_S1(
      accum_output_enq);

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(input_channel);
  Connections::In<ac_int<WEIGHT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      weight_channel);
  Connections::In<Pack1D<Buffer, cols>> CCS_INIT_S1(bias_channel);

  Connections::Out<Pack1D<Buffer, cols>> CCS_INIT_S1(output_channel);

  Connections::Out<ac_int<16, false>>
      accumulation_buffer_read_address[ACCUM_BUFFER_BANKS];
  Connections::In<Pack1D<Buffer, cols>>
      accumulation_buffer_read_data[ACCUM_BUFFER_BANKS];
  Connections::Out<BufferWriteRequest<Pack1D<Buffer, cols>>>
      accumulation_buffer_write_request[ACCUM_BUFFER_BANKS];

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::SyncOut accumulation_buffer_done[ACCUM_BUFFER_BANKS];
#endif

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(push_weights_params);

  Connections::Fifo<MatrixParams, 1> CCS_INIT_S1(
      process_accumulation_params_fifo);
  Connections::Combinational<MatrixParams> process_accumulation_params_enq;
  Connections::Combinational<MatrixParams> process_accumulation_params_deq;

  Connections::Fifo<MatrixParams, 1> CCS_INIT_S1(write_back_params_fifo);
  Connections::Combinational<MatrixParams> write_back_params_enq;
  Connections::Combinational<MatrixParams> write_back_params_deq;

  Connections::Combinational<PEInput<Input>> systolic_array_inputs[rows];
  Connections::Combinational<PEWeight<Weight>> systolic_array_weights[cols];
  Connections::Combinational<Psum> systolic_array_psums[cols];
  Connections::Combinational<Psum> systolic_array_outputs[cols];

#if SUPPORT_MX
  Connections::In<ac_int<Scale::width, false>> CCS_INIT_S1(input_scale_channel);
  Connections::In<ac_int<Scale::width * cols, false>> CCS_INIT_S1(
      weight_scale_channel);
#endif

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  SC_CTOR(MatrixProcessor) {
    input_skewer.clk(clk);
    input_skewer.rstn(rstn);
    input_skewer.din(input_skewer_din);
    for (int i = 0; i < rows; i++) {
      input_skewer.dout[i](systolic_array_inputs[i]);
    }

    weight_skewer.clk(clk);
    weight_skewer.rstn(rstn);
    weight_skewer.din(weight_skewer_din);
    for (int i = 0; i < cols; i++) {
      weight_skewer.dout[i](systolic_array_weights[i]);
    }

    psum_out_skewer.clk(clk);
    psum_out_skewer.rstn(rstn);
    for (int i = 0; i < cols; i++) {
      psum_out_skewer.din[i](systolic_array_outputs[i]);
    }
    psum_out_skewer.dout(psum_out_skewer_dout);

    systolic_array.clk(clk);
    systolic_array.rstn(rstn);
    for (int i = 0; i < rows; i++) {
      systolic_array.inputs[i](systolic_array_inputs[i]);
    }
    for (int i = 0; i < cols; i++) {
      systolic_array.outputs[i](systolic_array_outputs[i]);
    }
    for (int i = 0; i < cols; i++) {
      systolic_array.weights[i](systolic_array_weights[i]);
    }

    accum_to_wb_fifo.clk(clk);
    accum_to_wb_fifo.rst(rstn);
    accum_to_wb_fifo.enq(accum_to_wb_enq);
    accum_to_wb_fifo.deq(accum_to_wb_deq);

    accum_output_fifo.clk(clk);
    accum_output_fifo.rst(rstn);
    accum_output_fifo.enq(accum_output_enq);
    accum_output_fifo.deq(output_channel);

    process_accumulation_params_fifo.clk(clk);
    process_accumulation_params_fifo.rst(rstn);
    process_accumulation_params_fifo.enq(process_accumulation_params_enq);
    process_accumulation_params_fifo.deq(process_accumulation_params_deq);

    write_back_params_fifo.clk(clk);
    write_back_params_fifo.rst(rstn);
    write_back_params_fifo.enq(write_back_params_enq);
    write_back_params_fifo.deq(write_back_params_deq);

    SC_THREAD(push_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(push_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_back);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void push_weights() {
    push_weights_params.ResetRead();
    weight_channel.Reset();
    weight_skewer_din.ResetWrite();

    wait();

    while (true) {
      MatrixParams params = push_weights_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weight_reuse_idx[0]] = 1;
      loop_bounds[1][params.weight_reuse_idx[1]] = 1;

      // extra loop to control reuse which only occurs during transpose and when
      // cols > rows
      int rep_bound = 1;

      if (params.weight_transpose && cols > rows) {
        if (loop_bounds[0][params.reduction_loop_idx[0]] >= (cols / rows)) {
          // we are able to reuse the weights already in the buffer
          loop_bounds[0][params.reduction_loop_idx[0]] /= (cols / rows);
          rep_bound = (cols / rows);
        }
      }

      auto FY1 = params.loops[0][params.fy_loop_idx[0]];
      auto C2 = params.loops[0][params.reduction_loop_idx[0]];
      auto FY0 = params.loops[1][params.fy_loop_idx[1]];
      auto FX = params.loops[1][params.fx_loop_idx];
      auto C1 = params.loops[1][params.reduction_loop_idx[1]];
      auto K1 = params.loops[1][params.weight_loop_idx[1]];

      bool x_inner = params.weight_loop_idx[0] < params.x_loop_idx[0];
      bool y_inner = params.weight_loop_idx[0] < params.y_loop_idx[0];

      bool reuse_weights = FY1 == 1 && C2 == 1 && FY0 == 1 && FX == 1 &&
                           C1 == 1 && K1 == 1 && (x_inner || y_inner);

      if (reuse_weights) {
        if (x_inner) {
          loop_bounds[0][params.x_loop_idx[0]] = 1;
        }
        if (y_inner) {
          loop_bounds[0][params.y_loop_idx[0]] = 1;
        }
      }

      ac_int<32, false> total_loops =
          loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
          loop_bounds[0][3] * loop_bounds[0][4] * loop_bounds[1][0] *
          loop_bounds[1][1] * loop_bounds[1][2] * loop_bounds[1][3] *
          loop_bounds[1][4] * loop_bounds[1][5] * rep_bound;

      ac_int<32, false> step = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step++ < total_loops) {
        for (int row = 0; row < rows; row++) {
          Pack1D<PEWeight<Weight>, cols> weights;
          auto data = weight_channel.Pop();

          constexpr int weight_width = WEIGHT_BUFFER_WIDTH / cols;

#pragma hls_unroll yes
          for (int i = 0; i < cols; i++) {
            Weight decoded_weight = Weight::zero();
            auto bits = data.template slc<weight_width>(i * weight_width);

#if SUPPORT_CODEBOOK_QUANT
            if (params.use_weight_codebook) {
              auto value = params.weight_code[bits];
              decoded_weight.set_bits(value);
            } else
#endif
            {
              bool success = (decode_type<WeightTypes, Weight, weight_width,
                                          WeightTypes...>(
                                  params.weight_dtype, bits, decoded_weight) ||
                              ...);
#ifndef __SYNTHESIS__
              if (!success) {
                std::cerr << "Error: matrix weight dtype '"
                          << params.weight_dtype << "' is not valid"
                          << std::endl;
              }
#endif
            }

            weights[i].data = decoded_weight;
            weights[i].tag = row;
          }

          weight_skewer_din.Push(weights);
        }
      }
    }
  }

  void push_inputs() {
    params_in.Reset();
    input_channel.Reset();
    push_weights_params.ResetWrite();
    process_accumulation_params_enq.ResetWrite();
    write_back_params_enq.ResetWrite();
    input_skewer_din.ResetWrite();
    start.Reset();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();
      push_weights_params.Push(params);
      process_accumulation_params_enq.Push(params);
      write_back_params_enq.Push(params);

      start.SyncPush();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_counters[i][j] = 0;
        }
      }

      // We can reuse weights and avoid reloading them if FY, FX, C1, and K1
      // are all 1 (e.g. 1x1 pointwise conv with C and K equal to IC and OC)
      // and either OX or OY is the innermost L2 loop
      auto FY1 = params.loops[0][params.fy_loop_idx[0]];
      auto C2 = params.loops[0][params.reduction_loop_idx[0]];
      auto FY0 = params.loops[1][params.fy_loop_idx[1]];
      auto FX = params.loops[1][params.fx_loop_idx];
      auto C1 = params.loops[1][params.reduction_loop_idx[1]];
      auto K1 = params.loops[1][params.weight_loop_idx[1]];

      bool x_inner = params.weight_loop_idx[0] < params.x_loop_idx[0];
      bool y_inner = params.weight_loop_idx[0] < params.y_loop_idx[0];

      bool reuse_weights = FY1 == 1 && C2 == 1 && FY0 == 1 && FX == 1 &&
                           C1 == 1 && K1 == 1 && (x_inner || y_inner);

      ac_int<3, false> weight_reuse_idx[2];
      if (x_inner && y_inner) {
        weight_reuse_idx[0] = params.x_loop_idx[0];
        weight_reuse_idx[1] = params.y_loop_idx[0];
      } else {
        auto idx = x_inner ? params.x_loop_idx[0] : params.y_loop_idx[0];
        weight_reuse_idx[0] = idx;
        weight_reuse_idx[1] = idx;
      }

      ac_int<32, false> total_ops =
          params.loops[0][0] * params.loops[0][1] * params.loops[0][2] *
          params.loops[0][3] * params.loops[0][4] * params.loops[1][0] *
          params.loops[1][1] * params.loops[1][2] * params.loops[1][3] *
          params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < total_ops) {
#ifndef __SYNTHESIS__
        if (step % 1000 == 0 and step > 0) {
          CCS_LOG("step " << step << " out of " << total_ops);
        }
#endif

        Pack1D<PEInput<Input>, rows> inputs;
        auto data = input_channel.Pop();

        bool swap_weights_inner =
            loop_counters[1][params.weight_reuse_idx[0]] == 0 &&
            loop_counters[1][params.weight_reuse_idx[1]] == 0;
        bool swap_weights_outer = loop_counters[0][weight_reuse_idx[0]] == 0 &&
                                  loop_counters[0][weight_reuse_idx[1]] == 0;
        bool swap_weights =
            step == 0 ||
            (swap_weights_inner && (!reuse_weights || swap_weights_outer));

        constexpr int input_width = INPUT_BUFFER_WIDTH / rows;

#pragma hls_unroll yes
        for (int i = 0; i < rows; i++) {
          Input decoded_input = Input::zero();
          auto bits = data.template slc<input_width>(i * input_width);

#if SUPPORT_CODEBOOK_QUANT
          if (params.use_input_codebook) {
            auto value = params.input_code[bits];
            decoded_input.set_bits(value);
          } else
#endif
          {
            bool success =
                (decode_type<InputTypes, Input, input_width, InputTypes...>(
                     params.input_dtype, bits, decoded_input) ||
                 ...);
#ifndef __SYNTHESIS__
            if (!success) {
              std::cerr << "Error: matrix input dtype '" << params.input_dtype
                        << "' is not valid" << std::endl;
            }
#endif
          }

          inputs[i].data = decoded_input;
          inputs[i].swap_weights = swap_weights;
        }

        input_skewer_din.Push(inputs);

        step++;
        loop_counters[1][5]++;
#pragma hls_unroll yes
        for (int i = 1; i >= 0; i--) {
#pragma hls_unroll yes
          for (int j = 5; j >= 0; j--) {
            if (loop_counters[i][j] == params.loops[i][j]) {
              loop_counters[i][j] = 0;
              if (j > 0) {
                loop_counters[i][j - 1]++;
              } else {
                if (i > 0) {
                  loop_counters[i - 1][5]++;
                }
              }
            }
          }
        }
      }
    }
  }

  void process_accumulation() {
    process_accumulation_params_deq.ResetRead();
    psum_out_skewer_dout.ResetRead();
#if SUPPORT_MX
    input_scale_channel.Reset();
    weight_scale_channel.Reset();
#endif
    bias_channel.Reset();
    accumulation_buffer_read_data[0].Reset();
    accumulation_buffer_read_address[0].Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_read_data[1].Reset();
    accumulation_buffer_read_address[1].Reset();
#endif
    accum_to_wb_enq.ResetWrite();

    bool accumulation_buffer_bank = 0;

    wait();

    while (true) {
      const MatrixParams params = process_accumulation_params_deq.Pop();
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_counters[i][j] = 0;
        }
      }

      ac_int<32, false> total_ops =
          params.loops[0][0] * params.loops[0][1] * params.loops[0][2] *
          params.loops[0][3] * params.loops[0][4] * params.loops[1][0] *
          params.loops[1][1] * params.loops[1][2] * params.loops[1][3] *
          params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;

      ac_int<LOOP_WIDTH, false> c2_bound =
          params.loops[0][params.reduction_loop_idx[0]] - 1;
      ac_int<LOOP_WIDTH, false> c1_bound =
          params.loops[1][params.reduction_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> k1_bound =
          params.loops[1][params.weight_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> fx_bound =
          params.loops[1][params.fx_loop_idx] - 1;
      ac_int<LOOP_WIDTH, false> fy0_bound =
          params.loops[0][params.fy_loop_idx[0]] - 1;
      ac_int<LOOP_WIDTH, false> fy1_bound =
          params.loops[1][params.fy_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> y0_bound =
          params.loops[1][params.y_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> x0_bound =
          params.loops[1][params.x_loop_idx[1]] - 1;

      ac_int<LOOP_WIDTH, false> Y0 = params.loops[1][params.y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.x_loop_idx[1]];
      ac_int<16, false> k_stride = Y0 * X0;

      auto FY1 = params.loops[0][params.fy_loop_idx[0]];
      auto C2 = params.loops[0][params.reduction_loop_idx[0]];
      auto FY0 = params.loops[1][params.fy_loop_idx[1]];
      auto FX = params.loops[1][params.fx_loop_idx];
      auto C1 = params.loops[1][params.reduction_loop_idx[1]];
      auto K1 = params.loops[1][params.weight_loop_idx[1]];

      bool x_inner = params.weight_loop_idx[0] < params.x_loop_idx[0];
      bool y_inner = params.weight_loop_idx[0] < params.y_loop_idx[0];

      bool reuse_weights = FY1 == 1 && C2 == 1 && FY0 == 1 && FX == 1 &&
                           C1 == 1 && K1 == 1 && (x_inner || y_inner);

      ac_int<3, false> weight_reuse_idx[2];
      if (x_inner && y_inner) {
        weight_reuse_idx[0] = params.x_loop_idx[0];
        weight_reuse_idx[1] = params.y_loop_idx[0];
      } else {
        auto idx = x_inner ? params.x_loop_idx[0] : params.y_loop_idx[0];
        weight_reuse_idx[0] = idx;
        weight_reuse_idx[1] = idx;
      }

      Pack1D<Buffer, cols> bias;
      Pack1D<Scale, cols> weight_scales;

#pragma hls_unroll yes
      for (int i = 0; i < cols; i++) {
        weight_scales[i] = Scale::one();
      }

      // loop indices that are used to determine when to read in a new bias
      int bias_reuse_indices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weight_loop_idx[1]; i--) {
        bias_reuse_indices[5 - i] = i;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < total_ops) {
        Pack1D<Psum, cols> outputs = psum_out_skewer_dout.Pop();

#if SUPPORT_MX
        Scale input_scale = Scale::one();
        if (params.is_mx_op) {
          input_scale.set_bits(input_scale_channel.Pop());
        }

        bool swap_weights_inner =
            loop_counters[1][params.weight_reuse_idx[0]] == 0 &&
            loop_counters[1][params.weight_reuse_idx[1]] == 0;
        bool swap_weights_outer = loop_counters[0][weight_reuse_idx[0]] == 0 &&
                                  loop_counters[0][weight_reuse_idx[1]] == 0;
        bool swap_weights =
            step == 0 ||
            (swap_weights_inner && (!reuse_weights || swap_weights_outer));

        if (params.is_mx_op && swap_weights) {
          auto bits = weight_scale_channel.Pop();
          weight_scales = BitsToType<Pack1D<Scale, cols>>(TypeToBits(bits));
        }
#endif

        bool is_non_accumulating_tile =
            loop_counters[0][params.reduction_loop_idx[0]] == 0 &&
            loop_counters[1][params.reduction_loop_idx[1]] == 0 &&
            loop_counters[1][params.fx_loop_idx] == 0 &&
            loop_counters[0][params.fy_loop_idx[0]] == 0 &&
            loop_counters[1][params.fy_loop_idx[1]] == 0;

        Pack1D<Buffer, cols> previous_accumulation;

#pragma hls_unroll yes
        for (int i = 0; i < cols; i++) {
          previous_accumulation[i] = Buffer::zero();
        }

        if (is_non_accumulating_tile) {
          if (params.has_bias) {
            bool read_bias = loop_counters[1][bias_reuse_indices[0]] == 0 &&
                             loop_counters[1][bias_reuse_indices[1]] == 0 &&
                             loop_counters[1][bias_reuse_indices[2]] == 0 &&
                             loop_counters[1][bias_reuse_indices[3]] == 0;

            if (read_bias) {
              bias = bias_channel.Pop();
            }

            previous_accumulation = bias;
          }
        } else {
          ac_int<16, false> address =
              loop_counters[1][params.weight_loop_idx[1]] * k_stride +
              loop_counters[1][params.y_loop_idx[1]] * X0 +
              loop_counters[1][params.x_loop_idx[1]];

          accumulation_buffer_read_address[accumulation_buffer_bank].Push(
              address);
          previous_accumulation =
              accumulation_buffer_read_data[accumulation_buffer_bank].Pop();
        }

#if SUPPORT_MX
#pragma hls_unroll yes
        for (int i = 0; i < cols; i++) {
          previous_accumulation[i] =
              dequantize_mx_op(outputs[i], input_scale, weight_scales[i],
                               previous_accumulation[i]);
        }
#else
#pragma hls_unroll yes
        for (int i = 0; i < cols; i++) {
          previous_accumulation[i] += static_cast<Buffer>(outputs[i]);
        }
#endif
        accum_to_wb_enq.Push(previous_accumulation);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
        if (params.write_output_to_accum_buffer) {
          bool output_tile_completed =
              (loop_counters[0][params.reduction_loop_idx[0]] == c2_bound) &&
              (loop_counters[1][params.reduction_loop_idx[1]] == c1_bound) &&
              (loop_counters[1][params.weight_loop_idx[1]] == k1_bound) &&
              (loop_counters[1][params.fx_loop_idx] == fx_bound) &&
              (loop_counters[0][params.fy_loop_idx[0]] == fy0_bound) &&
              (loop_counters[1][params.fy_loop_idx[1]] == fy1_bound) &&
              (loop_counters[1][params.x_loop_idx[1]] == x0_bound) &&
              (loop_counters[1][params.y_loop_idx[1]] == y0_bound);

          if (output_tile_completed) {
            accumulation_buffer_bank = !accumulation_buffer_bank;
          }
        }
#endif
        step++;
        loop_counters[1][5]++;
#pragma hls_unroll yes
        for (int i = 1; i >= 0; i--) {
#pragma hls_unroll yes
          for (int j = 5; j >= 0; j--) {
            if (loop_counters[i][j] == params.loops[i][j]) {
              loop_counters[i][j] = 0;
              if (j > 0) {
                loop_counters[i][j - 1]++;
              } else {
                if (i > 0) {
                  loop_counters[i - 1][5]++;
                }
              }
            }
          }
        }
      }
    }
  }

  void write_back() {
    write_back_params_deq.ResetRead();
    accum_to_wb_deq.ResetRead();
    accumulation_buffer_write_request[0].Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_write_request[1].Reset();
    accumulation_buffer_done[0].Reset();
    accumulation_buffer_done[1].Reset();
#endif
    accum_output_enq.ResetWrite();
    done.Reset();

    bool accumulation_buffer_bank = 0;

    wait();

    while (true) {
      const MatrixParams params = write_back_params_deq.Pop();
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_counters[i][j] = 0;
        }
      }

      ac_int<32, false> total_ops =
          params.loops[0][0] * params.loops[0][1] * params.loops[0][2] *
          params.loops[0][3] * params.loops[0][4] * params.loops[1][0] *
          params.loops[1][1] * params.loops[1][2] * params.loops[1][3] *
          params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;

      ac_int<LOOP_WIDTH, false> c2_bound =
          params.loops[0][params.reduction_loop_idx[0]] - 1;
      ac_int<LOOP_WIDTH, false> c1_bound =
          params.loops[1][params.reduction_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> k1_bound =
          params.loops[1][params.weight_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> fx_bound =
          params.loops[1][params.fx_loop_idx] - 1;
      ac_int<LOOP_WIDTH, false> fy0_bound =
          params.loops[0][params.fy_loop_idx[0]] - 1;
      ac_int<LOOP_WIDTH, false> fy1_bound =
          params.loops[1][params.fy_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> y0_bound =
          params.loops[1][params.y_loop_idx[1]] - 1;
      ac_int<LOOP_WIDTH, false> x0_bound =
          params.loops[1][params.x_loop_idx[1]] - 1;

      ac_int<LOOP_WIDTH, false> Y0 = params.loops[1][params.y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 = params.loops[1][params.x_loop_idx[1]];
      ac_int<16, false> k_stride = Y0 * X0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < total_ops) {
        Pack1D<Buffer, cols> previous_accumulation = accum_to_wb_deq.Pop();

        bool accumulation_finished =
            (loop_counters[0][params.reduction_loop_idx[0]] == c2_bound) &&
            (loop_counters[1][params.reduction_loop_idx[1]] == c1_bound) &&
            (loop_counters[1][params.fx_loop_idx] == fx_bound) &&
            (loop_counters[0][params.fy_loop_idx[0]] == fy0_bound) &&
            (loop_counters[1][params.fy_loop_idx[1]] == fy1_bound);

        if (accumulation_finished && !(DOUBLE_BUFFERED_ACCUM_BUFFER &&
                                       params.write_output_to_accum_buffer)) {
          // write out to vector unit directly
          accum_output_enq.Push(previous_accumulation);
        } else {
          ac_int<16, false> address =
              loop_counters[1][params.weight_loop_idx[1]] * k_stride +
              loop_counters[1][params.y_loop_idx[1]] * X0 +
              loop_counters[1][params.x_loop_idx[1]];

          BufferWriteRequest<Pack1D<Buffer, cols>> req;
          req.address = address;
          req.data = previous_accumulation;
          accumulation_buffer_write_request[accumulation_buffer_bank].Push(req);
        }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
        if (params.write_output_to_accum_buffer) {
          bool output_tile_completed =
              (loop_counters[0][params.reduction_loop_idx[0]] == c2_bound) &&
              (loop_counters[1][params.reduction_loop_idx[1]] == c1_bound) &&
              (loop_counters[1][params.weight_loop_idx[1]] == k1_bound) &&
              (loop_counters[1][params.fx_loop_idx] == fx_bound) &&
              (loop_counters[0][params.fy_loop_idx[0]] == fy0_bound) &&
              (loop_counters[1][params.fy_loop_idx[1]] == fy1_bound) &&
              (loop_counters[1][params.x_loop_idx[1]] == x0_bound) &&
              (loop_counters[1][params.y_loop_idx[1]] == y0_bound);

          if (output_tile_completed) {
            accumulation_buffer_done[accumulation_buffer_bank].SyncPush();
            accumulation_buffer_bank = !accumulation_buffer_bank;
          }
        }
#endif

        step++;
        loop_counters[1][5]++;
#pragma hls_unroll yes
        for (int i = 1; i >= 0; i--) {
#pragma hls_unroll yes
          for (int j = 5; j >= 0; j--) {
            if (loop_counters[i][j] == params.loops[i][j]) {
              loop_counters[i][j] = 0;
              if (j > 0) {
                loop_counters[i][j - 1]++;
              } else {
                if (i > 0) {
                  loop_counters[i - 1][5]++;
                }
              }
            }
          }
        }
      }
      done.SyncPush();
    }
  }
};
