#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"
#include "Skewer.h"
#include "SystolicArray.h"

template <typename InputTypeTuple, typename WeightTypeTuple, typename Input,
          typename Weight, typename Psum, typename Buffer, typename Scale,
          int NRows, int NCols, int BufferSize>
struct MatrixProcessor;

template <typename... InputTypes, typename... WeightTypes, typename Input,
          typename Weight, typename Psum, typename Buffer, typename Scale,
          int NRows, int NCols, int BufferSize>
struct MatrixProcessor<std::tuple<InputTypes...>, std::tuple<WeightTypes...>,
                       Input, Weight, Psum, Buffer, Scale, NRows, NCols,
                       BufferSize> : public sc_module {
 private:
  InputSerializedSkewer<Input, NRows> CCS_INIT_S1(inputSkewer);
  Connections::Combinational<Pack1D<PEInput<Input>, NRows>> CCS_INIT_S1(
      inputSkewerDin);

  WeightSerializedSkewer<Weight, NCols> CCS_INIT_S1(weightSkewer);
  Connections::Combinational<Pack1D<PEWeight<Weight>, NCols>> CCS_INIT_S1(
      weightSkewerDin);

  DeserializedSkewer<Psum, NCols> CCS_INIT_S1(psumOutSkewer);
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(
      psumOutSkewerDout);

  SystolicArray<Input, Weight, Psum, NRows, NCols> CCS_INIT_S1(systolicArray);

  MatrixParamsDeserializer<1> CCS_INIT_S1(paramsDeserializer);

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int LOOP_WIDTH = 10;

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  static constexpr int ACCUM_BUFFER_BANKS = 2;
#else
  static constexpr int ACCUM_BUFFER_BANKS = 1;
#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(inputsChannel);
  Connections::In<ac_int<WEIGHT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      weightsChannel);
  Connections::In<Pack1D<Buffer, NCols>> CCS_INIT_S1(biasChannel);

  Connections::Out<Pack1D<Buffer, NCols>> CCS_INIT_S1(matrixUnitOutputChannel);

  Connections::Out<ac_int<16, false>>
      accumulation_buffer_read_address[ACCUM_BUFFER_BANKS];
  Connections::In<Pack1D<Buffer, NCols>>
      accumulation_buffer_read_data[ACCUM_BUFFER_BANKS];
  Connections::Out<BufferWriteRequest<Pack1D<Buffer, NCols>>>
      accumulation_buffer_write_request[ACCUM_BUFFER_BANKS];

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::SyncOut accumulation_buffer_done[ACCUM_BUFFER_BANKS];
#endif

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialParamsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(push_weights_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      accumulation_buffer_params);

  Connections::Combinational<PEInput<Input>> inputsToSystolicArray[NRows];
  Connections::Combinational<PEWeight<Weight>> weightsToSystolicArray[NCols];
  Connections::Combinational<Psum> psumsToSystolicArray[NCols];
  Connections::Combinational<Psum> outputsFromSystolicArray[NCols];

#if SUPPORT_MX
  Connections::In<ac_int<Scale::width, false>> CCS_INIT_S1(inputScaleChannel);
  Connections::In<ac_int<Scale::width * NCols, false>> CCS_INIT_S1(
      weightScaleChannel);
#endif

  Connections::SyncOut CCS_INIT_S1(startSignal);
  Connections::SyncOut CCS_INIT_S1(doneSignal);

  SC_CTOR(MatrixProcessor) {
    paramsDeserializer.clk(clk);
    paramsDeserializer.rstn(rstn);
    paramsDeserializer.serialParamsIn(serialParamsIn);
    paramsDeserializer.paramsOut(paramsIn);

    inputSkewer.clk(clk);
    inputSkewer.rstn(rstn);
    inputSkewer.din(inputSkewerDin);
    for (int i = 0; i < NRows; i++) {
      inputSkewer.dout[i](inputsToSystolicArray[i]);
    }

    weightSkewer.clk(clk);
    weightSkewer.rstn(rstn);
    weightSkewer.din(weightSkewerDin);
    for (int i = 0; i < NCols; i++) {
      weightSkewer.dout[i](weightsToSystolicArray[i]);
    }

    psumOutSkewer.clk(clk);
    psumOutSkewer.rstn(rstn);
    for (int i = 0; i < NCols; i++) {
      psumOutSkewer.din[i](outputsFromSystolicArray[i]);
    }
    psumOutSkewer.dout(psumOutSkewerDout);

    systolicArray.clk(clk);
    systolicArray.rstn(rstn);
    for (int i = 0; i < NRows; i++) {
      systolicArray.inputs[i](inputsToSystolicArray[i]);
    }
    for (int i = 0; i < NCols; i++) {
      systolicArray.outputs[i](outputsFromSystolicArray[i]);
    }
    for (int i = 0; i < NCols; i++) {
      systolicArray.weights[i](weightsToSystolicArray[i]);
    }

    SC_THREAD(push_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(push_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void push_weights() {
    push_weights_params.ResetRead();
    weightsChannel.Reset();
    weightSkewerDin.ResetWrite();

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
      loop_bounds[1][params.weightReuseIndex[0]] = 1;
      loop_bounds[1][params.weightReuseIndex[1]] = 1;

      // extra loop to control reuse which only occurs during transpose and when
      // NCols > NRows
      int rep_bound = 1;

      if (params.has_weight_transpose && NCols > NRows) {
        if (loop_bounds[0][params.reductionLoopIndex[0]] >= (NCols / NRows)) {
          // we are able to reuse the weights already in the buffer
          loop_bounds[0][params.reductionLoopIndex[0]] /= (NCols / NRows);
          rep_bound = (NCols / NRows);
        }
      }

      // extra loop for exploiting L1 buffer reuse.
      // this loop is used when OX and OY are the innermost L2 loops. when this
      // occurs, we can move OX and/or OY into the buffer reuse L1 loop
      int buffer_reuse = 1;
      if (params.loops[0][params.reductionLoopIndex[0]] == 1) {
        // OX loop can be absorbed
        if (params.weightLoopIndex[0] < params.inputXLoopIndex[0]) {
          buffer_reuse *= loop_bounds[0][params.inputXLoopIndex[0]];
          loop_bounds[0][params.inputXLoopIndex[0]] = 1;
        }
        // OY loop can be absorbed
        if (params.weightLoopIndex[0] < params.inputYLoopIndex[0]) {
          buffer_reuse *= loop_bounds[0][params.inputYLoopIndex[0]];
          loop_bounds[0][params.inputYLoopIndex[0]] = 1;
        }
      }

      ac_int<32, false> total_loops =
          loop_bounds[0][0] * loop_bounds[0][1] * loop_bounds[0][2] *
          loop_bounds[0][3] * loop_bounds[1][0] * loop_bounds[1][1] *
          loop_bounds[1][2] * loop_bounds[1][3] * loop_bounds[1][4] *
          loop_bounds[1][5] * buffer_reuse * rep_bound;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < total_loops; i++) {
        for (int weight_count = 0; weight_count < NRows; weight_count++) {
          Pack1D<PEWeight<Weight>, NCols> weights;
          auto bits = weightsChannel.Pop();

#pragma hls_unroll yes
          for (int i = 0; i < NCols; i++) {
            auto data =
                bits.template slc<WEIGHT_DTYPE_WIDTH>(i * WEIGHT_DTYPE_WIDTH);

#if SUPPORT_CODEBOOK_QUANT
            if (params.use_weight_codebook) {
              auto value = params.weight_code[data];
              weights[i].data.set_bits(value);
            } else
#endif
            {
              bool success = (decode_type<WeightTypes, Weight,
                                          WEIGHT_DTYPE_WIDTH, WeightTypes...>(
                                  params.weight_dtype, data, weights[i].data) ||
                              ...);
#ifndef __SYNTHESIS__
              if (!success) {
                std::cerr << "Error: matrix weight dtype '"
                          << params.weight_dtype << "' is not valid"
                          << std::endl;
              }
#endif
            }
            weights[i].tag = weight_count;
          }

          weightSkewerDin.Push(weights);
        }
      }
    }
  }

  int max3(int a, int b, int c) {
    if (a > b) {
      return a > c ? a : c;
    } else {
      return b > c ? b : c;
    }
  }

  void push_inputs() {
    paramsIn.ResetRead();
    inputsChannel.Reset();
    psumOutSkewerDout.ResetRead();
    push_weights_params.ResetWrite();
    accumulation_buffer_params.ResetWrite();
    inputSkewerDin.ResetWrite();

    startSignal.Reset();

    wait();

    while (true) {
      const MatrixParams params = paramsIn.Pop();
      push_weights_params.Push(params);
      accumulation_buffer_params.Push(params);

      startSignal.SyncPush();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_counters_out[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_counters[i][j] = 0;
          loop_counters_out[i][j] = 0;
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      ac_int<32, false> total_ops = params.loops[0][0] * params.loops[0][1] *
                                    params.loops[0][2] * params.loops[0][3] *
                                    params.loops[1][0] * params.loops[1][1] *
                                    params.loops[1][2] * params.loops[1][3] *
                                    params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;

      // Push inputs into the array
      // Pipelined across tiles
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < total_ops) {
#ifndef __SYNTHESIS__
        if (step % 1000 == 0 and step > 0) {
          CCS_LOG("step " << step << " out of " << total_ops);
        }
#endif

        Pack1D<PEInput<Input>, NRows> inputs;

        bool swap_weights;
        if (params.weightReuseIndex[0] != params.weightReuseIndex[1]) {
          swap_weights = (loop_counters[1][params.weightReuseIndex[1]] == 0) &&
                         (loop_counters[1][params.weightReuseIndex[0]] == 0);
        } else {
          swap_weights = (loop_counters[1][params.weightReuseIndex[1]] == 0);
        }

#pragma hls_unroll yes
        for (int i = 0; i < NRows; i++) {
          inputs[i].swapWeights =
              (swap_weights || step == 0) && step < total_ops - 1;
        }

        auto bits = inputsChannel.Pop();

#pragma hls_unroll yes
        for (int i = 0; i < NRows; i++) {
          auto data =
              bits.template slc<INPUT_DTYPE_WIDTH>(i * INPUT_DTYPE_WIDTH);
#if SUPPORT_CODEBOOK_QUANT
          if (params.use_input_codebook) {
            auto value = params.input_code[data];
            inputs[i].data.set_bits(value);
          } else
#endif
          {
            bool success = (decode_type<InputTypes, Input, INPUT_DTYPE_WIDTH,
                                        InputTypes...>(params.input_dtype, data,
                                                       inputs[i].data) ||
                            ...);
#ifndef __SYNTHESIS__
            if (!success) {
              std::cerr << "Error: matrix input dtype '" << params.input_dtype
                        << "' is not valid" << std::endl;
            }
#endif
          }
        }

        inputSkewerDin.Push(inputs);

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
    biasChannel.Reset();
    accumulation_buffer_params.ResetRead();
    psumOutSkewerDout.ResetRead();
    matrixUnitOutputChannel.Reset();

#if SUPPORT_MX
    inputScaleChannel.Reset();
    weightScaleChannel.Reset();
#endif

    accumulation_buffer_read_data[0].Reset();
    accumulation_buffer_read_address[0].Reset();
    accumulation_buffer_write_request[0].Reset();

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_read_data[1].Reset();
    accumulation_buffer_read_address[1].Reset();
    accumulation_buffer_write_request[1].Reset();

    accumulation_buffer_done[0].Reset();
    accumulation_buffer_done[1].Reset();
#endif
    doneSignal.Reset();

    bool accumulation_buffer_bank = 0;

    wait();

    while (true) {
      const MatrixParams params = accumulation_buffer_params.Pop();
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];
#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_counters[i][j] = 0;
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      ac_int<32, false> total_ops = params.loops[0][0] * params.loops[0][1] *
                                    params.loops[0][2] * params.loops[0][3] *
                                    params.loops[1][0] * params.loops[1][1] *
                                    params.loops[1][2] * params.loops[1][3] *
                                    params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;

      Pack1D<Buffer, NCols> bias;

#if SUPPORT_MX
      Pack1D<Scale, NCols> weight_scales;
#endif

      // loop indices that are used to determine when to read in a new bias
      int bias_reuse_indices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weightLoopIndex[1]; i--) {
        bias_reuse_indices[5 - i] = i;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < total_ops) {
        bool is_non_accumulating_tile =
            loop_counters[0][params.reductionLoopIndex[0]] == 0 &&
            loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
            loop_counters[1][params.fxIndex] == 0 &&
            loop_counters[1][params.fyIndex] == 0;

        Pack1D<Buffer, NCols> previous_accumulation;

#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          previous_accumulation.value[i] = Buffer::zero();
        }

        if (is_non_accumulating_tile) {
          if (params.has_bias) {
            bool read_bias = loop_counters[1][bias_reuse_indices[0]] == 0 &&
                             loop_counters[1][bias_reuse_indices[1]] == 0 &&
                             loop_counters[1][bias_reuse_indices[2]] == 0 &&
                             loop_counters[1][bias_reuse_indices[3]] == 0;

            if (read_bias) {
              bias = biasChannel.Pop();
            }

            previous_accumulation = bias;
          }
        } else {
          ac_int<int_log2(BufferSize), false> readAddress =
              loop_counters[1][params.weightLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] *
                  params.loops[1][params.inputYLoopIndex[1]] +
              loop_counters[1][params.inputYLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] +
              loop_counters[1][params.inputXLoopIndex[1]];

          accumulation_buffer_read_address[accumulation_buffer_bank].Push(
              readAddress);

          previous_accumulation =
              accumulation_buffer_read_data[accumulation_buffer_bank].Pop();
        }

        Pack1D<Psum, NCols> outputs = psumOutSkewerDout.Pop();

#if SUPPORT_MX
        Scale input_scale;
        if (params.is_mx_op) {
          input_scale.set_bits(inputScaleChannel.Pop());
        }

        bool swap_weights;
        if (params.weightReuseIndex[0] != params.weightReuseIndex[1]) {
          swap_weights = (loop_counters[1][params.weightReuseIndex[1]] == 0) &&
                         (loop_counters[1][params.weightReuseIndex[0]] == 0);
        } else {
          swap_weights = (loop_counters[1][params.weightReuseIndex[1]] == 0);
        }

        swap_weights = swap_weights || step == 0;

        if (swap_weights && params.is_mx_op) {
          auto bits = weightScaleChannel.Pop();
          weight_scales = BitsToType<Pack1D<Scale, NCols>>(TypeToBits(bits));
        }
#endif

        Pack1D<Buffer, NCols> scaled_outputs;

#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
#if SUPPORT_MX
          Buffer scale = static_cast<Buffer>(input_scale) *
                         static_cast<Buffer>(weight_scales[i]);
          scaled_outputs[i] = static_cast<Buffer>(outputs[i]);
          if (params.is_mx_op) {
            scaled_outputs[i] = scaled_outputs[i] * scale;
          }
#else
          scaled_outputs[i] = outputs[i];
#endif
          previous_accumulation.value[i] += scaled_outputs[i];
        }

        bool accumulation_finished =
            (loop_counters[0][params.reductionLoopIndex[0]] ==
             params.loops[0][params.reductionLoopIndex[0]] - 1) &&
            (loop_counters[1][params.reductionLoopIndex[1]] ==
             params.loops[1][params.reductionLoopIndex[1]] - 1) &&
            (loop_counters[1][params.fxIndex] ==
             params.loops[1][params.fxIndex] - 1) &&
            (loop_counters[1][params.fyIndex] ==
             params.loops[1][params.fyIndex] - 1);

        if ((accumulation_finished && !DOUBLE_BUFFERED_ACCUM_BUFFER) ||
            (accumulation_finished && DOUBLE_BUFFERED_ACCUM_BUFFER &&
             !params.write_output_to_accum_buffer)) {
          // write out to vector unit directly
          matrixUnitOutputChannel.Push(previous_accumulation);
        } else {
          int write_address = static_cast<ac_int<10, false>>(
                                  loop_counters[1][params.weightLoopIndex[1]] *
                                  params.loops[1][params.inputXLoopIndex[1]] *
                                  params.loops[1][params.inputYLoopIndex[1]]) +
                              static_cast<ac_int<10, false>>(
                                  loop_counters[1][params.inputYLoopIndex[1]] *
                                  params.loops[1][params.inputXLoopIndex[1]]) +
                              loop_counters[1][params.inputXLoopIndex[1]];

          BufferWriteRequest<Pack1D<Buffer, NCols>> req;
          req.address = write_address;
          req.data = previous_accumulation;
          accumulation_buffer_write_request[accumulation_buffer_bank].Push(req);
        }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
        if (params.write_output_to_accum_buffer) {
          bool output_tile_completed =
              (loop_counters[0][params.reductionLoopIndex[0]] ==
               params.loops[0][params.reductionLoopIndex[0]] - 1) &&
              (loop_counters[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters[1][params.weightLoopIndex[1]] ==
               params.loops[1][params.weightLoopIndex[1]] - 1) &&
              (loop_counters[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1) &&
              (loop_counters[1][params.inputXLoopIndex[1]] ==
               params.loops[1][params.inputXLoopIndex[1]] - 1) &&
              (loop_counters[1][params.inputYLoopIndex[1]] ==
               params.loops[1][params.inputYLoopIndex[1]] - 1);

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
      doneSignal.SyncPush();
    }
  }
};
