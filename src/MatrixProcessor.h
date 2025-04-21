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
  Connections::Combinational<Pack1D<PEInput<Input>, NRows>> CCS_INIT_S1(
      inputsToSkewer);
  Connections::Combinational<Pack1D<PEInput<Input>, NRows>> CCS_INIT_S1(
      inputsToSkewer_delayed);

  WeightSerializedSkewer<Weight, NCols> CCS_INIT_S1(weightSkewer);
  Connections::Combinational<Pack1D<PEWeight<Weight>, NCols>> CCS_INIT_S1(
      weightSkewerDin);

  SerializedSkewer<Psum, NCols> CCS_INIT_S1(psumInSkewer);
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(psumInSkewerDin);

  DeserializedSkewer<Psum, NCols> CCS_INIT_S1(psumOutSkewer);
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(
      psumOutSkewerDout);

  SystolicArray<Input, Weight, Psum, NRows, NCols> CCS_INIT_S1(systolicArray);

  MatrixParamsDeserializer<1> CCS_INIT_S1(paramsDeserializer);

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int LOOP_WIDTH = 10;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(inputsChannel);
  Connections::In<ac_int<WEIGHT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      weightsChannel);
  Connections::In<Pack1D<Buffer, NCols>> CCS_INIT_S1(biasChannel);

  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::In<Pack1D<Buffer, NCols>> accumulation_buffer_read_data[2];
  Connections::Out<BufferWriteRequest<Pack1D<Buffer, NCols>>>
      accumulation_buffer_write_request[2];
  Connections::SyncOut accumulation_buffer_done[2];

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialParamsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(sa_controller_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(weight_buffer_params);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      accumulation_buffer_params);

  Connections::Combinational<PEInput<Input>> inputsToSystolicArray[NRows];
  Connections::Combinational<PEWeight<Weight>> weightsToSystolicArray[NCols];
  Connections::Combinational<Psum> psumsToSystolicArray[NCols];
  Connections::Combinational<Psum> outputsFromSystolicArray[NCols];

#if SUPPORT_MX
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(
      unscaledAccumulationChannel);

  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(
      unscaledAccumulationChannel_delayed);

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

    psumInSkewer.clk(clk);
    psumInSkewer.rstn(rstn);
    psumInSkewer.din(psumInSkewerDin);
    for (int i = 0; i < NCols; i++) {
      psumInSkewer.dout[i](psumsToSystolicArray[i]);
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
      systolicArray.psums[i](psumsToSystolicArray[i]);
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

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#if !SUPPORT_MX
    SC_THREAD(push_psum);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(delay_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif

#if SUPPORT_MX
#ifdef __SYNTHESIS__
    SC_THREAD(delay_outputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif

    SC_THREAD(process_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif

    SC_THREAD(send_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void push_weights() {
    weight_buffer_params.ResetRead();
    weightsChannel.Reset();
    weightSkewerDin.ResetWrite();

    wait();

    while (true) {
      MatrixParams params = weight_buffer_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
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

#if !SUPPORT_MX
  void push_psum() {
    biasChannel.Reset();
    inputSkewerDin.ResetWrite();
    psumInSkewerDin.ResetWrite();
    accumulation_buffer_params.ResetRead();
    inputsToSkewer_delayed.ResetRead();

    accumulation_buffer_read_data[0].Reset();
    accumulation_buffer_read_data[1].Reset();

    bool accumulation_buffer_bank = 0;

    wait();
    while (true) {
      const MatrixParams params = accumulation_buffer_params.Pop();
      ac_int<32, false> total_ops = params.loops[0][0] * params.loops[0][1] *
                                    params.loops[0][2] * params.loops[0][3] *
                                    params.loops[1][0] * params.loops[1][1] *
                                    params.loops[1][2] * params.loops[1][3] *
                                    params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_counters[i][j] = 0;
        }
      }

      Pack1D<Psum, NCols> bias;

      // loop indices that are used to determine when to read in a new bias
      int bias_reuse_indices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weightLoopIndex[1]; i--) {
        bias_reuse_indices[5 - i] = i;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < total_ops) {
        Pack1D<PEInput<Input>, NRows> inputs = inputsToSkewer_delayed.Pop();

        Pack1D<Psum, NCols> psum;
#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          psum.value[i] = Psum::zero();
        }

        bool is_accumulating =
            loop_counters[0][params.reductionLoopIndex[0]] != 0 ||
            loop_counters[1][params.reductionLoopIndex[1]] != 0 ||
            loop_counters[1][params.fxIndex] != 0 ||
            loop_counters[1][params.fyIndex] != 0;

        if (is_accumulating) {
          psum = accumulation_buffer_read_data[accumulation_buffer_bank].Pop();
        } else if (params.has_bias) {
          // we need to load in a new bias every time the weight loop index
          // changes
          bool read_bias = loop_counters[1][bias_reuse_indices[0]] == 0 &&
                           loop_counters[1][bias_reuse_indices[1]] == 0 &&
                           loop_counters[1][bias_reuse_indices[2]] == 0 &&
                           loop_counters[1][bias_reuse_indices[3]] == 0;

          if (read_bias) {
            bias = biasChannel.Pop();
          }

          psum = bias;
        }

        inputSkewerDin.Push(inputs);
        psumInSkewerDin.Push(psum);

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
          accumulation_buffer_bank = !accumulation_buffer_bank;
        }

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

  void delay_inputs() {
    inputsToSkewer.ResetRead();
    inputsToSkewer_delayed.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      inputsToSkewer_delayed.Push(inputsToSkewer.Pop());
    }
  }
#endif

  void run() {
    sa_controller_params.ResetRead();

    inputsChannel.Reset();
    inputsToSkewer.ResetWrite();
    psumOutSkewerDout.ResetRead();

#if SUPPORT_MX
    inputSkewerDin.ResetWrite();
    psumInSkewerDin.ResetWrite();
    unscaledAccumulationChannel.ResetWrite();
#endif

    startSignal.Reset();
    doneSignal.Reset();

    bool accumulation_buffer_bank_delayed = 0;
    bool accumulation_buffer_bank = 0;

    accumulation_buffer_read_address[0].Reset();
    accumulation_buffer_read_address[1].Reset();

#if !SUPPORT_MX
    accumulation_buffer_write_request[0].Reset();
    accumulation_buffer_write_request[1].Reset();
    accumulation_buffer_done[0].Reset();
    accumulation_buffer_done[1].Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = sa_controller_params.Pop();

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
      ac_int<32, false> output_step = 0;
      ac_int<32, false> last_output_step = 0;

      // non_accumulating_tile_size is the number of inputs to send before we
      // start accumulating. For example, for a loop order of (C, K, FX, FY,
      // Y, X), the non_accumulating_tile_size is X * Y. For a loop order of (C,
      // K, FX, Y, FY, X), the non_accumulating_tile_size is X.
      int non_accumulating_tile_size = 1;
      int max_reduction_loop_index =
          max3(params.reductionLoopIndex[1], params.fxIndex, params.fyIndex);
      for (int i = 5; i > max_reduction_loop_index; i--) {
        non_accumulating_tile_size *= params.loops[1][i];
      }

      // loop indices that are used to determine when to read in a new bias
      int bias_reuse_indices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weightLoopIndex[1]; i--) {
        bias_reuse_indices[5 - i] = i;
      }

      // Push inputs and psums into the array
      // Pipelined across tiles
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < total_ops) {
#ifndef __SYNTHESIS__
        if (step % 1000 == 0 and step > 0) {
          CCS_LOG("step " << step << " out of " << total_ops);
        }
#endif

        accumulation_buffer_bank_delayed = accumulation_buffer_bank;

        bool stall_inputs = false;

#if !SUPPORT_MX
        bool is_accumulating =
            loop_counters[0][params.reductionLoopIndex[0]] != 0 ||
            loop_counters[1][params.reductionLoopIndex[1]] != 0 ||
            loop_counters[1][params.fxIndex] != 0 ||
            loop_counters[1][params.fyIndex] != 0;

        stall_inputs = is_accumulating &&
                       last_output_step < step - non_accumulating_tile_size + 2;
#endif

        last_output_step = output_step;

        // std::cerr << "step: " << step << std::endl;
        // std::cerr << "accum_step: " << accum_step << std::endl;
        // std::cerr << "output_step: " << output_step << std::endl;

        // std::cerr << "is_accumulating: " << is_accumulating << std::endl;
        // std::cerr << "stall_inputs: " << stall_inputs << std::endl;
        // std::cerr << "non_accumulating_tile_size: "
        //           << non_accumulating_tile_size << std::endl;
        // std::cerr << std::endl;

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

        if (!stall_inputs) {
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
                                          InputTypes...>(
                                  params.input_dtype, data, inputs[i].data) ||
                              ...);
#ifndef __SYNTHESIS__
              if (!success) {
                std::cerr << "Error: matrix input dtype '" << params.input_dtype
                          << "' is not valid" << std::endl;
              }
#endif
            }
          }

#if !SUPPORT_MX
          inputsToSkewer.Push(inputs);

          if (is_accumulating) {
            ac_int<int_log2(BufferSize), false> readAddress =
                loop_counters[1][params.weightLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] *
                    params.loops[1][params.inputYLoopIndex[1]] +
                loop_counters[1][params.inputYLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] +
                loop_counters[1][params.inputXLoopIndex[1]];

            accumulation_buffer_read_address[accumulation_buffer_bank].Push(
                readAddress);
          }
#else
          inputSkewerDin.Push(inputs);

          Pack1D<Psum, NCols> psum;
#pragma hls_unroll yes
          for (int i = 0; i < NCols; i++) {
            psum.value[i] = Psum::zero();
          }
          psumInSkewerDin.Push(psum);
#endif
        }

        Pack1D<Psum, NCols> outputs;
        if (psumOutSkewerDout.PopNB(outputs)) {
        INCR_OUT_STEP:
          output_step++;
          // CCS_LOG("systolic array output: " << outputs);
          bool output_tile_completed =
              (loop_counters_out[0][params.reductionLoopIndex[0]] ==
               params.loops[0][params.reductionLoopIndex[0]] - 1) &&
              (loop_counters_out[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.weightLoopIndex[1]] ==
               params.loops[1][params.weightLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters_out[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1) &&
              (loop_counters_out[1][params.inputXLoopIndex[1]] ==
               params.loops[1][params.inputXLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.inputYLoopIndex[1]] ==
               params.loops[1][params.inputYLoopIndex[1]] - 1);

#if SUPPORT_MX
          bool is_output_accumulating =
              loop_counters_out[0][params.reductionLoopIndex[0]] != 0 ||
              loop_counters_out[1][params.reductionLoopIndex[1]] != 0 ||
              loop_counters_out[1][params.fxIndex] != 0 ||
              loop_counters_out[1][params.fyIndex] != 0;

          if (is_output_accumulating) {
            ac_int<int_log2(BufferSize), false> readAddress =
                loop_counters_out[1][params.weightLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] *
                    params.loops[1][params.inputYLoopIndex[1]] +
                loop_counters_out[1][params.inputYLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] +
                loop_counters_out[1][params.inputXLoopIndex[1]];

            accumulation_buffer_read_address[accumulation_buffer_bank].Push(
                readAddress);
          }

          if (output_tile_completed) {
            accumulation_buffer_bank = !accumulation_buffer_bank;
          }

          unscaledAccumulationChannel.Push(outputs);
#else
          ac_int<int_log2(BufferSize), false> write_address =
              loop_counters_out[1][params.weightLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] *
                  params.loops[1][params.inputYLoopIndex[1]] +
              loop_counters_out[1][params.inputYLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] +
              loop_counters_out[1][params.inputXLoopIndex[1]];

          BufferWriteRequest<Pack1D<Buffer, NCols>> req;
          req.address = write_address;
          req.data = outputs;
          accumulation_buffer_write_request[accumulation_buffer_bank].Push(req);

          if (output_tile_completed) {
            accumulation_buffer_done[accumulation_buffer_bank].SyncPush();
            accumulation_buffer_bank = !accumulation_buffer_bank;
          }
#endif

          loop_counters_out[1][5]++;
#pragma hls_unroll yes
          for (int i = 1; i >= 0; i--) {
#pragma hls_unroll yes
            for (int j = 5; j >= 0; j--) {
              if (loop_counters_out[i][j] == params.loops[i][j]) {
                loop_counters_out[i][j] = 0;
                if (j > 0) {
                  loop_counters_out[i][j - 1]++;
                } else {
                  if (i > 0) {
                    loop_counters_out[i - 1][5]++;
                  }
                }
              }
            }
          }
        }

        if (!stall_inputs) {
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

// when stalling, use wait() to yield control to other processes
#ifndef __SYNTHESIS__
        if (stall_inputs) {
          wait();
        }
#endif
      }

      CCS_LOG("draining");

// Drain out any remaining outputs
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (output_step < total_ops) {
        Pack1D<Psum, NCols> outputs;
        if (psumOutSkewerDout.PopNB(outputs)) {
          output_step++;

          DLOG("output_step: " << output_step);

          bool output_tile_completed =
              (loop_counters_out[0][params.reductionLoopIndex[0]] ==
               params.loops[0][params.reductionLoopIndex[0]] - 1) &&
              (loop_counters_out[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.weightLoopIndex[1]] ==
               params.loops[1][params.weightLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters_out[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1) &&
              (loop_counters_out[1][params.inputXLoopIndex[1]] ==
               params.loops[1][params.inputXLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.inputYLoopIndex[1]] ==
               params.loops[1][params.inputYLoopIndex[1]] - 1);

#if SUPPORT_MX
          bool is_output_accumulating =
              loop_counters_out[0][params.reductionLoopIndex[0]] != 0 ||
              loop_counters_out[1][params.reductionLoopIndex[1]] != 0 ||
              loop_counters_out[1][params.fxIndex] != 0 ||
              loop_counters_out[1][params.fyIndex] != 0;

          if (is_output_accumulating) {
            ac_int<int_log2(BufferSize), false> readAddress =
                loop_counters_out[1][params.weightLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] *
                    params.loops[1][params.inputYLoopIndex[1]] +
                loop_counters_out[1][params.inputYLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] +
                loop_counters_out[1][params.inputXLoopIndex[1]];

            accumulation_buffer_read_address[accumulation_buffer_bank].Push(
                readAddress);
          }

          if (output_tile_completed) {
            accumulation_buffer_bank = !accumulation_buffer_bank;
          }
          unscaledAccumulationChannel.Push(outputs);
#else
          ac_int<int_log2(BufferSize), false> write_address =
              loop_counters_out[1][params.weightLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] *
                  params.loops[1][params.inputYLoopIndex[1]] +
              loop_counters_out[1][params.inputYLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] +
              loop_counters_out[1][params.inputXLoopIndex[1]];

          BufferWriteRequest<Pack1D<Buffer, NCols>> req;
          req.address = write_address;
          req.data = outputs;
          accumulation_buffer_write_request[accumulation_buffer_bank].Push(req);

          if (output_tile_completed) {
            accumulation_buffer_done[accumulation_buffer_bank].SyncPush();
            accumulation_buffer_bank = !accumulation_buffer_bank;
          }
#endif

          loop_counters_out[1][5]++;
#pragma hls_unroll yes
          for (int i = 1; i >= 0; i--) {
#pragma hls_unroll yes
            for (int j = 5; j >= 0; j--) {
              if (loop_counters_out[i][j] == params.loops[i][j]) {
                loop_counters_out[i][j] = 0;
                if (j > 0) {
                  loop_counters_out[i][j - 1]++;
                } else {
                  if (i > 0) {
                    loop_counters_out[i - 1][5]++;
                  }
                }
              }
            }
          }
        } else {
          wait();
        }
      }

      CCS_LOG("done");

      doneSignal.SyncPush();
    }
  }
#if SUPPORT_MX
  void delay_outputs() {
    unscaledAccumulationChannel.ResetRead();
    unscaledAccumulationChannel_delayed.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      unscaledAccumulationChannel_delayed.Push(
          unscaledAccumulationChannel.Pop());
    }
  }

  void process_accumulation() {
    biasChannel.Reset();
#ifndef __SYNTHESIS__
    unscaledAccumulationChannel.ResetRead();
#else
    unscaledAccumulationChannel_delayed.ResetRead();
#endif
    accumulation_buffer_params.ResetRead();
    inputScaleChannel.Reset();
    weightScaleChannel.Reset();
    accumulation_buffer_done[0].Reset();
    accumulation_buffer_done[1].Reset();
    accumulation_buffer_read_data[0].Reset();
    accumulation_buffer_read_data[1].Reset();
    accumulation_buffer_write_request[0].Reset();
    accumulation_buffer_write_request[1].Reset();

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
      Pack1D<Scale, NCols> weight_scales;

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
          previous_accumulation =
              accumulation_buffer_read_data[accumulation_buffer_bank].Pop();
        }

        Pack1D<Psum, NCols> outputs;
#ifndef __SYNTHESIS__
        outputs = unscaledAccumulationChannel.Pop();
#else
        outputs = unscaledAccumulationChannel_delayed.Pop();
#endif

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

        Pack1D<Buffer, NCols> scaled_outputs;

#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          Buffer scale = static_cast<Buffer>(input_scale) *
                         static_cast<Buffer>(weight_scales[i]);
          scaled_outputs[i] = static_cast<Buffer>(outputs[i]);
          if (params.is_mx_op) {
            scaled_outputs[i] = scaled_outputs[i] * scale;
          }
          previous_accumulation.value[i] += scaled_outputs[i];
        }

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
#endif

  void send_params() {
    paramsIn.ResetRead();

    sa_controller_params.ResetWrite();
    weight_buffer_params.ResetWrite();
    accumulation_buffer_params.ResetWrite();

    wait();

    while (true) {
      MatrixParams params = paramsIn.Pop();

      sa_controller_params.Push(params);
      weight_buffer_params.Push(params);
      accumulation_buffer_params.Push(params);
    }
  }
};
