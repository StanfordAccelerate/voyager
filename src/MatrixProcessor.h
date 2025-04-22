#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ArchitectureParams.h"
#include "ParamsDeserializer.h"
#include "Skewer.h"
#include "SystolicArray.h"

template <typename Input, typename Weight, typename Psum, typename Buffer,
          typename Scale, int NRows, int NCols, int BufferSize>
SC_MODULE(MatrixProcessor) {
 private:
  MultiInputSerializedSkewer<Input, typename Input::decoded, NRows> CCS_INIT_S1(
      inputSkewer);
  Connections::Combinational<Pack1D<PEInput<Input>, NRows>> CCS_INIT_S1(
      inputSkewerDin);
  Connections::Combinational<Pack1D<PEInput<Input>, NRows>> CCS_INIT_S1(
      inputsToSkewer);
  Connections::Combinational<Pack1D<PEInput<Input>, NRows>> CCS_INIT_S1(
      inputsToSkewer_delayed);

  WeightSerializedSkewer<Weight, typename Weight::decoded, NCols> CCS_INIT_S1(
      weightSkewer);
  Connections::Combinational<Pack1D<PEWeight<Weight>, NCols>> CCS_INIT_S1(
      weightSkewerDin);

  DeserializedSkewer<Psum, Psum, NCols> CCS_INIT_S1(psumOutSkewer);
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(
      psumOutSkewerDout);

  SystolicArray<typename Input::decoded, typename Input::decoded, Psum, NRows,
                NCols>
      CCS_INIT_S1(systolicArray);

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

  Connections::In<IC_PORT_TYPE> CCS_INIT_S1(inputsChannel);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weightsChannel);
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

  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      accumulationBufferParams);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<PEInput<typename Input::decoded>>
      inputsToSystolicArray[NRows];
  Connections::Combinational<PEWeight<typename Weight::decoded>>
      weightsToSystolicArray[NCols];
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

    SC_THREAD(process_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(push_inputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void process_weights() {
    weightsChannel.Reset();
    weightSkewerDin.ResetWrite();

    wait();

    while (true) {
#pragma hls_pipeline_init_interval 1
      for (int weight_count = 0; weight_count < NRows; weight_count++) {
        auto bits = weightsChannel.Pop();
        Pack1D<PEWeight<Weight>, NCols> weights;
#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          weights[i].data.set_bits(
              bits.template slc<Weight::width>(i * Weight::width));
          weights[i].tag = weight_count;
        }

        weightSkewerDin.Push(weights);
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
    inputsToSkewer.ResetWrite();

    accumulationBufferParams.ResetWrite();
    inputSkewerDin.ResetWrite();

    startSignal.Reset();

    wait();

    while (true) {
      const MatrixParams params = paramsIn.Pop();

      accumulationBufferParams.Push(params);

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

      ac_int<32, false> totalOps = params.loops[0][0] * params.loops[0][1] *
                                   params.loops[0][2] * params.loops[0][3] *
                                   params.loops[1][0] * params.loops[1][1] *
                                   params.loops[1][2] * params.loops[1][3] *
                                   params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;

      // nonAccumulatingTileSize is the number of inputs to send before we
      // start accumulating. For example, for a loop order of (C, K, FX, FY,
      // Y, X), the nonAccumulatingTileSize is X * Y. For a loop order of (C,
      // K, FX, Y, FY, X), the nonAccumulatingTileSize is X.
      int nonAccumulatingTileSize = 1;
      int largestReductionLoopIndex =
          max3(params.reductionLoopIndex[1], params.fxIndex, params.fyIndex);
      for (int i = 5; i > largestReductionLoopIndex; i--) {
        nonAccumulatingTileSize *= params.loops[1][i];
      }

      // loop indices that are used to determine when to read in a new bias
      int biasReuseIndices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weightLoopIndex[1]; i--) {
        biasReuseIndices[5 - i] = i;
      }

      // Push inputs and psums into the array
      // Pipelined across tiles
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < totalOps) {
#ifndef __SYNTHESIS__
        if (step % 1000 == 0 and step > 0) {
          CCS_LOG("step " << step << " out of " << totalOps);
        }
#endif

        Pack1D<PEInput<Input>, NRows> inputs;
        bool stallInputs = false;

        bool sendWeights;
        if (params.weightReuseIndex[0] != params.weightReuseIndex[1]) {
          sendWeights = (loop_counters[1][params.weightReuseIndex[1]] == 0) &&
                        (loop_counters[1][params.weightReuseIndex[0]] == 0);
        } else {
          sendWeights = (loop_counters[1][params.weightReuseIndex[1]] == 0);
        }
        sendWeights = sendWeights || step == 0;

        if (sendWeights && step < totalOps - 1) {
#pragma hls_unroll yes
          for (int i = 0; i < NRows; i++) {
            inputs[i].swapWeights = true;
          }
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < NRows; i++) {
            inputs[i].swapWeights = false;
          }
        }

        if (step < totalOps && !stallInputs) {
          auto bits = inputsChannel.Pop();
#pragma hls_unroll yes
          for (int i = 0; i < NRows; i++) {
            inputs[i].data.set_bits(
                bits.template slc<Input::width>(i * Input::width));
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
    accumulationBufferParams.ResetRead();
    psumOutSkewerDout.ResetRead();

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
      const MatrixParams params = accumulationBufferParams.Pop();
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

      ac_int<32, false> totalOps = params.loops[0][0] * params.loops[0][1] *
                                   params.loops[0][2] * params.loops[0][3] *
                                   params.loops[1][0] * params.loops[1][1] *
                                   params.loops[1][2] * params.loops[1][3] *
                                   params.loops[1][4] * params.loops[1][5];

      ac_int<32, false> step = 0;

      Pack1D<Buffer, NCols> bias;

#if SUPPORT_MX
      Pack1D<Scale, NCols> weightScales;
#endif

      // loop indices that are used to determine when to read in a new bias
      int biasReuseIndices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weightLoopIndex[1]; i--) {
        biasReuseIndices[5 - i] = i;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < totalOps) {
        bool firstAccumulation =
            loop_counters[0][params.reductionLoopIndex[0]] == 0 &&
            loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
            loop_counters[1][params.fxIndex] == 0 &&
            loop_counters[1][params.fyIndex] == 0;

        Pack1D<Buffer, NCols> previous_accumulation;

#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          previous_accumulation.value[i].set_zero();
        }

        if (firstAccumulation) {
          if (params.has_bias) {
            bool readBias = true;
#pragma hls_unroll yes
            for (int i = 0; i < 4; i++) {
              readBias =
                  readBias && (loop_counters[1][biasReuseIndices[i]] == 0);
            }

            if (readBias) {
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
        Scale inputScale;
        if (params.is_mx_op) {
          inputScale.set_bits(inputScaleChannel.Pop());
        }

        bool readNewWeights;
        if (params.weightReuseIndex[0] != params.weightReuseIndex[1]) {
          readNewWeights =
              (loop_counters[1][params.weightReuseIndex[1]] == 0) &&
              (loop_counters[1][params.weightReuseIndex[0]] == 0);
        } else {
          readNewWeights = (loop_counters[1][params.weightReuseIndex[1]] == 0);
        }
        readNewWeights = readNewWeights || step == 0;
        if (readNewWeights && params.is_mx_op) {
          auto bits = weightScaleChannel.Pop();
          weightScales = BitsToType<Pack1D<Scale, NCols>>(TypeToBits(bits));
        }
#endif

        Pack1D<Buffer, NCols> scaled_outputs;

#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
#if SUPPORT_MX
          Buffer scale = static_cast<Buffer>(inputScale) *
                         static_cast<Buffer>(weightScales[i]);
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
          int writeAddress = static_cast<ac_int<10, false>>(
                                 loop_counters[1][params.weightLoopIndex[1]] *
                                 params.loops[1][params.inputXLoopIndex[1]] *
                                 params.loops[1][params.inputYLoopIndex[1]]) +
                             static_cast<ac_int<10, false>>(
                                 loop_counters[1][params.inputYLoopIndex[1]] *
                                 params.loops[1][params.inputXLoopIndex[1]]) +
                             loop_counters[1][params.inputXLoopIndex[1]];

          BufferWriteRequest<Pack1D<Buffer, NCols>> req;
          req.address = writeAddress;
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
