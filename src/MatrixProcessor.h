#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ParamsDeserializer.h"
#include "SystolicArray.h"

template <typename Input, typename Psum, typename Buffer, typename Scale,
          int NRows, int NCols, int BufferSize>
SC_MODULE(MatrixProcessor) {
 private:
  Connections::SyncChannel CCS_INIT_S1(weightLoadDone);
  Connections::SyncChannel CCS_INIT_S1(weightSwapDone);

  MultiInputSerializedSkewer<Input, typename Input::decoded, NRows> CCS_INIT_S1(
      inputSkewer);
  Connections::Combinational<Pack1D<PEInput<Input>, NRows>> CCS_INIT_S1(
      inputSkewerDin);

  WeightSerializedSkewer<Input, typename Input::decoded, NCols> CCS_INIT_S1(
      weightSkewer);
  Connections::Combinational<Pack1D<PEWeight<Input>, NCols>> CCS_INIT_S1(
      weightSkewerDin);

  SerializedSkewer<Psum, typename Psum::decoded, NCols> CCS_INIT_S1(
      psumInSkewer);
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(psumInSkewerDin);

  DeserializedSkewer<typename Psum::decoded, Psum, NCols> CCS_INIT_S1(
      psumOutSkewer);
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(
      psumOutSkewerDout);

  SystolicArray<typename Input::decoded, typename Input::decoded,
                typename Psum::decoded, NRows, NCols>
      CCS_INIT_S1(systolicArray);

  MatrixParamsDeserializer<1> CCS_INIT_S1(paramsDeserializer);

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int LOOP_WIDTH = 10;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<Input, NRows>> CCS_INIT_S1(inputsChannel);
  Connections::In<Pack1D<Input, NCols>> CCS_INIT_S1(weightsChannel);
  Connections::In<Pack1D<Buffer, NCols>> CCS_INIT_S1(biasChannel);
  Connections::Out<Pack1D<Buffer, NCols>> CCS_INIT_S1(outputsChannel);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<PEInput<typename Input::decoded>>
      inputsToSystolicArray[NRows];
  // Connections::Combinational<ac_int<1, false> >
  //     weightSwapToSystolicArray[NRows];
  Connections::Combinational<typename Psum::decoded>
      psumsToSystolicArray[NCols];
  Connections::Combinational<typename Psum::decoded>
      outputsFromSystolicArray[NCols];
  Connections::Combinational<PEWeight<typename Input::decoded>>
      weightsToSystolicArray[NCols];

#if SUPPORT_MX
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      accumulationBufferParams);
  Connections::Combinational<Pack1D<Psum, NCols>> CCS_INIT_S1(
      unscaledAccumulationChannel);

  Connections::In<Pack1D<Scale, 1>> CCS_INIT_S1(inputScaleChannel);
  Connections::In<Pack1D<Scale, NCols>> CCS_INIT_S1(weightScaleChannel);
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

    SC_THREAD(process_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#if SUPPORT_MX
    SC_THREAD(process_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
  }

  void process_weights() {
    weightsChannel.Reset();
    weightSkewerDin.ResetWrite();

    wait();

    while (true) {
#pragma hls_pipeline_init_interval 1
      for (int weight_count = 0; weight_count < NRows; weight_count++) {
        Pack1D<Input, NCols> arrayWeights = weightsChannel.Pop();
        Pack1D<PEWeight<Input>, NCols> weights;
#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          weights[i].data = arrayWeights[i];
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

  void run() {
    paramsIn.ResetRead();

    inputSkewerDin.ResetWrite();
    inputsChannel.Reset();
    biasChannel.Reset();
    psumInSkewerDin.ResetWrite();
    psumOutSkewerDout.ResetRead();

#if SUPPORT_MX
    accumulationBufferParams.ResetWrite();
    unscaledAccumulationChannel.ResetWrite();
#else
    outputsChannel.Reset();
#endif

    startSignal.Reset();
    doneSignal.Reset();

    wait();

    while (true) {
      const MatrixParams params = paramsIn.Pop();

#if SUPPORT_MX
      accumulationBufferParams.Push(params);
#endif

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

#ifdef __SYNTHESIS__
      Pack1D<Buffer, NCols> accumulation_buffer[BufferSize];
#else
      Pack1D<Buffer, NCols> *accumulation_buffer =
          new Pack1D<Buffer, NCols>[BufferSize];
#endif

      // TODO: Replace oldOutputStep2 with outputStep - 2
      ac_int<32, false> step = 0;
      ac_int<32, false> outputStep = 0;
      ac_int<32, false> oldOutputStep = 0;
      ac_int<32, false> oldOutputStep2 = 0;

      // nonAccumulatingTileSize is the number of inputs to send before we start
      // accumulating. For example, for a loop order of (C, K, FX, FY, Y, X),
      // the nonAccumulatingTileSize is X * Y. For a loop order of (C, K, FX, Y,
      // FY, X), the nonAccumulatingTileSize is X.
      int nonAccumulatingTileSize = 1;
      int largestReductionLoopIndex =
          max3(params.reductionLoopIndex[1], params.fxIndex, params.fyIndex);
      for (int i = 5; i > largestReductionLoopIndex; i--) {
        nonAccumulatingTileSize *= params.loops[1][i];
      }

      Pack1D<Psum, NCols> bias;

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
        oldOutputStep2 = oldOutputStep;
        oldOutputStep = outputStep;

        Pack1D<PEInput<Input>, NRows> inputs;
        bool stallInputs = false;

        bool isAccumulation =
            loop_counters[0][params.reductionLoopIndex[0]] != 0 ||
            loop_counters[1][params.reductionLoopIndex[1]] != 0 ||
            loop_counters[1][params.fxIndex] != 0 ||
            loop_counters[1][params.fyIndex] != 0;
        if (isAccumulation) {
          // if accumulating, make sure that the output loop counter has
          // received the accumulated value
          // TODO: Replace oldOutputStep2 with oldOutputStep - 2. Define a
          // constant and avoid magic number.
          stallInputs = !(oldOutputStep2 > step - nonAccumulatingTileSize + 1);
        }

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
          Pack1D<Input, NRows> inputsData = inputsChannel.Pop();
#pragma hls_unroll yes
          for (int i = 0; i < NRows; i++) {
            inputs[i].data = inputsData[i];
          }
        }

        Pack1D<Psum, NCols> psum;
#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          psum.value[i].set_zero();
        }

#if !SUPPORT_MX
        // TODO: duplicate variable with isAccumulation
        bool firstAccumulation =
            loop_counters[0][params.reductionLoopIndex[0]] == 0 &&
            loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
            loop_counters[1][params.fxIndex] == 0 &&
            loop_counters[1][params.fyIndex] == 0;

        // TODO: do we need firstAccumulation?
        if (!firstAccumulation && step < totalOps && !stallInputs) {
          ac_int<int_log2(BufferSize), false> readAddress =
              loop_counters[1][params.weightLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] *
                  params.loops[1][params.inputYLoopIndex[1]] +
              loop_counters[1][params.inputYLoopIndex[1]] *
                  params.loops[1][params.inputXLoopIndex[1]] +
              loop_counters[1][params.inputXLoopIndex[1]];
#ifdef __SYNTHESIS__
        READ_ACC_BUFFER:
#endif
          psum = accumulation_buffer[readAddress];
        } else if (params.has_bias && step < totalOps && !stallInputs) {
          // we need to load in a new bias every time the weight loop index
          // changes
          bool readBias = true;
#pragma hls_unroll yes
          for (int i = 0; i < 4; i++) {
            readBias = readBias && (loop_counters[1][biasReuseIndices[i]] == 0);
          }

          if (readBias) {
            bias = biasChannel.Pop();
          }

          psum = bias;
        }
#endif

        if (!stallInputs) {
          inputSkewerDin.Push(inputs);
          psumInSkewerDin.Push(psum);
        }

        Pack1D<Psum, NCols> outputs;
        if (psumOutSkewerDout.PopNB(outputs)) {
        INCR_OUT_STEP:
          outputStep++;
          // CCS_LOG("systolic array output: " << outputs);
          bool accumulationFinished =
              (loop_counters_out[0][params.reductionLoopIndex[0]] ==
               params.loops[0][params.reductionLoopIndex[0]] - 1) &&
              (loop_counters_out[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters_out[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1);

#if SUPPORT_MX
          unscaledAccumulationChannel.Push(outputs);
#else
          if (accumulationFinished) {
            outputsChannel.Push(outputs);
          } else {
            ac_int<int_log2(BufferSize), false> writeAddress =
                loop_counters_out[1][params.weightLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] *
                    params.loops[1][params.inputYLoopIndex[1]] +
                loop_counters_out[1][params.inputYLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] +
                loop_counters_out[1][params.inputXLoopIndex[1]];

#ifdef __SYNTHESIS__
          WRITE_ACC_BUFFER:
#endif
            accumulation_buffer[writeAddress] = outputs;
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

        if (!stallInputs) {
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
        if (stallInputs) {
          wait();
        }
#endif
      }

// Drain out any remaining outputs
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (outputStep < totalOps) {
        Pack1D<Psum, NCols> outputs;
        if (psumOutSkewerDout.PopNB(outputs)) {
          outputStep++;

          DLOG("systolic array output: " << outputs);
          bool accumulationFinished =
              (loop_counters_out[0][params.reductionLoopIndex[0]] ==
               params.loops[0][params.reductionLoopIndex[0]] - 1) &&
              (loop_counters_out[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters_out[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1);

#if SUPPORT_MX
          unscaledAccumulationChannel.Push(outputs);
#else
          if (accumulationFinished) {
            outputsChannel.Push(outputs);
            DLOG("matrix processor output: " << outputs);
          } else {
            ac_int<int_log2(BufferSize), false> writeAddress =
                loop_counters_out[1][params.weightLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] *
                    params.loops[1][params.inputYLoopIndex[1]] +
                loop_counters_out[1][params.inputYLoopIndex[1]] *
                    params.loops[1][params.inputXLoopIndex[1]] +
                loop_counters_out[1][params.inputXLoopIndex[1]];

#ifdef __SYNTHESIS__
          WRITE_ACC_BUFFER2:
#endif
            accumulation_buffer[writeAddress] = outputs;
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

      doneSignal.SyncPush();
    }
  }

#if SUPPORT_MX
  void process_accumulation() {
    unscaledAccumulationChannel.ResetRead();
    accumulationBufferParams.ResetRead();
    inputScaleChannel.Reset();
    weightScaleChannel.Reset();

    outputsChannel.Reset();

    wait();

    while (true) {
      CCS_LOG("start");

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

#ifdef __SYNTHESIS__
      Pack1D<Buffer, NCols> accumulation_buffer[BufferSize];
#else
      Pack1D<Buffer, NCols> *accumulation_buffer =
          new Pack1D<Buffer, NCols>[BufferSize];
#endif

      ac_int<32, false> step = 0;

      Pack1D<Buffer, NCols> bias;
      Pack1D<Scale, NCols> weightScales;

      // loop indices that are used to determine when to read in a new bias
      int biasReuseIndices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weightLoopIndex[1]; i--) {
        biasReuseIndices[5 - i] = i;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < totalOps) {
        // CCS_LOG("acc step " << step << " out of " << totalOps);
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
          int readAddress = static_cast<ac_int<10, false>>(
                                loop_counters[1][params.weightLoopIndex[1]] *
                                params.loops[1][params.inputXLoopIndex[1]] *
                                params.loops[1][params.inputYLoopIndex[1]]) +
                            static_cast<ac_int<10, false>>(
                                loop_counters[1][params.inputYLoopIndex[1]] *
                                params.loops[1][params.inputXLoopIndex[1]]) +
                            loop_counters[1][params.inputXLoopIndex[1]];

        READ_ACC_BUFFER:
          previous_accumulation = accumulation_buffer[readAddress];
        }

        Pack1D<Psum, NCols> outputs;
        outputs = unscaledAccumulationChannel.Pop();

        // CCS_LOG("outputs: " << outputs);

        Pack1D<Scale, 1> inputScale;
        if (params.is_mx_op) {
          inputScale = inputScaleChannel.Pop();
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
          weightScales = weightScaleChannel.Pop();
        }

        Pack1D<Buffer, NCols> scaled_outputs;

#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          Buffer scale = static_cast<Buffer>(inputScale[0]) *
                         static_cast<Buffer>(weightScales[i]);
          scaled_outputs[i] = static_cast<Buffer>(outputs[i]);
          if (params.is_mx_op) {
            scaled_outputs[i] = scaled_outputs[i] * scale;
          }
          previous_accumulation.value[i] += scaled_outputs[i];
        }
        int addr = static_cast<ac_int<10, false>>(
                       loop_counters[1][params.weightLoopIndex[1]] *
                       params.loops[1][params.inputXLoopIndex[1]] *
                       params.loops[1][params.inputYLoopIndex[1]]) +
                   static_cast<ac_int<10, false>>(
                       loop_counters[1][params.inputYLoopIndex[1]] *
                       params.loops[1][params.inputXLoopIndex[1]]) +
                   loop_counters[1][params.inputXLoopIndex[1]];

        bool accumulationFinished =
            (loop_counters[0][params.reductionLoopIndex[0]] ==
             params.loops[0][params.reductionLoopIndex[0]] - 1) &&
            (loop_counters[1][params.reductionLoopIndex[1]] ==
             params.loops[1][params.reductionLoopIndex[1]] - 1) &&
            (loop_counters[1][params.fxIndex] ==
             params.loops[1][params.fxIndex] - 1) &&
            (loop_counters[1][params.fyIndex] ==
             params.loops[1][params.fyIndex] - 1);

        if (accumulationFinished) {
          outputsChannel.Push(previous_accumulation);
          // CCS_LOG("matrix processor output: " << previous_accumulation);
        } else {
          int writeAddress = static_cast<ac_int<10, false>>(
                                 loop_counters[1][params.weightLoopIndex[1]] *
                                 params.loops[1][params.inputXLoopIndex[1]] *
                                 params.loops[1][params.inputYLoopIndex[1]]) +
                             static_cast<ac_int<10, false>>(
                                 loop_counters[1][params.inputYLoopIndex[1]] *
                                 params.loops[1][params.inputXLoopIndex[1]]) +
                             loop_counters[1][params.inputXLoopIndex[1]];

        WRITE_ACC_BUFFER:
          accumulation_buffer[writeAddress] = previous_accumulation;
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
};
