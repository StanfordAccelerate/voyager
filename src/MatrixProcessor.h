#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ParamsDeserializer.h"
#include "SystolicArray.h"

template <typename InputT, typename OutputT, typename BufferT, typename ScaleT,
          bool is_mx, int NRows, int NCols, int BufferSize>
SC_MODULE(MatrixProcessor) {
 private:
  Connections::SyncChannel CCS_INIT_S1(weightLoadDone);
  Connections::SyncChannel CCS_INIT_S1(weightSwapDone);

  MultiInputSerializedSkewer<InputT, typename InputT::Decoded, NRows>
      CCS_INIT_S1(inputSkewer);
  Connections::Combinational<Pack1D<PEInput<InputT>, NRows> > CCS_INIT_S1(
      inputSkewerDin);

  WeightSerializedSkewer<InputT, typename InputT::Decoded, NCols> CCS_INIT_S1(
      weightSkewer);
  Connections::Combinational<Pack1D<PEWeight<InputT>, NCols> > CCS_INIT_S1(
      weightSkewerDin);

  SerializedSkewer<OutputT, typename OutputT::Decoded, NCols> CCS_INIT_S1(
      psumInSkewer);
  Connections::Combinational<Pack1D<OutputT, NCols> > CCS_INIT_S1(
      psumInSkewerDin);

  DeserializedSkewer<typename OutputT::Decoded, OutputT, NCols> CCS_INIT_S1(
      psumOutSkewer);
  Connections::Combinational<Pack1D<OutputT, NCols> > CCS_INIT_S1(
      psumOutSkewerDout);

  SystolicArray<typename InputT::Decoded, typename InputT::Decoded,
                typename OutputT::Decoded, NRows, NCols>
      CCS_INIT_S1(systolicArray);

  MatrixParamsDeserializer<1> CCS_INIT_S1(paramsDeserializer);

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int LOOP_WIDTH = 10;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<InputT, NRows> > CCS_INIT_S1(inputsChannel);
  Connections::In<Pack1D<InputT, NCols> > CCS_INIT_S1(weightsChannel);
  Connections::In<Pack1D<BufferT, NCols> > CCS_INIT_S1(biasChannel);
  Connections::Out<Pack1D<BufferT, NCols> > CCS_INIT_S1(outputsChannel);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<PEInput<typename InputT::Decoded> >
      inputsToSystolicArray[NRows];
  // Connections::Combinational<ac_int<1, false> >
  //     weightSwapToSystolicArray[NRows];
  Connections::Combinational<typename OutputT::Decoded>
      psumsToSystolicArray[NCols];
  Connections::Combinational<typename OutputT::Decoded>
      outputsFromSystolicArray[NCols];
  Connections::Combinational<PEWeight<typename InputT::Decoded> >
      weightsToSystolicArray[NCols];

  // only used for MX
  Connections::ConditionalCombinational<MatrixParams, is_mx> CCS_INIT_S1(
      accumulationBufferParams);
  Connections::ConditionalCombinational<Pack1D<OutputT, NCols>, is_mx>
      CCS_INIT_S1(unscaledAccumulationChannel);

  Connections::ConditionalIn<Pack1D<ScaleT, 1>, is_mx> CCS_INIT_S1(
      inputScaleChannel);
  Connections::ConditionalIn<Pack1D<ScaleT, NCols>, is_mx> CCS_INIT_S1(
      weightScaleChannel);

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

    if constexpr (is_mx) {
      SC_THREAD(process_accumulation);
      sensitive << clk.pos();
      async_reset_signal_is(rstn, false);
    }
  }

  void process_weights() {
    weightsChannel.Reset();
    weightSkewerDin.ResetWrite();

    wait();

    while (true) {
#pragma hls_pipeline_init_interval 1
      for (int weight_count = 0; weight_count < NRows; weight_count++) {
        Pack1D<InputT, NCols> arrayWeights = weightsChannel.Pop();
        Pack1D<PEWeight<InputT>, NCols> weights;
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
      if (a > c) {
        return a;
      } else {
        return c;
      }
    } else {
      if (b > c) {
        return b;
      } else {
        return c;
      }
    }
  }

  void run() {
    paramsIn.ResetRead();

    inputSkewerDin.ResetWrite();
    inputsChannel.Reset();
    biasChannel.Reset();
    psumInSkewerDin.ResetWrite();
    psumOutSkewerDout.ResetRead();

    if constexpr (is_mx) {
      accumulationBufferParams.ResetWrite();
      unscaledAccumulationChannel.ResetWrite();
    } else {
      outputsChannel.Reset();
    }

    startSignal.Reset();
    doneSignal.Reset();

    wait();

    while (true) {
      const MatrixParams params = paramsIn.Pop();

      if constexpr (is_mx) {
        accumulationBufferParams.Push(params);
      }

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

      ac_int<32, false> totalOps =
          params.loops[0][0] * params.loops[0][1] * params.loops[0][2] *
          params.loops[1][0] * params.loops[1][1] * params.loops[1][2] *
          params.loops[1][3] * params.loops[1][4] * params.loops[1][5];

#ifdef __SYNTHESIS__
      Pack1D<BufferT, NCols> accumulation_buffer[BufferSize];
#else
      Pack1D<BufferT, NCols> *accumulation_buffer =
          new Pack1D<BufferT, NCols>[BufferSize];
#endif

      ac_int<32, false> step = 0;
      ac_int<32, false> outputStep = 0;
      ac_int<32, false> oldOutputStep = 0;
      ac_int<32, false> oldOutputStep2 = 0;

      int nonAccumulatingTileSize = 1;
      int largestReductionLoopIndex =
          max3(params.reductionLoopIndex[1], params.fxIndex, params.fyIndex);
      for (int i = 5; i > largestReductionLoopIndex; i--) {
        nonAccumulatingTileSize *= params.loops[1][i];
      }

      Pack1D<OutputT, NCols> bias;

      // loop indices that are used to determine when to read in a new bias
      int biasReuseIndices[4] = {5, 5, 5, 5};
      for (int i = 5; i > params.weightLoopIndex[1]; i--) {
        biasReuseIndices[5 - i] = i;
      }

      // first couple of outputs are garbage (bc of the swap)
      //       int NUM_GARBAGE = 0;
      //       int garbageCount = 0;

      // #pragma hls_pipeline_init_interval 1
      // #pragma hls_pipeline_stall_mode flush
      //       for (int count = 0; count < NUM_GARBAGE; count++) {
      //         Pack1D<PEInput<InputT>, NRows> inputs;
      // #pragma hls_unroll yes
      //         for (int i = 0; i < NRows; i++) {
      //           inputs[i].swapWeights = count == 0;
      //         }

      //         Pack1D<OutputT, NCols> psum;

      //         inputSkewerDin.Push(inputs);
      //         psumInSkewerDin.Push(psum);
      //       }

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

        // Pack1D<ac_int<1, false>, NRows> weightSwap;
        Pack1D<PEInput<InputT>, NRows> inputs;
        bool stallInputs = false;

        bool isAccumulation =
            loop_counters[1][params.reductionLoopIndex[1]] != 0 ||
            loop_counters[1][params.fxIndex] != 0 ||
            loop_counters[1][params.fyIndex] != 0;
        if (isAccumulation) {
          // if accumulating, make sure that the output loop counter has
          // received the accumulated value
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
          // CCS_LOG("sendWeights: " << sendWeights);
#pragma hls_unroll yes
          for (int i = 0; i < NRows; i++) {
            inputs[i].swapWeights = true;
            // weightSwap.value[i] = true;
          }
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < NRows; i++) {
            // weightSwap.value[i] = false;
            inputs[i].swapWeights = false;
          }
        }

        // Pack1D<InputT, NRows> inputs;
        if (step < totalOps && !stallInputs) {
          Pack1D<InputT, NRows> inputsData = inputsChannel.Pop();
#pragma hls_unroll yes
          for (int i = 0; i < NRows; i++) {
            inputs[i].data = inputsData[i];
            // #ifdef HYBRID_FP8
            // inputs[i].castToE5M2 = params.COMBINE_GRADS;
            // #endif
          }
        }

        Pack1D<OutputT, NCols> psum;
#pragma hls_unroll yes
        for (int i = 0; i < NCols; i++) {
          psum.value[i].set_zero();
        }

        if constexpr (!is_mx) {
          bool firstAccumulation =
              loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
              loop_counters[1][params.fxIndex] == 0 &&
              loop_counters[1][params.fyIndex] == 0;

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
              readBias =
                  readBias && (loop_counters[1][biasReuseIndices[i]] == 0);
            }

            if (readBias) {
              bias = biasChannel.Pop();
            }

            psum = bias;
          }
        }

        if (!stallInputs) {
          inputSkewerDin.Push(inputs);
          // weightSwapSkewerDin.Push(weightSwap);
          psumInSkewerDin.Push(psum);
        }

        Pack1D<OutputT, NCols> outputs;
        if (psumOutSkewerDout.PopNB(outputs)) {
          // if (garbageCount < NUM_GARBAGE) {
          //   garbageCount++;
          // } else {
        INCR_OUT_STEP:
          outputStep++;
          // CCS_LOG("systolic array output: " << outputs);
          bool accumulationFinished =
              (loop_counters_out[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters_out[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1);

          if constexpr (is_mx) {
            unscaledAccumulationChannel.Push(outputs);
          } else {
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
          }

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
          // }
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
        Pack1D<OutputT, NCols> outputs;
        if (psumOutSkewerDout.PopNB(outputs)) {
          outputStep++;

          DLOG("systolic array output: " << outputs);
          bool accumulationFinished =
              (loop_counters_out[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters_out[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1);

          if constexpr (is_mx) {
            unscaledAccumulationChannel.Push(outputs);
          } else {
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
              // DLOG("writeAddress: " << writeAddress << " val "
              //                          << outputs);
            }
          }

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

  // This thread is only used for MX
  void process_accumulation() {
    if constexpr (is_mx) {
      unscaledAccumulationChannel.ResetRead();
      accumulationBufferParams.ResetRead();
      inputScaleChannel.Reset();
      weightScaleChannel.Reset();

      outputsChannel.Reset();

      wait();

      while (true) {
        CCS_LOG("start");

        const MatrixParams params = accumulationBufferParams.Pop();
        ac_int<8, false> loop_counters[2][6];
        ac_int<8, false> loop_bounds[2][6];
#pragma hls_unroll yes
        for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
          for (int j = 0; j < 6; j++) {
            loop_counters[i][j] = 0;
            loop_bounds[i][j] = params.loops[i][j];
          }
        }

        ac_int<32, false> totalOps =
            params.loops[0][0] * params.loops[0][1] * params.loops[0][2] *
            params.loops[1][0] * params.loops[1][1] * params.loops[1][2] *
            params.loops[1][3] * params.loops[1][4] * params.loops[1][5];

#ifdef __SYNTHESIS__
        Pack1D<BufferT, NCols> accumulation_buffer[BufferSize];
#else
        Pack1D<BufferT, NCols> *accumulation_buffer =
            new Pack1D<BufferT, NCols>[BufferSize];
#endif

        ac_int<32, false> step = 0;

        Pack1D<BufferT, NCols> bias;
        Pack1D<ScaleT, NCols> weightScales;

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
              loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
              loop_counters[1][params.fxIndex] == 0 &&
              loop_counters[1][params.fyIndex] == 0;

          Pack1D<BufferT, NCols> previous_accumulation;

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
            int readAddress = static_cast<ac_int<10, false> >(
                                  loop_counters[1][params.weightLoopIndex[1]] *
                                  params.loops[1][params.inputXLoopIndex[1]] *
                                  params.loops[1][params.inputYLoopIndex[1]]) +
                              static_cast<ac_int<10, false> >(
                                  loop_counters[1][params.inputYLoopIndex[1]] *
                                  params.loops[1][params.inputXLoopIndex[1]]) +
                              loop_counters[1][params.inputXLoopIndex[1]];

          READ_ACC_BUFFER:
            previous_accumulation = accumulation_buffer[readAddress];
          }

          Pack1D<OutputT, NCols> outputs;
          outputs = unscaledAccumulationChannel.Pop();

          // CCS_LOG("outputs: " << outputs);

          Pack1D<ScaleT, 1> inputScale;
          if (params.is_mx_op) {
            inputScale = inputScaleChannel.Pop();
          }

          bool readNewWeights;
          if (params.weightReuseIndex[0] != params.weightReuseIndex[1]) {
            readNewWeights =
                (loop_counters[1][params.weightReuseIndex[1]] == 0) &&
                (loop_counters[1][params.weightReuseIndex[0]] == 0);
          } else {
            readNewWeights =
                (loop_counters[1][params.weightReuseIndex[1]] == 0);
          }
          readNewWeights = readNewWeights || step == 0;
          if (readNewWeights && params.is_mx_op) {
            weightScales = weightScaleChannel.Pop();
          }

          Pack1D<BufferT, NCols> scaled_outputs;

#pragma hls_unroll yes
          for (int i = 0; i < NCols; i++) {
            BufferT scale = static_cast<BufferT>(inputScale[0]) *
                            static_cast<BufferT>(weightScales[i]);
            scaled_outputs[i] = static_cast<BufferT>(outputs[i]);
            if (params.is_mx_op) {
              scaled_outputs[i] = scaled_outputs[i] * scale;
            }
            previous_accumulation.value[i] += scaled_outputs[i];
          }
          int addr = static_cast<ac_int<10, false> >(
                         loop_counters[1][params.weightLoopIndex[1]] *
                         params.loops[1][params.inputXLoopIndex[1]] *
                         params.loops[1][params.inputYLoopIndex[1]]) +
                     static_cast<ac_int<10, false> >(
                         loop_counters[1][params.inputYLoopIndex[1]] *
                         params.loops[1][params.inputXLoopIndex[1]]) +
                     loop_counters[1][params.inputXLoopIndex[1]];

          bool accumulationFinished =
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
            int writeAddress = static_cast<ac_int<10, false> >(
                                   loop_counters[1][params.weightLoopIndex[1]] *
                                   params.loops[1][params.inputXLoopIndex[1]] *
                                   params.loops[1][params.inputYLoopIndex[1]]) +
                               static_cast<ac_int<10, false> >(
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
  }
};
