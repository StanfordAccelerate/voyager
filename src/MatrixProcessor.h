#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ParamsDeserializer.h"
#include "SystolicArray.h"

template <typename IDTYPE, typename ODTYPE, typename ACCUM_BUFFER_TYPE,
          bool IS_MX, int NROWS, int NCOLS, int BUFFER_SIZE>
SC_MODULE(MatrixProcessor) {
 private:
  Connections::SyncChannel CCS_INIT_S1(weightLoadDone);
  Connections::SyncChannel CCS_INIT_S1(weightSwapDone);

  MultiInputSerializedSkewer<IDTYPE, typename IDTYPE::AccumulationDatatype,
                             NROWS>
      CCS_INIT_S1(inputSkewer);
  Connections::Combinational<Pack1D<PEInput<IDTYPE>, NROWS> > CCS_INIT_S1(
      inputSkewerDin);

  WeightSerializedSkewer<IDTYPE, typename IDTYPE::AccumulationDatatype, NCOLS>
      CCS_INIT_S1(weightSkewer);
  Connections::Combinational<Pack1D<PEWeight<IDTYPE>, NCOLS> > CCS_INIT_S1(
      weightSkewerDin);

  SerializedSkewer<ODTYPE, typename ODTYPE::AccumulationDatatype, NCOLS>
      CCS_INIT_S1(psumInSkewer);
  Connections::Combinational<Pack1D<ODTYPE, NCOLS> > CCS_INIT_S1(
      psumInSkewerDin);

  DeserializedSkewer<typename ODTYPE::AccumulationDatatype, ODTYPE, NCOLS>
      CCS_INIT_S1(psumOutSkewer);
  Connections::Combinational<Pack1D<ODTYPE, NCOLS> > CCS_INIT_S1(
      psumOutSkewerDout);

  SystolicArray<typename IDTYPE::AccumulationDatatype,
                typename IDTYPE::AccumulationDatatype,
                typename ODTYPE::AccumulationDatatype, NROWS, NCOLS>
      CCS_INIT_S1(systolicArray);

  MatrixParamsDeserializer<1> CCS_INIT_S1(paramsDeserializer);

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int LOOP_WIDTH =
      (10 + int_log2(16 / (IC_DIMENSION < OC_DIMENSION ? IC_DIMENSION
                                                       : OC_DIMENSION)));

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(inputsChannel);
  Connections::In<Pack1D<IDTYPE, NCOLS> > CCS_INIT_S1(weightsChannel);
  Connections::In<Pack1D<ACCUM_BUFFER_TYPE, NCOLS> > CCS_INIT_S1(biasChannel);
  Connections::Out<Pack1D<ACCUM_BUFFER_TYPE, NCOLS> > CCS_INIT_S1(
      outputsChannel);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<PEInput<typename IDTYPE::AccumulationDatatype> >
      inputsToSystolicArray[NROWS];
  // Connections::Combinational<ac_int<1, false> >
  //     weightSwapToSystolicArray[NROWS];
  Connections::Combinational<typename ODTYPE::AccumulationDatatype>
      psumsToSystolicArray[NCOLS];
  Connections::Combinational<typename ODTYPE::AccumulationDatatype>
      outputsFromSystolicArray[NCOLS];
  Connections::Combinational<PEWeight<typename IDTYPE::AccumulationDatatype> >
      weightsToSystolicArray[NCOLS];

  // only used for MX
  Connections::ConditionalCombinational<MatrixParams, IS_MX> CCS_INIT_S1(
      accumulationBufferParams);
  Connections::ConditionalCombinational<Pack1D<ODTYPE, NCOLS>, IS_MX>
      CCS_INIT_S1(unscaledAccumulationChannel);

  Connections::ConditionalIn<Pack1D<IDTYPE, 1>, IS_MX> CCS_INIT_S1(
      inputScaleChannel);
  Connections::ConditionalIn<Pack1D<IDTYPE, NCOLS>, IS_MX> CCS_INIT_S1(
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
    for (int i = 0; i < NROWS; i++) {
      inputSkewer.dout[i](inputsToSystolicArray[i]);
    }

    weightSkewer.clk(clk);
    weightSkewer.rstn(rstn);
    weightSkewer.din(weightSkewerDin);
    for (int i = 0; i < NCOLS; i++) {
      weightSkewer.dout[i](weightsToSystolicArray[i]);
    }

    psumInSkewer.clk(clk);
    psumInSkewer.rstn(rstn);
    psumInSkewer.din(psumInSkewerDin);
    for (int i = 0; i < NCOLS; i++) {
      psumInSkewer.dout[i](psumsToSystolicArray[i]);
    }

    psumOutSkewer.clk(clk);
    psumOutSkewer.rstn(rstn);
    for (int i = 0; i < NCOLS; i++) {
      psumOutSkewer.din[i](outputsFromSystolicArray[i]);
    }
    psumOutSkewer.dout(psumOutSkewerDout);

    systolicArray.clk(clk);
    systolicArray.rstn(rstn);
    for (int i = 0; i < NROWS; i++) {
      systolicArray.inputs[i](inputsToSystolicArray[i]);
    }
    for (int i = 0; i < NCOLS; i++) {
      systolicArray.psums[i](psumsToSystolicArray[i]);
    }
    for (int i = 0; i < NCOLS; i++) {
      systolicArray.outputs[i](outputsFromSystolicArray[i]);
    }
    for (int i = 0; i < NCOLS; i++) {
      systolicArray.weights[i](weightsToSystolicArray[i]);
    }

    SC_THREAD(process_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    if constexpr (IS_MX) {
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
      for (int weight_count = 0; weight_count < NROWS; weight_count++) {
        Pack1D<IDTYPE, NCOLS> arrayWeights = weightsChannel.Pop();
        // std::cout << "Weights: " << arrayWeights << std::endl;
        Pack1D<PEWeight<IDTYPE>, NCOLS> weights;
#pragma hls_unroll yes
        for (int i = 0; i < NCOLS; i++) {
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

    if constexpr (IS_MX) {
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

      if constexpr (IS_MX) {
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
      Pack1D<ACCUM_BUFFER_TYPE, NCOLS> accumulation_buffer[BUFFER_SIZE];
#else
      Pack1D<ACCUM_BUFFER_TYPE, NCOLS> *accumulation_buffer =
          new Pack1D<ACCUM_BUFFER_TYPE, NCOLS>[BUFFER_SIZE];
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

      Pack1D<ODTYPE, NCOLS> bias;

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
      //         Pack1D<PEInput<IDTYPE>, NROWS> inputs;
      // #pragma hls_unroll yes
      //         for (int i = 0; i < NROWS; i++) {
      //           inputs[i].swapWeights = count == 0;
      //         }

      //         Pack1D<ODTYPE, NCOLS> psum;

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

        // Pack1D<ac_int<1, false>, NROWS> weightSwap;
        Pack1D<PEInput<IDTYPE>, NROWS> inputs;
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
          for (int i = 0; i < NROWS; i++) {
            inputs[i].swapWeights = true;
            // weightSwap.value[i] = true;
          }
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < NROWS; i++) {
            // weightSwap.value[i] = false;
            inputs[i].swapWeights = false;
          }
        }

        // Pack1D<IDTYPE, NROWS> inputs;
        if (step < totalOps && !stallInputs) {
          Pack1D<IDTYPE, NROWS> inputsData = inputsChannel.Pop();
#pragma hls_unroll yes
          for (int i = 0; i < NROWS; i++) {
            inputs[i].data = inputsData[i];
            // #ifdef HYBRID_FP8
            // inputs[i].castToE5M2 = params.COMBINE_GRADS;
            // #endif
          }
        }

        Pack1D<ODTYPE, NCOLS> psum;
#pragma hls_unroll yes
        for (int i = 0; i < NCOLS; i++) {
          psum.value[i].setZero();
        }

        if constexpr (!IS_MX) {
          bool firstAccumulation =
              loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
              loop_counters[1][params.fxIndex] == 0 &&
              loop_counters[1][params.fyIndex] == 0;

          if ((!firstAccumulation || params.ACC_FROM_ACC) && step < totalOps &&
              !stallInputs) {
            ac_int<int_log2(BUFFER_SIZE), false> readAddress =
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
          } else if (params.BIAS && step < totalOps && !stallInputs) {
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

        Pack1D<ODTYPE, NCOLS> outputs;
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

          if constexpr (IS_MX) {
            unscaledAccumulationChannel.Push(outputs);
          } else {
            if (accumulationFinished && !params.STORE_IN_ACC) {
              outputsChannel.Push(outputs);
              // DLOG("matrix processor output: " << outputs);
              // std::cout << "Output: " << outputs << std::endl;
            } else {
              ac_int<int_log2(BUFFER_SIZE), false> writeAddress =
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
        Pack1D<ODTYPE, NCOLS> outputs;
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

          if constexpr (IS_MX) {
            unscaledAccumulationChannel.Push(outputs);

          } else {
            if (accumulationFinished && !params.STORE_IN_ACC) {
              outputsChannel.Push(outputs);
              DLOG("matrix processor output: " << outputs);
              // std::cout << "Output: " << outputs << std::endl;
            } else {
              ac_int<int_log2(BUFFER_SIZE), false> writeAddress =
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
    if constexpr (IS_MX) {
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
      UNROLL_0:
        for (int i = 0; i < 2; i++) {
        UNROLL_1:
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
        Pack1D<ACCUM_BUFFER_TYPE, NCOLS> accumulation_buffer[BUFFER_SIZE];
#else
        Pack1D<ACCUM_BUFFER_TYPE, NCOLS> *accumulation_buffer =
            new Pack1D<ACCUM_BUFFER_TYPE, NCOLS>[BUFFER_SIZE];
#endif

        ac_int<32, false> step = 0;

        Pack1D<ACCUM_BUFFER_TYPE, NCOLS> bias;
        Pack1D<IDTYPE, NCOLS> weightScales;

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

          Pack1D<ACCUM_BUFFER_TYPE, NCOLS> previous_accumulation;

// intentionally not labeled with UNROLL_2 as it gets optimized away
#pragma hls_unroll yes
          for (int i = 0; i < NCOLS; i++) {
            previous_accumulation.value[i].setZero();
          }

          if (firstAccumulation) {
            if (params.BIAS) {
              bool readBias = true;
            UNROLL_2:
              for (int i = 0; i < 4; i++) {
                readBias =
                    readBias && (loop_counters[1][biasReuseIndices[i]] == 0);
              }

              if (readBias) {
                bias = biasChannel.Pop();
                // CCS_LOG("bias: " << bias);
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

          Pack1D<ODTYPE, NCOLS> outputs;
          outputs = unscaledAccumulationChannel.Pop();

          // CCS_LOG("outputs: " << outputs);

          // TODO: add scale factors and scale the psum
          Pack1D<IDTYPE, 1> inputScale;
          if (params.MX) {
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
          if (readNewWeights && params.MX) {
            weightScales = weightScaleChannel.Pop();
            // CCS_LOG("weightScales: " << weightScales);
          }

          Pack1D<ACCUM_BUFFER_TYPE, NCOLS> scaled_outputs;

        UNROLL_3:
          for (int i = 0; i < NCOLS; i++) {
            IDTYPE scale = inputScale[0] + weightScales[i];
            // CCS_LOG("scale: " << scale);
            // IDTYPE scale = inputScale[0];
            scaled_outputs[i] = static_cast<ACCUM_BUFFER_TYPE>(outputs[i]);
            // CCS_LOG("scaled_outputs: " << scaled_outputs[i]);
            if (params.MX) {
              scaled_outputs[i].expScale(scale.int_val);
            }
            // CCS_LOG("scaled_outputs after scale: " << scaled_outputs[i]);
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
          if (addr == 0) {
            IDTYPE scale = inputScale[0] + weightScales[0];
            // CCS_LOG(inputScale[0] << " + " << weightScales[0] << " = " <<
            // scale); CCS_LOG("scaled psums: " << previous_accumulation);
          }

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
        UNROLL_4:
          for (int i = 1; i >= 0; i--) {
          UNROLL_5:
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
