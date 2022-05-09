#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ParamsDeserializer.h"
#include "SystolicArray.h"

template <typename IDTYPE, typename ODTYPE, int NROWS, int NCOLS,
          int BUFFER_SIZE>
SC_MODULE(MatrixProcessor) {
 private:
  Connections::SyncChannel CCS_INIT_S1(weightLoadDone);
  Connections::SyncChannel CCS_INIT_S1(weightSwapDone);

  SerializedSkewer<IDTYPE, NROWS> CCS_INIT_S1(inputSkewer);
  Connections::Combinational<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(
      inputSkewerDin);

  SerializedSkewer<ac_int<1, false>, NROWS> CCS_INIT_S1(weightSwapSkewer);
  Connections::Combinational<Pack1D<ac_int<1, false>, NROWS> > CCS_INIT_S1(
      weightSwapSkewerDin);

  SerializedSkewer<ODTYPE, NROWS> CCS_INIT_S1(psumInSkewer);
  Connections::Combinational<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(
      psumInSkewerDin);

  DeserializedSkewer<ODTYPE, NROWS> CCS_INIT_S1(psumOutSkewer);
  Connections::Combinational<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(
      psumOutSkewerDout);

  SystolicArray<IDTYPE, ODTYPE, NROWS, NCOLS> CCS_INIT_S1(systolicArray);

  MatrixParamsDeserializer CCS_INIT_S1(paramsDeserializer);

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(inputsChannel);
  Connections::In<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(weightsChannel);
  Connections::Out<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(outputsChannel);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);
  Connections::Combinational<IDTYPE> inputsToSystolicArray[NROWS];
  Connections::Combinational<ac_int<1, false> >
      weightSwapToSystolicArray[NROWS];
  Connections::Combinational<ODTYPE> psumsToSystolicArray[NCOLS];
  Connections::Combinational<ODTYPE> outputsFromSystolicArray[NCOLS];
  Connections::Combinational<Pack1D<IDTYPE, NCOLS> > CCS_INIT_S1(
      weightsToSystolicArray);

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

    weightSwapSkewer.clk(clk);
    weightSwapSkewer.rstn(rstn);
    weightSwapSkewer.din(weightSwapSkewerDin);
    for (int i = 0; i < NROWS; i++) {
      weightSwapSkewer.dout[i](weightSwapToSystolicArray[i]);
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
    for (int i = 0; i < NROWS; i++) {
      systolicArray.swapWeights[i](weightSwapToSystolicArray[i]);
    }
    for (int i = 0; i < NCOLS; i++) {
      systolicArray.psums[i](psumsToSystolicArray[i]);
    }
    for (int i = 0; i < NCOLS; i++) {
      systolicArray.outputs[i](outputsFromSystolicArray[i]);
    }
    systolicArray.weights(weightsToSystolicArray);
    systolicArray.weightSwapDone(weightSwapDone);

    SC_THREAD(process_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void process_weights() {
    weightsChannel.Reset();
    weightsToSystolicArray.ResetWrite();
    weightLoadDone.ResetWrite();
    weightSwapDone.ResetRead();

    wait();

    while (true) {
#pragma hls_pipeline_init_interval 1
      for (int weight_count = 0; weight_count < NROWS; weight_count++) {
        Pack1D<IDTYPE, NCOLS> arrayWeights = weightsChannel.Pop();
        // std::cout << "Weights: " << arrayWeights << std::endl;

        weightsToSystolicArray.Push(arrayWeights);
      }

      weightLoadDone.SyncPush();

      // wait for swap signal to propagate throughout the entire array
      weightSwapDone.SyncPop();
    }
  }

  void run() {
    paramsIn.ResetRead();

    inputSkewerDin.ResetWrite();
    inputsChannel.Reset();
    weightLoadDone.ResetRead();
    psumInSkewerDin.ResetWrite();
    outputsChannel.Reset();
    // weightSwapToSystolicArray.ResetWrite();
    psumOutSkewerDout.ResetRead();
    weightSwapSkewerDin.ResetWrite();

    bool toggle = false;
    bool weightFillToggle = false;

    int swapDoneCount = 0;

    wait();

    while (true) {
      MatrixParams params = paramsIn.Pop();

      int loop_counters[2][6];
      int loop_counters_out[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_counters[i][j] = 0;
          loop_counters_out[i][j] = 0;
        }
      }

      int totalOps = 1;
#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          totalOps *= params.loops[i][j];
        }
      }

      Pack1D<ODTYPE, NCOLS> accumulation_buffer[BUFFER_SIZE];

      int step = 0;
      int outputStep = 0;
      int outputCounter = 0;

// TODO: figure out why this is the case
#ifdef __SYNTHESIS__
      const int latency = 0;
#elif CONNECTIONS_FAST_SIM
      const int latency = 0;
#else
      const int latency = 0;
#endif

      // Push inputs and psums into the array
      // Pipelined across tiles

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < totalOps) {
#ifndef __SYNTHESIS__
        if (step % 1000 == 0) {
          CCS_LOG("step " << step << " out of " << totalOps);
        }
#endif

        Pack1D<ac_int<1, false>, NROWS> weightSwap;
        bool newWeights = loop_counters[1][params.weightReuseIndex[0]] == 0 &&
                          loop_counters[1][params.weightReuseIndex[1]] == 0;
        if (newWeights && step < totalOps) {
          // wait for weight loading to finish
          weightLoadDone.SyncPop();

          swapDoneCount = 0;

#pragma hls_unroll yes
          for (int i = 0; i < NROWS; i++) {
            weightSwap.value[i] = true;
          }
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < NROWS; i++) {
            weightSwap.value[i] = false;
          }
        }

        Pack1D<IDTYPE, NROWS> inputs;
        if (step < totalOps) {
          inputs = inputsChannel.Pop();
        }

        inputSkewerDin.Push(inputs);
        weightSwapSkewerDin.Push(weightSwap);

        Pack1D<ODTYPE, NCOLS> psum;
#pragma hls_unroll yes
        for (int i = 0; i < NCOLS; i++) {
          psum.value[i].setZero();
        }

        bool firstAccumulation =
            loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
            loop_counters[1][params.fxIndex] == 0 &&
            loop_counters[1][params.fyIndex] == 0;

        if ((!firstAccumulation || params.ACC_FROM_ACC) && step < totalOps) {
          int readAddress = loop_counters[1][params.weightLoopIndex[1]] *
                                params.loops[1][params.inputXLoopIndex[1]] *
                                params.loops[1][params.inputYLoopIndex[1]] +
                            loop_counters[1][params.inputYLoopIndex[1]] *
                                params.loops[1][params.inputXLoopIndex[1]] +
                            loop_counters[1][params.inputXLoopIndex[1]];
#ifdef __SYNTHESIS__
        READ_ACC_BUFFER:
#endif
          psum = accumulation_buffer[readAddress];
          // DLOG("readAddress: " << readAddress << " psum " << psum);
        }

        psumInSkewerDin.Push(psum);

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

          if (accumulationFinished && !params.STORE_IN_ACC) {
            outputsChannel.Push(outputs);
            DLOG("matrix processor output: " << outputs);
            // std::cout << "Output: " << outputs << std::endl;
          } else {
            int writeAddress = loop_counters_out[1][params.weightLoopIndex[1]] *
                                   params.loops[1][params.inputXLoopIndex[1]] *
                                   params.loops[1][params.inputYLoopIndex[1]] +
                               loop_counters_out[1][params.inputYLoopIndex[1]] *
                                   params.loops[1][params.inputXLoopIndex[1]] +
                               loop_counters_out[1][params.inputXLoopIndex[1]];

#ifdef __SYNTHESIS__
          WRITE_ACC_BUFFER:
#endif
            accumulation_buffer[writeAddress] = outputs;
            // DLOG("writeAddress: " << writeAddress << " val "
            //                          << outputs);
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
          if (accumulationFinished && !params.STORE_IN_ACC) {
            outputsChannel.Push(outputs);
            DLOG("matrix processor output: " << outputs);
            // std::cout << "Output: " << outputs << std::endl;
          } else {
            int writeAddress = loop_counters_out[1][params.weightLoopIndex[1]] *
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
    }
  }
};
