#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "SystolicArray.h"

template <typename IDTYPE, typename INTERMEDIATE_DTYPE, typename ODTYPE,
          int NROWS, int NCOLS, int BUFFER_SIZE>
SC_MODULE(MatrixProcessor) {
 private:
  Connections::SyncChannel CCS_INIT_S1(weightSync);
  sc_signal<bool> CCS_INIT_S1(weightReady);
  sc_signal<bool> CCS_INIT_S1(weightFill);

  sc_signal<bool> CCS_INIT_S1(paramsReady);

  sc_signal<bool> CCS_INIT_S1(toggleOut);

  Skewer<IDTYPE, NROWS> CCS_INIT_S1(inputSkewer);
  Connections::Combinational<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(
      inputSkewerDin);

  Skewer<ODTYPE, NROWS> CCS_INIT_S1(psumInSkewer);
  Connections::Combinational<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(
      psumInSkewerDin);

  Skewer<ODTYPE, NROWS> CCS_INIT_S1(psumOutSkewer);
  Connections::Combinational<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(
      psumOutSkewerDout);

  Skewer<ac_int<1, false>, NROWS> CCS_INIT_S1(weightSwapSkewer);
  Connections::Combinational<Pack1D<ac_int<1, false>, NROWS> > CCS_INIT_S1(
      weightSwapSkewerDin);
  SystolicArray<IDTYPE, INTERMEDIATE_DTYPE, ODTYPE, NROWS, NCOLS> CCS_INIT_S1(
      systolicArray);

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(inputsChannel);
  Connections::In<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(weightsChannel);
  Connections::Out<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(outputsChannel);

  Connections::In<MatrixParams> CCS_INIT_S1(paramsIn);

  Connections::Combinational<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(
      inputsToSystolicArray);
  Connections::Combinational<Pack1D<ODTYPE, NCOLS> > CCS_INIT_S1(
      psumsToSystolicArray);
  Connections::Combinational<Pack1D<ODTYPE, NCOLS> > CCS_INIT_S1(
      outputsFromSystolicArray);
  Connections::Combinational<Pack1D<ac_int<1, false>, NROWS> > CCS_INIT_S1(
      weightSwapToSystolicArray);

  sc_signal<Pack1D<IDTYPE, NCOLS> > CCS_INIT_S1(weightsToSystolicArray);
  sc_signal<bool> CCS_INIT_S1(weightsToggle);

  SC_CTOR(MatrixProcessor) {
    systolicArray.clk(clk);
    systolicArray.rstn(rstn);

    systolicArray.inputs(inputsToSystolicArray);
    systolicArray.weights(weightsToSystolicArray);
    systolicArray.weightsToggle(weightsToggle);
    systolicArray.psums(psumsToSystolicArray);
    systolicArray.outputs(outputsFromSystolicArray);
    systolicArray.swapWeights(weightSwapToSystolicArray);

    inputSkewer.clk(clk);
    inputSkewer.rstn(rstn);
    inputSkewer.din(inputSkewerDin);
    inputSkewer.dout(inputsToSystolicArray);

    psumInSkewer.clk(clk);
    psumInSkewer.rstn(rstn);
    psumInSkewer.din(psumInSkewerDin);
    psumInSkewer.dout(psumsToSystolicArray);

    weightSwapSkewer.clk(clk);
    weightSwapSkewer.rstn(rstn);
    weightSwapSkewer.din(weightSwapSkewerDin);
    weightSwapSkewer.dout(weightSwapToSystolicArray);

    psumOutSkewer.clk(clk);
    psumOutSkewer.rstn(rstn);
    psumOutSkewer.din(outputsFromSystolicArray);
    psumOutSkewer.dout(psumOutSkewerDout);

    SC_THREAD(process_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  //   void convert_inputs() {
  //     inputsChannel.Reset()
  //     convertedInputsChannel.ResetWrite();
  //     wait();

  // #pragma hls_pipeline_init_interval 1
  //     while (true) {
  //       Pack1D<IDTYPE, NROWS> val = inputsChannel.Pop();
  //       Pack1D<IDTYPE, NROWS> convertedVal;
  // #pragma hls_unroll yes
  //       for (int i = 0; i < NROWS; i++) {
  //         convertedVal[i] = val[i];
  //       }
  //     }
  //   }
  //   void convert_weights() {
  //     weightsChannel.Reset();
  //     convertedWeightsChannel.ResetWrite();
  //     wait();

  // #pragma hls_pipeline_init_interval 1
  //     while (true) {
  //       Pack1D<IDTYPE, NROWS> val = weightsChannel.Pop();
  //       Pack1D<IDTYPE, NROWS> convertedVal;
  // #pragma hls_unroll yes
  //       for (int i = 0; i < NROWS; i++) {
  //         convertedVal[i] = val[i];
  //       }
  //     }
  //   }
  //   void convert_inputs() {
  //     outputsChannel.Reset();
  //     unconvertedOutputsChannel.ResetRead();
  //     wait();

  // #pragma hls_pipeline_init_interval 1
  //     while (true) {
  //       Pack1D<IDTYPE, NROWS> val = unconvertedOutputsChannel.Pop();
  //       Pack1D<IDTYPE, NROWS> convertedVal;
  // #pragma hls_unroll yes
  //       for (int i = 0; i < NROWS; i++) {
  //         convertedVal[i] = val[i];
  //       }
  //     }
  //   }

  void process_weights() {
    weightsChannel.Reset();
    weightsToSystolicArray.write(Pack1D<IDTYPE, NCOLS>());
    weightReady.write(false);
    weightSync.ResetRead();
    weightsToggle.write(false);

    bool toggle = false;
    bool weightFillToggle = false;

    wait();

    while (true) {
#pragma hls_pipeline_init_interval 1
      for (int weight_count = 0; weight_count < NROWS; weight_count++) {
        Pack1D<IDTYPE, NCOLS> arrayWeights = weightsChannel.Pop();
        // std::cout << "Weights: " << arrayWeights << std::endl;

        weightsToSystolicArray.write(arrayWeights);
        toggle = !toggle;
        weightsToggle.write(toggle);
        wait();
      }

      weightReady.write(true);

      while (weightFillToggle == weightFill.read()) {
        wait();
      }
      weightFillToggle = weightFill.read();
      weightReady.write(false);

      // wait for swap signal to propagate throughout the entire array
      for (int i = 0; i < NROWS + NCOLS; i++) {
        wait();
      }
    }
  }

  void run() {
    paramsIn.Reset();

    inputSkewerDin.ResetWrite();
    toggleOut.write(false);
    inputsChannel.Reset();
    weightSync.ResetWrite();
    psumInSkewerDin.ResetWrite();
    outputsChannel.Reset();
    weightFill.write(false);
    weightSwapSkewerDin.ResetWrite();
    psumOutSkewerDout.ResetRead();

    bool toggle = false;
    bool weightFillToggle = false;

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
      int outputCounter = 0;

// TODO: figure out why this is the case
#ifdef __SYNTHESIS__
      const int latency = 7;
#else
      const int latency = 2;
#endif

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (step < totalOps + (NROWS - 1) + (NCOLS - 1) + latency) {
#ifndef __SYNTHESIS__
        if (step % 10000 == 0) {
          CCS_LOG("step " << step << " out of " << totalOps);
        }
#endif

        Pack1D<ac_int<1, false>, NROWS> weightSwap;
        bool newWeights = loop_counters[1][params.weightReuseIndex[0]] == 0 &&
                          loop_counters[1][params.weightReuseIndex[1]] == 0;
        if (newWeights && step < totalOps) {
          // CCS_LOG("*****");
          while (!weightReady) {
            wait();
          }
          // CCS_LOG("weight ready");

#pragma hls_unroll yes
          for (int i = 0; i < NROWS; i++) {
            weightSwap.value[i] = true;
          }

          weightFillToggle = !weightFillToggle;
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < NROWS; i++) {
            weightSwap.value[i] = false;
          }
        }

        Pack1D<IDTYPE, NROWS> inputs;
        if (step < totalOps) {
          inputs = inputsChannel.Pop();

          // CCS_LOG("input: " << inputs);
        }
        toggle = !toggle;
        toggleOut.write(toggle);

        inputSkewerDin.Push(inputs);
        weightSwapSkewerDin.Push(weightSwap);

        weightFill.write(weightFillToggle);

        Pack1D<ODTYPE, NCOLS> psum;
#pragma hls_unroll yes
        for (int i = 0; i < NCOLS; i++) {
          psum.value[i] = 0;
        }

        if (!(loop_counters[1][params.reductionLoopIndex[1]] == 0 &&
              loop_counters[1][params.fxIndex] == 0 &&
              loop_counters[1][params.fyIndex] == 0) &&
            step < totalOps) {
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
          // CCS_LOG("readAddress: " << readAddress << " psum " << psum);
        }

        psumInSkewerDin.Push(psum);

        wait();

        Pack1D<ODTYPE, NCOLS> outputs;
        psumOutSkewerDout.PopNB(outputs);
        Pack1D<ODTYPE, NCOLS> flippedOutputs;

#pragma hls_unroll yes
        for (int i = 0; i < NCOLS; i++) {
          flippedOutputs[i] = outputs[NCOLS - 1 - i];
        }

        if (step >= (NCOLS - 1) + (NROWS - 1) + latency) {
          bool accumulationFinished =
              (loop_counters_out[1][params.reductionLoopIndex[1]] ==
               params.loops[1][params.reductionLoopIndex[1]] - 1) &&
              (loop_counters_out[1][params.fxIndex] ==
               params.loops[1][params.fxIndex] - 1) &&
              (loop_counters_out[1][params.fyIndex] ==
               params.loops[1][params.fyIndex] - 1);
          if (accumulationFinished) {
            outputsChannel.Push(flippedOutputs);
            // std::cout << "Output: " << flippedOutputs << std::endl;
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
            accumulation_buffer[writeAddress] = flippedOutputs;
            // CCS_LOG("writeAddress: " << writeAddress << " val "
            //                          << flippedOutputs);
          }
        }

        if (step >= (NCOLS - 1) + (NROWS - 1) + latency) {
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
    }
  }
};
