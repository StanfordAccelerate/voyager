#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"

template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int NROWS,
          int NCOLS>
SC_MODULE(SystolicArray) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  sc_in<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(inputs);
  sc_in<bool> CCS_INIT_S1(inputsToggle);

  sc_in<Pack1D<WDTYPE, NCOLS> > CCS_INIT_S1(weights);
  sc_in<bool> CCS_INIT_S1(weightsToggle);

  sc_in<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(psums);
  // sc_in<bool> CCS_INIT_S1(psumsValid);

  sc_out<Pack1D<ODTYPE, NCOLS> > CCS_INIT_S1(outputs);
  sc_out<bool> CCS_INIT_S1(outputsValid);

  sc_in<Pack1D<ac_int<1, false>, NROWS> > CCS_INIT_S1(swap_weights);

  sc_signal<IDTYPE> inputConnection[NROWS][NCOLS + 1];
  sc_signal<ODTYPE> psumConnection[NROWS + 1][NCOLS];
  sc_signal<WDTYPE> weightConnection[NROWS + 1][NCOLS];
  sc_signal<ac_int<1, false> > weightSwap[NROWS][NCOLS + 1];
  sc_signal<bool> weightPush;
  sc_signal<bool> enable;

  SC_CTOR(SystolicArray) {
    ProcessingElement<IDTYPE, WDTYPE, ODTYPE> *pe[NROWS][NCOLS];
    for (int i = 0; i < NROWS; i++) {
      for (int j = 0; j < NCOLS; j++) {
        pe[i][j] = new ProcessingElement<IDTYPE, WDTYPE, ODTYPE>(
            sc_gen_unique_name("pe_inst"));
        pe[i][j]->clk(clk);
        pe[i][j]->rstn(rstn);
        pe[i][j]->input_in(inputConnection[i][j]);
        pe[i][j]->weight_in(weightConnection[i][j]);
        pe[i][j]->psum_in(psumConnection[i][j]);
        pe[i][j]->swap_weights_in(weightSwap[i][j]);
        pe[i][j]->swap_weights_out(weightSwap[i][j + 1]);
        pe[i][j]->input_out(inputConnection[i][j + 1]);
        pe[i][j]->weight_out(weightConnection[i + 1][j]);
        pe[i][j]->psum_out(psumConnection[i + 1][j]);
        pe[i][j]->enable(enable);
        pe[i][j]->push_weights(weightPush);
      }
    }

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(push_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void push_weights() {
    for (int i = 0; i < NCOLS; i++) {
      weightConnection[0][i].write(0);
    }
    weightPush.write(false);
    bool oldToggle = false;
    wait();

#pragma hls_pipeline_init_interval 1
    while (true) {
      Pack1D<WDTYPE, NCOLS> arrayWeights = weights.read();
#pragma hls_unroll yes
      for (int i = 0; i < NCOLS; i++) {
        weightConnection[0][i].write(arrayWeights[i]);
      }

      weightPush.write(weightsToggle.read() != oldToggle);
      oldToggle = weightsToggle.read();

      wait();
    }
  }

  void run() {
    bool oldToggle = false;
    outputs.write(Pack1D<ODTYPE, NCOLS>());
    enable.write(false);

    for (int i = 0; i < NCOLS; i++) {
      psumConnection[0][i].write(0);
    }

    for (int i = 0; i < NROWS; i++) {
      inputConnection[i][0].write(0);
      weightSwap[i][0].write(0);
    }

    outputsValid.write(0);

    wait();

#pragma hls_pipeline_init_interval 1
    while (true) {
      enable.write(inputsToggle != oldToggle);
      CCS_LOG("enable: " << enable);
      Pack1D<IDTYPE, NROWS> arrayInput = inputs.read();
      Pack1D<ODTYPE, NCOLS> arrayPsum = psums.read();
      Pack1D<ac_int<1, false>, NROWS> arraySwap = swap_weights.read();

#pragma hls_unroll yes
      for (int i = 0; i < NROWS; i++) {
        inputConnection[i][0].write(arrayInput[i]);
        weightSwap[i][0].write(arraySwap[i]);
      }

#pragma hls_unroll yes
      for (int i = 0; i < NCOLS; i++) {
        psumConnection[0][i].write(arrayPsum[i]);
      }

      Pack1D<ODTYPE, NCOLS> arrayPsumOut;

#pragma hls_unroll yes
      for (int i = 0; i < NCOLS; i++) {
        arrayPsumOut[i] = psumConnection[NCOLS][i];
      }

      outputs.write(arrayPsumOut);
      outputsValid.write(inputsToggle != oldToggle);
      oldToggle = inputsToggle;
      wait();
    }
  }
};
