#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"
#include "Skewer.h"

template <typename IDTYPE, typename INTERMEDIATE_DTYPE, typename ODTYPE,
          int NROWS, int NCOLS>
SC_MODULE(SystolicArray) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<IDTYPE, NROWS> > CCS_INIT_S1(inputs);
  Connections::In<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(psums);
  Connections::Out<Pack1D<ODTYPE, NROWS> > CCS_INIT_S1(outputs);
  Connections::In<Pack1D<ac_int<1, false>, NROWS> > CCS_INIT_S1(swapWeights);

  sc_in<Pack1D<IDTYPE, NCOLS> > CCS_INIT_S1(weights);
  sc_in<bool> CCS_INIT_S1(weightsToggle);

  sc_signal<IDTYPE> inputConnection[NROWS][NCOLS + 1];
  sc_signal<ODTYPE> psumConnection[NROWS + 1][NCOLS];
  sc_signal<IDTYPE> weightConnection[NROWS + 1][NCOLS];
  sc_signal<ac_int<1, false> > weightSwap[NROWS][NCOLS + 1];
  sc_signal<bool> weightPush;
  sc_signal<bool> enable;
  sc_signal<bool> peToggle;
  sc_signal<bool> validOutput[NROWS][NCOLS];

  sc_fifo<Pack1D<ODTYPE, NROWS> > outputFifo;

  SC_CTOR(SystolicArray) {
    ProcessingElement<IDTYPE, INTERMEDIATE_DTYPE, ODTYPE> *pe[NROWS * NCOLS];
    for (int i = 0; i < NROWS; i++) {
      for (int j = 0; j < NCOLS; j++) {
        pe[i * NCOLS + j] =
            new ProcessingElement<IDTYPE, INTERMEDIATE_DTYPE, ODTYPE>(
                sc_gen_unique_name("pe_inst"));
        pe[i * NCOLS + j]->clk(clk);
        pe[i * NCOLS + j]->rstn(rstn);
        pe[i * NCOLS + j]->input_in(inputConnection[i][j]);
        pe[i * NCOLS + j]->weight_in(weightConnection[i][j]);
        pe[i * NCOLS + j]->psum_in(psumConnection[i][j]);
        pe[i * NCOLS + j]->swap_weights_in(weightSwap[i][j]);
        pe[i * NCOLS + j]->swap_weights_out(weightSwap[i][j + 1]);
        pe[i * NCOLS + j]->input_out(inputConnection[i][j + 1]);
        pe[i * NCOLS + j]->weight_out(weightConnection[i + 1][j]);
        pe[i * NCOLS + j]->psum_out(psumConnection[i + 1][j]);
        pe[i * NCOLS + j]->valid_output(validOutput[i][j]);
        pe[i * NCOLS + j]->toggle(peToggle);
        pe[i * NCOLS + j]->push_weights(weightPush);
      }
    }

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(collect_outputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_outputs);
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
      Pack1D<IDTYPE, NCOLS> arrayWeights = weights.read();
#pragma hls_unroll yes
      for (int i = 0; i < NCOLS; i++) {
        weightConnection[0][i].write(arrayWeights[i]);
      }

      weightPush.write(weightsToggle.read() != oldToggle);
      oldToggle = weightsToggle.read();

      wait();
    }
  }

#pragma design modulario < in>
  void run() {
    bool toggle = false;
    bool oldToggle = false;
    enable.write(false);
    peToggle.write(false);

    inputs.Reset();
    psums.Reset();
    swapWeights.Reset();

    for (int i = 0; i < NCOLS; i++) {
      psumConnection[0][i].write(0);
    }

    for (int i = 0; i < NROWS; i++) {
      inputConnection[i][0].write(0);
      weightSwap[i][0].write(0);
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<IDTYPE, NROWS> arrayInput = inputs.Pop();
      Pack1D<ODTYPE, NCOLS> arrayPsum = psums.Pop();
      Pack1D<ac_int<1, false>, NROWS> arraySwap = swapWeights.Pop();

      toggle = !toggle;
      peToggle.write(toggle);

#pragma hls_unroll yes
      for (int i = 0; i < NROWS; i++) {
        inputConnection[i][0].write(arrayInput[i]);
        weightSwap[i][0].write(arraySwap[i]);
      }

#pragma hls_unroll yes
      for (int i = 0; i < NCOLS; i++) {
        psumConnection[0][i].write(arrayPsum[i]);
      }

      oldToggle = toggle;
      wait();
    }
  }

#pragma design modulario < in>
  void collect_outputs() {
    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<ODTYPE, NCOLS> finalOutputs;
#pragma hls_unroll yes
      for (int i = 0; i < NCOLS; i++) {
        finalOutputs[i] = psumConnection[NCOLS][NCOLS - 1 - i];
      }
      if (validOutput[NROWS - 1][NCOLS - 1]) {
        outputFifo.write(finalOutputs);
      }

      wait();
    }
  }

#pragma design modulario < in>
  void send_outputs() {
    outputs.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      outputs.Push(outputFifo.read());
      wait();
    }
  }
};
