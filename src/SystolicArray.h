#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"
#include "Skewer.h"

template <typename IDTYPE, typename ODTYPE, int NROWS, int NCOLS>
SC_MODULE(SystolicArray) {
 private:
  Connections::Combinational<IDTYPE> inputConnection[NROWS][NCOLS];
  Connections::Combinational<ODTYPE> psumConnection[NROWS - 1][NCOLS];
  sc_signal<IDTYPE> weightConnection[NROWS + 1][NCOLS];
  sc_signal<bool> weightValid;
  Connections::Combinational<ac_int<1, false> > weightSwap[NROWS * NCOLS - 1];
  Connections::Combinational<ac_int<1, false> > weightSwapFinal;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<IDTYPE> inputs[NROWS];
  Connections::In<ac_int<1, false> > swapWeights[NROWS];
  Connections::In<Pack1D<IDTYPE, NCOLS> > CCS_INIT_S1(weights);
  Connections::In<ODTYPE> psums[NCOLS];
  Connections::Out<ODTYPE> outputs[NCOLS];
  Connections::SyncOut CCS_INIT_S1(weightSwapDone);

  SC_CTOR(SystolicArray) {
    ProcessingElement<IDTYPE, ODTYPE> *pe[NROWS * NCOLS];
    for (int i = 0; i < NROWS; i++) {
      for (int j = 0; j < NCOLS; j++) {
        pe[i * NCOLS + j] = new ProcessingElement<IDTYPE, ODTYPE>(
            sc_gen_unique_name("pe_inst"));
        pe[i * NCOLS + j]->clk(clk);
        pe[i * NCOLS + j]->rstn(rstn);
        if (j == 0) {
          pe[i * NCOLS + j]->inputIn(inputs[i]);
        } else {
          pe[i * NCOLS + j]->inputIn(inputConnection[i][j - 1]);
        }
        pe[i * NCOLS + j]->weightIn(weightConnection[i][j]);
        pe[i * NCOLS + j]->weightValid(weightValid);
        if (i == 0) {
          pe[i * NCOLS + j]->psumIn(psums[j]);
        } else {
          pe[i * NCOLS + j]->psumIn(psumConnection[i - 1][j]);
        }
        if (j == 0) {
          pe[i * NCOLS + j]->weightSwapIn(swapWeights[i]);
        } else {
          pe[i * NCOLS + j]->weightSwapIn(weightSwap[i * NCOLS + j - 1]);
        }
        if (i == NROWS - 1 && j == NCOLS - 1) {
          pe[i * NCOLS + j]->weightSwapOut(weightSwapFinal);
        } else {
          pe[i * NCOLS + j]->weightSwapOut(weightSwap[i * NCOLS + j]);
        }
        pe[i * NCOLS + j]->inputOut(inputConnection[i][j]);
        pe[i * NCOLS + j]->weightOut(weightConnection[i + 1][j]);
        if (i == NROWS - 1) {
          pe[i * NCOLS + j]->psumOut(outputs[j]);
        } else {
          pe[i * NCOLS + j]->psumOut(psumConnection[i][j]);
        }
      }
    }

    SC_THREAD(sendWeights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(tieoff);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(checkSwapDone);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  // void sendWeightSwap() {
  //   swapWeights.Reset();
  //   for (int i = 0; i < NROWS; i++) {
  //     weightSwap[i][0].ResetWrite();
  //   }

  //   wait();

  //   while (true) {
  //     swapWeights.SyncPop();

  //     // push the weight swap to the next row every cycle
  //     // cycle 0- row 0
  //     // cycle 1- row 1
  //     // cycle 2- row 2
  //     // and so on...
  //     for (int i = 0; i < NROWS; i++) {
  //       weightSwap[i][0].SyncPush();
  //     }
  //   }
  // }

  void checkSwapDone() {
    weightSwapFinal.ResetRead();
    weightSwapDone.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      ac_int<1, false> swap = weightSwapFinal.Pop();
      if (swap) {
        weightSwapDone.SyncPush();
      }
    }
  }

  void tieoff() {
    // Reset all the unused Connections
    for (int i = 0; i < NROWS; i++) {
      inputConnection[i][NCOLS - 1].ResetRead();
    }

    for (int i = 0; i < NROWS - 1; i++) {
      weightSwap[i * NCOLS + NCOLS - 1].ResetRead();
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
#pragma hls_unroll yes
      for (int i = 0; i < NROWS; i++) {
        IDTYPE unusedInput;
        inputConnection[i][NCOLS - 1].PopNB(unusedInput);
      }
#pragma hls_unroll yes
      for (int i = 0; i < NROWS - 1; i++) {
        ac_int<1, false> unusedSwap;
        weightSwap[i * NCOLS + NCOLS - 1].PopNB(unusedSwap);
      }
      wait();
    }
  }

  void sendWeights() {
    for (int j = 0; j < NCOLS; j++) {
      weightConnection[0][j].write(IDTYPE());
    }
    weightValid.write(false);

    weights.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<IDTYPE, NCOLS> arrayWeights;
      if (weights.PopNB(arrayWeights)) {
        weightValid.write(true);
      } else {
        weightValid.write(false);
      }
#pragma hls_unroll yes
      for (int i = 0; i < NCOLS; i++) {
        weightConnection[0][i].write(arrayWeights[i]);
      }

      wait();
    }
  }
};
