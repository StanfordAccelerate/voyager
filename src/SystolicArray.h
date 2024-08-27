#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"
#include "Skewer.h"
#include "Tieoff.h"
#include "mc_scverify.h"

template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int NROWS,
          int NCOLS>
SC_MODULE(SystolicArray) {
 private:
  Connections::Combinational<PEInput<IDTYPE> > inputConnection[NROWS][NCOLS];
  Connections::Combinational<ODTYPE> psumConnection[NROWS - 1][NCOLS];
  Connections::Combinational<PEWeight<WDTYPE> > weightConnection[NROWS][NCOLS];

// To speed up HLS synthesis, we instantiate arrays of SC_MODULE on
// the stack. However, for simulation, we will run into stack overflow issues,
// so we have to instantiate them on the heap.
#ifdef __SYNTHESIS__
  ProcessingElement<IDTYPE, WDTYPE, ODTYPE> pe[NROWS * NCOLS];
  Tieoff<PEInput<IDTYPE> > inputConnectionTieoff[NROWS];
  Tieoff<PEWeight<IDTYPE> > weightConnectionTieoff[NCOLS];

#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<IDTYPE> > inputs[NROWS];
  Connections::In<PEWeight<WDTYPE> > weights[NCOLS];
  Connections::In<ODTYPE> psums[NCOLS];
  Connections::Out<ODTYPE> outputs[NCOLS];

  SC_CTOR(SystolicArray) {
    ProcessingElement<IDTYPE, WDTYPE, ODTYPE> *pe_ptr[NROWS * NCOLS];

    for (int i = 0; i < NROWS; i++) {
      for (int j = 0; j < NCOLS; j++) {
#ifdef __SYNTHESIS__
        pe_ptr[i * NCOLS + j] = &pe[i * NCOLS + j];
#else
        pe_ptr[i * NCOLS + j] = new ProcessingElement<IDTYPE, WDTYPE, ODTYPE>(
            sc_gen_unique_name("pe"));
#endif

        pe_ptr[i * NCOLS + j]->clk(clk);
        pe_ptr[i * NCOLS + j]->rstn(rstn);

        if (j == 0) {
          pe_ptr[i * NCOLS + j]->inputIn(inputs[i]);
        } else {
          pe_ptr[i * NCOLS + j]->inputIn(inputConnection[i][j - 1]);
        }

        if (i == 0) {
          pe_ptr[i * NCOLS + j]->psumIn(psums[j]);
        } else {
          pe_ptr[i * NCOLS + j]->psumIn(psumConnection[i - 1][j]);
        }

        if (i == 0) {
          pe_ptr[i * NCOLS + j]->weightIn(weights[j]);
        } else {
          pe_ptr[i * NCOLS + j]->weightIn(weightConnection[i - 1][j]);
        }

        pe_ptr[i * NCOLS + j]->weightOut(weightConnection[i][j]);
        pe_ptr[i * NCOLS + j]->inputOut(inputConnection[i][j]);

        if (i == NROWS - 1) {
          pe_ptr[i * NCOLS + j]->psumOut(outputs[j]);
        } else {
          pe_ptr[i * NCOLS + j]->psumOut(psumConnection[i][j]);
        }
      }
    }

    // Tie off unused Connections

    // last column of array for inputs
    Tieoff<PEInput<IDTYPE> > *inputConnectionTieoff_ptr[NROWS];
    for (int i = 0; i < NROWS; i++) {
#ifdef __SYNTHESIS__
      inputConnectionTieoff_ptr[i] = &inputConnectionTieoff[i];
#else
      inputConnectionTieoff_ptr[i] =
          new Tieoff<PEInput<IDTYPE> >(sc_gen_unique_name("tieoff"));
#endif
      inputConnectionTieoff_ptr[i]->in(inputConnection[i][NCOLS - 1]);
#ifdef CONNECTIONS_FAST_SIM
      // we need to connect clock and reset if using fast sim
      inputConnectionTieoff_ptr[i]->clk(clk);
      inputConnectionTieoff_ptr[i]->rstn(rstn);
#endif
    }

    // last row for weights
    Tieoff<PEWeight<IDTYPE> > *weightConnectionTieoff_ptr[NCOLS];
    for (int i = 0; i < NCOLS; i++) {
#ifdef __SYNTHESIS__
      weightConnectionTieoff_ptr[i] = &weightConnectionTieoff[i];
#else
      weightConnectionTieoff_ptr[i] =
          new Tieoff<PEWeight<IDTYPE> >(sc_gen_unique_name("tieoff"));
#endif
      weightConnectionTieoff_ptr[i]->in(weightConnection[NROWS - 1][i]);
#ifdef CONNECTIONS_FAST_SIM
      // we need to connect clock and reset if using fast sim
      weightConnectionTieoff_ptr[i]->clk(clk);
      weightConnectionTieoff_ptr[i]->rstn(rstn);
#endif
    }
  }
};
