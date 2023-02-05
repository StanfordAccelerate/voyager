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

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<IDTYPE> > inputs[NROWS];
  Connections::In<PEWeight<WDTYPE> > weights[NCOLS];
  Connections::In<ODTYPE> psums[NCOLS];
  Connections::Out<ODTYPE> outputs[NCOLS];

  SC_CTOR(SystolicArray) {
#ifdef SIM_ProcessingElement
    // clang-format off
    CCS_DESIGN((ProcessingElement<IDTYPE, WDTYPE, ODTYPE>))
    // clang-format on
#else
    ProcessingElement<IDTYPE, WDTYPE, ODTYPE>
#endif
    *pe[NROWS * NCOLS];

    for (int i = 0; i < NROWS; i++) {
      for (int j = 0; j < NCOLS; j++) {
        pe[i * NCOLS + j] = new
#ifdef SIM_ProcessingElement
            // clang-format off
            CCS_DESIGN((ProcessingElement<IDTYPE, WDTYPE, ODTYPE>))
        // clang-format on
#else
            ProcessingElement<IDTYPE, WDTYPE, ODTYPE>
#endif
                (sc_gen_unique_name("pe_inst"));
        pe[i * NCOLS + j]->clk(clk);
        pe[i * NCOLS + j]->rstn(rstn);

        if (j == 0) {
          pe[i * NCOLS + j]->inputIn(inputs[i]);
        } else {
          pe[i * NCOLS + j]->inputIn(inputConnection[i][j - 1]);
        }

        if (i == 0) {
          pe[i * NCOLS + j]->psumIn(psums[j]);
        } else {
          pe[i * NCOLS + j]->psumIn(psumConnection[i - 1][j]);
        }

        if (i == 0) {
          pe[i * NCOLS + j]->weightIn(weights[j]);
        } else {
          pe[i * NCOLS + j]->weightIn(weightConnection[i - 1][j]);
        }

        pe[i * NCOLS + j]->weightOut(weightConnection[i][j]);
        pe[i * NCOLS + j]->inputOut(inputConnection[i][j]);

        if (i == NROWS - 1) {
          pe[i * NCOLS + j]->psumOut(outputs[j]);
        } else {
          pe[i * NCOLS + j]->psumOut(psumConnection[i][j]);
        }
      }
    }

    // Tie off unused Connections

    // last column of array for inputs
    Tieoff<PEInput<IDTYPE> > *inputConnectionTieoff[NROWS];
    for (int i = 0; i < NROWS; i++) {
      inputConnectionTieoff[i] =
          new Tieoff<PEInput<IDTYPE> >(sc_gen_unique_name("tieoff"));
      inputConnectionTieoff[i]->in(inputConnection[i][NCOLS - 1]);
    }

    // last row for weights
    Tieoff<PEWeight<IDTYPE> > *weightConnectionTieoff[NCOLS];
    for (int i = 0; i < NCOLS; i++) {
      weightConnectionTieoff[i] =
          new Tieoff<PEWeight<IDTYPE> >(sc_gen_unique_name("tieoff"));
      weightConnectionTieoff[i]->in(weightConnection[NROWS - 1][i]);
    }
  }
};
