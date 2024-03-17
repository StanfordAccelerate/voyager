#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"
#include "Skewer.h"
#include "Tieoff.h"
#include "mc_scverify.h"

template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int NCOLS>
SC_MODULE(SystolicArrayRow) {
 private:
  Connections::Combinational<PEInput<IDTYPE> > inputConnection[NCOLS];

  // tie off unused connection for last input
  Tieoff<PEInput<IDTYPE> > CCS_INIT_S1(inputConnectionTieoff);

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<IDTYPE> > inputs;
  Connections::In<PEWeight<WDTYPE> > weightsIn[NCOLS];
  Connections::Out<PEWeight<WDTYPE> > weightsOut[NCOLS];
  Connections::In<ODTYPE> psumIn[NCOLS];
  Connections::Out<ODTYPE> psumOut[NCOLS];

  SC_CTOR(SystolicArrayRow) {
    ProcessingElement<IDTYPE, WDTYPE, ODTYPE> *pe[NCOLS];

    for (int i = 0; i < NCOLS; i++) {
      pe[i] = new ProcessingElement<IDTYPE, WDTYPE, ODTYPE>(
          sc_gen_unique_name("pe_inst"));
      pe[i]->clk(clk);
      pe[i]->rstn(rstn);

      if (i == 0) {
        pe[i]->inputIn(inputs);
      } else {
        pe[i]->inputIn(inputConnection[i - 1]);
      }

      pe[i]->psumIn(psumIn[i]);
      pe[i]->weightIn(weightsIn[i]);
      pe[i]->weightOut(weightsOut[i]);
      pe[i]->inputOut(inputConnection[i]);
      pe[i]->psumOut(psumOut[i]);
    }

    inputConnectionTieoff.in(inputConnection[NCOLS - 1]);
  };
};

template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int NROWS,
          int NCOLS>
SC_MODULE(SystolicArrayChunk) {
 private:
  Connections::Combinational<ODTYPE> psumConnection[NROWS - 1][NCOLS];
  Connections::Combinational<PEWeight<WDTYPE> > weightConnection[NROWS - 1]
                                                                [NCOLS];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<IDTYPE> > inputs[NROWS];
  Connections::In<PEWeight<WDTYPE> > weightsIn[NCOLS];
  Connections::Out<PEWeight<WDTYPE> > weightsOut[NCOLS];
  Connections::In<ODTYPE> psumIn[NCOLS];
  Connections::Out<ODTYPE> psumOut[NCOLS];

  SC_CTOR(SystolicArrayChunk) {
    SystolicArrayRow<IDTYPE, WDTYPE, ODTYPE, NCOLS> *rows[NROWS];

    for (int i = 0; i < NROWS; i++) {
      rows[i] = new SystolicArrayRow<IDTYPE, WDTYPE, ODTYPE, NCOLS>(
          sc_gen_unique_name("systolic_row_inst"));
      rows[i]->clk(clk);
      rows[i]->rstn(rstn);
      rows[i]->inputs(inputs[i]);
      for (int j = 0; j < NCOLS; j++) {
        if (i == 0) {
          rows[i]->psumIn[j](psumIn[j]);

        } else {
          rows[i]->psumIn[j](psumConnection[i - 1][j]);
        }

        if (i == 0) {
          rows[i]->weightsIn[j](weightsIn[j]);
        } else {
          rows[i]->weightsIn[j](weightConnection[i - 1][j]);
        }

        if (i == NROWS - 1) {
          rows[i]->weightsOut[j](weightsOut[j]);
        } else {
          rows[i]->weightsOut[j](weightConnection[i][j]);
        }

        if (i == NROWS - 1) {
          rows[i]->psumOut[j](psumOut[j]);
        } else {
          rows[i]->psumOut[j](psumConnection[i][j]);
        }
      }
    }
  }
};

template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int NROWS,
          int NCOLS>
SC_MODULE(SystolicArray) {
 private:
  static constexpr int CHUNK_SIZE = 4;
  static constexpr int NCHUNKS = NROWS / 4;

  Connections::Combinational<ODTYPE> psumConnection[NCHUNKS - 1][NCOLS];
  Connections::Combinational<PEWeight<WDTYPE> > weightConnection[NCHUNKS]
                                                                [NCOLS];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<IDTYPE> > inputs[NROWS];
  Connections::In<PEWeight<WDTYPE> > weights[NCOLS];
  Connections::In<ODTYPE> psums[NCOLS];
  Connections::Out<ODTYPE> outputs[NCOLS];

  SC_CTOR(SystolicArray) {
    SystolicArrayChunk<IDTYPE, WDTYPE, ODTYPE, CHUNK_SIZE, NCOLS>
        *chunks[NCHUNKS];

    for (int i = 0; i < NCHUNKS; i++) {
      chunks[i] =
          new SystolicArrayChunk<IDTYPE, WDTYPE, ODTYPE, CHUNK_SIZE, NCOLS>(
              sc_gen_unique_name("systolic_chunk_inst"));

      chunks[i]->clk(clk);
      chunks[i]->rstn(rstn);
      for (int j = 0; j < CHUNK_SIZE; j++) {
        chunks[i]->inputs[j](inputs[i * CHUNK_SIZE + j]);
      }
      for (int j = 0; j < NCOLS; j++) {
        if (i == 0) {
          chunks[i]->psumIn[j](psums[j]);
        } else {
          chunks[i]->psumIn[j](psumConnection[i - 1][j]);
        }

        if (i == 0) {
          chunks[i]->weightsIn[j](weights[j]);
        } else {
          chunks[i]->weightsIn[j](weightConnection[i - 1][j]);
        }

        chunks[i]->weightsOut[j](weightConnection[i][j]);

        if (i == NCHUNKS - 1) {
          chunks[i]->psumOut[j](outputs[j]);
        } else {
          chunks[i]->psumOut[j](psumConnection[i][j]);
        }
      }
    }

    // last row for weights
    Tieoff<PEWeight<IDTYPE> > *weightConnectionTieoff[NCOLS];
    for (int i = 0; i < NCOLS; i++) {
      weightConnectionTieoff[i] =
          new Tieoff<PEWeight<IDTYPE> >(sc_gen_unique_name("tieoff"));
      weightConnectionTieoff[i]->in(weightConnection[NCHUNKS - 1][i]);
    }

    /*
      for (int i = 0; i < NROWS; i++) {
        rows[i] = new SystolicArrayRow<IDTYPE, WDTYPE, ODTYPE, NROWS, NCOLS>(
            sc_gen_unique_name("systolic_row_inst"));
        rows[i]->clk(clk);
        rows[i]->rstn(rstn);
        rows[i]->inputs(inputs[i]);
        for (int j = 0; j < NCOLS; j++) {
          if (i == 0) {
            rows[i]->psumIn[j](psums[j]);
          } else {
            rows[i]->psumIn[j](psumConnection[i - 1][j]);
          }

          if (i == 0) {
            rows[i]->weightsIn[j](weights[j]);
          } else {
            rows[i]->weightsIn[j](weightConnection[i - 1][j]);
          }

          rows[i]->weightsOut[j](weightConnection[i][j]);

          if (i == NROWS - 1) {
            rows[i]->psumOut[j](outputs[j]);
          } else {
            rows[i]->psumOut[j](psumConnection[i][j]);
          }
        }
      }

      // last row for weights
      Tieoff<PEWeight<IDTYPE> > *weightConnectionTieoff[NCOLS];
      for (int i = 0; i < NCOLS; i++) {
        weightConnectionTieoff[i] =
            new Tieoff<PEWeight<IDTYPE> >(sc_gen_unique_name("tieoff"));
        weightConnectionTieoff[i]->in(weightConnection[NROWS - 1][i]);
      }
      */
  }
};

/*
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
*/
