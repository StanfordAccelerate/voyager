#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"
#include "Tieoff.h"
#include "mc_scverify.h"

template <typename Input, typename Weight, typename Psum, int cols>
SC_MODULE(SystolicArrayRow) {
 private:
  Connections::Combinational<PEInput<Input>> input_wires[cols];

#ifdef __SYNTHESIS__
  ProcessingElement<Input, Weight, Psum> pe[cols];
  Tieoff<PEInput<Input>> input_tieoff;
#else
  ProcessingElement<Input, Weight, Psum>* pe_ptr[cols];
  Tieoff<PEInput<Input>>* input_tieoff_ptr;
#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<Input>> input;
  Connections::In<PEWeight<Weight>> weights_in[cols];
  Connections::Out<PEWeight<Weight>> weights_out[cols];
  Connections::In<Psum> psums_in[cols];
  Connections::Out<Psum> psums_out[cols];

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(SystolicArrayRow);
  SystolicArrayRow() : sc_module(sc_gen_unique_name("SystolicArrayRow")) {
    for (int j = 0; j < cols; j++) {
      pe[j].clk(clk);
      pe[j].rstn(rstn);

      if (j == 0) {
        pe[j].input_in(input);
      } else {
        pe[j].input_in(input_wires[j - 1]);
      }
      pe[j].input_out(input_wires[j]);

      pe[j].weight_in(weights_in[j]);
      pe[j].weight_out(weights_out[j]);

      pe[j].psum_in(psums_in[j]);
      pe[j].psum_out(psums_out[j]);
    }

#ifdef CONNECTIONS_FAST_SIM
    input_tieoff.clk(clk);
    input_tieoff.rstn(rstn);
#endif
    input_tieoff.in(input_wires[cols - 1]);
  }

#else  // !__SYNTHESIS__
  SC_CTOR(SystolicArrayRow) {
    for (int j = 0; j < cols; j++) {
      pe_ptr[j] =
          new ProcessingElement<Input, Weight, Psum>(sc_gen_unique_name("pe"));

      pe_ptr[j]->clk(clk);
      pe_ptr[j]->rstn(rstn);

      if (j == 0) {
        pe_ptr[j]->input_in(input);
      } else {
        pe_ptr[j]->input_in(input_wires[j - 1]);
      }
      pe_ptr[j]->input_out(input_wires[j]);

      pe_ptr[j]->weight_in(weights_in[j]);
      pe_ptr[j]->weight_out(weights_out[j]);

      pe_ptr[j]->psum_in(psums_in[j]);
      pe_ptr[j]->psum_out(psums_out[j]);
    }

    input_tieoff_ptr =
        new Tieoff<PEInput<Input>>(sc_gen_unique_name("input_tieoff"));
#ifdef CONNECTIONS_FAST_SIM
    input_tieoff_ptr->clk(clk);
    input_tieoff_ptr->rstn(rstn);
#endif
    input_tieoff_ptr->in(input_wires[cols - 1]);
  };
#endif  // !__SYNTHESIS__
};

template <typename Input, typename Weight, typename Psum, int rows, int cols>
SC_MODULE(SystolicArray) {
 private:
  Connections::Combinational<Psum> psum_wires[rows][cols];
  Connections::Combinational<PEWeight<Weight>> weight_wires[rows][cols];

// To speed up HLS synthesis, we instantiate arrays of SC_MODULE on
// the stack. However, for simulation, we will run into stack overflow issues,
// so we have to instantiate them on the heap.
#ifdef __SYNTHESIS__
  SystolicArrayRow<Input, Weight, Psum, cols> sa_rows[rows];
  Tieoff<PEWeight<Input>> weight_wires_tieoff[cols];
  ZeroTieoff<Psum> psum_wires_tieoff[cols];
#else
  SystolicArrayRow<Input, Weight, Psum, cols>* sa_rows_ptr[rows];
  Tieoff<PEWeight<Input>>* weight_wires_tieoff_pt[cols];
  ZeroTieoff<Psum>* psum_wires_tieoff_ptr[cols];
#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<Input>> inputs[rows];
  Connections::In<PEWeight<Weight>> weights[cols];
  Connections::Out<Psum> outputs[cols];

  SC_CTOR(SystolicArray) {
#ifdef __SYNTHESIS__
    // Instantiate and connect SA rows
    for (int i = 0; i < rows; i++) {
      sa_rows[i].clk(clk);
      sa_rows[i].rstn(rstn);
      sa_rows[i].input(inputs[i]);

      for (int j = 0; j < cols; j++) {
        if (i == 0) {
          sa_rows[i].weights_in[j](weights[j]);
        } else {
          sa_rows[i].weights_in[j](weight_wires[i - 1][j]);
        }

        sa_rows[i].weights_out[j](weight_wires[i][j]);
        sa_rows[i].psums_in[j](psum_wires[i][j]);

        if (i == rows - 1) {
          sa_rows[i].psums_out[j](outputs[j]);
        } else {
          sa_rows[i].psums_out[j](psum_wires[i + 1][j]);
        }
      }
    }

    // Tie first row of psum wires to zero
    for (int i = 0; i < cols; i++) {
#ifdef CONNECTIONS_FAST_SIM
      psum_wires_tieoff[i].clk(clk);
      psum_wires_tieoff[i].rstn(rstn);
#endif
      psum_wires_tieoff[i].out(psum_wires[0][i]);
    }

    // Tie last row of weight wires
    for (int i = 0; i < cols; i++) {
#ifdef CONNECTIONS_FAST_SIM
      weight_wires_tieoff[i].clk(clk);
      weight_wires_tieoff[i].rstn(rstn);
#endif
      weight_wires_tieoff[i].in(weight_wires[rows - 1][i]);
    }

#else  // !__SYNTHESIS__
    // Instantiate and connect SA rows
    for (int i = 0; i < rows; i++) {
      sa_rows_ptr[i] = new SystolicArrayRow<Input, Weight, Psum, cols>(
          sc_gen_unique_name("row"));
      sa_rows_ptr[i]->clk(clk);
      sa_rows_ptr[i]->rstn(rstn);
      sa_rows_ptr[i]->input(inputs[i]);

      for (int j = 0; j < cols; j++) {
        if (i == 0) {
          sa_rows_ptr[i]->weights_in[j](weights[j]);
        } else {
          sa_rows_ptr[i]->weights_in[j](weight_wires[i - 1][j]);
        }

        sa_rows_ptr[i]->weights_out[j](weight_wires[i][j]);
        sa_rows_ptr[i]->psums_in[j](psum_wires[i][j]);

        if (i == rows - 1) {
          sa_rows_ptr[i]->psums_out[j](outputs[j]);
        } else {
          sa_rows_ptr[i]->psums_out[j](psum_wires[i + 1][j]);
        }
      }
    }

    // Tie first row of psum wires to zero
    for (int i = 0; i < cols; i++) {
      psum_wires_tieoff_ptr[i] =
          new ZeroTieoff<Psum>(sc_gen_unique_name("psum_tieoff"));
#ifdef CONNECTIONS_FAST_SIM
      psum_wires_tieoff_ptr[i]->clk(clk);
      psum_wires_tieoff_ptr[i]->rstn(rstn);
#endif
      psum_wires_tieoff_ptr[i]->out(psum_wires[0][i]);
    }

    // Tie last row of weight wires
    for (int i = 0; i < cols; i++) {
      weight_wires_tieoff_pt[i] =
          new Tieoff<PEWeight<Input>>(sc_gen_unique_name("weight_tieoff"));
#ifdef CONNECTIONS_FAST_SIM
      weight_wires_tieoff_pt[i]->clk(clk);
      weight_wires_tieoff_pt[i]->rstn(rstn);
#endif
      weight_wires_tieoff_pt[i]->in(weight_wires[rows - 1][i]);
    }
#endif  // __SYNTHESIS__
  }
};
