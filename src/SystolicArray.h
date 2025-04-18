#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"
#include "Tieoff.h"
#include "mc_scverify.h"

template <typename Input, typename Weight, typename Psum, int NRows, int NCols>
SC_MODULE(SystolicArray) {
 private:
  Connections::Combinational<PEInput<Input> > input_wires[NRows][NCols];
  Connections::Combinational<Psum> psum_wires[NRows - 1][NCols];
  Connections::Combinational<PEWeight<Weight> > weight_wires[NRows][NCols];

// To speed up HLS synthesis, we instantiate arrays of SC_MODULE on
// the stack. However, for simulation, we will run into stack overflow issues,
// so we have to instantiate them on the heap.
#ifdef __SYNTHESIS__
  ProcessingElement<Input, Weight, Psum> pe[NRows * NCols];
  Tieoff<PEInput<Input> > input_wires_tieoff[NRows];
  Tieoff<PEWeight<Input> > weight_wires_tieoff[NCols];

#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<Input> > inputs[NRows];
  Connections::In<PEWeight<Weight> > weights[NCols];
  Connections::In<Psum> psums[NCols];
  Connections::Out<Psum> outputs[NCols];

  SC_CTOR(SystolicArray) {
    ProcessingElement<Input, Weight, Psum> *pe_ptr[NRows * NCols];

    for (int i = 0; i < NRows; i++) {
      for (int j = 0; j < NCols; j++) {
#ifdef __SYNTHESIS__
        pe_ptr[i * NCols + j] = &pe[i * NCols + j];
#else
        pe_ptr[i * NCols + j] = new ProcessingElement<Input, Weight, Psum>(
            sc_gen_unique_name("pe"));
#endif

        pe_ptr[i * NCols + j]->clk(clk);
        pe_ptr[i * NCols + j]->rstn(rstn);

        if (j == 0) {
          pe_ptr[i * NCols + j]->input_in(inputs[i]);
        } else {
          pe_ptr[i * NCols + j]->input_in(input_wires[i][j - 1]);
        }

        if (i == 0) {
          pe_ptr[i * NCols + j]->psum_in(psums[j]);
        } else {
          pe_ptr[i * NCols + j]->psum_in(psum_wires[i - 1][j]);
        }

        if (i == 0) {
          pe_ptr[i * NCols + j]->weight_in(weights[j]);
        } else {
          pe_ptr[i * NCols + j]->weight_in(weight_wires[i - 1][j]);
        }

        pe_ptr[i * NCols + j]->weight_out(weight_wires[i][j]);
        pe_ptr[i * NCols + j]->input_out(input_wires[i][j]);

        if (i == NRows - 1) {
          pe_ptr[i * NCols + j]->psum_out(outputs[j]);
        } else {
          pe_ptr[i * NCols + j]->psum_out(psum_wires[i][j]);
        }
      }
    }

    // Tie off unused Connections

    // last column of array for inputs
    Tieoff<PEInput<Input> > *inputConnectionTieoff_ptr[NRows];
    for (int i = 0; i < NRows; i++) {
#ifdef __SYNTHESIS__
      inputConnectionTieoff_ptr[i] = &input_wires_tieoff[i];
#else
      inputConnectionTieoff_ptr[i] =
          new Tieoff<PEInput<Input> >(sc_gen_unique_name("tieoff"));
#endif
      inputConnectionTieoff_ptr[i]->in(input_wires[i][NCols - 1]);
#ifdef CONNECTIONS_FAST_SIM
      // we need to connect clock and reset if using fast sim
      inputConnectionTieoff_ptr[i]->clk(clk);
      inputConnectionTieoff_ptr[i]->rstn(rstn);
#endif
    }

    // last row for weights
    Tieoff<PEWeight<Input> > *weight_wires_tieoff_pt[NCols];
    for (int i = 0; i < NCols; i++) {
#ifdef __SYNTHESIS__
      weight_wires_tieoff_pt[i] = &weight_wires_tieoff[i];
#else
      weight_wires_tieoff_pt[i] =
          new Tieoff<PEWeight<Input> >(sc_gen_unique_name("tieoff"));
#endif
      weight_wires_tieoff_pt[i]->in(weight_wires[NRows - 1][i]);
#ifdef CONNECTIONS_FAST_SIM
      // we need to connect clock and reset if using fast sim
      weight_wires_tieoff_pt[i]->clk(clk);
      weight_wires_tieoff_pt[i]->rstn(rstn);
#endif
    }
  }
};
