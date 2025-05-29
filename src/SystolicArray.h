#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ProcessingElement.h"
#include "Tieoff.h"
#include "mc_scverify.h"

template <typename Input, typename Weight, typename Psum, int NCols>
SC_MODULE(SystolicArrayRow) {
 private:
  Connections::Combinational<PEInput<Input>> input_wires[NCols];

#ifdef __SYNTHESIS__
  ProcessingElement<Input, Weight, Psum> pe[NCols];
#endif
  Tieoff<PEInput<Input>> input_tieoff;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<Input>> input;
  Connections::In<PEWeight<Weight>> weights_in[NCols];
  Connections::Out<PEWeight<Weight>> weights_out[NCols];
  Connections::In<Psum> psums_in[NCols];
  Connections::Out<Psum> psums_out[NCols];

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(SystolicArrayRow);
  SystolicArrayRow()
      : sc_module(sc_gen_unique_name("SystolicArrayRow"))
#else
  SC_CTOR(SystolicArrayRow)
#endif
  {
    ProcessingElement<Input, Weight, Psum> *pe_ptr[NCols];

    for (int j = 0; j < NCols; j++) {
#ifdef __SYNTHESIS__
      pe_ptr[j] = &pe[j];
#else
      pe_ptr[j] =
          new ProcessingElement<Input, Weight, Psum>(sc_gen_unique_name("pe"));
#endif

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

    // tie off unused connection for last input
    input_tieoff.in(input_wires[NCols - 1]);
#ifdef CONNECTIONS_FAST_SIM
    // we need to connect clock and reset if using fast sim
    input_tieoff.clk(clk);
    input_tieoff.rstn(rstn);
#endif
  };
};

template <typename Input, typename Weight, typename Psum, int NRows, int NCols>
SC_MODULE(SystolicArray) {
 private:
  Connections::Combinational<PEInput<Input>> input_wires[NRows][NCols];
  Connections::Combinational<Psum> psum_wires[NRows][NCols];
  Connections::Combinational<PEWeight<Weight>> weight_wires[NRows][NCols];

// To speed up HLS synthesis, we instantiate arrays of SC_MODULE on
// the stack. However, for simulation, we will run into stack overflow issues,
// so we have to instantiate them on the heap.
#ifdef __SYNTHESIS__
  SystolicArrayRow<Input, Weight, Psum, NCols> sa_rows[NRows];
  Tieoff<PEWeight<Input>> weight_wires_tieoff[NCols];
  ZeroTieoff<Psum> psum_wires_tieoff[NCols];
#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEInput<Input>> inputs[NRows];
  Connections::In<PEWeight<Weight>> weights[NCols];
  Connections::Out<Psum> outputs[NCols];

  SC_CTOR(SystolicArray) {
    SystolicArrayRow<Input, Weight, Psum, NCols> *sa_rows_ptr[NRows];

    for (int i = 0; i < NRows; i++) {
#ifdef __SYNTHESIS__
      sa_rows_ptr[i] = &sa_rows[i];
#else
      sa_rows_ptr[i] = new SystolicArrayRow<Input, Weight, Psum, NCols>(
          sc_gen_unique_name("row"));
#endif

      sa_rows_ptr[i]->clk(clk);
      sa_rows_ptr[i]->rstn(rstn);
      sa_rows_ptr[i]->input(inputs[i]);

      for (int j = 0; j < NCols; j++) {
        if (i == 0) {
          sa_rows_ptr[i]->weights_in[j](weights[j]);
        } else {
          sa_rows_ptr[i]->weights_in[j](weight_wires[i - 1][j]);
        }

        sa_rows_ptr[i]->weights_out[j](weight_wires[i][j]);

        sa_rows_ptr[i]->psums_in[j](psum_wires[i][j]);

        if (i == NRows - 1) {
          sa_rows_ptr[i]->psums_out[j](outputs[j]);
        } else {
          sa_rows_ptr[i]->psums_out[j](psum_wires[i + 1][j]);
        }
      }
    }

    // Tie the first row of psum wires to zero
    ZeroTieoff<Psum> *psum_wires_tieoff_ptr[NCols];
    for (int i = 0; i < NCols; i++) {
#ifdef __SYNTHESIS__
      psum_wires_tieoff_ptr[i] = &psum_wires_tieoff[i];
#else
      psum_wires_tieoff_ptr[i] =
          new ZeroTieoff<Psum>(sc_gen_unique_name("tieoff"));
#endif
      psum_wires_tieoff_ptr[i]->out(psum_wires[0][i]);
#ifdef CONNECTIONS_FAST_SIM
      // we need to connect clock and reset if using fast sim
      psum_wires_tieoff_ptr[i]->clk(clk);
      psum_wires_tieoff_ptr[i]->rstn(rstn);
#endif
    }

    // Tie the last row of weight wires
    Tieoff<PEWeight<Input>> *weight_wires_tieoff_pt[NCols];
    for (int i = 0; i < NCols; i++) {
#ifdef __SYNTHESIS__
      weight_wires_tieoff_pt[i] = &weight_wires_tieoff[i];
#else
      weight_wires_tieoff_pt[i] =
          new Tieoff<PEWeight<Input>>(sc_gen_unique_name("tieoff"));
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

// template <typename Input, typename Weight, typename Psum, int NRows, int
// NCols> SC_MODULE(SystolicArray) {
//  private:
//   Connections::Combinational<PEInput<Input>> input_wires[NRows][NCols];
//   Connections::Combinational<Psum> psum_wires[NRows][NCols];
//   Connections::Combinational<PEWeight<Weight>> weight_wires[NRows][NCols];

// // To speed up HLS synthesis, we instantiate arrays of SC_MODULE on
// // the stack. However, for simulation, we will run into stack overflow
// issues,
// // so we have to instantiate them on the heap.
// #ifdef __SYNTHESIS__
//   ProcessingElement<Input, Weight, Psum> pe[NRows * NCols];
//   Tieoff<PEInput<Input>> input_wires_tieoff[NRows];
//   Tieoff<PEWeight<Input>> weight_wires_tieoff[NCols];
//   ZeroTieoff<Psum> psum_wires_tieoff[NCols];
// #endif

//  public:
//   sc_in<bool> CCS_INIT_S1(clk);
//   sc_in<bool> CCS_INIT_S1(rstn);

//   Connections::In<PEInput<Input>> inputs[NRows];
//   Connections::In<PEWeight<Weight>> weights[NCols];
//   // Connections::In<Psum> psums[NCols];
//   Connections::Out<Psum> outputs[NCols];

//   SC_CTOR(SystolicArray) {
//     ProcessingElement<Input, Weight, Psum> *pe_ptr[NRows * NCols];

//     for (int i = 0; i < NRows; i++) {
//       for (int j = 0; j < NCols; j++) {
// #ifdef __SYNTHESIS__
//         pe_ptr[i * NCols + j] = &pe[i * NCols + j];
// #else
//         pe_ptr[i * NCols + j] = new ProcessingElement<Input, Weight, Psum>(
//             sc_gen_unique_name("pe"));
// #endif

//         pe_ptr[i * NCols + j]->clk(clk);
//         pe_ptr[i * NCols + j]->rstn(rstn);

//         if (j == 0) {
//           pe_ptr[i * NCols + j]->input_in(inputs[i]);
//         } else {
//           pe_ptr[i * NCols + j]->input_in(input_wires[i][j - 1]);
//         }

//         pe_ptr[i * NCols + j]->psum_in(psum_wires[i][j]);

//         if (i == 0) {
//           pe_ptr[i * NCols + j]->weight_in(weights[j]);
//         } else {
//           pe_ptr[i * NCols + j]->weight_in(weight_wires[i - 1][j]);
//         }

//         pe_ptr[i * NCols + j]->weight_out(weight_wires[i][j]);
//         pe_ptr[i * NCols + j]->input_out(input_wires[i][j]);

//         if (i == NRows - 1) {
//           pe_ptr[i * NCols + j]->psum_out(outputs[j]);
//         } else {
//           pe_ptr[i * NCols + j]->psum_out(psum_wires[i + 1][j]);
//         }
//       }
//     }

//     // Tie off unused Connections
//     // first row of array for psums
//     ZeroTieoff<Psum> *psum_wires_tieoff_ptr[NCols];
//     for (int i = 0; i < NCols; i++) {
// #ifdef __SYNTHESIS__
//       psum_wires_tieoff_ptr[i] = &psum_wires_tieoff[i];
// #else
//       psum_wires_tieoff_ptr[i] =
//           new ZeroTieoff<Psum>(sc_gen_unique_name("tieoff"));
// #endif
//       psum_wires_tieoff_ptr[i]->out(psum_wires[0][i]);
// #ifdef CONNECTIONS_FAST_SIM
//       // we need to connect clock and reset if using fast sim
//       psum_wires_tieoff_ptr[i]->clk(clk);
//       psum_wires_tieoff_ptr[i]->rstn(rstn);
// #endif
//     }

//     // last column of array for inputs
//     Tieoff<PEInput<Input>> *inputConnectionTieoff_ptr[NRows];
//     for (int i = 0; i < NRows; i++) {
// #ifdef __SYNTHESIS__
//       inputConnectionTieoff_ptr[i] = &input_wires_tieoff[i];
// #else
//       inputConnectionTieoff_ptr[i] =
//           new Tieoff<PEInput<Input>>(sc_gen_unique_name("tieoff"));
// #endif
//       inputConnectionTieoff_ptr[i]->in(input_wires[i][NCols - 1]);
// #ifdef CONNECTIONS_FAST_SIM
//       // we need to connect clock and reset if using fast sim
//       inputConnectionTieoff_ptr[i]->clk(clk);
//       inputConnectionTieoff_ptr[i]->rstn(rstn);
// #endif
//     }

//     // last row for weights
//     Tieoff<PEWeight<Input>> *weight_wires_tieoff_pt[NCols];
//     for (int i = 0; i < NCols; i++) {
// #ifdef __SYNTHESIS__
//       weight_wires_tieoff_pt[i] = &weight_wires_tieoff[i];
// #else
//       weight_wires_tieoff_pt[i] =
//           new Tieoff<PEWeight<Input>>(sc_gen_unique_name("tieoff"));
// #endif
//       weight_wires_tieoff_pt[i]->in(weight_wires[NRows - 1][i]);
// #ifdef CONNECTIONS_FAST_SIM
//       // we need to connect clock and reset if using fast sim
//       weight_wires_tieoff_pt[i]->clk(clk);
//       weight_wires_tieoff_pt[i]->rstn(rstn);
// #endif
//     }
//   }
// };
