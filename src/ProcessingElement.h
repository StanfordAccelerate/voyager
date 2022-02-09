#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename IDTYPE, typename INTERMEDIATE_DTYPE, typename ODTYPE>
SC_MODULE(ProcessingElement) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  sc_in<bool> CCS_INIT_S1(toggle);

  sc_in<IDTYPE> CCS_INIT_S1(weight_in);
  sc_out<IDTYPE> CCS_INIT_S1(weight_out);
  sc_in<bool> CCS_INIT_S1(push_weights);

  sc_in<ac_int<1, false> > CCS_INIT_S1(swap_weights_in);
  sc_out<ac_int<1, false> > CCS_INIT_S1(swap_weights_out);

  sc_in<IDTYPE> CCS_INIT_S1(input_in);
  sc_in<ODTYPE> CCS_INIT_S1(psum_in);

  sc_out<IDTYPE> CCS_INIT_S1(input_out);
  sc_out<ODTYPE> CCS_INIT_S1(psum_out);

  sc_out<bool> CCS_INIT_S1(valid_output);

  INTERMEDIATE_DTYPE weight_reg;
  IDTYPE weight_fifo;

  IDTYPE input_reg;
  ODTYPE psum_reg;

  IDTYPE old_input_reg;
  ODTYPE old_psum_reg;

  SC_CTOR(ProcessingElement) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    bool oldToggle = false;
    bool swap_weights_reg;
    bool old_swap_weights_reg;

    input_out.write(0);
    psum_out.write(0);
    weight_out.write(0);
    swap_weights_out.write(false);
    valid_output.write(false);

    wait();

#pragma hls_pipeline_init_interval 1
    while (true) {
      if (push_weights.read()) {
        weight_fifo = weight_in.read();
      }

      if (toggle != oldToggle) {
        input_reg = input_in.read();
        ODTYPE psum = psum_in.read();

        swap_weights_reg = swap_weights_in.read();

        if (swap_weights_reg) {
          // CCS_LOG("swap");
          weight_reg = weight_fifo;
        }

        psum_reg = fma(input_reg, weight_reg, psum);

        input_out.write(input_reg);
        psum_out.write(psum_reg);
        swap_weights_out.write(swap_weights_reg);
        valid_output.write(true);
      } else {
        input_out.write(old_input_reg);
        psum_out.write(old_psum_reg);
        swap_weights_out.write(old_swap_weights_reg);
        valid_output.write(false);
      }

      old_input_reg = input_reg;
      old_psum_reg = psum_reg;
      old_swap_weights_reg = swap_weights_reg;

      weight_out.write(weight_fifo);
      oldToggle = toggle;
      wait();
    }
  }

  template <typename DTYPE>
  DTYPE fma(DTYPE input, DTYPE weight, DTYPE psum) {
    // CCS_LOG(input << " * " << weight << " + " << psum);
    return input * weight + psum;
  }

  // Posit<16, 1, 8, 16> fma(Posit<8, 1, 8, 16> input, PositFP<8, 16> weight,
  //                         Posit<16, 1, 8, 16> psum) {
  //   PositFP<8, 16> intermediate = weight.fma(input, psum);
  //   Posit<16, 1, 8, 16> output = intermediate;
  //   return output;
  // }

  ODTYPE fma(IDTYPE input, INTERMEDIATE_DTYPE weight,
             ODTYPE psum) {
    INTERMEDIATE_DTYPE intermediate = weight.fma(input, psum);
    ODTYPE output = intermediate;
    return output;
  }
};
