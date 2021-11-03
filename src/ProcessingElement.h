#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename IDTYPE, typename WDTYPE, typename ODTYPE>
SC_MODULE(ProcessingElement) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  sc_in<bool> CCS_INIT_S1(enable);

  sc_in<WDTYPE> CCS_INIT_S1(weight_in);
  sc_out<WDTYPE> CCS_INIT_S1(weight_out);
  sc_in<bool> CCS_INIT_S1(push_weights);

  sc_in<ac_int<1, false> > CCS_INIT_S1(swap_weights_in);
  sc_out<ac_int<1, false> > CCS_INIT_S1(swap_weights_out);

  sc_in<IDTYPE> CCS_INIT_S1(input_in);
  sc_in<ODTYPE> CCS_INIT_S1(psum_in);

  sc_out<IDTYPE> CCS_INIT_S1(input_out);
  sc_out<ODTYPE> CCS_INIT_S1(psum_out);

  WDTYPE weight_reg;
  WDTYPE weight_fifo;

  IDTYPE input_reg;
  ODTYPE psum_reg;

  SC_CTOR(ProcessingElement) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    bool swap_weights_reg;

    input_out.write(0);
    psum_out.write(0);
    weight_out.write(0);
    swap_weights_out.write(false);

    wait();

#pragma hls_pipeline_init_interval 1
    while (true) {
      if (push_weights.read()) {
        weight_fifo = weight_in.read();
      }

      if (enable.read()) {
        input_reg = input_in.read();
        ODTYPE psum = psum_in.read();

        swap_weights_reg = swap_weights_in.read();

        if (swap_weights_reg) {
          weight_reg = weight_fifo;
        }

        psum_reg = fma(input_reg, weight_reg, psum);
      }

      input_out.write(input_reg);
      psum_out.write(psum_reg);
      weight_out.write(weight_fifo);
      swap_weights_out.write(swap_weights_reg);

      wait();
    }
  }

  template <typename DTYPE>
  DTYPE fma(DTYPE input, DTYPE weight, DTYPE psum) {
    CCS_LOG(input << " * " << weight << " + " << psum);
    return input * weight + psum;
  }
};
