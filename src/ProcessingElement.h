#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename IDTYPE, typename WDTYPE, typename ODTYPE>
SC_MODULE(ProcessingElement) {
 private:
  sc_signal<WDTYPE> updatedWeight;

  IDTYPE weight_reg;
  WDTYPE weight_fifo;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  sc_in<WDTYPE> CCS_INIT_S1(weightIn);
  sc_in<bool> CCS_INIT_S1(weightValid);

  sc_out<WDTYPE> CCS_INIT_S1(weightOut);

  Connections::In<ac_int<1, false> > CCS_INIT_S1(weightSwapIn);
  Connections::Out<ac_int<1, false> > CCS_INIT_S1(weightSwapOut);

  Connections::In<IDTYPE> CCS_INIT_S1(inputIn);
  Connections::Out<IDTYPE> CCS_INIT_S1(inputOut);

  Connections::In<ODTYPE> CCS_INIT_S1(psumIn);
  Connections::Out<ODTYPE> CCS_INIT_S1(psumOut);

  SC_CTOR(ProcessingElement) {
    SC_THREAD(weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void weights() {
    weightOut.write(WDTYPE());

    wait();

    while (true) {
      if (weightValid.read()) {
        weight_fifo = weightIn.read();
        updatedWeight = weight_fifo;
      }
      weightOut.write(weight_fifo);

      wait();

      // weightOut.Push(weight_fifo);
      // CCS_LOG("pushed: " << weight_fifo);

      // weight_fifo = weightIn.Pop();
      // CCS_LOG("popped: " << weight_fifo);
      // updatedWeight = weight_fifo;
    }
  }

  void run() {
    inputIn.Reset();
    psumIn.Reset();
    inputOut.Reset();
    psumOut.Reset();
    weightSwapIn.Reset();
    weightSwapOut.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      IDTYPE input = inputIn.Pop();
      ODTYPE psum = psumIn.Pop();
      ac_int<1, false> weightSwap = weightSwapIn.Pop();

      if (weightSwap) {
        typename WDTYPE::DecomposedPosit decodedWeight(updatedWeight.read());
        weight_reg = decodedWeight;
      }

      ODTYPE output = pe_fma(input, weight_reg, psum);

      inputOut.Push(input);
      psumOut.Push(output);
      weightSwapOut.Push(weightSwap);
    }
  }

  template <typename DTYPE>
  DTYPE pe_fma(DTYPE input, DTYPE weight, DTYPE psum) {
    // CCS_LOG(input << " * " << weight << " + " << psum);
    return input * weight + psum;
  }

  ODTYPE pe_fma(IDTYPE input, WDTYPE weight, ODTYPE psum) {
    // CCS_LOG(input << " * " << weight << " + " << psum);
    return decomposed_fma<8, 1, 16, 1>(input, weight, psum);
  }
};
