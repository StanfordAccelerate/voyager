#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename IDTYPE, typename WDTYPE, typename ODTYPE>
SC_MODULE(ProcessingElement) {
 private:
  IDTYPE weight_reg;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEWeight<WDTYPE> > CCS_INIT_S1(weightIn);
  Connections::Out<PEWeight<WDTYPE> > CCS_INIT_S1(weightOut);

  Connections::In<PEInput<IDTYPE> > CCS_INIT_S1(inputIn);
  Connections::Out<PEInput<IDTYPE> > CCS_INIT_S1(inputOut);

  Connections::In<ODTYPE> CCS_INIT_S1(psumIn);
  Connections::Out<ODTYPE> CCS_INIT_S1(psumOut);

  Connections::Combinational<WDTYPE> CCS_INIT_S1(newWeightFifo);
  Connections::Combinational<WDTYPE> CCS_INIT_S1(storedWeight);

  SC_CTOR(ProcessingElement) {
    SC_THREAD(storeWeights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(weightFifo);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void weightFifo() {
    weightIn.Reset();
    weightOut.Reset();
    newWeightFifo.ResetWrite();

    wait();

    while (true) {
      PEWeight<WDTYPE> weightStruct = weightIn.Pop();

      if (weightStruct.tag == 0) {
        newWeightFifo.Push(weightStruct.data);
      } else {
        weightStruct.tag--;
        weightOut.Push(weightStruct);
      }
    }
  }

  void storeWeights() {
    storedWeight.ResetWrite();
    newWeightFifo.ResetRead();

    wait();

    while (true) {
      storedWeight.Push(newWeightFifo.Pop());
    }
  }

  void run() {
    inputIn.Reset();
    psumIn.Reset();
    inputOut.Reset();
    psumOut.Reset();

    storedWeight.ResetRead();

    WDTYPE nextWeight0 = WDTYPE();
    WDTYPE nextWeight1 = WDTYPE();
    WDTYPE nextWeight2 = WDTYPE();
    bool swapWeight0 = false;
    bool swapWeight1 = false;
    bool swapWeight2 = false;

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      if (swapWeight0) {
        weight_reg = nextWeight0;
      }

      swapWeight0 = swapWeight1;
      swapWeight1 = swapWeight2;

      nextWeight0 = nextWeight1;
      nextWeight1 = nextWeight2;
      
      PEInput<IDTYPE> inputStruct = inputIn.Pop();

      ODTYPE psum = psumIn.Pop();
      inputOut.Push(inputStruct);
      ODTYPE output = pe_fma(inputStruct.data, weight_reg, psum);

      psumOut.Push(output);

      swapWeight2 = inputStruct.swapWeights;
      if (inputStruct.swapWeights) {  // for next iteration
        nextWeight2 = storedWeight.Pop();
      }
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
