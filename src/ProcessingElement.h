#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

#ifdef CHECK_PE
#include "test/checker/PEChecker.h"
#endif

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

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(ProcessingElement);
  ProcessingElement()
      : sc_module(sc_gen_unique_name("ProcessingElement"))
#else
  SC_CTOR(ProcessingElement)
#endif
  {
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

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
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

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
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

    // name is a string separated by '.'
    // get the last part of the string
    // bool debug = std::string(name()) ==
    //              "harness.accelerator.matrixUnit.matrixProcessor.systolicArray."
    //              "systolic_chunk_inst_0.systolic_row_inst_0.pe_inst_0";

    // wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      // if (swapWeight0) {
      //   weight_reg = nextWeight0;
      // }

      // swapWeight0 = swapWeight1;
      // swapWeight1 = swapWeight2;

      // nextWeight0 = nextWeight1;
      // nextWeight1 = nextWeight2;

      PEInput<IDTYPE> inputStruct = inputIn.Pop();
      // CCS_LOG("PE: " << inputStruct.data << " " << inputStruct.swapWeights);
      ODTYPE psum = psumIn.Pop();

      if (inputStruct.swapWeights) {
        weight_reg = storedWeight.Pop();
      }

      inputOut.Push(inputStruct);
      // #ifdef HYBRID_FP8
      //       ODTYPE output;
      //       if (inputStruct.castToE5M2) {
      //         StdFloat<2, 5> castedInput;
      //         castedInput.float_val.d = inputStruct.data.float_val.d;
      //         StdFloat<2, 5> castedWeight;
      //         castedWeight.float_val.d = weight_reg.float_val.d;

      //         output = castedInput.fma(castedWeight, psum);

      //       } else {
      //         output = inputStruct.data.fma(weight_reg, psum);
      //       }
      // #else
      ODTYPE output = pe_fma(inputStruct.data, weight_reg, psum);
      // #endif

      // if (debug) {
      // CCS_LOG(inputStruct.data << " * " << weight_reg << " + " << psum);
      // }

      psumOut.Push(output);

      // swapWeight2 = inputStruct.swapWeights;
      // if (inputStruct.swapWeights) {  // for next iteration
      //   nextWeight2 = storedWeight.Pop();
      // }
    }
  }

  template <typename DTYPE>
  DTYPE pe_fma(DTYPE input, DTYPE weight, DTYPE psum) {
    // CCS_LOG(input << " * " << weight << " + " << psum);
    return input * weight + psum;
  }

  ODTYPE pe_fma(IDTYPE input, WDTYPE weight, ODTYPE psum) {
// CCS_LOG(input << " * " << weight << " + " << psum << " = "
//               << input.fma(weight, psum));
#ifdef CHECK_PE
    std::string inst_name = name();
    int pe_num = std::stoi(inst_name.substr(inst_name.find_last_of('_') + 1));
    pe_checker.checkReference(pe_num, input, weight, psum);
#endif
    return input.fma(weight, psum);
  }
};
