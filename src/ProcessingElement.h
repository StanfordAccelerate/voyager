#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

#ifdef CHECK_PE
#include "test/checker/PEChecker.h"
#endif

template <typename Input, typename Weight, typename Psum>
SC_MODULE(ProcessingElement) {
 private:
  Input weight_reg;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEWeight<Weight> > CCS_INIT_S1(weight_in);
  Connections::Out<PEWeight<Weight> > CCS_INIT_S1(weight_out);

  Connections::In<PEInput<Input> > CCS_INIT_S1(input_in);
  Connections::Out<PEInput<Input> > CCS_INIT_S1(input_out);

  Connections::In<Psum> CCS_INIT_S1(psum_in);
  Connections::Out<Psum> CCS_INIT_S1(psum_out);

  Connections::Combinational<Weight> CCS_INIT_S1(new_weight_fifo);
  Connections::Combinational<Weight> CCS_INIT_S1(stored_weight);

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(ProcessingElement);
  ProcessingElement()
      : sc_module(sc_gen_unique_name("ProcessingElement"))
#else
  SC_CTOR(ProcessingElement)
#endif
  {
    SC_THREAD(store_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(weight_fifo);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void weight_fifo() {
    weight_in.Reset();
    weight_out.Reset();
    new_weight_fifo.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      PEWeight<Weight> weightStruct = weight_in.Pop();

      if (weightStruct.tag == 0) {
        new_weight_fifo.Push(weightStruct.data);
      } else {
        weightStruct.tag--;
        weight_out.Push(weightStruct);
      }
    }
  }

  void store_weights() {
    stored_weight.ResetWrite();
    new_weight_fifo.ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      stored_weight.Push(new_weight_fifo.Pop());
    }
  }

  void run() {
    input_in.Reset();
    psum_in.Reset();
    input_out.Reset();
    psum_out.Reset();

    stored_weight.ResetRead();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      PEInput<Input> input = input_in.Pop();
      Psum psum = psum_in.Pop();

      if (input.swapWeights) {
        weight_reg = stored_weight.Pop();
      }

      input_out.Push(input);

      Psum output = pe_fma(input.data, weight_reg, psum);
      psum_out.Push(output);
    }
  }

  Psum pe_fma(Input input, Weight weight, Psum psum) {
#ifdef CHECK_PE
    std::string inst_name = name();
    int pe_num = std::stoi(inst_name.substr(inst_name.find_last_of('_') + 1));
    pe_checker.check_reference(pe_num, input, weight, psum);
#endif
    return input.fma(weight, psum);
  }
};
