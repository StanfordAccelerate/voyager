#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

SC_MODULE(ParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  Connections::Out<Params> CCS_INIT_S1(paramsOut);

  SC_CTOR(ParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    serialParamsIn.Reset();
    paramsOut.Reset();

    wait();
    while (true) {
      Params params;

      params.M0 = serialParamsIn.Pop();
      params.P1 = serialParamsIn.Pop();
      params.N1 = serialParamsIn.Pop();
      params.M1 = serialParamsIn.Pop();
      params.P2 = serialParamsIn.Pop();
      params.INPUT_OFFSET = serialParamsIn.Pop();
      params.WEIGHT_OFFSET = serialParamsIn.Pop();
      params.OUTPUT_OFFSET = serialParamsIn.Pop();
      params.SOFTMAX = serialParamsIn.Pop();
      params.SCALE = serialParamsIn.Pop();
      params.TRANSPOSE = serialParamsIn.Pop();
      params.VECTOR_OFFSET = serialParamsIn.Pop();
      params.VEC_OP = serialParamsIn.Pop();
      params.VEC_SUB = serialParamsIn.Pop();
      params.VEC_SQUARE = serialParamsIn.Pop();
      params.VEC_REDUCE = serialParamsIn.Pop();
      params.CONST_SCALE = serialParamsIn.Pop();
      params.VEC_SCALE_OFFSET = serialParamsIn.Pop();
      params.VEC_SUB_OFFSET = serialParamsIn.Pop();
      params.RELU = serialParamsIn.Pop();

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          params.loops[i][j] = serialParamsIn.Pop();
        }
      }
      for (int i = 0; i < 2; i++) {
        params.inputLoopIndex[i] = serialParamsIn.Pop();
      }
      for (int i = 0; i < 2; i++) {
        params.reductionLoopIndex[i] = serialParamsIn.Pop();
      }
      for (int i = 0; i < 2; i++) {
        params.weightLoopIndex[i] = serialParamsIn.Pop();
      }

      paramsOut.Push(params);
    }
  }
};
