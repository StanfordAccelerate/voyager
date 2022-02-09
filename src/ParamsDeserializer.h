#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "TypeToBits.h"

SC_MODULE(ParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  Connections::Out<MatrixParams> CCS_INIT_S1(paramsOut);
  Connections::Out<VectorParams> CCS_INIT_S1(vectorParamsOut);
  Connections::Out<VectorInstructionConfig> CCS_INIT_S1(vectorInstructionsOut);

  SC_CTOR(ParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  template <typename T, unsigned int interfaceWidth>
  T getSerializedParams() {
    ac_int<((T::width + interfaceWidth - 1) / interfaceWidth) * interfaceWidth,
           false>
        serializedParamsPadded;
    for (int i = 0; i < serializedParamsPadded.width / interfaceWidth; i++) {
      ac_int<interfaceWidth, false> val = serialParamsIn.Pop();
      serializedParamsPadded.set_slc(i * interfaceWidth, val);
    }
    ac_int<T::width, false> serializedParams =
        serializedParamsPadded.template slc<T::width>(0);
    sc_lv<T::width> serializedParamsLV;
    type_to_vector(serializedParams, T::width, serializedParamsLV);
    return BitsToType<T>(serializedParamsLV);
  }

  void run() {
    serialParamsIn.Reset();
    paramsOut.Reset();
    vectorParamsOut.Reset();

    wait();
    while (true) {
      // Matrix Unit Params
      while (serialParamsIn.Pop() == 1) {
        MatrixParams params = getSerializedParams<MatrixParams, 32>();

        std::cout << "Matrix Params Received" << std::endl;
        std::cout << params << std::endl;
        paramsOut.Push(params);
      }

      // Vector Unit Params
      while (serialParamsIn.Pop() == 1) {
        VectorParams vectorParams = getSerializedParams<VectorParams, 32>();

        std::cout << "Vector Params Received" << std::endl;
        std::cout << vectorParams << std::endl;

        vectorParamsOut.Push(vectorParams);

        VectorInstructionConfig vectorInstructionConfig =
            getSerializedParams<VectorInstructionConfig, 32>();

        std::cout << "Vector Instructions Received" << std::endl;
        std::cout << vectorInstructionConfig << std::endl;

        vectorInstructionsOut.Push(vectorInstructionConfig);
      }
    }
  }
};
