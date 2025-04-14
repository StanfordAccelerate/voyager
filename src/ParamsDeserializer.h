#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "TypeToBits.h"

#ifndef __SYNTHESIS__
#include "spdlog/spdlog.h"
#endif

template <typename T, unsigned int interfaceWidth>
T getSerializedParams(Connections::In<ac_int<64, false>> &serialParamsIn) {
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

// stupid trick to uniquify it
// otherwise, this module ends up repeated for each subblock in the final top
// RTL
template <int id>
SC_MODULE(MatrixParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialParamsIn);
  Connections::Out<MatrixParams> CCS_INIT_S1(paramsOut);

  SC_CTOR(MatrixParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    serialParamsIn.Reset();
    paramsOut.Reset();

    wait();

    while (true) {
      MatrixParams params =
          getSerializedParams<MatrixParams, 64>(serialParamsIn);

// This module gets instantiated 3 times:
// inputController, weightController, matrixProcessor
// But we only want to see the print once
#ifndef __SYNTHESIS__
      if (static_cast<std::string>(name()).find("matrixProcessor") !=
          std::string::npos) {
        std::ostringstream oss;
        oss << "Matrix Params: " << std::endl << params << std::endl;
        spdlog::debug(oss.str());
      }
#endif
      paramsOut.Push(params);
    }
  }
};

SC_MODULE(VectorParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialParamsIn);
  Connections::Out<VectorParams> CCS_INIT_S1(vectorParamsOut);
  Connections::Out<VectorInstructionConfig> CCS_INIT_S1(vectorInstructionsOut);

  SC_CTOR(VectorParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    serialParamsIn.Reset();
    vectorParamsOut.Reset();
    vectorInstructionsOut.Reset();

    wait();

    while (true) {
      VectorParams vectorParams =
          getSerializedParams<VectorParams, 64>(serialParamsIn);

#ifndef __SYNTHESIS__
      std::ostringstream oss;
      oss << "Vector Params: " << std::endl << vectorParams << std::endl;
      spdlog::debug(oss.str());
      oss.clear();
#endif

      vectorParamsOut.Push(vectorParams);

      VectorInstructionConfig vectorInstructionConfig =
          getSerializedParams<VectorInstructionConfig, 64>(serialParamsIn);

#ifndef __SYNTHESIS__
      oss << "Vector Instructions: " << std::endl
          << vectorInstructionConfig << std::endl;
      spdlog::debug(oss.str());
#endif

      vectorInstructionsOut.Push(vectorInstructionConfig);
    }
  }
};

template <int MODULE_COUNT>
SC_MODULE(MatrixParamsRouter) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialParamsIn);
  Connections::Out<ac_int<64, false>> serialMatrixParams[MODULE_COUNT];

  SC_CTOR(MatrixParamsRouter) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    serialParamsIn.Reset();
    for (int i = 0; i < MODULE_COUNT; i++) {
      serialMatrixParams[i].Reset();
    }

    wait();
    while (true) {
      // Matrix Params
      ac_int<64, false> serialParam = serialParamsIn.Pop();
#pragma hls_unroll yes
      for (int i = 0; i < MODULE_COUNT; i++) {
        serialMatrixParams[i].Push(serialParam);
      }
    }
  }
};
