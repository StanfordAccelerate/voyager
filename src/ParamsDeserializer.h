#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "TypeToBits.h"

SC_MODULE(MatrixParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  Connections::Out<MatrixParams> CCS_INIT_S1(paramsOut);

  SC_CTOR(MatrixParamsDeserializer) {
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

    wait();

    while (true) {
      MatrixParams params = getSerializedParams<MatrixParams, 32>();

      std::cout << "Matrix Params Received" << std::endl;
      std::cout << params << std::endl;
      paramsOut.Push(params);
    }
  }
};

SC_MODULE(VectorParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  Connections::Out<VectorParams> CCS_INIT_S1(vectorParamsOut);
  Connections::Out<VectorInstructionConfig> CCS_INIT_S1(vectorInstructionsOut);

  SC_CTOR(VectorParamsDeserializer) {
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
    vectorParamsOut.Reset();
    vectorInstructionsOut.Reset();

    wait();

    while (true) {
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
};

SC_MODULE(ParamsRouter) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  Connections::Out<int> serialMatrixParams[3];
  Connections::Out<int> CCS_INIT_S1(serialVectorParams);

  Connections::SyncOut CCS_INIT_S1(startSignal);

  SC_CTOR(ParamsRouter) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    startSignal.Reset();

    serialParamsIn.Reset();
    for (int i = 0; i < 3; i++) {
      serialMatrixParams[i].Reset();
    }
    serialVectorParams.Reset();

    wait();
    while (true) {
      // Matrix Params
      while (serialParamsIn.Pop() == 1) {
        for (int i = 0; i < (((MatrixParams::width + 32 - 1) / 32) * 32) / 32;
             i++) {
          int serialParam = serialParamsIn.Pop();
#pragma hls_unroll yes
          for (int i = 0; i < 3; i++) {
            serialMatrixParams[i].Push(serialParam);
          }
          CCS_LOG("writing " << i);
        }
        startSignal.SyncPush();
      }

      // Vector Unit Params
      while (serialParamsIn.Pop() == 1) {
        for (int i = 0; i < (((VectorParams::width + 32 - 1) / 32) * 32) / 32;
             i++) {
          int serialParam = serialParamsIn.Pop();
          serialVectorParams.Push(serialParam);
        }

        for (int i = 0;
             i < (((VectorInstructionConfig::width + 32 - 1) / 32) * 32) / 32;
             i++) {
          int serialParam = serialParamsIn.Pop();
          serialVectorParams.Push(serialParam);
        }
      }
    }
  }
};
