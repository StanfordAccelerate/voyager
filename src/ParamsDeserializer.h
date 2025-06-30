#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "TypeToBits.h"

#ifndef __SYNTHESIS__
#include "spdlog/spdlog.h"
#endif

#include <ac_blackbox.h>

template <typename T, unsigned int interfaceWidth>
ac_int<T::width, false> getSerializedParams(
    Connections::In<ac_int<64, false>> &serialParamsIn) {
  ac_int<((T::width + interfaceWidth - 1) / interfaceWidth) * interfaceWidth,
         false>
      serializedParamsPadded;
  for (int i = 0; i < serializedParamsPadded.width / interfaceWidth; i++) {
    ac_int<interfaceWidth, false> val = serialParamsIn.Pop();
    serializedParamsPadded.set_slc(i * interfaceWidth, val);
  }
  ac_int<T::width, false> serializedParams =
      serializedParamsPadded.template slc<T::width>(0);
  return serializedParams;
}

/*
 * Blackbox the type conversion to a passthrough module to avoid long synthesis
 * times in Catapult when dealing with types with many fields.
 */
template <typename T>
SC_MODULE(TypeConverter) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<T::width, false>> CCS_INIT_S1(bitsIn);
  Connections::Out<T> CCS_INIT_S1(typeOut);

  SC_CTOR(TypeConverter) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    ac_blackbox()
        .entity("passthrough")
        .verilog_files("passthrough.v")
        .parameter("WIDTH", T::width)
        .end();
  }

  void run() {
    bitsIn.Reset();
    typeOut.Reset();

    wait();

    while (true) {
      ac_int<T::width, false> bits = bitsIn.Pop();
      sc_lv<T::width> bitsLV;
      type_to_vector(bits, T::width, bitsLV);
      T out = BitsToType<T>(bitsLV);
      typeOut.Push(out);
    }
  }
};

template <int MODULE_COUNT>
SC_MODULE(MatrixParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialParamsIn);
  Connections::Out<MatrixParams> paramsOut[MODULE_COUNT];

  Connections::Combinational<ac_int<MatrixParams::width, false>> paramsBits;
  Connections::Combinational<MatrixParams> convertedParams;
  TypeConverter<MatrixParams> CCS_INIT_S1(typeConverter);

  SC_CTOR(MatrixParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    typeConverter.clk(clk);
    typeConverter.rstn(rstn);
    typeConverter.bitsIn(paramsBits);
    typeConverter.typeOut(convertedParams);
  }

  void run() {
    serialParamsIn.Reset();
    paramsBits.ResetWrite();
    convertedParams.ResetRead();

    for (int i = 0; i < MODULE_COUNT; i++) {
      paramsOut[i].Reset();
    }

    wait();

    while (true) {
      ac_int<MatrixParams::width, false> bits =
          getSerializedParams<MatrixParams, 64>(serialParamsIn);
      paramsBits.Push(bits);
      MatrixParams params = convertedParams.Pop();

#ifndef __SYNTHESIS__
      std::ostringstream oss;
      oss << "Matrix Params: " << std::endl << params << std::endl;
      spdlog::debug(oss.str());
#endif

      for (int i = 0; i < MODULE_COUNT; i++) {
        paramsOut[i].Push(params);
      }
    }
  }
};

SC_MODULE(VectorParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialParamsIn);
  Connections::Out<VectorParams> CCS_INIT_S1(vectorParamsOut);
  Connections::Out<VectorInstructionConfig> CCS_INIT_S1(vectorInstructionsOut);

  Connections::Combinational<ac_int<VectorParams::width, false>>
      vectorParamsBits;
  Connections::Combinational<VectorParams> convertedVectorParams;
  TypeConverter<VectorParams> CCS_INIT_S1(vectorParamsConverter);

  Connections::Combinational<ac_int<VectorInstructionConfig::width, false>>
      vectorInstructionsBits;
  Connections::Combinational<VectorInstructionConfig>
      convertedVectorInstructions;
  TypeConverter<VectorInstructionConfig> CCS_INIT_S1(
      vectorInstructionsConverter);

  SC_CTOR(VectorParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    vectorParamsConverter.clk(clk);
    vectorParamsConverter.rstn(rstn);
    vectorParamsConverter.bitsIn(vectorParamsBits);
    vectorParamsConverter.typeOut(convertedVectorParams);

    vectorInstructionsConverter.clk(clk);
    vectorInstructionsConverter.rstn(rstn);
    vectorInstructionsConverter.bitsIn(vectorInstructionsBits);
    vectorInstructionsConverter.typeOut(convertedVectorInstructions);
  }

  void run() {
    serialParamsIn.Reset();
    vectorInstructionsOut.Reset();
    vectorParamsOut.Reset();
    vectorParamsBits.ResetWrite();
    vectorInstructionsBits.ResetWrite();
    convertedVectorParams.ResetRead();
    convertedVectorInstructions.ResetRead();

    wait();

    while (true) {
      ac_int<VectorParams::width, false> vectorParamsBitsVal =
          getSerializedParams<VectorParams, 64>(serialParamsIn);
      vectorParamsBits.Push(vectorParamsBitsVal);
      VectorParams vectorParams = convertedVectorParams.Pop();

#ifndef __SYNTHESIS__
      std::ostringstream oss;
      oss << "Vector Params: " << std::endl << vectorParams << std::endl;
      spdlog::debug(oss.str());
      oss.clear();
#endif

      vectorParamsOut.Push(vectorParams);

      ac_int<VectorInstructionConfig::width, false> vectorInstructionsBitsVal =
          getSerializedParams<VectorInstructionConfig, 64>(serialParamsIn);
      vectorInstructionsBits.Push(vectorInstructionsBitsVal);
      VectorInstructionConfig vectorInstructionConfig =
          convertedVectorInstructions.Pop();

#ifndef __SYNTHESIS__
      oss << "Vector Instructions: " << std::endl
          << vectorInstructionConfig << std::endl;
      spdlog::debug(oss.str());
#endif

      vectorInstructionsOut.Push(vectorInstructionConfig);
    }
  }
};
