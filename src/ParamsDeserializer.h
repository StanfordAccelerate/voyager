#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "TypeToBits.h"

#ifndef __SYNTHESIS__
#include "spdlog/spdlog.h"
#endif

#include <ac_blackbox.h>

template <typename T, unsigned int width>
ac_int<T::width, false> get_serialized_params(
    Connections::In<ac_int<64, false>>& serial_params_in) {
  ac_int<((T::width + width - 1) / width) * width, false> padded_params;
  for (int i = 0; i < padded_params.width / width; i++) {
    ac_int<width, false> bits = serial_params_in.Pop();
    padded_params.set_slc(i * width, bits);
  }
  ac_int<T::width, false> serialized_params =
      padded_params.template slc<T::width>(0);
  return serialized_params;
}

/*
 * Blackbox the type conversion to a passthrough module to avoid long synthesis
 * times in Catapult when dealing with types with many fields.
 */
template <typename T>
SC_MODULE(TypeConverter) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<T::width, false>> CCS_INIT_S1(bits_in);
  Connections::Out<T> CCS_INIT_S1(type_out);

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
    bits_in.Reset();
    type_out.Reset();

    wait();

    while (true) {
      ac_int<T::width, false> bits = bits_in.Pop();
      sc_lv<T::width> bits_lv;
      type_to_vector(bits, T::width, bits_lv);
      T out = BitsToType<T>(bits_lv);
      type_out.Push(out);
    }
  }
};

template <int id, int module_count>
SC_MODULE(MatrixParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Out<MatrixParams> params_out[module_count];

  Connections::Combinational<ac_int<MatrixParams::width, false>> params_bits;
  Connections::Combinational<MatrixParams> converted_params;
  TypeConverter<MatrixParams> CCS_INIT_S1(converter);

  SC_CTOR(MatrixParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    converter.clk(clk);
    converter.rstn(rstn);
    converter.bits_in(params_bits);
    converter.type_out(converted_params);
  }

  void run() {
    serial_params_in.Reset();
    params_bits.ResetWrite();
    converted_params.ResetRead();

    for (int i = 0; i < module_count; i++) {
      params_out[i].Reset();
    }

    wait();

    while (true) {
      ac_int<MatrixParams::width, false> bits =
          get_serialized_params<MatrixParams, 64>(serial_params_in);
      params_bits.Push(bits);
      MatrixParams params = converted_params.Pop();

#ifndef __SYNTHESIS__
      std::ostringstream oss;
      oss << "Matrix Params: " << std::endl << params << std::endl;
      spdlog::debug(oss.str());
#endif

      for (int i = 0; i < module_count; i++) {
        params_out[i].Push(params);
      }
    }
  }
};

SC_MODULE(VectorParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Out<VectorParams> CCS_INIT_S1(vector_params_out);
  Connections::Out<VectorInstructionConfig> CCS_INIT_S1(
      vector_instructions_out);

  Connections::Combinational<ac_int<VectorParams::width, false>>
      vector_params_bits;
  Connections::Combinational<VectorParams> converted_vector_params;
  TypeConverter<VectorParams> CCS_INIT_S1(vector_params_converter);

  Connections::Combinational<ac_int<VectorInstructionConfig::width, false>>
      vector_instructions_bit;
  Connections::Combinational<VectorInstructionConfig>
      converted_vector_instructions;
  TypeConverter<VectorInstructionConfig> CCS_INIT_S1(
      vector_instructions_converter);

  SC_CTOR(VectorParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    vector_params_converter.clk(clk);
    vector_params_converter.rstn(rstn);
    vector_params_converter.bits_in(vector_params_bits);
    vector_params_converter.type_out(converted_vector_params);

    vector_instructions_converter.clk(clk);
    vector_instructions_converter.rstn(rstn);
    vector_instructions_converter.bits_in(vector_instructions_bit);
    vector_instructions_converter.type_out(converted_vector_instructions);
  }

  void run() {
    serial_params_in.Reset();
    vector_instructions_out.Reset();
    vector_params_out.Reset();
    vector_params_bits.ResetWrite();
    vector_instructions_bit.ResetWrite();
    converted_vector_params.ResetRead();
    converted_vector_instructions.ResetRead();

    wait();

    while (true) {
      ac_int<VectorParams::width, false> vector_params_serialized =
          get_serialized_params<VectorParams, 64>(serial_params_in);
      vector_params_bits.Push(vector_params_serialized);
      VectorParams vector_params = converted_vector_params.Pop();

#ifndef __SYNTHESIS__
      std::ostringstream oss;
      oss << "Vector Params: " << std::endl << vector_params << std::endl;
      spdlog::debug(oss.str());
      oss.clear();
#endif

      vector_params_out.Push(vector_params);

      ac_int<VectorInstructionConfig::width, false>
          vector_instructions_serialized =
              get_serialized_params<VectorInstructionConfig, 64>(
                  serial_params_in);
      vector_instructions_bit.Push(vector_instructions_serialized);
      VectorInstructionConfig vector_instruction_config =
          converted_vector_instructions.Pop();

#ifndef __SYNTHESIS__
      oss << "Vector Instructions: " << std::endl
          << vector_instruction_config << std::endl;
      spdlog::debug(oss.str());
#endif

      vector_instructions_out.Push(vector_instruction_config);
    }
  }
};

SC_MODULE(DwCParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Out<DwCParams> CCS_INIT_S1(dwc_params_out);

  Connections::Combinational<ac_int<DwCParams::width, false>> CCS_INIT_S1(
      deserialized_params);
  Connections::Combinational<DwCParams> converted_params;
  TypeConverter<DwCParams> CCS_INIT_S1(converter);

  SC_CTOR(DwCParamsDeserializer) {
    converter.clk(clk);
    converter.rstn(rstn);
    converter.bits_in(deserialized_params);
    converter.type_out(converted_params);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    serial_params_in.Reset();
    deserialized_params.ResetWrite();
    converted_params.ResetRead();
    dwc_params_out.Reset();

    wait();

    while (true) {
      ac_int<DwCParams::width, false> dwcParams =
          get_serialized_params<DwCParams, 64>(serial_params_in);
      deserialized_params.Push(dwcParams);
      DwCParams params = converted_params.Pop();

#ifndef __SYNTHESIS__
      std::ostringstream oss;
      oss << "DwC Params: " << std::endl << params << std::endl;
      spdlog::debug(oss.str());
      oss.clear();
#endif

      dwc_params_out.Push(params);
    }
  }
};
