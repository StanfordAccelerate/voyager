#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "DoubleBuffer.h"
#include "DualPortBuffer.h"
#include "InputController.h"
#include "InputScaleController.h"
#include "MatrixProcessor.h"
#include "WeightController.h"
#include "WeightScaleController.h"
#include "mc_scverify.h"

SC_MODULE(MatrixUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

#if SUPPORT_MX
  static constexpr int PARAMS_MODULE_COUNT = 5;
  static constexpr int SCALE_PORT_WIDTH = SCALE_DATATYPE::width * OC_DIMENSION;
  typedef ac_int<SCALE_PORT_WIDTH, false> SCALE_PORT_TYPE;
#else
  static constexpr int PARAMS_MODULE_COUNT = 3;
#endif

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  static constexpr int ACCUM_BUFFER_BANKS = 2;
#else
  static constexpr int ACCUM_BUFFER_BANKS = 1;
#endif

  MatrixParamsDeserializer<0, PARAMS_MODULE_COUNT> CCS_INIT_S1(
      params_deserializer);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<MatrixParams> matrix_params[PARAMS_MODULE_COUNT];

  InputController<InputTypeList, IC_DIMENSION, IC_PORT_WIDTH,
                  INPUT_BUFFER_WIDTH>
      CCS_INIT_S1(input_controller);

  DoubleBuffer<INPUT_BUFFER_SIZE, INPUT_BUFFER_WIDTH> CCS_INIT_S1(input_buffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(input_resp);
  Connections::Combinational<
      BufferWriteRequest<ac_int<INPUT_BUFFER_WIDTH, false>>>
      input_buffer_write_req[2];
  Connections::Combinational<BufferReadRequest> input_buffer_read_req[2];
  Connections::Combinational<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      window_buffer_in);
  Connections::Combinational<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      window_buffer_out);

#if SUPPORT_MX
  InputScaleController<SCALE_DATATYPE, IC_DIMENSION> CCS_INIT_S1(
      input_scale_controller);
  DoubleBuffer<INPUT_BUFFER_SIZE, SCALE_DATATYPE::width> CCS_INIT_S1(
      input_scale_buffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_scale_req);
  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      input_scale_resp);
  Connections::Combinational<
      BufferWriteRequest<ac_int<SCALE_DATATYPE::width, false>>>
      input_scale_write_req[2];
  Connections::Combinational<BufferReadRequest> input_scale_read_req[2];
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      input_scale_read_resp);
#endif

  WeightController<WeightTypeList, ACCUM_BUFFER_DATATYPE, IC_DIMENSION,
                   OC_DIMENSION, OC_PORT_WIDTH, WEIGHT_BUFFER_WIDTH>
      CCS_INIT_S1(weight_controller);

  DoubleBuffer<WEIGHT_BUFFER_SIZE, WEIGHT_BUFFER_WIDTH> CCS_INIT_S1(
      weight_buffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weight_resp);
  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(bias_resp);
  Connections::Combinational<
      BufferWriteRequest<ac_int<WEIGHT_BUFFER_WIDTH, false>>>
      weight_buffer_write_req[2];
  Connections::Combinational<BufferReadRequest> weight_buffer_read_req[2];
  Connections::Combinational<ac_int<WEIGHT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      weight_buffer_read_resp);

#if SUPPORT_MX
  WeightScaleController<SCALE_DATATYPE, IC_DIMENSION, OC_DIMENSION,
                        OC_PORT_WIDTH>
      CCS_INIT_S1(weight_scale_controller);
  DoubleBuffer<WEIGHT_BUFFER_SIZE / IC_DIMENSION,
               SCALE_DATATYPE::width * OC_DIMENSION>
      CCS_INIT_S1(weight_scale_buffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weight_scale_resp);
  Connections::Combinational<BufferWriteRequest<SCALE_PORT_TYPE>>
      weight_scale_write_req[2];
  Connections::Combinational<BufferReadRequest> weight_scale_read_req[2];
  Connections::Combinational<ac_int<SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      weight_scale_read_resp);
#endif

  MatrixProcessor<InputTypeList, WeightTypeList, SA_INPUT_TYPE, SA_WEIGHT_TYPE,
                  ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE,
                  IC_DIMENSION, OC_DIMENSION, ACCUM_BUFFER_SIZE>
      CCS_INIT_S1(matrix_processor);
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(bias_data);

  DualPortBuffer<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>, ACCUM_BUFFER_SIZE>
      CCS_INIT_S1(accumulation_buffer);
  Connections::Combinational<ac_int<16, false>>
      accumulation_buffer_mu_read_address[ACCUM_BUFFER_BANKS];
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_mu_read_data[ACCUM_BUFFER_BANKS];
  Connections::Combinational<
      BufferWriteRequest<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>>
      accumulation_buffer_mu_write_request[ACCUM_BUFFER_BANKS];

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::SyncChannel accumulation_buffer_mu_done[ACCUM_BUFFER_BANKS];

  Connections::In<ac_int<16, false>> accumulation_buffer_vu_read_address[2];
  Connections::Out<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_vu_read_data[2];
  // Write request from Vector Unit, unused for now
  Connections::Combinational<
      BufferWriteRequest<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>>
      accumulation_buffer_vu_write_request[2];
  Connections::SyncIn accumulation_buffer_vu_done[2];
#endif

  Connections::Out<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>> output_channel;

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  SC_CTOR(MatrixUnit) {
    params_deserializer.clk(clk);
    params_deserializer.rstn(rstn);
    params_deserializer.serial_params_in(serial_params_in);
    for (int i = 0; i < PARAMS_MODULE_COUNT; i++) {
      params_deserializer.params_out[i](matrix_params[i]);
    }

    input_controller.clk(clk);
    input_controller.rstn(rstn);
    input_controller.input_req(input_req);
    input_controller.input_resp(input_resp);
    input_controller.params_in(matrix_params[0]);
    input_controller.window_buffer_in(window_buffer_in);
    input_controller.window_buffer_out(window_buffer_out);

    input_buffer.clk(clk);
    input_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      input_controller.write_request[i](input_buffer_write_req[i]);
      input_controller.read_request[i](input_buffer_read_req[i]);

      input_buffer.write_request[i](input_buffer_write_req[i]);
      input_buffer.read_request[i](input_buffer_read_req[i]);
    }
    input_buffer.output(window_buffer_in);

#if SUPPORT_MX
    input_scale_controller.clk(clk);
    input_scale_controller.rstn(rstn);
    input_scale_controller.scale_req(input_scale_req);
    input_scale_controller.scale_resp(input_scale_resp);
    input_scale_controller.params_in(matrix_params[3]);

    input_scale_buffer.clk(clk);
    input_scale_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      input_scale_controller.write_request[i](input_scale_write_req[i]);
      input_scale_controller.read_request[i](input_scale_read_req[i]);

      input_scale_buffer.write_request[i](input_scale_write_req[i]);
      input_scale_buffer.read_request[i](input_scale_read_req[i]);
    }
    input_scale_buffer.output(input_scale_read_resp);
#endif

    weight_controller.clk(clk);
    weight_controller.rstn(rstn);
    weight_controller.weight_req(weight_req);
    weight_controller.weight_resp(weight_resp);
    weight_controller.params_in(matrix_params[1]);
    weight_controller.bias_req(bias_req);
    weight_controller.bias_resp(bias_resp);
    weight_controller.bias_data(bias_data);

    weight_buffer.clk(clk);
    weight_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weight_controller.write_request[i](weight_buffer_write_req[i]);
      weight_controller.read_request[i](weight_buffer_read_req[i]);

      weight_buffer.write_request[i](weight_buffer_write_req[i]);
      weight_buffer.read_request[i](weight_buffer_read_req[i]);
    }
    weight_buffer.output(weight_buffer_read_resp);

#if SUPPORT_MX
    weight_scale_controller.clk(clk);
    weight_scale_controller.rstn(rstn);
    weight_scale_controller.weight_scale_req(weight_scale_req);
    weight_scale_controller.weight_scale_resp(weight_scale_resp);
    weight_scale_controller.params_in(matrix_params[4]);

    weight_scale_buffer.clk(clk);
    weight_scale_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weight_scale_controller.write_request[i](weight_scale_write_req[i]);
      weight_scale_controller.read_request[i](weight_scale_read_req[i]);

      weight_scale_buffer.write_request[i](weight_scale_write_req[i]);
      weight_scale_buffer.read_request[i](weight_scale_read_req[i]);
    }
    weight_scale_buffer.output(weight_scale_read_resp);
#endif

    matrix_processor.clk(clk);
    matrix_processor.rstn(rstn);
    matrix_processor.input_channel(window_buffer_out);
    matrix_processor.weight_channel(weight_buffer_read_resp);
    matrix_processor.bias_channel(bias_data);
    matrix_processor.params_in(matrix_params[2]);
    matrix_processor.start(start);
    matrix_processor.done(done);

    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
      matrix_processor.accumulation_buffer_read_address[i](
          accumulation_buffer_mu_read_address[i]);
      matrix_processor.accumulation_buffer_read_data[i](
          accumulation_buffer_mu_read_data[i]);
      matrix_processor.accumulation_buffer_write_request[i](
          accumulation_buffer_mu_write_request[i]);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      matrix_processor.accumulation_buffer_done[i](
          accumulation_buffer_mu_done[i]);
#endif
    }

    matrix_processor.output_channel(output_channel);

    accumulation_buffer.clk(clk);
    accumulation_buffer.rstn(rstn);

    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
      accumulation_buffer.read_address[i * 2](
          accumulation_buffer_mu_read_address[i]);
      accumulation_buffer.read_data[i * 2](accumulation_buffer_mu_read_data[i]);
      accumulation_buffer.write_request[i * 2](
          accumulation_buffer_mu_write_request[i]);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      accumulation_buffer.done[i * 2](accumulation_buffer_mu_done[i]);
#endif
    }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
      accumulation_buffer.read_address[i * 2 + 1](
          accumulation_buffer_vu_read_address[i]);
      accumulation_buffer.read_data[i * 2 + 1](
          accumulation_buffer_vu_read_data[i]);
      accumulation_buffer.write_request[i * 2 + 1](
          accumulation_buffer_vu_write_request[i]);
      accumulation_buffer.done[i * 2 + 1](accumulation_buffer_vu_done[i]);
    }
#endif

#if SUPPORT_MX
    matrix_processor.input_scale_channel(input_scale_read_resp);
    matrix_processor.weight_scale_channel(weight_scale_read_resp);
#endif
  }
};
