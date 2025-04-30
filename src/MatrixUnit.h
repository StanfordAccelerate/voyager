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

  MatrixParamsRouter<PARAMS_MODULE_COUNT> CCS_INIT_S1(paramsRouter);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serialMatrixParamsIn);
  Connections::Combinational<ac_int<64, false>>
      serialMatrixParams[PARAMS_MODULE_COUNT];

  InputController<InputTypeList, IC_DIMENSION, IC_PORT_WIDTH,
                  INPUT_BUFFER_WIDTH>
      CCS_INIT_S1(inputController);

  DoubleBuffer<INPUT_BUFFER_SIZE, INPUT_BUFFER_WIDTH> CCS_INIT_S1(inputBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputAddressRequest);

  Connections::In<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(inputDataResponse);
  Connections::Combinational<
      BufferWriteRequest<ac_int<INPUT_BUFFER_WIDTH, false>>>
      inputBufferWriteReq[2];
  Connections::Combinational<BufferReadRequest> inputBufferReadAddress[2];
  Connections::Combinational<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      inputsToWindowBuffer);
  Connections::Combinational<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      inputsFromBuffer);

#if SUPPORT_MX
  InputScaleController<SCALE_DATATYPE, IC_DIMENSION> CCS_INIT_S1(
      inputScaleController);
  DoubleBuffer<INPUT_BUFFER_SIZE, SCALE_DATATYPE::width> CCS_INIT_S1(
      inputScaleBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputScaleAddressRequest);
  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleDataResponse);
  Connections::Combinational<
      BufferWriteRequest<ac_int<SCALE_DATATYPE::width, false>>>
      inputScaleWriteRequest[2];
  Connections::Combinational<BufferReadRequest> inputScaleReadAddress[2];
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleFromBuffer);
#endif

  WeightController<WeightTypeList, ACCUM_BUFFER_DATATYPE, IC_DIMENSION,
                   OC_DIMENSION, OC_PORT_WIDTH, WEIGHT_BUFFER_WIDTH>
      CCS_INIT_S1(weightController);

  DoubleBuffer<WEIGHT_BUFFER_SIZE, WEIGHT_BUFFER_WIDTH> CCS_INIT_S1(
      weightBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weightDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(biasDataResponse);
  Connections::Combinational<
      BufferWriteRequest<ac_int<WEIGHT_BUFFER_WIDTH, false>>>
      weightBufferWriteReq[2];
  Connections::Combinational<BufferReadRequest> weightBufferReadAddress[2];
  Connections::Combinational<ac_int<WEIGHT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      weightsFromBuffer);

#if SUPPORT_MX
  WeightScaleController<SCALE_DATATYPE, IC_DIMENSION, OC_DIMENSION,
                        OC_PORT_WIDTH>
      CCS_INIT_S1(weightScaleController);
  DoubleBuffer<WEIGHT_BUFFER_SIZE, SCALE_DATATYPE::width * OC_DIMENSION>
      CCS_INIT_S1(weightScaleBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightScaleAddressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleDataResponse);
  Connections::Combinational<BufferWriteRequest<SCALE_PORT_TYPE>>
      weightScaleWriteRequest[2];
  Connections::Combinational<BufferReadRequest> weightScaleReadAddress[2];
  Connections::Combinational<ac_int<SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleFromBuffer);
#endif

  MatrixProcessor<InputTypeList, WeightTypeList, SA_INPUT_TYPE, SA_WEIGHT_TYPE,
                  ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE,
                  IC_DIMENSION, OC_DIMENSION, ACCUM_BUFFER_SIZE>
      CCS_INIT_S1(matrixProcessor);
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(biasToSystolicArray);

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
  Connections::In<
      BufferWriteRequest<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>>
      accumulation_buffer_vu_write_request[2];
  Connections::SyncIn accumulation_buffer_vu_done[2];
#endif

  Connections::Out<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      matrixUnitOutputChannel;

  Connections::SyncOut CCS_INIT_S1(startSignal);
  Connections::SyncOut CCS_INIT_S1(doneSignal);

  SC_CTOR(MatrixUnit) {
    paramsRouter.clk(clk);
    paramsRouter.rstn(rstn);
    paramsRouter.serialParamsIn(serialMatrixParamsIn);
    for (int i = 0; i < PARAMS_MODULE_COUNT; i++) {
      paramsRouter.serialMatrixParams[i](serialMatrixParams[i]);
    }

    inputController.clk(clk);
    inputController.rstn(rstn);
    inputController.addressRequest(inputAddressRequest);
    inputController.dataResponse(inputDataResponse);
    inputController.serialParamsIn(serialMatrixParams[0]);
    inputController.windowBufferIn(inputsToWindowBuffer);
    inputController.windowBufferOut(inputsFromBuffer);

    inputBuffer.clk(clk);
    inputBuffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      inputController.writeRequest[i](inputBufferWriteReq[i]);
      inputController.readAddress[i](inputBufferReadAddress[i]);

      inputBuffer.writeRequest[i](inputBufferWriteReq[i]);
      inputBuffer.readAddress[i](inputBufferReadAddress[i]);
    }
    inputBuffer.output(inputsToWindowBuffer);

#if SUPPORT_MX
    inputScaleController.clk(clk);
    inputScaleController.rstn(rstn);
    inputScaleController.addressRequest(inputScaleAddressRequest);
    inputScaleController.dataResponse(inputScaleDataResponse);
    inputScaleController.serialParamsIn(serialMatrixParams[3]);

    inputScaleBuffer.clk(clk);
    inputScaleBuffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      inputScaleController.writeRequest[i](inputScaleWriteRequest[i]);
      inputScaleController.readAddress[i](inputScaleReadAddress[i]);

      inputScaleBuffer.writeRequest[i](inputScaleWriteRequest[i]);
      inputScaleBuffer.readAddress[i](inputScaleReadAddress[i]);
    }
    inputScaleBuffer.output(inputScaleFromBuffer);
#endif

    weightController.clk(clk);
    weightController.rstn(rstn);
    weightController.addressRequest(weightAddressRequest);
    weightController.dataResponse(weightDataResponse);
    weightController.serialParamsIn(serialMatrixParams[1]);
    weightController.biasAddressRequest(biasAddressRequest);
    weightController.biasDataResponse(biasDataResponse);
    weightController.biasToSystolicArray(biasToSystolicArray);

    weightBuffer.clk(clk);
    weightBuffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weightController.writeRequest[i](weightBufferWriteReq[i]);
      weightController.readAddress[i](weightBufferReadAddress[i]);

      weightBuffer.writeRequest[i](weightBufferWriteReq[i]);
      weightBuffer.readAddress[i](weightBufferReadAddress[i]);
    }
    weightBuffer.output(weightsFromBuffer);

#if SUPPORT_MX
    weightScaleController.clk(clk);
    weightScaleController.rstn(rstn);
    weightScaleController.addressRequest(weightScaleAddressRequest);
    weightScaleController.dataResponse(weightScaleDataResponse);
    weightScaleController.serialParamsIn(serialMatrixParams[4]);

    weightScaleBuffer.clk(clk);
    weightScaleBuffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weightScaleController.writeRequest[i](weightScaleWriteRequest[i]);
      weightScaleController.readAddress[i](weightScaleReadAddress[i]);

      weightScaleBuffer.writeRequest[i](weightScaleWriteRequest[i]);
      weightScaleBuffer.readAddress[i](weightScaleReadAddress[i]);
    }
    weightScaleBuffer.output(weightScaleFromBuffer);
#endif

    matrixProcessor.clk(clk);
    matrixProcessor.rstn(rstn);
    matrixProcessor.inputsChannel(inputsFromBuffer);
    matrixProcessor.weightsChannel(weightsFromBuffer);
    matrixProcessor.biasChannel(biasToSystolicArray);
    matrixProcessor.serialParamsIn(serialMatrixParams[2]);
    matrixProcessor.startSignal(startSignal);
    matrixProcessor.doneSignal(doneSignal);

    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
      matrixProcessor.accumulation_buffer_read_address[i](
          accumulation_buffer_mu_read_address[i]);
      matrixProcessor.accumulation_buffer_read_data[i](
          accumulation_buffer_mu_read_data[i]);
      matrixProcessor.accumulation_buffer_write_request[i](
          accumulation_buffer_mu_write_request[i]);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      matrixProcessor.accumulation_buffer_done[i](
          accumulation_buffer_mu_done[i]);
#endif
    }

    matrixProcessor.matrixUnitOutputChannel(matrixUnitOutputChannel);

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
    matrixProcessor.inputScaleChannel(inputScaleFromBuffer);
    matrixProcessor.weightScaleChannel(weightScaleFromBuffer);
#endif
  }
};
