#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "DoubleBuffer.h"
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
#else
  static constexpr int PARAMS_MODULE_COUNT = 3;
#endif

  MatrixParamsRouter<PARAMS_MODULE_COUNT> CCS_INIT_S1(paramsRouter);
  Connections::In<int> CCS_INIT_S1(serialMatrixParamsIn);
  Connections::Combinational<int> serialMatrixParams[PARAMS_MODULE_COUNT];

  // clang-format off
#ifdef SIM_InputController
  CCS_DESIGN((InputController<INPUT_DATATYPE, IC_DIMENSION>)) CCS_INIT_S1(inputController);
#else
  InputController<INPUT_DATATYPE, IC_DIMENSION> CCS_INIT_S1(inputController);
#endif
  // clang-format on

  DoubleBuffer<IC_PORT_WIDTH, INPUT_BUFFER_SIZE> CCS_INIT_S1(inputBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  Connections::In<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(inputDataResponse);
  Connections::Combinational<BufferWriteRequest<IC_PORT_WIDTH>>
      inputBufferWriteReq[2];
  Connections::Combinational<ac_int<32, false>> inputBufferWriteControl[2];
  Connections::Combinational<ac_int<16, false>> inputBufferReadAddress[2];
  Connections::Combinational<ac_int<32, false>> inputBufferReadControl[2];
  Connections::Combinational<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(
      inputsToWindowBuffer);
  Connections::Combinational<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(
      inputsFromBuffer);

#if SUPPORT_MX
  InputScaleController<SCALE_DATATYPE, IC_DIMENSION> CCS_INIT_S1(
      inputScaleController);
  DoubleBuffer<SCALE_DATATYPE::width, INPUT_BUFFER_SIZE> CCS_INIT_S1(
      inputScaleBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputScaleAddressRequest);
  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleDataResponse);
  Connections::Combinational<BufferWriteRequest<SCALE_DATATYPE::width>>
      inputScaleWriteRequest[2];
  Connections::Combinational<ac_int<32, false>> inputScaleWriteControl[2];
  Connections::Combinational<ac_int<16, false>> inputScaleReadAddress[2];
  Connections::Combinational<ac_int<32, false>> inputScaleReadControl[2];
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      inputScaleFromBuffer);
#endif

#ifdef SIM_WeightController
  // clang-format off
  CCS_DESIGN( (WeightController<WEIGHT_DATATYPE, ACCUM_DATATYPE, IC_DIMENSION, OC_DIMENSION>) ) CCS_INIT_S1(weightController);
  // clang-format on
#else
  WeightController<WEIGHT_DATATYPE, ACCUM_BUFFER_DATATYPE, IC_DIMENSION,
                   OC_DIMENSION>
      CCS_INIT_S1(weightController);
#endif

  DoubleBuffer<OC_PORT_WIDTH, WEIGHT_BUFFER_SIZE> CCS_INIT_S1(weightBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weightDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(biasDataResponse);
  Connections::Combinational<BufferWriteRequest<OC_PORT_WIDTH>>
      weightBufferWriteReq[2];
  Connections::Combinational<ac_int<32, false>> weightBufferWriteControl[2];
  Connections::Combinational<ac_int<16, false>> weightBufferReadAddress[2];
  Connections::Combinational<ac_int<32, false>> weightBufferReadControl[2];
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightsFromBuffer);

#if SUPPORT_MX
  WeightScaleController<SCALE_DATATYPE, IC_DIMENSION, OC_DIMENSION> CCS_INIT_S1(
      weightScaleController);
  DoubleBuffer<SCALE_DATATYPE::width * OC_DIMENSION, WEIGHT_BUFFER_SIZE>
      CCS_INIT_S1(weightScaleBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightScaleAddressRequest);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleDataResponse);
  Connections::Combinational<BufferWriteRequest<SCALE_PORT_WIDTH>>
      weightScaleWriteRequest[2];
  Connections::Combinational<ac_int<32, false>> weightScaleWriteControl[2];
  Connections::Combinational<ac_int<16, false>> weightScaleReadAddress[2];
  Connections::Combinational<ac_int<32, false>> weightScaleReadControl[2];
  Connections::Combinational<ac_int<SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      weightScaleFromBuffer);
#endif

#ifdef SIM_MatrixProcessor
  // clang-format off
  CCS_DESIGN( (MatrixProcessor<INPUT_DATATYPE, WEIGHT_DATATYPE, ACCUM_DATATYPE,
    ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE, IC_DIMENSION, OC_DIMENSION, ACCUM_BUFFER_SIZE>) ) CCS_INIT_S1(matrixProcessor);
  // clang-format on
#else
  MatrixProcessor<INPUT_DATATYPE, WEIGHT_DATATYPE, ACCUM_DATATYPE,
                  ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE, IC_DIMENSION,
                  OC_DIMENSION, ACCUM_BUFFER_SIZE>
      CCS_INIT_S1(matrixProcessor);
#endif

  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(biasToSystolicArray);
  Connections::Out<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>> CCS_INIT_S1(
      outputsFromSystolicArray);

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
      inputController.writeControl[i](inputBufferWriteControl[i]);
      inputController.readAddress[i](inputBufferReadAddress[i]);
      inputController.readControl[i](inputBufferReadControl[i]);

      inputBuffer.writeRequest[i](inputBufferWriteReq[i]);
      inputBuffer.writeControl[i](inputBufferWriteControl[i]);
      inputBuffer.readAddress[i](inputBufferReadAddress[i]);
      inputBuffer.readControl[i](inputBufferReadControl[i]);
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
      inputScaleController.writeControl[i](inputScaleWriteControl[i]);
      inputScaleController.readAddress[i](inputScaleReadAddress[i]);
      inputScaleController.readControl[i](inputScaleReadControl[i]);

      inputScaleBuffer.writeRequest[i](inputScaleWriteRequest[i]);
      inputScaleBuffer.writeControl[i](inputScaleWriteControl[i]);
      inputScaleBuffer.readAddress[i](inputScaleReadAddress[i]);
      inputScaleBuffer.readControl[i](inputScaleReadControl[i]);
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
      weightController.writeControl[i](weightBufferWriteControl[i]);
      weightController.readAddress[i](weightBufferReadAddress[i]);
      weightController.readControl[i](weightBufferReadControl[i]);

      weightBuffer.writeRequest[i](weightBufferWriteReq[i]);
      weightBuffer.writeControl[i](weightBufferWriteControl[i]);
      weightBuffer.readAddress[i](weightBufferReadAddress[i]);
      weightBuffer.readControl[i](weightBufferReadControl[i]);
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
      weightScaleController.writeControl[i](weightScaleWriteControl[i]);
      weightScaleController.readAddress[i](weightScaleReadAddress[i]);
      weightScaleController.readControl[i](weightScaleReadControl[i]);

      weightScaleBuffer.writeRequest[i](weightScaleWriteRequest[i]);
      weightScaleBuffer.writeControl[i](weightScaleWriteControl[i]);
      weightScaleBuffer.readAddress[i](weightScaleReadAddress[i]);
      weightScaleBuffer.readControl[i](weightScaleReadControl[i]);
    }
    weightScaleBuffer.output(weightScaleFromBuffer);
#endif

    matrixProcessor.clk(clk);
    matrixProcessor.rstn(rstn);
    matrixProcessor.inputsChannel(inputsFromBuffer);
    matrixProcessor.weightsChannel(weightsFromBuffer);
    matrixProcessor.biasChannel(biasToSystolicArray);
    matrixProcessor.outputsChannel(outputsFromSystolicArray);
    matrixProcessor.serialParamsIn(serialMatrixParams[2]);
    matrixProcessor.startSignal(startSignal);
    matrixProcessor.doneSignal(doneSignal);

#if SUPPORT_MX
    matrixProcessor.inputScaleChannel(inputScaleFromBuffer);
    matrixProcessor.weightScaleChannel(weightScaleFromBuffer);
#endif
  }
};
