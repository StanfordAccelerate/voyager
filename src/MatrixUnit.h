#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "DoubleBuffer.h"
#include "InputController.h"
#include "MatrixProcessor.h"
#include "WeightController.h"
#include "mc_scverify.h"

SC_MODULE(MatrixUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixParamsRouter CCS_INIT_S1(paramsRouter);
  Connections::In<int> CCS_INIT_S1(serialMatrixParamsIn);
  Connections::Combinational<int> serialMatrixParams[3];

  // clang-format off
  #ifdef SIM_InputController  
  CCS_DESIGN((InputController<INPUT_DATATYPE, DIMENSION>)) CCS_INIT_S1(inputController);
  #else
  InputController<INPUT_DATATYPE, DIMENSION> CCS_INIT_S1(inputController);
  #endif
  // clang-format on

  DoubleBuffer<INPUT_DATATYPE, DIMENSION, INPUT_BUFFER_SIZE> CCS_INIT_S1(
      inputBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputDataResponse);
  Connections::Combinational<BufferWriteRequest<INPUT_DATATYPE, DIMENSION> >
      inputBufferWriteReq[2];
  Connections::Combinational<int> inputBufferWriteControl[2];
  Connections::Combinational<int> inputBufferReadAddress[2];
  Connections::Combinational<int> inputBufferReadControl[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputsToWindowBuffer);

#ifdef SIM_WeightController
  // clang-format off
  CCS_DESIGN( (WeightController<INPUT_DATATYPE, ACCUM_DATATYPE, DIMENSION, DIMENSION>) ) CCS_INIT_S1(weightController);
// clang-format on
#else
  WeightController<INPUT_DATATYPE, ACCUM_DATATYPE, DIMENSION, DIMENSION>
      CCS_INIT_S1(weightController);
#endif

  DoubleBuffer<INPUT_DATATYPE, DIMENSION, WEIGHT_BUFFER_SIZE> CCS_INIT_S1(
      weightBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      weightDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(gradAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      gradDataResponse);
  Connections::Combinational<BufferWriteRequest<INPUT_DATATYPE, DIMENSION> >
      weightBufferWriteReq[2];
  Connections::Combinational<int> weightBufferWriteControl[2];
  Connections::Combinational<int> weightBufferReadAddress[2];
  Connections::Combinational<int> weightBufferReadControl[2];
  Connections::Combinational<Pack1D<P8, DIMENSION> > CCS_INIT_S1(
      weightsFromBuffer);

#ifdef SIM_MatrixProcessor
  // clang-format off
  CCS_DESIGN( (MatrixProcessor<INPUT_DATATYPE, ACCUM_DATATYPE, DIMENSION, DIMENSION, ACCUMULATION_BUFFER_SIZE>) ) CCS_INIT_S1(matrixProcessor);
// clang-format on
#else
  MatrixProcessor<INPUT_DATATYPE, ACCUM_DATATYPE, DIMENSION, DIMENSION,
                  ACCUMULATION_BUFFER_SIZE>
      CCS_INIT_S1(matrixProcessor);
#endif
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputsToSystolicArray);
  Connections::Combinational<Pack1D<P8D, DIMENSION> > CCS_INIT_S1(
      weightsToSystolicArray);
  Connections::Out<Pack1D<ACCUM_DATATYPE, DIMENSION> > CCS_INIT_S1(
      outputsFromSystolicArray);

  Connections::SyncOut CCS_INIT_S1(startSignal);
  Connections::SyncOut CCS_INIT_S1(doneSignal);

  SC_CTOR(MatrixUnit) {
    paramsRouter.clk(clk);
    paramsRouter.rstn(rstn);
    paramsRouter.serialParamsIn(serialMatrixParamsIn);
    for (int i = 0; i < 3; i++) {
      paramsRouter.serialMatrixParams[i](serialMatrixParams[i]);
    }

    inputController.clk(clk);
    inputController.rstn(rstn);
    inputController.addressRequest(inputAddressRequest);
    inputController.dataResponse(inputDataResponse);
    inputController.serialParamsIn(serialMatrixParams[0]);
    inputController.windowBufferIn(inputsToWindowBuffer);
    inputController.windowBufferOut(inputsToSystolicArray);

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

    weightController.clk(clk);
    weightController.rstn(rstn);
    weightController.addressRequest(weightAddressRequest);
    weightController.dataResponse(weightDataResponse);
    weightController.serialParamsIn(serialMatrixParams[1]);
    weightController.gradAddressRequest(gradAddressRequest);
    weightController.gradDataResponse(gradDataResponse);
    weightController.weightsFromBuffer(weightsFromBuffer);
    weightController.weightsToSystolicArray(weightsToSystolicArray);

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

    matrixProcessor.clk(clk);
    matrixProcessor.rstn(rstn);
    matrixProcessor.inputsChannel(inputsToSystolicArray);
    matrixProcessor.weightsChannel(weightsToSystolicArray);
    matrixProcessor.outputsChannel(outputsFromSystolicArray);
    matrixProcessor.serialParamsIn(serialMatrixParams[2]);
    matrixProcessor.startSignal(startSignal);
    matrixProcessor.doneSignal(doneSignal);
  }
};