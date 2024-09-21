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

  MatrixParamsRouter CCS_INIT_S1(paramsRouter);
  Connections::In<int> CCS_INIT_S1(serialMatrixParamsIn);
  Connections::Combinational<int> serialMatrixParams[5];

  // clang-format off
  #ifdef SIM_InputController
  CCS_DESIGN((InputController<INPUT_DATATYPE, IC_DIMENSION>)) CCS_INIT_S1(inputController);
  #else
  InputController<INPUT_DATATYPE, IC_DIMENSION> CCS_INIT_S1(inputController);
  #endif
  // clang-format on

  DoubleBuffer<INPUT_DATATYPE, IC_DIMENSION, INPUT_BUFFER_SIZE> CCS_INIT_S1(
      inputBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, IC_DIMENSION> > CCS_INIT_S1(
      inputDataResponse);
  Connections::Combinational<BufferWriteRequest<INPUT_DATATYPE, IC_DIMENSION> >
      inputBufferWriteReq[2];
  Connections::Combinational<int> inputBufferWriteControl[2];
  Connections::Combinational<int> inputBufferReadAddress[2];
  Connections::Combinational<int> inputBufferReadControl[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, IC_DIMENSION> > CCS_INIT_S1(
      inputsToWindowBuffer);

  // TODO: conditional initialization
  InputScaleController<INPUT_DATATYPE, IC_DIMENSION> CCS_INIT_S1(
      inputScaleController);
  DoubleBuffer<INPUT_DATATYPE, 1, INPUT_BUFFER_SIZE> CCS_INIT_S1(
      inputScaleBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputScaleAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, 1> > CCS_INIT_S1(
      inputScaleDataResponse);
  Connections::Combinational<BufferWriteRequest<INPUT_DATATYPE, 1> >
      inputScaleWriteRequest[2];
  Connections::Combinational<int> inputScaleWriteControl[2];
  Connections::Combinational<int> inputScaleReadAddress[2];
  Connections::Combinational<int> inputScaleReadControl[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, 1> > CCS_INIT_S1(
      inputScaleFromBuffer);

#ifdef SIM_WeightController
  // clang-format off
  CCS_DESIGN( (WeightController<INPUT_DATATYPE, ACCUM_DATATYPE, IC_DIMENSION, OC_DIMENSION>) ) CCS_INIT_S1(weightController);
// clang-format on
#else
  WeightController<INPUT_DATATYPE, ACCUM_BUFFER_DATATYPE, IC_DIMENSION,
                   OC_DIMENSION>
      CCS_INIT_S1(weightController);
#endif

  DoubleBuffer<INPUT_DATATYPE, OC_DIMENSION, WEIGHT_BUFFER_SIZE> CCS_INIT_S1(
      weightBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      weightDataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(biasAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      biasDataResponse);
  Connections::Combinational<BufferWriteRequest<INPUT_DATATYPE, OC_DIMENSION> >
      weightBufferWriteReq[2];
  Connections::Combinational<int> weightBufferWriteControl[2];
  Connections::Combinational<int> weightBufferReadAddress[2];
  Connections::Combinational<int> weightBufferReadControl[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      weightsFromBuffer);

  // TODO: conditional initialization
  WeightScaleController<INPUT_DATATYPE, ACCUM_BUFFER_DATATYPE, IC_DIMENSION,
                        OC_DIMENSION>
      CCS_INIT_S1(weightScaleController);
  DoubleBuffer<INPUT_DATATYPE, OC_DIMENSION, WEIGHT_BUFFER_SIZE> CCS_INIT_S1(
      weightScaleBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightScaleAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      weightScaleDataResponse);
  Connections::Combinational<BufferWriteRequest<INPUT_DATATYPE, OC_DIMENSION> >
      weightScaleWriteRequest[2];
  Connections::Combinational<int> weightScaleWriteControl[2];
  Connections::Combinational<int> weightScaleReadAddress[2];
  Connections::Combinational<int> weightScaleReadControl[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      weightScaleFromBuffer);

#ifdef SIM_MatrixProcessor
  // clang-format off
  CCS_DESIGN( (MatrixProcessor<INPUT_DATATYPE, ACCUM_DATATYPE, IC_DIMENSION, OC_DIMENSION, ACCUMULATION_BUFFER_SIZE>) ) CCS_INIT_S1(matrixProcessor);
// clang-format on
#else
#ifdef HYBRID_FP8
  MatrixProcessor<HYBRID_TYPE, ACCUM_DATATYPE, IC_DIMENSION, OC_DIMENSION,
                  ACCUMULATION_BUFFER_SIZE>
      CCS_INIT_S1(matrixProcessor);
#else
  MatrixProcessor<INPUT_DATATYPE, ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE, MX,
                  IC_DIMENSION, OC_DIMENSION, ACCUMULATION_BUFFER_SIZE>
      CCS_INIT_S1(matrixProcessor);
#endif
#endif

#ifdef HYBRID_FP8
  Connections::Combinational<Pack1D<HYBRID_TYPE, IC_DIMENSION> > CCS_INIT_S1(
      inputsToSystolicArray);
  Connections::Combinational<Pack1D<HYBRID_TYPE, OC_DIMENSION> > CCS_INIT_S1(
      weightsToSystolicArray);
#else
  Connections::Combinational<Pack1D<INPUT_DATATYPE, IC_DIMENSION> > CCS_INIT_S1(
      inputsToSystolicArray);

  Connections::Combinational<
      Pack1D<INPUT_DATATYPE::AccumulationDatatype, OC_DIMENSION> >
      CCS_INIT_S1(weightsToSystolicArray);
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION> >
      CCS_INIT_S1(biasToSystolicArray);
#endif
  Connections::Out<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION> > CCS_INIT_S1(
      outputsFromSystolicArray);

  Connections::SyncOut CCS_INIT_S1(startSignal);
  Connections::SyncOut CCS_INIT_S1(doneSignal);

  SC_CTOR(MatrixUnit) {
    paramsRouter.clk(clk);
    paramsRouter.rstn(rstn);
    paramsRouter.serialParamsIn(serialMatrixParamsIn);
    for (int i = 0; i < 5; i++) {
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

    weightController.clk(clk);
    weightController.rstn(rstn);
    weightController.addressRequest(weightAddressRequest);
    weightController.dataResponse(weightDataResponse);
    weightController.serialParamsIn(serialMatrixParams[1]);
    weightController.biasAddressRequest(biasAddressRequest);
    weightController.biasDataResponse(biasDataResponse);
    weightController.weightsToSystolicArray(weightsToSystolicArray);
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

    matrixProcessor.clk(clk);
    matrixProcessor.rstn(rstn);
    matrixProcessor.inputsChannel(inputsToSystolicArray);
    matrixProcessor.inputScaleChannel(inputScaleFromBuffer);
    matrixProcessor.weightsChannel(weightsFromBuffer);
    matrixProcessor.weightScaleChannel(weightScaleFromBuffer);
    matrixProcessor.biasChannel(biasToSystolicArray);
    matrixProcessor.outputsChannel(outputsFromSystolicArray);
    matrixProcessor.serialParamsIn(serialMatrixParams[2]);
    matrixProcessor.startSignal(startSignal);
    matrixProcessor.doneSignal(doneSignal);
  }
};