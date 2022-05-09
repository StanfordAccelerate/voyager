#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "DoubleBuffer.h"
#include "InputController.h"
#include "MatrixProcessor.h"
#include "ParamsDeserializer.h"
#include "VectorUnit.h"
#include "WeightController.h"
#include "mc_scverify.h"

SC_MODULE(Accelerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  ParamsRouter CCS_INIT_S1(paramsRouter);
  Connections::Combinational<int> serialMatrixParams[3];
  Connections::Combinational<int> serialVectorParams;

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
  CCS_DESIGN( (WeightController<INPUT_DATATYPE, DIMENSION, DIMENSION>) ) CCS_INIT_S1(weightController);
// clang-format on
#else
  WeightController<INPUT_DATATYPE, DIMENSION, DIMENSION> CCS_INIT_S1(
      weightController);
#endif

  DoubleBuffer<INPUT_DATATYPE, DIMENSION, WEIGHT_BUFFER_SIZE> CCS_INIT_S1(
      weightBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      weightDataResponse);
  Connections::Combinational<BufferWriteRequest<INPUT_DATATYPE, DIMENSION> >
      weightBufferWriteReq[2];
  Connections::Combinational<int> weightBufferWriteControl[2];
  Connections::Combinational<int> weightBufferReadAddress[2];
  Connections::Combinational<int> weightBufferReadControl[2];

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
  Connections::Combinational<Pack1D<WEIGHT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      weightsToSystolicArray);
  Connections::Combinational<Pack1D<ACCUM_DATATYPE, DIMENSION> > CCS_INIT_S1(
      outputsFromSystolicArray);

#ifdef SIM_VectorUnit
  // clang-format off
  CCS_DESIGN((VectorUnit<OUTPUT_DATATYPE, ACCUM_DATATYPE, DIMENSION>)) CCS_INIT_S1(vectorUnit);
  // clang-format on
#else
  VectorUnit<OUTPUT_DATATYPE, ACCUM_DATATYPE, DIMENSION> CCS_INIT_S1(
      vectorUnit);
#endif
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::In<Pack1D<OUTPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch0DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::In<Pack1D<OUTPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch1DataResponse);
  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::In<Pack1D<OUTPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorFetch2DataResponse);
  Connections::Out<Pack1D<OUTPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      vectorOutput);
  Connections::Out<int> CCS_INIT_S1(vectorOutputAddress);
  Connections::Out<Pack1D<OUTPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      scalarUnitOutput);
  Connections::Out<int> CCS_INIT_S1(scalarOutputAddress);

  Connections::SyncOut startSignal;
  Connections::SyncOut doneSignal;

  SC_CTOR(Accelerator) {
    paramsRouter.clk(clk);
    paramsRouter.rstn(rstn);
    paramsRouter.serialParamsIn(serialParamsIn);
    for (int i = 0; i < 3; i++) {
      paramsRouter.serialMatrixParams[i](serialMatrixParams[i]);
    }
    paramsRouter.serialVectorParams(serialVectorParams);
    paramsRouter.startSignal(startSignal);

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
    weightBuffer.output(weightsToSystolicArray);

    matrixProcessor.clk(clk);
    matrixProcessor.rstn(rstn);
    matrixProcessor.inputsChannel(inputsToSystolicArray);
    matrixProcessor.weightsChannel(weightsToSystolicArray);
    matrixProcessor.outputsChannel(outputsFromSystolicArray);
    matrixProcessor.serialParamsIn(serialMatrixParams[2]);

    vectorUnit.clk(clk);
    vectorUnit.rstn(rstn);
    vectorUnit.serialParamsIn(serialVectorParams);
    vectorUnit.systolicArrayOutput(outputsFromSystolicArray);
    vectorUnit.vectorFetch0AddressRequest(vectorFetch0AddressRequest);
    vectorUnit.vectorFetch0DataResponse(vectorFetch0DataResponse);
    vectorUnit.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
    vectorUnit.vectorFetch1DataResponse(vectorFetch1DataResponse);
    vectorUnit.vectorFetch2AddressRequest(vectorFetch2AddressRequest);
    vectorUnit.vectorFetch2DataResponse(vectorFetch2DataResponse);
    vectorUnit.scalarOutputAddress(scalarOutputAddress);
    vectorUnit.scalarUnitOutput(scalarUnitOutput);
    vectorUnit.vectorOutputAddress(vectorOutputAddress);
    vectorUnit.finalVectorOutput(vectorOutput);
    vectorUnit.done(doneSignal);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run();
};
