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

SC_MODULE(Accelerator) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  ParamsDeserializer CCS_INIT_S1(paramsDeserializer);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(paramsIn);

  InputController<INPUT_DATATYPE, DIMENSION> CCS_INIT_S1(inputController);
  DoubleBuffer<INPUT_DATATYPE, DIMENSION, INPUT_BUFFER_SIZE> CCS_INIT_S1(
      inputBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(inputAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputDataResponse);
  Connections::Combinational<int> inputBufferWriteAddress[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
      inputBufferWriteData[2];
  Connections::Combinational<int> inputBufferWriteControl[2];
  Connections::Combinational<int> inputBufferReadAddress[2];
  Connections::Combinational<int> inputBufferReadControl[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputsToWindowBuffer);
  Connections::Combinational<MatrixParams> inputControllerParams;

  WeightController<INPUT_DATATYPE, DIMENSION, DIMENSION> CCS_INIT_S1(
      weightController);
  DoubleBuffer<INPUT_DATATYPE, DIMENSION, WEIGHT_BUFFER_SIZE> CCS_INIT_S1(
      weightBuffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weightAddressRequest);
  Connections::In<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      weightDataResponse);
  Connections::Combinational<int> weightBufferWriteAddress[2];
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
      weightBufferWriteData[2];
  Connections::Combinational<int> weightBufferWriteControl[2];
  Connections::Combinational<int> weightBufferReadAddress[2];
  Connections::Combinational<int> weightBufferReadControl[2];
  Connections::Combinational<MatrixParams> weightControllerParams;

  MatrixProcessor<INPUT_DATATYPE, INTERMEDIATE_DATATYPE, ACCUM_DATATYPE,
                  DIMENSION, DIMENSION, ACCUMULATION_BUFFER_SIZE>
      CCS_INIT_S1(matrixProcessor);
  Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      inputsToSystolicArray);
  Connections::Combinational<Pack1D<WEIGHT_DATATYPE, DIMENSION> > CCS_INIT_S1(
      weightsToSystolicArray);
  Connections::Combinational<Pack1D<ACCUM_DATATYPE, DIMENSION> > CCS_INIT_S1(
      outputsFromSystolicArray);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(matrixProcessorParams);

  VectorUnit<OUTPUT_DATATYPE, ACCUM_DATATYPE, INTERMEDIATE_DATATYPE, DIMENSION>
      CCS_INIT_S1(vectorUnit);
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

  Connections::Combinational<VectorParams> CCS_INIT_S1(vectorUnitParams);
  Connections::Combinational<VectorInstructionConfig> CCS_INIT_S1(
      vectorInstructionConfig);

  Connections::SyncOut startSignal;
  Connections::SyncOut doneSignal;

  SC_CTOR(Accelerator) {
    paramsDeserializer.clk(clk);
    paramsDeserializer.rstn(rstn);
    paramsDeserializer.serialParamsIn(serialParamsIn);
    paramsDeserializer.paramsOut(paramsIn);
    paramsDeserializer.vectorParamsOut(vectorUnitParams);
    paramsDeserializer.vectorInstructionsOut(vectorInstructionConfig);

    inputController.clk(clk);
    inputController.rstn(rstn);
    inputController.addressRequest(inputAddressRequest);
    inputController.dataResponse(inputDataResponse);
    inputController.paramsIn(inputControllerParams);
    inputController.windowBufferIn(inputsToWindowBuffer);
    inputController.windowBufferOut(inputsToSystolicArray);

    inputBuffer.clk(clk);
    inputBuffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      inputController.writeAddress[i](inputBufferWriteAddress[i]);
      inputController.writeData[i](inputBufferWriteData[i]);
      inputController.writeControl[i](inputBufferWriteControl[i]);
      inputController.readAddress[i](inputBufferReadAddress[i]);
      inputController.readControl[i](inputBufferReadControl[i]);

      inputBuffer.writeAddress[i](inputBufferWriteAddress[i]);
      inputBuffer.writeData[i](inputBufferWriteData[i]);
      inputBuffer.writeControl[i](inputBufferWriteControl[i]);
      inputBuffer.readAddress[i](inputBufferReadAddress[i]);
      inputBuffer.readControl[i](inputBufferReadControl[i]);
    }
    inputBuffer.output(inputsToWindowBuffer);

    weightController.clk(clk);
    weightController.rstn(rstn);
    weightController.addressRequest(weightAddressRequest);
    weightController.dataResponse(weightDataResponse);
    weightController.paramsIn(weightControllerParams);

    weightBuffer.clk(clk);
    weightBuffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weightController.writeAddress[i](weightBufferWriteAddress[i]);
      weightController.writeData[i](weightBufferWriteData[i]);
      weightController.writeControl[i](weightBufferWriteControl[i]);
      weightController.readAddress[i](weightBufferReadAddress[i]);
      weightController.readControl[i](weightBufferReadControl[i]);

      weightBuffer.writeAddress[i](weightBufferWriteAddress[i]);
      weightBuffer.writeData[i](weightBufferWriteData[i]);
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
    matrixProcessor.paramsIn(matrixProcessorParams);

    vectorUnit.clk(clk);
    vectorUnit.rstn(rstn);
    vectorUnit.paramsIn(vectorUnitParams);
    vectorUnit.vectorInstructionsIn(vectorInstructionConfig);
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
