#include "Accelerator.h"

void Accelerator::run() {
  startSignal.Reset();
  paramsIn.ResetRead();
  inputControllerParams.ResetWrite();
  weightControllerParams.ResetWrite();
  matrixProcessorParams.ResetWrite();
  vectorUnitParams.ResetWrite();

  wait();

  while (true) {
    Params params = paramsIn.Pop();
    startSignal.SyncPush();
    inputControllerParams.Push(params);
    weightControllerParams.Push(params);
    matrixProcessorParams.Push(params);
    vectorUnitParams.Push(params);
  }
}
