#include "Accelerator.h"

void Accelerator::run() {
  startSignal.Reset();
  paramsIn.ResetRead();
  inputControllerParams.ResetWrite();
  weightControllerParams.ResetWrite();
  matrixProcessorParams.ResetWrite();

  wait();

  while (true) {
    MatrixParams params = paramsIn.Pop();
    startSignal.SyncPush();
    inputControllerParams.Push(params);
    weightControllerParams.Push(params);
    matrixProcessorParams.Push(params);
  }
}
