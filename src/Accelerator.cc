#include "Accelerator.h"

void Accelerator::run() {
  paramsIn.ResetRead();
  inputControllerParams.ResetWrite();
  weightControllerParams.ResetWrite();
  matrixProcessorParams.ResetWrite();
  vectorUnitParams.ResetWrite();

  wait();

  while (true) {
    Params params = paramsIn.Pop();
    inputControllerParams.Push(params);
    weightControllerParams.Push(params);
    matrixProcessorParams.Push(params);
    vectorUnitParams.Push(params);
  }
}
