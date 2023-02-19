#include "test/toolchain/MapOperation.h"

#include "test/toolchain/operations/Operations.h"

// create Matrix and Vector Params from SimplifiedParams
void MapOperation(const SimplifiedParams &params, const MemoryMap &memoryMap,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;

  if (params.GRAD_CLIPPING_UNIT_TEST) {
    MapGradNormUnitTest(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.SOFTMAX) {
    MapSoftmax(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.SOFTMAX_GRAD) {
    MapSoftmaxGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.FC_GRAD) {
    MapFCGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.FC) {
    MapFC(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.NO_NORM) {
    MapNoNorm(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.NO_NORM_GRAD) {
    MapNoNormGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.MSE_GRAD || params.BCE_WITH_LOGITS_GRAD) {
    MapGenericErrorGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.BIAS_GRAD) {
    MapBiasGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.CROSS_ENTROPY_GRAD) {
    MapCrossEntropyGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.WEIGHT_UPDATE) {
    MapWeightUpdate(params, memoryMap, mappedParams, opMemoryMaps);
  } else {
    MapMatrixOp(params, memoryMap, mappedParams, opMemoryMaps);
  }
}
