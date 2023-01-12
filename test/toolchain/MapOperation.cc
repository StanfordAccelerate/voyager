#include "test/toolchain/MapOperation.h"

#include "test/toolchain/operations/Operations.h"

// create Matrix and Vector Params from SimplifiedParams
void MapOperation(const SimplifiedParams &params,
                  std::deque<BaseParams *> &mappedParams) {
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
    MapGradNormClipping(params, mappedParams, X * C);
  } else if (params.SOFTMAX) {
    MapSoftmax(params, mappedParams);
  } else if (params.SOFTMAX_GRAD) {
    MapSoftmaxGrad(params, mappedParams);
  } else if (params.FC_GRAD) {
    if (params.GRAD_CLIPPING) {
      MapFCGradWithNormClipping(params, mappedParams);
    } else {
      MapFCGrad(params, mappedParams);
    }
  } else if (params.FC) {
    MapFC(params, mappedParams);
  } else if (params.NO_NORM) {
    MapNoNorm(params, mappedParams);
  } else if (params.NO_NORM_GRAD) {
    MapNoNormGrad(params, mappedParams);
  } else if (params.MSE_GRAD || params.BCE_WITH_LOGITS_GRAD) {
    MapGenericErrorGrad(params, mappedParams);
  } else if (params.BIAS_GRAD) {
    MapBiasGrad(params, mappedParams);
  } else if (params.CROSS_ENTROPY_GRAD) {
    MapCrossEntropyGrad(params, mappedParams);
  } else if (params.WEIGHT_UPDATE) {
    MapWeightUpdate(params, mappedParams);
  } else {
    MapMatrixOp(params, mappedParams);
  }
}
