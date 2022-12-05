#pragma once

#include "src/AccelTypes.h"
#include "src/ArchitectureParams.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"

void MapSoftmax(const SimplifiedParams &params,
                std::deque<BaseParams *> &mappedParams);

void MapSoftmaxGrad(const SimplifiedParams &params,
                    std::deque<BaseParams *> &mappedParams);

void MapFCGrad(const SimplifiedParams &params,
               std::deque<BaseParams *> &mappedParams);

void MapFC(const SimplifiedParams &params,
           std::deque<BaseParams *> &mappedParams);

void MapNoNorm(const SimplifiedParams &params,
               std::deque<BaseParams *> &mappedParams);

void MapNoNormGrad(const SimplifiedParams &params,
                   std::deque<BaseParams *> &mappedParams);

void MapGenericErrorGrad(const SimplifiedParams &params,
                         std::deque<BaseParams *> &mappedParams);

void MapMatrixOp(const SimplifiedParams &params,
                 std::deque<BaseParams *> &mappedParams);
