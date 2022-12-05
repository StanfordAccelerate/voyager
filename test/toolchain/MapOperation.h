#pragma once

#include "src/AccelTypes.h"
#include "src/ArchitectureParams.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"

void MapOperation(const SimplifiedParams &params,
                  std::deque<BaseParams *> &mappedParams);
