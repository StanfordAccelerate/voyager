#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/Network.h"

void MapOperation(const Operation &operation,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps);