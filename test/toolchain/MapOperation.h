#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

void MapOperation(const Operation &operation,
                  std::deque<BaseParams *> &mapped_params,
                  std::deque<AcceleratorMemoryMap> &memory_maps);
