#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/compiler/proto/param.pb.h"

void MapOperation(const codegen::Operation &param,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps);