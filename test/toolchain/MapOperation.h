#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"

void map_operation(const Operation& operation,
                   std::deque<BaseParams*>& mapped_params);
