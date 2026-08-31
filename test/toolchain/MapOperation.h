#pragma once

#include "src/Params.h"
#include "test/common/GraphUtils.h"
#include "test/common/Utils.h"

void map_operation(const voyager::Operation& operation, const ScalarEnv& env,
                   std::deque<BaseParams*>& mapped_params);
