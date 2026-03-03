#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"

void map_operation(const Operation& operation,
                   std::deque<BaseParams*>& mapped_params);
std::deque<BaseParams*> get_ping_pong_params(std::deque<BaseParams*> params,
                                             int offset);
