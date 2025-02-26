#pragma once
#define NO_SYSC

#include <any>

// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/Network.h"
#include "test/compiler/proto/param.pb.h"

void run_gold_model(const Operation &operation, std::vector<std::any> args);
