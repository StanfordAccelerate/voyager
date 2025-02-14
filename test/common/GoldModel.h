#pragma once
#define NO_SYSC

#include <any>

// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/compiler/proto/param.pb.h"

std::any run_gold_model(const codegen::Operation &param,
                        std::map<std::string, std::any> kwargs);
