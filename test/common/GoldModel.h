#pragma once
#define NO_SYSC

#include <any>

#include "test/common/Network.h"

std::vector<std::any> run_gold_model(const Operation& operation,
                                     std::map<std::string, std::any> kwargs);
