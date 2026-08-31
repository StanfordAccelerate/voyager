#pragma once

#include <string>

#include "test/common/Simulation.h"

class JtagSimChecker : public Simulation {
 public:
  explicit JtagSimChecker(const std::string& scratchpad_bin_path);
};
