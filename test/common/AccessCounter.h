#pragma once

#include <map>
#include <string>

#include "test/compiler/proto/tiling.pb.h"

class AccessCounter {
 public:
  AccessCounter();

  void increment(const std::string& module_name);
  void increment(const std::string& module_name, int count);

  void print_summary(const voyager::Tiling& tiling, bool check_expected);
  void reset();

 private:
  // Map of module name to access count
  std::map<std::string, int> access_counts;
};
