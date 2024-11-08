#pragma once

#include <map>
#include <string>

class AccessCounter {
 public:
  AccessCounter();

  void increment(const std::string& module_name);
  void increment(const std::string& module_name, int count);

  void print_summary();

 private:
  // Map of module name to access count
  std::map<std::string, int> access_counts;
};
