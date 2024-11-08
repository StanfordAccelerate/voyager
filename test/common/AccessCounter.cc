#include "AccessCounter.h"

#include <iostream>

AccessCounter::AccessCounter() {}

void AccessCounter::increment(const std::string& module_name) {
  access_counts[module_name]++;
}

void AccessCounter::increment(const std::string& module_name, int count) {
  access_counts[module_name] += count;
}

void AccessCounter::print_summary() {
  std::cout << "Access counts:" << std::endl;

  for (const auto& pair : access_counts) {
    std::cout << pair.first << ": " << pair.second << std::endl;
  }
  std::cout << "----------------" << std::endl;
}
