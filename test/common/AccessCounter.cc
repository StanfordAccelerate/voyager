#include "AccessCounter.h"

#include <iostream>

AccessCounter::AccessCounter() {}

void AccessCounter::increment(const std::string& module_name) {
  access_counts[module_name]++;
}

void AccessCounter::increment(const std::string& module_name, int count) {
  access_counts[module_name] += count;
}

void AccessCounter::print_summary(const voyager::Tiling& tiling,
                                  bool check_expected) {
  std::cout << "Access counts:" << std::endl;

  bool mismatched_access_counts = false;
  for (const auto& pair : access_counts) {
    std::cout << pair.first << ": " << pair.second;

    if (check_expected) {
      if (pair.first.find("input_buffer") != std::string::npos) {
        int expected_access_count =
            tiling.level_access_counts(0).input_access_count();
        std::cout << " (expected: " << expected_access_count << ")";

        mismatched_access_counts = pair.second != expected_access_count;
      } else if (pair.first.find("weight_buffer") != std::string::npos) {
        int expected_access_count =
            tiling.level_access_counts(0).weight_access_count();

        std::cout << " (expected: " << expected_access_count << ")";

        mismatched_access_counts = pair.second != expected_access_count;
      } else if (pair.first.find("inputs") != std::string::npos) {
        int expected_access_count =
            tiling.level_access_counts(1).input_access_count();
        std::cout << " (expected: " << expected_access_count << ")";

        // mismatch only if access count is not within 10% of expected
        mismatched_access_counts = pair.second < expected_access_count * 0.9 ||
                                   pair.second > expected_access_count * 1.1;

      } else if (pair.first.find("weights") != std::string::npos) {
        int expected_access_count =
            tiling.level_access_counts(1).weight_access_count();
        std::cout << " (expected: " << expected_access_count << ")";

        mismatched_access_counts = pair.second != expected_access_count;
      }
    }
    std::cout << std::endl;
  }
  if (mismatched_access_counts) {
    std::cout
        << "WARNING: Access counts from simulation do not match expected!!"
        << std::endl;
  }
  std::cout << "----------------" << std::endl;
}

void AccessCounter::reset() { access_counts.clear(); }
