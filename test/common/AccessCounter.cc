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

  bool mismatchedAccessCounts = false;
  for (const auto& pair : access_counts) {
    std::cout << pair.first << ": " << pair.second;

    if (check_expected) {
      if (pair.first.find("inputBuffer") != std::string::npos) {
        int expectedAccessCount =
            tiling.level_access_counts(0).input_access_count();
        std::cout << " (expected: " << expectedAccessCount << ")";

        mismatchedAccessCounts = pair.second != expectedAccessCount;
      } else if (pair.first.find("weightBuffer") != std::string::npos) {
        int expectedAccessCount =
            tiling.level_access_counts(0).weight_access_count();

        std::cout << " (expected: " << expectedAccessCount << ")";

        mismatchedAccessCounts = pair.second != expectedAccessCount;
      } else if (pair.first.find("inputs") != std::string::npos) {
        int expectedAccessCount =
            tiling.level_access_counts(1).input_access_count();
        std::cout << " (expected: " << expectedAccessCount << ")";

        // mismatch only if access count is not within 10% of expected
        mismatchedAccessCounts = pair.second < expectedAccessCount * 0.9 ||
                                 pair.second > expectedAccessCount * 1.1;

      } else if (pair.first.find("weights") != std::string::npos) {
        int expectedAccessCount =
            tiling.level_access_counts(1).weight_access_count();
        std::cout << " (expected: " << expectedAccessCount << ")";

        mismatchedAccessCounts = pair.second != expectedAccessCount;
      }
    }
    std::cout << std::endl;
  }
  if (mismatchedAccessCounts) {
    std::cout
        << "WARNING: Access counts from simulation do not match expected!!"
        << std::endl;
  }
  std::cout << "----------------" << std::endl;
}

void AccessCounter::reset() { access_counts.clear(); }
