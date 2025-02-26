#pragma once

#define NO_SYSC

#include <fstream>
#include <iostream>
#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include <any>

#include "VerificationTypes.h"
#include "src/ArchitectureParams.h"
#include "test/common/Tiling.h"

int validateMapping(Tiling tiling);

template <typename TA, typename TB>
float compare_arrays(std::any matrixA, std::string matrixA_name,
                     std::any matrixB, std::string matrixB_name, size_t size,
                     std::string filename, bool doublePrecision) {
  std::cout << "Writing comparison between " << matrixA_name << " and "
            << matrixB_name << " to file: " << filename << std::endl;
  std::ofstream diffFile(filename);
  diffFile << matrixA_name << " vs. " << matrixB_name << std::endl;

  // Records absolute differences
  int abs_diff_buckets[5] = {0, 0, 0, 0, 0};
  // Records relative differences
  int rel_diff_buckets[5] = {0, 0, 0, 0, 0};

  double always_zero = 0.0;

  TA *matrixA_ptr = std::any_cast<TA *>(matrixA);
  TB *matrixB_ptr = std::any_cast<TB *>(matrixB);

  for (int index = 0; index < size; index++) {
    // Calculate absolute difference
    float a = matrixA_ptr[index];
    float b = matrixB_ptr[index];
    always_zero += abs(a) + abs(b);
    float abs_diff = abs(a - b);

    // Write the two values + error scale indicator to file
    diffFile << a << " vs. " << b << " ";
    for (float i = 0.001; i < abs_diff; i *= 10.0) {
      diffFile << "*";
    }
    diffFile << std::endl;

    if (abs_diff < 0.001) {
      abs_diff_buckets[0]++;
    }
    if (abs_diff < 0.01) {
      abs_diff_buckets[1]++;
    }
    if (abs_diff < 0.1) {
      abs_diff_buckets[2]++;
    }
    if (abs_diff < 1) {
      abs_diff_buckets[3]++;
    } else if (!std::isinf(abs_diff) && !std::isnan(abs_diff)) {
      abs_diff_buckets[4]++;
    }

    // Does not fully protect against overflow, but lets not over engineer
    if ((a == 0 && b == 0) || std::isinf(a) || std::isinf(b)) {
      rel_diff_buckets[0]++;
      rel_diff_buckets[1]++;
      rel_diff_buckets[2]++;
      rel_diff_buckets[3]++;
      continue;
    } else {
      // See https://en.wikipedia.org/wiki/Relative_change_and_difference
      float rel_diff = abs_diff / ((abs(a) + abs(b)) / 2);
      if (rel_diff < 0.001) {
        rel_diff_buckets[0]++;
      }
      if (rel_diff < 0.01) {
        rel_diff_buckets[1]++;
      }
      if (rel_diff < 0.1) {
        rel_diff_buckets[2]++;
      }
      if (rel_diff < 1) {
        rel_diff_buckets[3]++;
      } else {
        rel_diff_buckets[4]++;
      }
    }
  }

  std::cout << "Difference Count:" << std::endl;
  std::cout << "< 0.001: " << abs_diff_buckets[0] << "("
            << (float)abs_diff_buckets[0] / size * 100.0 << "%)" << std::endl;
  std::cout << "< 0.01: " << abs_diff_buckets[1] << "("
            << (float)abs_diff_buckets[1] / size * 100.0 << "%)" << std::endl;
  std::cout << "< 0.1: " << abs_diff_buckets[2] << "("
            << (float)abs_diff_buckets[2] / size * 100.0 << "%)" << std::endl;
  std::cout << "< 1: " << abs_diff_buckets[3] << "("
            << (float)abs_diff_buckets[3] / size * 100.0 << "%)" << std::endl;
  std::cout << "> 1: " << abs_diff_buckets[4] << "("
            << (float)abs_diff_buckets[4] / size * 100.0 << "%)" << std::endl;

  std::cout << "Percent Difference Count:" << std::endl;
  std::cout << "< 0.001: " << rel_diff_buckets[0] << "("
            << (float)rel_diff_buckets[0] / size * 100.0 << "%)" << std::endl;
  std::cout << "< 0.01: " << rel_diff_buckets[1] << "("
            << (float)rel_diff_buckets[1] / size * 100.0 << "%)" << std::endl;
  std::cout << "< 0.1: " << rel_diff_buckets[2] << "("
            << (float)rel_diff_buckets[2] / size * 100.0 << "%)" << std::endl;
  std::cout << "< 1: " << rel_diff_buckets[3] << "("
            << (float)rel_diff_buckets[3] / size * 100.0 << "%)" << std::endl;
  std::cout << "> 1: " << rel_diff_buckets[4] << "("
            << (float)rel_diff_buckets[4] / size * 100.0 << "%)" << std::endl;
  std::cout << std::endl;

  if (always_zero == 0.0) {
    std::cout << "WARNING: All compared values are zero!" << std::endl;
  }

  // Ideally, these buckets should be non-overlapping...
  // TODO(fpedd): Subtract the next smaller bucket to make them non-overlapping
  float err = (1 - (float)rel_diff_buckets[1] / size) * 0.001 +
              (1 - (float)rel_diff_buckets[2] / size) * 0.01 +
              (1 - (float)rel_diff_buckets[3] / size) * 0.1 +
              (float)rel_diff_buckets[4] / size;
  return err * 100;
}
