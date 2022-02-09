#include "test/common/Utils.h"

#include <fstream>
#include <iostream>
#include <string>

int compare_arrays(INPUT_DATATYPE *matrixA, INPUT_DATATYPE *matrixB,
                   size_t size, std::string &filename) {
  // buckets of <0.001, <0.01, <0.1, <1, >1
  int diff_buckets[5] = {0, 0, 0, 0, 0};
  int percent_diff_buckets[5] = {0, 0, 0, 0, 0};

  std::ofstream diffFile(filename);
  for (int index = 0; index < size; index++) {
    diffFile << matrixA[index] << " vs. " << matrixB[index] << std::endl;
    float diff = abs(((float)matrixA[index] - (float)matrixB[index]));

    if (diff < 0.001) {
      diff_buckets[0]++;
    }
    if (diff < 0.01) {
      diff_buckets[1]++;
    }
    if (diff < 0.1) {
      diff_buckets[2]++;
    }
    if (diff < 1) {
      diff_buckets[3]++;
    } else {
      diff_buckets[4]++;
    }
    if (matrixA[index] != 0){
    float percent_diff = abs(((float)matrixA[index] - (float)matrixB[index])) / (float)matrixA[index];
    if (percent_diff < 0.001) {
      percent_diff_buckets[0]++;
    }
    if (percent_diff < 0.01) {
      percent_diff_buckets[1]++;
    }
    if (percent_diff < 0.1) {
      percent_diff_buckets[2]++;
    }
    if (percent_diff < 1) {
      percent_diff_buckets[3]++;
    } else {
      percent_diff_buckets[4]++;

    }
    }

  }

  std::cout << "Difference Count:" << std::endl;
  std::cout << "< 0.001: " << diff_buckets[0] << "("
            << (float)diff_buckets[0] / (size)*100.0 << "%)" << std::endl;
  std::cout << "< 0.01: " << diff_buckets[1] << "("
            << (float)diff_buckets[1] / (size)*100.0 << "%)" << std::endl;
  std::cout << "< 0.1: " << diff_buckets[2] << "("
            << (float)diff_buckets[2] / (size)*100.0 << "%)" << std::endl;
  std::cout << "< 1: " << diff_buckets[3] << "("
            << (float)diff_buckets[3] / (size)*100.0 << "%)" << std::endl;
  std::cout << "> 1: " << diff_buckets[4] << "("
            << (float)diff_buckets[4] / (size)*100.0 << "%)" << std::endl;

  std::cout << "Percent Difference Count:" << std::endl;
  std::cout << "< 0.001: " << percent_diff_buckets[0] << "("
            << (float)percent_diff_buckets[0] / (size)*100.0 << "%)" << std::endl;
  std::cout << "< 0.01: " << percent_diff_buckets[1] << "("
            << (float)percent_diff_buckets[1] / (size)*100.0 << "%)" << std::endl;
  std::cout << "< 0.1: " << percent_diff_buckets[2] << "("
            << (float)percent_diff_buckets[2] / (size)*100.0 << "%)" << std::endl;
  std::cout << "< 1: " << percent_diff_buckets[3] << "("
            << (float)percent_diff_buckets[3] / (size)*100.0 << "%)" << std::endl;
  std::cout << "> 1: " << percent_diff_buckets[4] << "("
            << (float)percent_diff_buckets[4] / (size)*100.0 << "%)" << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;

  return diff_buckets[4];
}