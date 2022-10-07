#include "test/common/Utils.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "matplotlib-cpp/matplotlibcpp.h"
// namespace plt = matplotlibcpp;

#ifndef NO_UNIVERSAL
inline float readInput(UniversalPosit *matrix, int index, bool accType) {
  if (!accType) {
    return static_cast<float>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].encoding();
  int encoding2 = matrix[2 * index + 1].encoding();
  UniversalPositAccum p16;
  p16.setbits((encoding2 << 8) + encoding1);
  return static_cast<float>(p16);
}
#endif

inline float readInput(INPUT_DATATYPE *matrix, int index, bool accType) {
  if (!accType) {
    return static_cast<float>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].bits;
  int encoding2 = matrix[2 * index + 1].bits;
  ACCUM_DATATYPE p16;
  p16.setbits((encoding2 << 8) + encoding1);
  return static_cast<float>(p16);
}

inline float readInput(float *matrix, int index, bool accType) {
  return accType ? matrix[2 * index] : matrix[index];
}

template <typename T>
std::vector<float> get_float_vector(T *matrix, size_t size) {
  std::vector<float> vector(size);
  for (int i = 0; i < size; i++) {
    vector[i] = (float)matrix[i];
  }
  return vector;
}

template <typename TA, typename TB>
void plot_histograms(TA *matrixA, TB *matrixB, size_t size,
                     std::string &filename) {
  // convert to float
  std::vector<float> vectorA = get_float_vector<TA>(matrixA, size);
  std::vector<float> vectorB = get_float_vector<TB>(matrixB, size);

  // plt::hist(vectorA, 100, "b");
  // plt::hist(vectorB, 100, "r");
  // plt::save(filename + ".png");
}

template <typename TA, typename TB>
int compare_arrays_internal(TA *matrixA, TB *matrixB, size_t size,
                            std::string filename, bool accType) {
  // plot_histograms<TA, TB>(matrixA, matrixB, size, filename);
  // buckets of <0.001, <0.01, <0.1, <1, >1
  int diff_buckets[5] = {0, 0, 0, 0, 0};
  int percent_diff_buckets[5] = {0, 0, 0, 0, 0};

  // std::cout << "Dir: " << std::filesystem::current_path().str() << std::endl;
  std::cout << "Writing comparison to file: " << filename << std::endl;
  std::ofstream diffFile(filename);
  for (int index = 0; index < size; index++) {
    float a = readInput(matrixA, index, accType);
    float b = readInput(matrixB, index, accType);
    diffFile << a << " vs. " << b << " ";
    float diff = abs(a - b);

    for (float i = 0.001; i < diff; i *= 10.0) {
      diffFile << "*";
    }
    diffFile << std::endl;

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

    if (a != 0) {
      float percent_diff = abs(diff / a);
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
            << (float)percent_diff_buckets[0] / (size)*100.0 << "%)"
            << std::endl;
  std::cout << "< 0.01: " << percent_diff_buckets[1] << "("
            << (float)percent_diff_buckets[1] / (size)*100.0 << "%)"
            << std::endl;
  std::cout << "< 0.1: " << percent_diff_buckets[2] << "("
            << (float)percent_diff_buckets[2] / (size)*100.0 << "%)"
            << std::endl;
  std::cout << "< 1: " << percent_diff_buckets[3] << "("
            << (float)percent_diff_buckets[3] / (size)*100.0 << "%)"
            << std::endl;
  std::cout << "> 1: " << percent_diff_buckets[4] << "("
            << (float)percent_diff_buckets[4] / (size)*100.0 << "%)"
            << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;

  return percent_diff_buckets[4];
}

int compare_arrays(INPUT_DATATYPE *matrixA, INPUT_DATATYPE *matrixB,
                   size_t size, std::string filename, bool accType) {
  return compare_arrays_internal<INPUT_DATATYPE, INPUT_DATATYPE>(
      matrixA, matrixB, size, filename, accType);
}

int compare_arrays(INPUT_DATATYPE *matrixA, float *matrixB, size_t size,
                   std::string filename, bool accType) {
  return compare_arrays_internal<INPUT_DATATYPE, float>(matrixA, matrixB, size,
                                                        filename, accType);
}

#ifndef NO_UNIVERSAL
int compare_arrays(INPUT_DATATYPE *matrixA, UniversalPosit *matrixB,
                   size_t size, std::string filename, bool accType) {
  return compare_arrays_internal<INPUT_DATATYPE, UniversalPosit>(
      matrixA, matrixB, size, filename, accType);
}

int compare_arrays(UniversalPosit *matrixA, UniversalPosit *matrixB,
                   size_t size, std::string filename, bool accType) {
  return compare_arrays_internal<UniversalPosit, UniversalPosit>(
      matrixA, matrixB, size, filename, accType);
}

int compare_arrays(UniversalPosit *matrixA, float *matrixB, size_t size,
                   std::string filename, bool accType) {
  return compare_arrays_internal<UniversalPosit, float>(matrixA, matrixB, size,
                                                        filename, accType);
}
#endif

int compare_arrays(float *matrixA, float *matrixB, size_t size,
                   std::string filename, bool accType) {
  return compare_arrays_internal<float, float>(matrixA, matrixB, size, filename,
                                               accType);
}

int validateMapping(SimplifiedParams params) {
  int x0 = params.loops[1][params.inputXLoopIndex[1]];
  int y0 = params.loops[1][params.inputYLoopIndex[1]];
  int c0 = params.loops[1][params.reductionLoopIndex[1]];
  int k0 = params.loops[1][params.weightLoopIndex[1]];
  int fx = params.loops[1][params.fxIndex];
  int fy = params.loops[1][params.fyIndex];
  int stride = params.STRIDE;

  if (params.FC || params.SOFTMAX || params.SOFTMAX_GRAD ||
      params.NO_NORM) {  // don't check for vector ops
    return 0;
  }

  // Input buffer
  int input_buffer_tile_size = (x0 * stride + fx - 1) * (y0 * stride + fy - 1);
  if (params.REPLICATION) {
    // don't check temporarily TODO(fpedd): Why not?
    input_buffer_tile_size = 1;
  }
  if (input_buffer_tile_size > INPUT_BUFFER_SIZE) {
    std::cerr << "ERROR: Input buffer tile size violation." << std::endl;
    return -1;
  }

  // Weight buffer
  if (fx * fy * k0 > WEIGHT_BUFFER_SIZE) {
    std::cerr << "ERROR: Weight buffer tile size violation." << std::endl;
    return -1;
  }

  // Accumulation buffer
  if (x0 * y0 * k0 > ACCUMULATION_BUFFER_SIZE) {
    std::cerr << "ERROR: Accumulation buffer tile size violation." << std::endl;
    return -1;
  }

  return 0;
}