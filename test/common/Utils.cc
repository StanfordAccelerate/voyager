#include "test/common/Utils.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "matplotlib-cpp/matplotlibcpp.h"
// namespace plt = matplotlibcpp;

#ifndef NO_UNIVERSAL
inline float readValue(UniversalPosit *matrix, int index, bool accType) {
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

inline float readValue(INPUT_DATATYPE *matrix, int index, bool accType) {
  if (!accType) {
    return static_cast<float>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].bits;
  int encoding2 = matrix[2 * index + 1].bits;
  ACCUM_DATATYPE p16;
  p16.setbits((encoding2 << 8) + encoding1);
  return static_cast<float>(p16);
}

inline float readValue(float *matrix, int index, bool accType) {
  return accType ? matrix[2 * index] : matrix[index];
}

// template <typename T>
// std::vector<float> get_float_vector(T *matrix, size_t size) {
//   std::vector<float> vector(size);
//   for (int i = 0; i < size; i++) {
//     vector[i] = (float)matrix[i];
//   }
//   return vector;
// }

// template <typename TA, typename TB>
// void plot_histograms(TA *matrixA, TB *matrixB, size_t size,
//                      std::string &filename) {
//   // convert to float
//   std::vector<float> vectorA = get_float_vector<TA>(matrixA, size);
//   std::vector<float> vectorB = get_float_vector<TB>(matrixB, size);

//   // plt::hist(vectorA, 100, "b");
//   // plt::hist(vectorB, 100, "r");
//   // plt::save(filename + ".png");
// }

template <typename TA, typename TB>
float compare_arrays_internal(TA *matrixA, std::string matrixA_name,
                              TB *matrixB, std::string matrixB_name,
                              size_t size, std::string filename, bool accType) {
  std::cout << "Writing comparison between " << matrixA_name << " and "
            << matrixB_name << " to file: " << filename << std::endl;
  std::ofstream diffFile(filename);
  diffFile << matrixA_name << " vs. " << matrixB_name << std::endl;

  // Records absolute differences
  int abs_diff_buckets[5] = {0, 0, 0, 0, 0};
  // Records relative differences
  int rel_diff_buckets[5] = {0, 0, 0, 0, 0};

  double always_zero = 0.0;

  for (int index = 0; index < size; index++) {
    // Calculate absolute difference
    float a = readValue(matrixA, index, accType);
    float b = readValue(matrixB, index, accType);
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
    } else {
      abs_diff_buckets[4]++;
    }

    // Does not fully protect against overflow, but lets not over engineer
    if (a == 0 && b == 0) {
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

float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     INPUT_DATATYPE *matrixB, std::string matrixB_name,
                     size_t size, std::string filename, bool accType) {
  return compare_arrays_internal<INPUT_DATATYPE, INPUT_DATATYPE>(
      matrixA, matrixA_name, matrixB, matrixB_name, size, filename, accType);
}

float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     float *matrixB, std::string matrixB_name, size_t size,
                     std::string filename, bool accType) {
  return compare_arrays_internal<INPUT_DATATYPE, float>(
      matrixA, matrixA_name, matrixB, matrixB_name, size, filename, accType);
}

#ifndef NO_UNIVERSAL
float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     UniversalPosit *matrixB, std::string matrixB_name,
                     size_t size, std::string filename, bool accType) {
  return compare_arrays_internal<INPUT_DATATYPE, UniversalPosit>(
      matrixA, matrixA_name, matrixB, matrixB_name, size, filename, accType);
}

float compare_arrays(UniversalPosit *matrixA, std::string matrixA_name,
                     UniversalPosit *matrixB, std::string matrixB_name,
                     size_t size, std::string filename, bool accType) {
  return compare_arrays_internal<UniversalPosit, UniversalPosit>(
      matrixA, matrixA_name, matrixB, matrixB_name, size, filename, accType);
}

float compare_arrays(UniversalPosit *matrixA, std::string matrixA_name,
                     float *matrixB, std::string matrixB_name, size_t size,
                     std::string filename, bool accType) {
  return compare_arrays_internal<UniversalPosit, float>(
      matrixA, matrixA_name, matrixB, matrixB_name, size, filename, accType);
}
#endif

float compare_arrays(float *matrixA, std::string matrixA_name, float *matrixB,
                     std::string matrixB_name, size_t size,
                     std::string filename, bool accType) {
  return compare_arrays_internal<float, float>(
      matrixA, matrixA_name, matrixB, matrixB_name, size, filename, accType);
}

int validateMapping(SimplifiedParams params) {
  int x0 = params.loops[1][params.inputXLoopIndex[1]];
  int y0 = params.loops[1][params.inputYLoopIndex[1]];
  int c0 = params.loops[1][params.reductionLoopIndex[1]];
  int k0 = params.loops[1][params.weightLoopIndex[1]];
  int fx = params.loops[1][params.fxIndex];
  int fy = params.loops[1][params.fyIndex];
  int stride = params.STRIDE;

  if (params.FC || params.FC_GRAD || params.SOFTMAX || params.SOFTMAX_GRAD ||
      params.NO_NORM || params.NO_NORM_GRAD || params.CROSS_ENTROPY_GRAD ||
      params.GRAD_CLIPPING_UNIT_TEST ||
      params.BIAS_GRAD) {  // don't check for vector ops
    return 0;
  }

  // // Input buffer
  // int input_buffer_tile_size = (x0 * stride + fx - 1) * (y0 * stride + fy -
  // 1); if (params.REPLICATION) {
  //   // don't check temporarily TODO(fpedd): Why not?
  //   input_buffer_tile_size = 1;
  // }
  // if (input_buffer_tile_size > INPUT_BUFFER_SIZE) {
  //   std::cerr << "ERROR: Input buffer tile size violation." << std::endl;
  //   std::cerr << "Constraint " << INPUT_BUFFER_SIZE << " but is " <<
  //   input_buffer_tile_size << std::endl; return -1;
  // }

  // Weight buffer
  if (fx * fy * k0 * (params.REPLICATION ? 3 : 16) > WEIGHT_BUFFER_SIZE) {
    std::cerr << "ERROR: Weight buffer tile size violation." << std::endl;
    return -1;
  }

  // Accumulation buffer
  if (x0 * y0 * k0 > ACCUMULATION_BUFFER_SIZE) {
    std::cerr << "ERROR: Accumulation buffer tile size violation." << std::endl;
    return -1;
  }

  int x_check = params.inputXLoopIndex[1] >= 4
                    ? params.loops[1][params.inputXLoopIndex[1]]
                    : 1;
  int y_check = params.inputYLoopIndex[1] >= 4
                    ? params.loops[1][params.inputYLoopIndex[1]]
                    : 1;
  if (x_check * y_check < 32) {
    std::cerr << "ERROR: Innermost X*Y must be >= 32." << std::endl;
    std::cerr << "X -> params.loops[1][" << params.inputXLoopIndex[1]
              << "] = " << params.loops[1][params.inputXLoopIndex[1]]
              << std::endl;
    std::cerr << "Y -> params.loops[1][" << params.inputYLoopIndex[1]
              << "] = " << params.loops[1][params.inputYLoopIndex[1]]
              << std::endl;
    std::cerr << "X*Y (with index >= 4) is " << x_check * y_check << std::endl;
    return -1;
  }

  if (params.reductionLoopIndex[1] != 0) {
    std::cerr
        << "ERROR: Input channel needs to be outermost loop of buffer level."
        << std::endl;
    return -1;
  }

  return 0;
}