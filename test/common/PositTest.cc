/* Running a number of tests and comparing our posit implementation to both the
 * Universal Numbers Library and IEEE (GCC/C++) float reference.
 * Loops are parallelized using OpenMP.
 */

// TODO(fpedd): We should probably add more tests cases to ensure full coverage
// of all posit functions, maybe take some inspiration from here
// https://github.com/stillwater-sc/universal/tree/main/tests/posit
// TODO(fpedd): We are not always matching the Universal Numbers Library
// TODO(fpedd): I am unsure about the portability of the != (==) on the float

#include <math.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "src/PositTypes.h"                  // Our Posit library
#include "universal/number/posit/posit.hpp"  // Universal Numbers Library reference

// For larger tests (lower part of this file) we don't want to exhaustively run
// through all possible combinations. When SUBSAMPLED_TESTS is defined we
// instead randomly sample a subset
#define SUBSAMPLED_TESTS

// Turn on extra print statements for more information
#define VERBOSE_PRINTS

using posit8e1 = sw::universal::posit<8, 1>;
using posit16e1 = sw::universal::posit<16, 1>;
using Internal = sw::universal::value<15>;

template <int nbits, int es>
int testPositEncoding(const float f) {
  Posit<nbits, es> p = f;
  sw::universal::posit<nbits, es> ref = f;
  int errors = 0;

  // Round toward 0
  // if (f < -(nbits - 1) * pow(2, es) + pow(2, es - 1)) {
  //   if (p.bits != 0) {
  //     std::cerr << "ERROR: incorrect encoding produced!" << std::endl
  //               << "Input: " << f << std::endl
  //               << "HLS Posit: " << p.bits.to_string(AC_BIN) << std::endl;
  //     return 1;
  //   }
  //   return 0;
  // }

  // Encoding test
  long encoding = ref.encoding();
  if (p.bits != encoding) {
    std::bitset<nbits> bitstring(encoding);
    std::cerr << "ERROR: incorrect encoding produced!" << std::endl
              << "Input: " << f << std::endl
              << "Universal: " << bitstring.to_string() << std::endl
              << "HLS Posit: " << p.bits.to_string(AC_BIN) << std::endl;
    errors++;
  }

  // Decoding test
  typename Posit<nbits, es>::DecomposedPosit positfp(p);
  if ((float)positfp != (float)ref) {
    std::cerr << "ERROR: incorrect decoded value produced!" << std::endl
              << "Input: " << f << std::endl
              << "Universal: " << (float)ref << std::endl
              << "HLS Posit: " << (float)positfp << std::endl;
    errors++;
  }

  return errors;
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cerr << "ERROR: Please provide the name of the tests you want to run!"
              << std::endl;
    return 1;
  }
  std::string tests(argv[1]);

  std::vector<std::string> tests_list;

  std::istringstream ss(tests);
  std::string item;
  while (std::getline(ss, item, ',')) {
    tests_list.push_back(item);
  }

  // Fixed random seed, but could also be time(NULL)
  srand(42);

  // Error counter for OpenMP loops as one can't use return -1;
  int errors = 0;

  union {
    float output;
    unsigned int input;
  } ufloat;

  if (std::find(tests_list.begin(), tests_list.end(), "encoding") !=
      tests_list.end()) {
    std::cout << "Test HLS Posit encoding and decoding... " << std::flush;

    // #pragma omp parallel for reduction(+ : errors)
    for (unsigned int i = 0; i < 0xffffffff; i++) {
      ufloat.input = i;
      if (std::isfinite(ufloat.output)) {
        errors += testPositEncoding<8, 0>(ufloat.output);
        errors += testPositEncoding<8, 1>(ufloat.output);
        errors += testPositEncoding<16, 1>(ufloat.output);
        errors += testPositEncoding<32, 2>(ufloat.output);
      }
    }

    std::cout << "done." << std::endl;
    std::cout << errors << " errors found." << std::endl;
  }

  if (std::find(tests_list.begin(), tests_list.end(), "fma") !=
      tests_list.end()) {
    std::cout << "Test 8-bit Posit fused multiply-add... " << std::flush;
    for (int k = 0; k < 0xff; k++) {
      for (int i = 0; i < 0xff; i++) {
        for (int j = 0; j < 0xff; j++) {
          // Skip NaN
          if (i == 128 || j == 128) continue;

          Posit<8, 1> pA, pB, pC;
          pA.setbits(i);
          pB.setbits(j);
          pC.setbits(k);

          Posit<8, 1>::DecomposedPosit positfpA = pA;
          Posit<8, 1>::DecomposedPosit positfpB = pB;
          PositFP<8, 15> positfpC = pC;
          positfpC = fma(positfpA, positfpB, positfpC);
          double hlsResult = (float)positfpC;

          posit8e1 universalA, universalB, universalC;
          universalA.setbits(i);
          universalB.setbits(j);
          universalC.setbits(k);
          auto internal =
              sw::universal::fma(universalA, universalB, universalC);
          double universalResult = (float)internal;

          double floatA = (float)universalA;
          double floatB = (float)universalB;
          double floatC = (float)universalC;
          double floatResult = floatA * floatB + floatC;

          double hlsDiff =
              abs((hlsResult - floatResult) / (floatResult + 1e-6));
          double universalDiff =
              abs((universalResult - floatResult) / (floatResult + 1e-6));

          if (hlsDiff > universalDiff) {
            errors++;
            fprintf(stderr, "bits: a: %d, b: %d, c: %d\n", i, j, k);
            fprintf(stderr, "values: a: %.15f, b: %.15f, c: %.15f\n",
                    (float)universalA, (float)universalB, (float)universalC);
            fprintf(stderr, "Float:     %.18f\n", floatResult);
            fprintf(stderr, "HLS Posit: %.18f\n", hlsResult);
            fprintf(stderr, "Universal: %.18f\n", universalResult);
            fprintf(stderr, "HLS diff: %.6f, Universal diff: %.6f\n",
                    hlsDiff * 100, universalDiff * 100);
          }
        }
      }
    }
    std::cout << "done." << std::endl;
    std::cout << errors << " errors found." << std::endl;
  }

  if (std::find(tests_list.begin(), tests_list.end(), "posit_add") !=
      tests_list.end()) {
    std::cout << "Test 16-bit posit addition/subtraction... " << std::flush;
    for (int i = 0; i < 65536; i++) {
      for (int j = 0; j < 65536; j++) {
        // Skip NaN
        if (i == 32768 || j == 32768) continue;

        Posit<16, 1> pA, pB, pC;
        pA.setbits(i);
        pB.setbits(j);
        pC = pA + pB;
        double hlsResult = (float)pC;

        posit16e1 universalA, universalB, universalC;
        universalA.setbits(i);
        universalB.setbits(j);
        universalC = universalA + universalB;
        double universalResult = (float)universalC;

        double a = (float)universalA;
        double b = (float)universalB;
        double gold = a + b;

        float hlsDiff = abs((hlsResult - gold) / gold);
        float universalDiff = abs((universalResult - gold) / gold);

        if (hlsDiff > universalDiff) {
          errors++;
          fprintf(stderr, "bits: a: %d, b: %d\n", i, j);
          fprintf(stderr, "values: a: %.15f, b: %.15f\n", (float)universalA,
                  (float)universalB);
          fprintf(stderr, "Float:     %.18f\n", gold);
          fprintf(stderr, "HLS Posit: %.18f\n", hlsResult);
          fprintf(stderr, "Universal: %.18f\n", universalResult);
          fprintf(stderr, "HLS diff: %.6f, Universal diff: %.6f\n",
                  hlsDiff * 100, universalDiff * 100);
        }
      }
    }
    std::cout << "done." << std::endl;
    std::cout << errors << " errors found." << std::endl;
  }

  if (std::find(tests_list.begin(), tests_list.end(), "posit_multiply") !=
      tests_list.end()) {
    std::cout << "Test 16-bit posit multiplication... " << std::flush;
    for (int i = 0; i < 65536; i++) {
      for (int j = 0; j < 65536; j++) {
        // Skip NaN
        if (i == 32768 || j == 32768) continue;

        Posit<16, 1> pA, pB, pC;
        pA.setbits(i);
        pB.setbits(j);
        pC = pA + pB;
        double hlsResult = (float)pC;

        posit16e1 universalA, universalB, universalC;
        universalA.setbits(i);
        universalB.setbits(j);
        universalC = universalA * universalB;
        double universalResult = (float)universalC;

        double a = (float)universalA;
        double b = (float)universalB;
        double gold = a * b;

        float hlsDiff = abs((hlsResult - gold) / gold);
        float universalDiff = abs((universalResult - gold) / gold);

        if (hlsDiff > universalDiff) {
          errors++;
          fprintf(stderr, "bits: a: %d, b: %d\n", i, j);
          fprintf(stderr, "values: a: %.15f, b: %.15f\n", (float)universalA,
                  (float)universalB);
          fprintf(stderr, "Float:     %.18f\n", gold);
          fprintf(stderr, "HLS Posit: %.18f\n", hlsResult);
          fprintf(stderr, "Universal: %.18f\n", universalResult);
          fprintf(stderr, "HLS diff: %.6f, Universal diff: %.6f\n",
                  hlsDiff * 100, universalDiff * 100);
        }
      }
    }
    std::cout << "done." << std::endl;
    std::cout << errors << " errors found." << std::endl;
  }

  if (std::find(tests_list.begin(), tests_list.end(), "posit_exp") !=
      tests_list.end()) {
#if defined(SUBSAMPLED_TESTS)
    // Using 65536 * 64 instead of 65536 * 65536 in supsampled case
    long long size = 65536 * 64ll;
#else
    // Full search-space tests
    long long size = 65536ll * 65536ll;
#endif
    double diff_buckets[5] = {0};

    std::cout << "Test 16bit posit exp... " << std::flush;
    memset(diff_buckets, 0, sizeof(diff_buckets));

    // #pragma omp parallel for reduction(+ : errors, diff_buckets)
    for (int i = 0; i < 65536; i++) {
      if (errors) continue;

      // Skip NaN
      if (i == 32768) continue;

      Posit<16, 1> p16A;
      p16A.setbits(i);

      // Since the posit exp is implemented in an optimized fashion using
      // approximations it is only valid for negative values. So we skip
      // positive test values
      if (p16A >= 0) continue;

      Posit<16, 1> p16B = posit_exp(p16A);

      posit16e1 uni16A, uni16B;
      uni16A.setbits(i);
      uni16B = sw::universal::exp(uni16B);

      float a = (float)uni16A;
      float gold = exp(a);

      if ((float)p16A != a) {
        errors++;
        fprintf(stderr, "ERROR: i: %d, a: %f,  p16A: %f\n", i, a, (float)p16A);
      }

      float hlsDiff = abs(((float)p16B - gold) / gold);
      float universalDiff = abs(((float)uni16B - gold) / gold);

      if (hlsDiff < 0.001) {
        diff_buckets[0]++;
      }
      if (hlsDiff < 0.01) {
        diff_buckets[1]++;
      }
      if (hlsDiff < 0.1) {
        diff_buckets[2]++;
      }
      if (hlsDiff < 1.0) {
        diff_buckets[3]++;
      } else {
        diff_buckets[4]++;
      }

      // For most cases the bit representation should be same. If not, at
      // least our error should be less than universals.
      if (p16B.bits != uni16B.encoding() && hlsDiff > universalDiff) {
        errors++;
        fprintf(stderr, "ERROR: i: %d\n", i);
        fprintf(stderr, "a: %f\n", a);
        fprintf(stderr, "float: %f,  hls: %lf, universal: %lf\n", gold,
                (float)p16B, (float)uni16B);
        fprintf(stderr, "hlsDiff: %f, universalDiff: %f\n", hlsDiff,
                universalDiff);
        std::cerr << p16B.bits.to_string(AC_BIN) << std::endl;
        std::cerr << uni16B.get() << std::endl;
      }
    }
    std::cout << "done." << std::endl;

#if defined(VERBOSE_PRINTS)
    std::cout << "Percentage Difference Count:" << std::endl;
    std::cout << " < 0.001: " << diff_buckets[0] << "("
              << diff_buckets[0] / (double)size * 100.0 << "%)" << std::endl;
    std::cout << " < 0.01:  " << diff_buckets[1] << "("
              << diff_buckets[1] / (double)size * 100.0 << "%)" << std::endl;
    std::cout << " < 0.1:   " << diff_buckets[2] << "("
              << diff_buckets[2] / (double)size * 100.0 << "%)" << std::endl;
    std::cout << " < 1.0:   " << diff_buckets[3] << "("
              << diff_buckets[3] / (double)size * 100.0 << "%)" << std::endl;
    std::cout << ">= 1.0:   " << diff_buckets[4] << "("
              << diff_buckets[4] / (double)size * 100.0 << "%)" << std::endl;
#endif
  }

  return errors;
}
