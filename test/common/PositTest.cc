/* Running a number of tests and comparing our posit implementation to both the
 * Universal Numbers Library and IEEE (GCC/C++) float reference.
 * Loops are parallelized using OpenMP.
 */

// TODO(fpedd): We should probably add more tests cases to ensure full coverage
// of all posit functions, maybe take some inspiration from here
// https://github.com/stillwater-sc/universal/tree/main/tests/posit
// TODO(fpedd): We are not always matching the Universal Numbers Library
// TODO(fpedd): I am unsure about the portability of the != (==) on the float

#include <cmath>

#include "src/PositTypes.h"                  // Our Posit library
#include "universal/number/posit/posit.hpp"  // Universal Numbers Library reference

// For larger tests (lower part of this file) we don't want to exhaustively run
// through all possible combinations. When SUBSAMPLED_TESTS is defined we
// instead randomly sample a subset
#define SUBSAMPLED_TESTS

// Turn on extra print statements for more information
#define VERBOSE_PRINTS

using Universal8 = sw::universal::posit<8, 1>;
using Universal16 = sw::universal::posit<16, 1>;
using Internal = sw::universal::value<15>;

template <int nbits, int es, int sbits, int fbits>
int testEncodeDecode(float f) {
  Posit<nbits, es> p = f;
  sw::universal::posit<nbits, es> ref = f;
  long encoding = ref.encoding();
  if (p.bits != encoding) {
    std::bitset<nbits> bitstring(encoding);
    std::cerr << "ERROR: incorrect encoding produced! \n"
              << "input: " << f << "  gold: " << bitstring.to_string()
              << "\thls: " << p.bits.to_string(AC_BIN) << std::endl;
    return 1;
  }

  PositFP<sbits, fbits> fp(p);
  if ((float)fp != (float)ref) {
    std::cerr << "ERROR: incorrect decoded value produced! \n"
              << "gold: " << (float)ref << " hls: " << (float)fp << std::endl;
    return 1;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  // Fixed random seed, but could also be time(NULL)
  srand(42);

  // Error counter for OpenMP loops as one can't use return -1;
  int errors = 0;

  union {
    float output;
    unsigned int input;
  } data;

  std::cout << "Test posit encoding and decoding... " << std::flush;

#pragma omp parallel for reduction(+ : errors)
  for (unsigned int i = 0; i < 0xffffffff; i++) {
    data.input = i;
    // errors += testEncodeDecode<8, 0, 8, 5>(data.output);
    errors += testEncodeDecode<8, 1, 8, 4>(data.output);
    errors += testEncodeDecode<16, 1, 8, 12>(data.output);
    // errors += testEncodeDecode<32, 2, 8, 26>(data.output);
    if (i % (unsigned int)1e7 == 0) {
      std::cerr << ".";
    }
  }
  std::cout << "done." << std::endl;
  std::cerr << errors << " errors found." << std::endl;

  if (errors) {
    return 1;
  }

  std::cout << "Test posit range... " << std::flush;
#pragma omp parallel for reduction(+ : errors)
  for (int i = -1e6; i < (int)1e6; i++) {
    if (errors) continue;
    errors += testEncodeDecode<8, 0, 8, 5>((float)i);
    errors += testEncodeDecode<8, 1, 8, 4>((float)i);
    errors += testEncodeDecode<16, 1, 8, 12>((float)i);
    errors += testEncodeDecode<32, 2, 8, 26>((float)i);
  }
  std::cout << "done." << std::endl;

  Posit<8, 1> pA, pB;
  Posit<16, 1> pC;
  Universal8 uniA, uniB, uniC;
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      if (errors) continue;

      // Skip NaN
      if (i == 128 || j == 128) continue;
      pA.setbits(1);
      pB.setbits(1);
      pC.setbits(0);

      typename Posit<8, 1>::DecomposedPosit decomposed_pA =
          typename Posit<8, 1>::DecomposedPosit(pA);
      typename Posit<8, 1>::DecomposedPosit decomposed_pB =
          typename Posit<8, 1>::DecomposedPosit(pB);
      typename Posit<16, 1>::DecomposedPosit decomposed_pC =
          typename Posit<16, 1>::DecomposedPosit(pC);

      pC = fma(pA, pB, pC);
      decomposed_pC = decomposed_fma<8, 1, 16, 1>(decomposed_pA, decomposed_pB,
                                                  decomposed_pC);

      typename Posit<16, 1>::DecomposedPosit ref_pC =
          typename Posit<16, 1>::DecomposedPosit(pC);

      if ((float)ref_pC != (float)decomposed_pC) {
        fprintf(stderr, "ERROR: received: %f != %f\n", (float)decomposed_pC,
                (float)ref_pC);
        return -1;
      }
    }
  }

  std::cout << "Test 8bit posit multiplication... " << std::flush;
#pragma omp parallel for reduction(+ : errors)
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      if (errors) continue;

      // Skip NaN
      if (i == 128 || j == 128) continue;

      Posit<8, 1> pA, pB;
      Posit<16, 1> pC;
      pA.setbits(i);
      pB.setbits(j);
      pC.setbits(0);
      pC = fma(pA, pB, pC);

      Universal8 uniA, uniB, uniC;
      uniA.setbits(i);
      uniB.setbits(j);
      uniC = uniA * uniB;

      float a = (float)uniA;
      float b = (float)uniB;
      float gold = a * b;

      if ((float)pA != a) {
        errors++;
        fprintf(stderr, "i: %d, a: %f,  pA: %f\n", i, a, (float)pA);
      }

      if ((float)pB != b) {
        errors++;
        fprintf(stderr, "j: %d, b: %f,  pB: %f\n", j, b, (float)pB);
      }

      // TODO(fpedd): Results not equal to universal ??
      // float: 0.000000, posit: 0.000000, universal: 0.000244
      if ((float)pC != gold /*|| (float)pC != (float)uniC*/) {
        errors++;
        fprintf(stderr, "ERROR: i: %d, j: %d\n", i, j);
        fprintf(stderr, "a: %f, b: %f\n", a, b);
        fprintf(stderr, "float: %f, posit: %f, universal: %f\n", gold,
                (float)pC, (float)uniC);
      }
    }
  }
  std::cout << "done." << std::endl;

#if defined(SUBSAMPLED_TESTS)
  // Using 65536 * 64 instead of 65536 * 65536 in supsampled case
  long long size = 65536 * 64ll;
#else
  // Full search-space tests
  long long size = 65536ll * 65536ll;
#endif
  double diff_buckets[5] = {0};

  std::cout << "Test 16bit posit addition... " << std::flush;
#if defined(SUBSAMPLED_TESTS)
#pragma omp parallel for reduction(+ : errors, diff_buckets)
  for (int n = 0; n < size; n++) {
    {
      int i = rand() % 65536;
      int j = rand() % 65536;
#else
#pragma omp parallel for reduction(+ : errors, diff_buckets)
  for (int i = 0; i < 65536; i++) {
    for (int j = 0; j < 65536; j++) {
#endif
      if (errors) continue;

      // Skip NaN
      if (i == 32768 || j == 32768) continue;

      Posit<16, 1> p16A, p16B, p16C;
      Universal16 uni16A, uni16B, uni16C;

      p16A.setbits(i);
      p16B.setbits(j);
      p16C = p16A + p16B;

      uni16A.setbits(i);
      uni16B.setbits(j);
      uni16C = uni16A + uni16B;

      float a = (float)uni16A;
      float b = (float)uni16B;
      float gold = a + b;

      if ((float)p16A != a) {
        errors++;
        fprintf(stderr, "ERROR: i: %d, a: %f,  p16A: %f\n", i, a, (float)p16A);
      }
      if ((float)p16B != b) {
        errors++;
        fprintf(stderr, "ERROR: j: %d, b: %f,  p16B: %f\n", j, b, (float)p16B);
      }

      float hlsDiff = abs(((float)p16C - gold) / gold);
      float universalDiff = abs(((float)uni16C - gold) / gold);

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
      if (p16C.bits != uni16C.encoding() && hlsDiff > universalDiff) {
        errors++;
        fprintf(stderr, "ERROR: i: %d, j: %d\n", i, j);
        fprintf(stderr, "a: %f, b: %f\n", a, b);
        fprintf(stderr, "float: %f,  hls: %lf, universal: %lf\n", gold,
                (float)p16C, (float)uni16C);
        fprintf(stderr, "hlsDiff: %f, universalDiff: %f\n", hlsDiff,
                universalDiff);
        std::cerr << p16C.bits.to_string(AC_BIN) << std::endl;
        std::cerr << uni16C.get() << std::endl;
      }
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

  std::cout << "Test 16bit posit substraction... " << std::flush;
  memset(diff_buckets, 0, sizeof(diff_buckets));

#if defined(SUBSAMPLED_TESTS)
#pragma omp parallel for reduction(+ : errors, diff_buckets)
  for (int n = 0; n < size; n++) {
    {
      int i = rand() % 65536;
      int j = rand() % 65536;
#else
#pragma omp parallel for reduction(+ : errors, diff_buckets)
  for (int i = 0; i < 65536; i++) {
    for (int j = 0; j < 65536; j++) {
#endif
      if (errors) continue;

      // Skip NaN
      if (i == 32768 || j == 32768) continue;

      Posit<16, 1> p16A, p16B, p16C;
      Universal16 uni16A, uni16B, uni16C;

      p16A.setbits(i);
      p16B.setbits(j);
      p16C = p16A - p16B;

      uni16A.setbits(i);
      uni16B.setbits(j);
      uni16C = uni16A - uni16B;

      float a = (float)uni16A;
      float b = (float)uni16B;
      float gold = a - b;

      if ((float)p16A != a) {
        errors++;
        fprintf(stderr, "ERROR: i: %d, a: %f,  p16A: %f\n", i, a, (float)p16A);
      }
      if ((float)p16B != b) {
        errors++;
        fprintf(stderr, "ERROR: j: %d, b: %f,  p16B: %f\n", j, b, (float)p16B);
      }

      float hlsDiff = abs(((float)p16C - gold) / gold);
      float universalDiff = abs(((float)uni16C - gold) / gold);

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
      if (p16C.bits != uni16C.encoding() && hlsDiff > universalDiff) {
        errors++;
        fprintf(stderr, "ERROR: i: %d, j: %d\n", i, j);
        fprintf(stderr, "a: %f, b: %f\n", a, b);
        fprintf(stderr, "float: %f,  hls: %lf, universal: %lf\n", gold,
                (float)p16C, (float)uni16C);
        fprintf(stderr, "hlsDiff: %f, universalDiff: %f\n", hlsDiff,
                universalDiff);
        std::cerr << p16C.bits.to_string(AC_BIN) << std::endl;
        std::cerr << uni16C.get() << std::endl;
      }
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
    // approximations it is only valid for negative values. So we skip positive
    // test values
    if (p16A >= 0) continue;

    Posit<16, 1> p16B = posit_exp(p16A);

    Universal16 uni16A, uni16B;
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

  return errors;
}
