#include "src/PositTypes.h"
#include "universal/number/posit/posit.hpp"
#include "cmath"

using Universal8 = sw::universal::posit<8, 1>;
using Universal16 = sw::universal::posit<16, 1>;;
using Internal = sw::universal::value<15>;

template<int nbits, int es, int sbits, int fbits>
bool test(float f) {
  Posit<nbits, es> p = f;
  sw::universal::posit<nbits, es> ref = f;
  long encoding = ref.encoding();
  if (p.bits != encoding) {
    std::bitset<nbits> bitstring(encoding);
    std::cerr << "ERROR: incorrect encoding produced!" << std::endl
              << "input: " << f << "  gold: " << bitstring.to_string()
              << "\thls: " << p.bits.to_string(AC_BIN) << std::endl;
    return false;
  }

  PositFP<sbits, fbits> fp(p);
  if ((float) fp != (float) ref) {
      std::cerr << "ERROR: incorrect decoded value produced!" << std::endl
                << "gold: " << (float) ref << "  hls: " << (float) fp << std::endl;
      return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  // std::cerr << "Testing posit encoding and decoding." << std::endl;

  // // precision test
  // for (float f = 0; f < 10; f += 1e-6) {
  //   if (!test<8, 0, 8, 5>(f)) return -1;
  //   if (!test<8, 1, 8, 4>(f)) return -1;
  //   if (!test<16, 1, 8, 12>(f)) return -1;
  //   if (!test<32, 2, 8, 26>(f)) return -1;
  // }

  // // range test
  // for (float f = 1e-6; f < 1e6; f += 1) {
  //   if (!test<8, 0, 8, 5>(f)) return -1;
  //   if (!test<8, 1, 8, 4>(f)) return -1;
  //   if (!test<16, 1, 8, 12>(f)) return -1;
  //   if (!test<32, 2, 8, 27>(f)) return -1;
  // }

  // Posit<8, 1> pA, pB;
  // Posit<16, 1> mul;
  // Universal8 uniA, uniB, uniSum;
  // for (int i = 0; i < 256; i++) {
  //   for (int j = 0; j < 256; j++) {
  //     if (i == 128 || j == 128) continue;

  //     pA.setbits(i);
  //     pB.setbits(j);
  //     // mul = fma(pA, pB, 0);
  //     mul = pA * pB;

  //     uniA.setbits(i);
  //     uniB.setbits(j);
  //     uniSum = uniA * uniB;
  //     float a = (float) uniA;
  //     float b = (float) uniB;

  //     if ((float) pA != a) {
  //       printf("i: %d, a: %f,  pA: %f\n", i, a, (float) pA);
  //       return -1;
  //     }
  //     if ((float) pB != b) {
  //       printf("j: %d, b: %f,  pB: %f\n", j, b, (float) pB);
  //       return -1;
  //     }
      
  //     float gold = a * b;
  //     if ((float) mul != (float) uniSum) {
  //       printf("i: %d, j: %d\n", i, j);
  //       printf("a: %f, b: %f\n", a, b);
  //       printf("float: %f, posit: %f, universal: %f\n",  gold, (float) mul, (float) uniSum);
  //       return -1;
  //     }
  //   }
  // }

  Posit<16, 1> p16A, p16B, sum;
  Universal16 uni16A, uni16B, uniResult;
  long diff_buckets[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 65536; i++) {
    for (int j = 0; j < 65536; j++) {
      if (i == 32768 || j == 32768) continue;
      // i = 33;
      // j = 64000;

      p16A.setbits(i);
      p16B.setbits(j);
      sum = p16A + p16B;

      uni16A.setbits(i);
      uni16B.setbits(j);
      uniResult = uni16A + uni16B;

      float a = (float) uni16A;
      float b = (float) uni16B;

      if ((float) p16A != a) {
        printf("i: %d, a: %f,  p16A: %f\n", i, a, (float) p16A);
        return -1;
      }
      if ((float) p16B != b) {
        printf("j: %d, b: %f,  p16B: %f\n", j, b, (float) p16B);
        return -1;
      }

      float gold = a + b;
      float hlsDiff = abs(((float) sum - gold) / gold);
      float universalDiff = abs(((float) uniResult - gold) / gold);

      if (hlsDiff < 0.001) {
        diff_buckets[0]++;
      }
      if (hlsDiff < 0.01) {
        diff_buckets[1]++;
      }
      if (hlsDiff < 0.1) {
        diff_buckets[2]++;
      }
      if (hlsDiff < 1) {
        diff_buckets[3]++;
      } else {
        diff_buckets[4]++;
      }

      if (sum.bits != uniResult.encoding() && hlsDiff > universalDiff) {
        printf("i: %d, j: %d\n", i, j);
        printf("a: %f, b: %f\n", a, b);
        printf("float: %f,  sum: %lf, universal: %lf\n",  gold, (float) sum, (float) uniResult);
        std::cerr << sum.bits.to_string(AC_BIN) << std::endl;
        std::cerr << uniResult.get() << std::endl;
        return -1;
      }

      // if ((float) sum != gold && hlsDiff > universalDiff) {
      //   printf("i: %d, j: %d\n", i, j);
      //   printf("a: %f, b: %f\n", a, b);
      //   printf("float: %f,  sum: %lf, universal: %lf\n",  gold, (float) sum, (float) uniResult);
      //   printf("hlsDiff: %f, universalDiff: %f\n", hlsDiff, universalDiff);
      //   return -1;
      // }
    }
  }

  float size = 65536l * 65536l;
  std::cout << "Percentage Difference Count:" << std::endl;
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
}