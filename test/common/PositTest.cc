#include "src/PositTypes.h"
#include "universal/number/posit/posit.hpp"

using Posit8 = sw::universal::posit<8, 1>;
using Posit16 = sw::universal::posit<16, 1>;;
using Internal = sw::universal::value<15>;

template<int nbits, int es, int sbits, int fbits>
bool test(float f) {
  Posit<nbits, es> p = f;
  sw::universal::posit<nbits, es> ref = f;
  long encoding = ref.encoding();
  if (p.bits != encoding) {
    std::bitset<nbits> bitstring(encoding);
    std::cerr << "ERROR: incorrect encoding produced!" << std::endl
              << "input: " << f << "  gold: " <<  bitstring.to_string()
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
  //   if (!test<8, 0, 8, 6>(f)) return -1;
  //   if (!test<8, 1, 8, 5>(f)) return -1;
  //   if (!test<16, 1, 8, 13>(f)) return -1;
  //   if (!test<32, 2, 8, 28>(f)) return -1;
  // }

  // // range test
  // for (float f = 1e-6; f < 1e6; f += 1) {
  //   if (!test<8, 0, 8, 6>(f)) return -1;
  //   if (!test<8, 1, 8, 5>(f)) return -1;
  //   if (!test<16, 1, 8, 13>(f)) return -1;
  //   if (!test<32, 2, 8, 28>(f)) return -1;
  // }

  // return 0;

  Posit8 refA, refB;
  Posit16 refC, refOut;
  int errCount = 0;
  for (int i = 10000; i < 256 * 256; i++) {
    for (int j = 0; j < 256; j++) {
      for (int k = 0; k < 256; k++) {
        if (i == 32768 || j == 128 || k == 128) continue;
        // i = 65;
        // j = 1;
        // k = 124;

        Posit<8, 1> pA(j);
        Posit<8, 1> pB(k);
        Posit<16, 1> pC(i);
        Posit<16, 1> pOut = fma(pA, pB, pC);

        refA.setbits(j);
        refB.setbits(k);
        refC.setbits(i);
        Internal internal = sw::universal::fma<8, 1>(refA, refB, 0);
        sw::universal::convert<16, 1, 15>(internal, refOut);
        refOut += refC;

        // check float conversion
        if ((float) pA != (float) refA) {
          printf("pA: %f, refA: %f\n", (float) pA, (float) refA);
          return -1;
        }
        if ((float) pB != (float) refB) {
          std::cerr << k << std::endl;
          std::cerr << pB.bits.to_string(AC_BIN) << std::endl;
          printf("pB: %f, refB: %f\n", (float) pB, (float) refB);
          return -1;
        }
        if ((float) pC != (float) refC) {
          printf("pC: %f, refC: %f\n", (float) pC, (float) refC);
          return -1;
        }

        float a = (float) refA;
        float b = (float) refB;
        float c = (float) refC;
        float gold = a * b + c;

        float hlsDiff = abs(((float) pOut - gold) / gold);
        float universalDiff = abs(((float) refOut - gold) / gold);

        if ((float) pOut != gold && hlsDiff > universalDiff) {
          printf("i: %d, j: %d, k: %d\n", i, j, k);
          printf("a: %f, b: %f, c: %f\n", a, b, c);
          printf("float: %f,  hls: %lf, universal: %lf\n",  gold, (float) pOut, (float) refOut);
          printf("hlsDiff: %f, universalDiff: %f\n", hlsDiff, universalDiff);
          if (++errCount == 100) return -1;
        }
      }
    }
  }
}