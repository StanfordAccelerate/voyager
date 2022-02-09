#include "src/PositTypes.h"
#include "universal/number/posit/posit.hpp"

#define ACCUM_WIDTH 23
#define SCALE_WIDTH 8

using Real = sw::universal::posit<8, 1>;
using Internal = sw::universal::value<15>;

template<int nbits, int es, int sbits, int fbits>
bool test(float f) {
  sw::universal::posit<nbits, es> ref = f;
  Posit<nbits, es, sbits, fbits> p = f;
  long encoding = ref.encoding();
  if (p.bits != encoding) {
    std::bitset<nbits> bitstring(encoding);
    std::cerr << "ERROR: incorrect encoding produced!" << std::endl
              << "input: " << f << "  gold: " <<  bitstring.to_string()
              << "\thls: " << p.bits.to_string(AC_BIN) << std::endl;
    return false;
  }

  PositFP<sbits, fbits> fp(p);
  if ((float) fp != (float) p) {
      std::cerr << "ERROR: incorrect decoded value produced!" << std::endl
                << "gold: " << (float) p << "  hls: " << (float) fp << std::endl;
      return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  // std::cerr << "Testing posit encoding and decoding." << std::endl;
  // for (float f = 0; f < 10; f += 1e-5) {
  //   if (!test<8, 0, 8, 23>(f)) return -1;
  //   if (!test<8, 1, 8, 23>(f)) return -1;
  //   if (!test<16, 1, 8, 23>(f)) return -1;
  //   if (!test<32, 2, 8, 23>(f)) return -1;
  // }

  std::cerr << "Testing posit fma." << std::endl;
  for (int i = 0; i < 10000; i++) {
    float a = (double) rand() / RAND_MAX;
    float b = (double) rand() / RAND_MAX;
    float c = (double) rand() / RAND_MAX;

    Posit<8, 1, 8, 16> pA(a);
    Posit<8, 1, 8, 16> pB(b);
    Posit<8, 1, 8, 16> pC(c);
    PositFP<8, 16> fp1(pA);
    PositFP<8, 16> fp2(pB);
    PositFP<8, 16> fp3(pC);
    fp1 = fp1.fma(pB, pC);

    Real rA = a;
    Real rB = b;
    Real rC = c;
    Internal ref = sw::universal::fma<8, 1>(rA, rB, rC);

    float gold = a * b + c;

    float positDiff = abs(((float) fp1 - gold) / gold);
    float refDiff = abs(((float) ref - gold) / gold);

    if (gold != (float) fp1 && positDiff > refDiff) {
        // printf("a: %f, b: %f, c: %f\n", a, b, c);
        printf("float: %f,  hls: %f, universal: %f\n",  gold, (float) fp1, (float) ref);
    }
  }
}