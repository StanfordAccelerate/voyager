#pragma once

// clang-format off
#include <ac_std_float.h>
// clang-format on
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl.h>
#include <ccs_dw_fp_lib.h>

template <typename T>
inline ac_int<4, false> get_nf4_index(T val) {
  // -31, -22, -16, -12, -9, -6, -3, 0, 2, 5, 8, 10, 14, 17, 22, 31
  if (val <= 1) {
    if (val <= -10.5) {
      if (val <= -19.0) {
        return val <= -26.5 ? 0 : 1;
      } else {
        return val <= -14.0 ? 2 : 3;
      }
    } else {
      if (val <= -4.5) {
        return val <= -7.5 ? 4 : 5;
      } else {
        return val <= -1.5 ? 6 : 7;
      }
    }
  } else {
    if (val <= 12.0) {
      if (val < 6.5) {
        return val < 3.5 ? 8 : 9;
      } else {
        return val < 9.0 ? 10 : 11;
      }
    } else {
      if (val < 19.5) {
        return val < 15.5 ? 12 : 13;
      } else {
        return val < 26.5 ? 14 : 15;
      }
    }
  }
}

template <int nbits, int ibits>
class NormalFloat {
 public:
  typedef ac_int<nbits, false> ac_int_rep;

  static constexpr unsigned int width = nbits;

  typedef Int<ibits + 1, true> Decoded;

  ac_int<ibits + 1, true> code[16] = {-31, -22, -16, -12, -9, -6, -3, 0,
                                      2,   5,   8,   10,  14, 17, 22, 31};

  ac_int_rep encoding;

  NormalFloat() {}
  NormalFloat(const ac_int_rep &encoding);

#ifndef __SYNTHESIS__
  NormalFloat(const float val);
#endif

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  NormalFloat(
      const StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &input);

  ac_int<nbits, false> bits_rep() { return encoding; }

  void set_bits(int i) { encoding = i; }

  void set_zero() { encoding = 7; }

  template <int wdth, bool sgnd>
  operator Int<wdth, sgnd>() const {
    return code[encoding];
  }

  operator float() const { return code[encoding]; }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & encoding;
  }
#endif
};

#ifndef __SYNTHESIS__

#include <float.h>
#include <math.h>

template <int nbits, int ibits>
NormalFloat<nbits, ibits>::NormalFloat(const float val) {
  encoding = get_nf4_index(val);
}
#endif

template <int nbits, int ibits>
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
NormalFloat<nbits, ibits>::NormalFloat(
    const StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &input) {
  encoding = get_nf4_index(input);
}
