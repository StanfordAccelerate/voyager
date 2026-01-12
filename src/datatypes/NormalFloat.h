#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

#include <ac_std_float.h>

#include "IntTypes.h"
#include "StdFloatTypes.h"

class NormalFloat4 {
 public:
  static constexpr unsigned int width = 4;
  static constexpr int emax = 3;  // max normal exponent

  typedef ac_int<4, false> ac_int_rep;
  typedef Int<6, true> decoded;

  static inline const ac_int<6, true> code[16] = {
      -31, -22, -16, -12, -9, -6, -3, 0, 2, 5, 8, 10, 14, 17, 22, 31};

  ac_int_rep index;

  NormalFloat4() : index(7) {}
  NormalFloat4(const ac_int_rep& other) : index(other) {}

#ifndef __SYNTHESIS__
  NormalFloat4(const float other) : index(other) {}
#endif

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  NormalFloat4(
      const StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>& other);

  ac_int<4, false> bits_rep() { return index; }

  void set_bits(int i) { index = i; }

  static NormalFloat4 zero() {
    NormalFloat4 r;
    r.index = 7;
    return r;
  }

#ifndef __SYNTHESIS__
  operator float() const { return code[index]; }
#endif

  template <int W, bool S>
  operator Int<W, S>() const {
    return code[index];
  }

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>() const {
    using std_float_t =
        StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>;
    std_float_t f;
    f.float_val = typename std_float_t::ac_float_rep(code[index]);
    return f;
  }

  friend std::ostream& operator<<(std::ostream& os, const NormalFloat4& nf) {
    os << NormalFloat4::code[nf.index];
    return os;
  }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & index;
  }
#endif
};

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
NormalFloat4::NormalFloat4(
    const StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>& other) {
  using ac_float_t = typename StdFloat<mantissa, exp, use_dw_impl,
                                       ieee_compliance, Q>::ac_float_rep;

  const ac_float_t float_val = other.float_val;

  if (float_val <= ac_float_t(1.0)) {
    if (float_val <= ac_float_t(-10.5)) {
      if (float_val <= ac_float_t(-19.0)) {
        index = float_val <= ac_float_t(-26.5) ? 0 : 1;
      } else {
        index = float_val <= ac_float_t(-14.0) ? 2 : 3;
      }
    } else {
      if (float_val <= ac_float_t(-4.5)) {
        index = float_val <= ac_float_t(-7.5) ? 4 : 5;
      } else {
        index = float_val <= ac_float_t(-1.5) ? 6 : 7;
      }
    }
  } else {
    if (float_val <= ac_float_t(12.0)) {
      if (float_val <= ac_float_t(6.5)) {
        index = float_val <= ac_float_t(3.5) ? 8 : 9;
      } else {
        index = float_val <= ac_float_t(9.0) ? 10 : 11;
      }
    } else {
      if (float_val <= ac_float_t(19.5)) {
        index = float_val <= ac_float_t(15.5) ? 12 : 13;
      } else {
        index = float_val <= ac_float_t(26.5) ? 14 : 15;
      }
    }
  }
}
