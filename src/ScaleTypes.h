#pragma once

#include <ac_int.h>

// Represents a microscaling type used for scaling, e.g., E8M0
// Currently in this implementation, we don't actually store it according to the
// E8M0 standard as specified in the OCP MX specification, but instead as a
// signed unbiased exponent.
template <int W, int E>
class UFloat {
 public:
  typedef ac_int<W, false> ac_int_rep;
  typedef ac_std_float<W + 1, E> ac_float_rep;

  static constexpr unsigned int width = W;

  ac_int_rep d;

  UFloat() : d(ac_float_rep::one().data()) {}

#ifndef __SYNTHESIS__
  UFloat(const float val) {
    ac_float_rep tmp(abs(val));
    tmp = (tmp.d == 0) ? ac_float_rep::one() : tmp;
    d = tmp.data();
  }
#endif

  template <int W2, int E2>
  UFloat(const ac_std_float<W2, E2> &i) {
    ac_float_rep tmp = i.abs();
    d = tmp.data();
  }

  ac_float_rep to_ac_float() const {
    ac_float_rep result;
    result.set_data(d);
    return result;
  }

  ac_int<W, true> bits_rep() { return d; }

  void set_bits(const ac_int<W, true> &rhs) { d = rhs; }
  void set_zero() { d = 0; }
  void set_one() { d = ac_float_rep::one().data(); }

  UFloat operator*(const UFloat &rhs) const {
    if constexpr (W == E) {
      int exp = d + rhs.d - ac_float_rep::exp_bias;
      ac_int<E, true> e_r = exp;
      UFloat result;
      result.set_bits(e_r);
      return result;
    } else {
      return to_ac_float() * rhs.to_ac_float();
    }
  }

#ifndef __SYNTHESIS__
  operator float() const { return to_ac_float().to_float(); }
#endif

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>() const {
    return to_ac_float();
  }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & d;
  }
#endif
};
