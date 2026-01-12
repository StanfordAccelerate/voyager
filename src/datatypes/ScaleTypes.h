#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

#include <ac_int.h>
#include <ac_std_float.h>

#include "StdFloatTypes.h"

template <int W, int E>
class UFloat {
 public:
  static constexpr unsigned int width = W;
  static constexpr unsigned int e_width = E;

  typedef ac_int<W, false> ac_int_rep;
  typedef ac_std_float<W + 1, E> ac_float_rep;
  typedef UFloat<W, E> decoded;

  ac_int_rep d;

  UFloat() : d(0) {}

#ifndef __SYNTHESIS__
  UFloat(const float val) {
    ac_float_rep tmp(abs(val));
    tmp = (tmp.d == 0) ? ac_float_rep::one() : tmp;
    d = tmp.data();
  }
#endif

  template <int W2, int E2>
  UFloat(const ac_std_float<W2, E2>& other) {
    if constexpr (width == e_width) {
      d = other.d.template slc<E2>(W2 - E2 - 1);
    } else {
      ac_float_rep r(other.abs());
      d = r.data();
    }
  }

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  UFloat(const StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>& other)
      : UFloat(other.float_val) {}

  ac_float_rep to_ac_float() const {
    ac_float_rep result;
    result.set_data(d);
    return result;
  }

  ac_int<W, true> bits_rep() { return d; }

  void set_bits(const ac_int<W, true>& rhs) { d = rhs; }

  bool is_zero() const { return d == ac_float_rep::zero().data(); }

  static UFloat zero() {
    UFloat r;
    r.d = ac_float_rep::zero().data();
    return r;
  }

  static UFloat one() {
    UFloat r;
    r.d = ac_float_rep::one().data();
    return r;
  }

  UFloat operator*(const UFloat& rhs) const {
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

  bool operator<(const UFloat& other) const {
    return to_ac_float() < other.to_ac_float();
  }

  bool operator==(const UFloat& other) const { return d == other.d; }

#ifndef __SYNTHESIS__
  operator float() const { return to_ac_float().to_float(); }
#endif

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>() const {
    return to_ac_float();
  }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & d;
  }
#endif
};
