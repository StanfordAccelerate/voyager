#pragma once

// clang-format off
#include <ac_std_float.h>
// clang-format on
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ccs_dw_fp_lib.h>

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
class StdFloat {
 public:
  typedef ac_std_float<mantissa + exp + 1, exp> ac_float_rep;
  static constexpr unsigned int width = ac_float_rep::width;
  static constexpr unsigned int exponent_width = ac_float_rep::e_width;
  static constexpr unsigned int mantissa_width = ac_float_rep::mant_bits;

  typedef StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> Decoded;
  typedef ac_fixed<2 * mantissa, mantissa, true> ac_float_to_fixed_rep;
  typedef ac_fixed<2 * mantissa, mantissa, false> ac_float_to_fixed_rep_out;

  ac_float_rep float_val;

  StdFloat() {}

#ifndef __SYNTHESIS__
  StdFloat(const float val);
#endif

  template <int W, int E>
  StdFloat(const ac_std_float<W, E> &rhs);

  template <int W, int I, bool S, ac_q_mode Q2, ac_o_mode O>
  StdFloat(const ac_fixed<W, I, S, Q2, O> &rhs);

  template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
            ac_q_mode Q2>
  StdFloat(
      const StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> &input);

  template <int WFX, int IFX, bool SFX, ac_q_mode QFX, ac_o_mode OFX>
  ac_fixed<WFX, IFX, SFX, QFX, OFX> to_ac_fixed() {
    return float_val.template convert_to_ac_fixed<WFX, IFX, SFX, QFX, OFX>();
  }

  ac_int<mantissa + exp + 1, false> bits_rep() { return float_val.d; }

  void negate() { float_val = -float_val; }

  void relu() {
    if (float_val.neg()) float_val = float_val.zero();
  }

  void masked_relu(const StdFloat &mask) {
    if (mask.float_val.d == 0) float_val = float_val.zero();
  }

  void set_bits(int i) { float_val.d = i; }

  void set_zero() { float_val = float_val.zero(); }

  void custom_converted_reciprocal() { this->reciprocal(); }

  void reciprocal() {
    float_val = ac_math::ac_reciprocal_pwl<ac_float_rep, AC_TRN, ac_float_rep>(
        float_val);
  }

  void exponential() {
    ac_float_rep result;
    ac_math::ac_exp_pwl(float_val, result);
    float_val = result;
  }

  StdFloat inv_sqrt() {
    StdFloat result = float_val.template sqrt<AC_RND_CONV, false>();
    result.reciprocal();
    return result;
  }

  StdFloat sqrt() { return float_val.template sqrt<AC_RND_CONV, false>(); }

  StdFloat max1() {
    ac_float_rep one;
    one = one.one();
    if (float_val > one) {
      float_val = one;
    }
    return float_val;
  }

  void sigmoid() {
    ac_float_to_fixed_rep converted_to_fixed = float_val.convert_to_ac_fixed();
    ac_float_to_fixed_rep_out sigmoid_in_fixed;
    ac_math::ac_sigmoid_pwl(converted_to_fixed, sigmoid_in_fixed);
    float_val = static_cast<ac_float_rep>(sigmoid_in_fixed);
  }

  template <int width, bool sign>
  void expScale(ac_int<width, sign> offset) {
    if (float_val == float_val.zero()) return;
    // TODO: handle subnormal numbers
    ac_int<exponent_width, true> exp_bits =
        float_val.d.template slc<exponent_width>(mantissa_width);
    exp_bits += offset;
    float_val.d.set_slc(mantissa_width, exp_bits);
  }

  ac_int<exponent_width, false> unbiased_exponent() {
    return float_val.d.template slc<exponent_width>(mantissa_width);
  }

  ac_int<exponent_width, true> exponent() {
    ac_int<exponent_width, true> exp_bits =
        float_val.d.template slc<exponent_width>(mantissa_width);

    return exp_bits - ac_int<exponent_width, true>(ac_float_rep::exp_bias);
  }

  template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
            ac_q_mode Q2>
  StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> fma(
      StdFloat &b,
      StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> &c);

  StdFloat operator+(const StdFloat &rhs);
  StdFloat operator*(const StdFloat &rhs);
  StdFloat operator/(const StdFloat &rhs);
  StdFloat &operator+=(const StdFloat &rhs);
  StdFloat &operator-=(const StdFloat &rhs);
  StdFloat &operator*=(const StdFloat &rhs);
  StdFloat &operator/=(const StdFloat &rhs);

  bool operator<(const StdFloat &rhs) const;
  operator float() const { return float_val.to_float(); }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & float_val.d;
  }
#endif
};

#ifndef __SYNTHESIS__
#include <float.h>
#include <math.h>
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const float val) {
  if (val == FLT_MAX) {
    float_val = float_val.inf();
  } else if (val == -FLT_MAX) {
    float_val = -float_val.inf();
  } else {
    float_val = ac_float_rep(val);
  }
}
#endif

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int W, int E>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const ac_std_float<W, E> &rhs) {
  float_val = ac_float_rep(rhs);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int W, int I, bool S, ac_q_mode Q2, ac_o_mode O>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const ac_fixed<W, I, S, Q2, O> &rhs) {
  float_val = ac_float_rep(rhs);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
          ac_q_mode Q2>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> &input) {
  float_val = static_cast<ac_float_rep>(input.float_val);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator+(
    const StdFloat &rhs) {
  return float_val.template add<Q, !ieee_compliance>(rhs.float_val);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator*(
    const StdFloat &rhs) {
  return float_val.template mult<Q, !ieee_compliance>(rhs.float_val);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator/(
    const StdFloat &rhs) {
  return float_val.template div<Q, !ieee_compliance>(rhs.float_val);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator+=(
    const StdFloat &rhs) {
  *this = float_val.template add<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator-=(
    const StdFloat &rhs) {
  *this = float_val.template sub<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator*=(
    const StdFloat &rhs) {
  *this = float_val.template mult<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator/=(
    const StdFloat &rhs) {
  *this = float_val.template div<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

#pragma hls_design ccore
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
inline bool StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::operator<(
    const StdFloat &rhs) const {
  return float_val < rhs.float_val;
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
          ac_q_mode Q2>
StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::fma(
    StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &b,
    StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> &c) {
  StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> a_higherprecision(
      *this);
  StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> b_higherprecision(
      b);

  if (useDWImpl) {
    return fp_mac<Q, ieee_compliance, mantissa2 + exp2 + 1, exp2>(
        a_higherprecision.float_val, b_higherprecision.float_val, c.float_val);
  } else {
    return a_higherprecision.float_val.template fma<Q, !ieee_compliance>(
        b_higherprecision.float_val, c.float_val);
  }
}
