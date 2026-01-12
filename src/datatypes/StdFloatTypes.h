#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

#include <ac_math/ac_gelu_pwl.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_std_float.h>
#include <ccs_dw_fp_lib.h>

template <int mantissa, int exp, bool use_dw_impl = false,
          bool ieee_compliance = true, ac_q_mode Q = AC_RND_CONV>
class StdFloat {
 public:
  static constexpr unsigned int width = mantissa + exp + 1;
  static constexpr unsigned int e_width = exp;
  static constexpr unsigned int mant_bits = mantissa;
  static constexpr int emax =
      (1 << (e_width - 1)) - 1;  // IEEE-754 maximum normal exponent

  typedef ac_std_float<width, exp> ac_float_rep;
  typedef ac_fixed<2 * mantissa, mantissa, true> ac_float_to_fixed_rep;
  typedef ac_fixed<2 * mantissa, mantissa, false> ac_float_to_fixed_rep_out;
  typedef StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q> decoded;

  ac_float_rep float_val;

  StdFloat() {}

#ifndef __SYNTHESIS__
  StdFloat(const float val);
#endif

  template <int W, int E>
  StdFloat(const ac_std_float<W, E>& other);

  template <int W, int I, bool S, ac_q_mode Q2, ac_o_mode O>
  StdFloat(const ac_fixed<W, I, S, Q2, O>& other);

  template <int mantissa2, int exp2, bool use_dw_impl2, bool ieee_compliance2,
            ac_q_mode Q2>
  StdFloat(const StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2>&
               other);

  template <int WFX, int IFX, bool SFX, ac_q_mode QFX, ac_o_mode OFX>
  ac_fixed<WFX, IFX, SFX, QFX, OFX> to_ac_fixed() const {
    return float_val.template convert_to_ac_fixed<WFX, IFX, SFX, QFX, OFX>();
  }

  template <int width, bool is_signed = true>
  ac_int<width, is_signed> to_ac_int() const {
    return float_val.template convert_to_ac_int<width, is_signed>();
  }

  ac_int<width, false> bits_rep() const { return float_val.d; }

  void set_bits(int i) { float_val.d = i; }

  bool is_zero() const { return float_val == ac_float_rep::zero(); }

  static StdFloat zero() {
    StdFloat r;
    r.float_val = ac_float_rep::zero();
    return r;
  }

  static StdFloat one() {
    StdFloat r;
    r.float_val = ac_float_rep::one();
    return r;
  }

  static StdFloat min() {
    StdFloat r;
    r.float_val = -ac_float_rep::max();
    return r;
  }

  StdFloat abs() const { return float_val.abs(); }

  StdFloat negate() const { return -float_val; }

#pragma hls_design ccore
  StdFloat exponential() const {
    return ac_math::ac_exp_pwl<ac_float_rep>(float_val);
  }

  StdFloat reciprocal() const {
    return ac_math::ac_reciprocal_pwl<ac_float_rep, AC_RND>(float_val);
  }

  StdFloat sqrt() const {
    return float_val.template sqrt<AC_RND_CONV, false>();
  }

  StdFloat inv_sqrt() const { return sqrt().reciprocal(); }

  StdFloat relu() const {
    return float_val.neg() ? ac_float_rep::zero() : float_val;
  }

#pragma hls_design ccore
  StdFloat gelu() const {
    typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
    typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> output_type;

    input_type x = to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
    return ac_math::ac_gelu_pwl<output_type>(x);
  }

#pragma hls_design ccore
  StdFloat silu() const {
    typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
    typedef ac_fixed<15, 3, false, AC_RND, AC_SAT> output_type;

    input_type x = to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
    return x * ac_math::ac_sigmoid_pwl<output_type>(x);
  }

#pragma hls_design ccore
  StdFloat tanh() const {
    typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
    typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> output_type;

    input_type x = to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
    return ac_math::ac_tanh_pwl<output_type>(x);
  }

  ac_int<e_width, false> unbiased_exponent() const {
    return float_val.d.template slc<e_width>(mant_bits);
  }

  void adjust_exponent(int bias) {
    ac_int<e_width, false> e = unbiased_exponent();
    e += bias;
    float_val.d.set_slc(mant_bits, e);
  }

  template <int mantissa2, int exp2, bool use_dw_impl2, bool ieee_compliance2,
            ac_q_mode Q2>
  StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2> fma(
      StdFloat& b,
      StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2>& c);

  StdFloat operator+(const StdFloat& rhs) const;
  StdFloat operator-(const StdFloat& rhs) const;
  StdFloat operator*(const StdFloat& rhs) const;
  StdFloat operator/(const StdFloat& rhs) const;

  StdFloat& operator+=(const StdFloat& rhs);
  StdFloat& operator-=(const StdFloat& rhs);
  StdFloat& operator*=(const StdFloat& rhs);
  StdFloat& operator/=(const StdFloat& rhs);

  bool operator==(const StdFloat& rhs) const;
  bool operator!=(const StdFloat& rhs) const;
  bool operator>(const StdFloat& rhs) const;
  bool operator>=(const StdFloat& rhs) const;
  bool operator<(const StdFloat& rhs) const;
  bool operator<=(const StdFloat& rhs) const;

#ifndef __SYNTHESIS__
  operator float() const { return float_val.to_float(); }
#endif

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & float_val.d;
  }
#endif
};

#ifndef __SYNTHESIS__
#include <float.h>
#include <math.h>
template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::StdFloat(
    const float val) {
  float max_float = ac_float_rep::max().to_float();
  if (val > max_float) {
    float_val = ac_float_rep::max();
  } else if (val < -max_float) {
    float_val = -ac_float_rep::max();
  } else {
    float_val = ac_float_rep(val);
  }
}
#endif

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
template <int W, int E>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::StdFloat(
    const ac_std_float<W, E>& other) {
  float_val = ac_float_rep(other);
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
template <int W, int I, bool S, ac_q_mode Q2, ac_o_mode O>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::StdFloat(
    const ac_fixed<W, I, S, Q2, O>& other) {
  float_val = ac_float_rep(other);
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
template <int mantissa2, int exp2, bool use_dw_impl2, bool ieee_compliance2,
          ac_q_mode Q2>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::StdFloat(
    const StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2>&
        other) {
  float_val = static_cast<ac_float_rep>(other.float_val);
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator+(
    const StdFloat& rhs) const {
  return float_val.template add<Q, !ieee_compliance>(rhs.float_val);
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator-(
    const StdFloat& rhs) const {
  return float_val.template sub<Q, !ieee_compliance>(rhs.float_val);
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator*(
    const StdFloat& rhs) const {
  return float_val.template mult<Q, !ieee_compliance>(rhs.float_val);
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator/(
    const StdFloat& rhs) const {
  return float_val.template div<Q, !ieee_compliance>(rhs.float_val);
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>&
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator+=(
    const StdFloat& rhs) {
  *this = float_val.template add<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>&
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator-=(
    const StdFloat& rhs) {
  *this = float_val.template sub<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>&
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator*=(
    const StdFloat& rhs) {
  *this = float_val.template mult<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>&
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator/=(
    const StdFloat& rhs) {
  *this = float_val.template div<Q, !ieee_compliance>(rhs.float_val);
  return *this;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline bool StdFloat<mantissa, exp, use_dw_impl, ieee_compliance,
                     Q>::operator==(const StdFloat& rhs) const {
  return float_val == rhs.float_val;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline bool StdFloat<mantissa, exp, use_dw_impl, ieee_compliance,
                     Q>::operator!=(const StdFloat& rhs) const {
  return float_val != rhs.float_val;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline bool StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator>(
    const StdFloat& rhs) const {
  return float_val > rhs.float_val;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline bool StdFloat<mantissa, exp, use_dw_impl, ieee_compliance,
                     Q>::operator>=(const StdFloat& rhs) const {
  return float_val >= rhs.float_val;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline bool StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::operator<(
    const StdFloat& rhs) const {
  return float_val < rhs.float_val;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
inline bool StdFloat<mantissa, exp, use_dw_impl, ieee_compliance,
                     Q>::operator<=(const StdFloat& rhs) const {
  return float_val <= rhs.float_val;
}

template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
template <int mantissa2, int exp2, bool use_dw_impl2, bool ieee_compliance2,
          ac_q_mode Q2>
StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2>
StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>::fma(
    StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>& b,
    StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2>& c) {
  StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2> a_upcast(*this);
  StdFloat<mantissa2, exp2, use_dw_impl2, ieee_compliance2, Q2> b_upcast(b);

  if (use_dw_impl) {
    return fp_mac<Q, ieee_compliance, mantissa2 + exp2 + 1, exp2>(
        a_upcast.float_val, b_upcast.float_val, c.float_val);
  } else {
    return a_upcast.float_val.template fma<Q, !ieee_compliance>(
        b_upcast.float_val, c.float_val);
  }
}
