#pragma once

// clang-format off
#include <ac_int.h>
// #include <ac_sc.h>
// clang-format on
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl_vha.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_sqrt_pwl.h>

template <int W, bool S>
class Int {
 public:
  typedef ac_int<W, S> ac_int_rep;

  static constexpr unsigned int width = W;

  typedef Int<W, S> Decoded;
  typedef ac_fixed<2 * W, W, true> ac_int_to_fixed_rep;
  typedef ac_fixed<2 * W, W, false> ac_int_to_fixed_rep_out;

  ac_int_rep int_val;

  Int() {}

#ifndef __SYNTHESIS__
  Int(const float val);
#endif

  template <int W2, bool S2>
  Int(const ac_int<W2, S2> &rhs);

  template <int W2, bool S2>
  Int(const Int<W2, S2> &input);

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  Int(const StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &input);

  ac_int<W, false> bits_rep() { return int_val; }

  void negate() { int_val = -int_val; }

  void relu() {
    if (int_val < 0) int_val = 0;
  }

  void masked_relu(const Int &mask) {
    if (mask.int_val == 0) int_val = 0;
  }

  void set_bits(int i) { int_val = i; }

  void set_zero() { int_val = 0; }

  void custom_converted_reciprocal() { this->reciprocal(); }

  void reciprocal(ac_int<8, false> scale) {
    ac_int_to_fixed_rep converted_to_fixed = int_val;
    ac_int_to_fixed_rep_out reciprocal_in_fixed;
    // Compute with larger width then scale / shift as required
    ac_int<8, false> shift = W - scale;
    ac_math::ac_reciprocal_pwl_vha(converted_to_fixed, reciprocal_in_fixed);
    // std::cout << "Reciprocal in Fixed " << reciprocal_in_fixed << std::endl;
    // int_val = static_cast<ac_int_rep>(reciprocal_in_fixed >> shift);
    reciprocal_in_fixed = reciprocal_in_fixed << shift;

    // std::cout << "Reciprocal in Fixed Shifted " << reciprocal_in_fixed <<
    // std::endl;
    int_val = reciprocal_in_fixed.to_ac_int();
    // std::cout << "Reciprocal in Int " << int_val << std::endl;
  }

  // TODO add unique scale factor for exponent similar to reciprocal ?
  void exponential() {
    // convert to fixed point
    ac_int_to_fixed_rep converted_to_fixed = int_val;
    ac_int_to_fixed_rep_out exponent_in_fixed;
    // take fixed point exponent
    ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
    // convert back to float
    int_val = exponent_in_fixed.to_ac_int();
  }

  Int inv_sqrt(ac_int<8, false> scale) {
    ac_int_to_fixed_rep_out converted_to_fixed = int_val;
    ac_int_to_fixed_rep_out inv_sqrt_in_fixed;
    // Compute with larger width then scale / shift as required
    ac_int<8, false> shift = W - scale;
    ac_math::ac_inverse_sqrt_pwl(converted_to_fixed, inv_sqrt_in_fixed);
    inv_sqrt_in_fixed = inv_sqrt_in_fixed >> shift;
    return inv_sqrt_in_fixed.to_ac_int();
  }

  Int sqrt(ac_int<8, false> scale) {
    ac_int_to_fixed_rep_out converted_to_fixed = int_val;
    ac_int_to_fixed_rep_out sqrt_in_fixed;
    // Compute with larger width then scale / shift as required
    ac_int<8, false> shift = W - scale;
    ac_math::ac_sqrt_pwl(converted_to_fixed, sqrt_in_fixed);
    sqrt_in_fixed = sqrt_in_fixed >> shift;
    return sqrt_in_fixed.to_ac_int();
  }

  Int max1() {
    ac_int_rep one;
    one = 1;
    if (int_val > one) {
      int_val = one;
    }
    return int_val;
  }

  // TODO add unique scale factor for exponent similar to reciprocal ?
  void sigmoid() {
    ac_int_to_fixed_rep converted_to_fixed = int_val.convert_to_ac_fixed();
    ac_int_to_fixed_rep_out sigmoid_in_fixed;
    ac_math::ac_sigmoid_pwl(converted_to_fixed, sigmoid_in_fixed);
    int_val = static_cast<ac_int_rep>(sigmoid_in_fixed);
  }

  void expScale(ac_int<8, false> offset) {
    // TODO: Temp implementation do we need this for int ?
    int_val = int_val << offset;
  }

  void scale(ac_int<8, true> scale) { int_val = int_val << scale; }

  template <int W2, bool S2>
  Int<W2, S2> fma(Int &b, Int<W2, S2> &c);

  Int operator+(const Int &rhs);
  Int operator*(const Int &rhs);
  Int operator/(const Int &rhs);
  Int &operator+=(const Int &rhs);
  Int &operator-=(const Int &rhs);
  Int &operator*=(const Int &rhs);
  Int &operator/=(const Int &rhs);

  bool operator<(const Int &rhs) const;
  operator float() const { return float(int_val); }

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>() const {
    using FloatType = StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>;
    FloatType f;
    f.float_val = typename FloatType::ac_float_rep(int_val);
    return f;
  }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & int_val;
  }
#endif
};

#ifndef __SYNTHESIS__
template <int W, bool S>
Int<W, S>::Int(const float val) {
  int_val = Int<W, S>::ac_int_rep(val);
}
#endif

template <int W, bool S>
template <int W2, bool S2>
Int<W, S>::Int(const ac_int<W2, S2> &rhs) {
  int_val = ac_int_rep(rhs);
}

template <int W, bool S>
template <int W2, bool S2>
Int<W, S>::Int(const Int<W2, S2> &input) {
  int_val = static_cast<ac_int_rep>(input);
}

template <int W, bool S>
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
Int<W, S>::Int(
    const StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &input) {
  int_val = input.float_val.template convert_to_ac_int<W, S>();
}

template <int W, bool S>
Int<W, S> exponent(Int<W, S> element) {
  // TODO: clean this up
  typedef ac_int<W, S> ac_int_rep;

  typedef ac_fixed<2 * W, W, true, AC_TRN, AC_WRAP> ac_int_to_fixed_rep;
  typedef ac_fixed<2 * W, W, false> ac_int_to_fixed_out_rep;
  // convert to fixed point
  ac_int_to_fixed_rep converted_to_fixed = element.int_val;

  ac_int_to_fixed_out_rep exponent_in_fixed;
  // take fixed point exponent
  ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
  // convert back to int
  ac_int_rep exponent_in_int = exponent_in_fixed.to_ac_int();

  return static_cast<Int<W, S> >(exponent_in_int);
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator+(const Int &rhs) {
  return int_val + rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator*(const Int &rhs) {
  return int_val * rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator/(const Int &rhs) {
  return int_val / rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator+=(const Int &rhs) {
  int_val += rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator-=(const Int &rhs) {
  int_val -= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator*=(const Int &rhs) {
  int_val *= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator/=(const Int &rhs) {
  int_val /= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline bool Int<W, S>::operator<(const Int &rhs) const {
  return int_val < rhs.int_val;
}

template <int W, bool S>
template <int W2, bool S2>
Int<W2, S2> Int<W, S>::fma(Int<W, S> &b, Int<W2, S2> &c) {
  // Int<W2, S2> a_higherprecision(*this);
  // Int<W2, S2> b_higherprecision(b);

  return static_cast<Int<W2, S2> >(*this) * static_cast<Int<W2, S2> >(b) + c;

  // if (useDWImpl) {
  //   return fp_mac<AC_TRN_ZERO, ieee_compliance, W2 + exp2 + 1, exp2>(
  //       a_higherprecision.int_val, b_higherprecision.int_val, c.int_val);
  // } else {
  //   return a_higherprecision.int_val.template fma<AC_TRN_ZERO, true>(
  //       b_higherprecision.int_val, c.int_val);
  // }
}

template <int W, bool S, int W2, bool S2>
typename Int<W2, S2>::Decoded decomposed_fma(
    const typename Int<W, S>::Decoded &a, const typename Int<W, S>::Decoded &b,
    const typename Int<W2, S2>::Decoded &c) {
  Int<W2, S2> a_higherprecision(a);
  Int<W2, S2> b_higherprecision(b);

  return a_higherprecision.int_val.template fma<AC_TRN_ZERO, true>(
      b_higherprecision.int_val, c.int_val);
}
