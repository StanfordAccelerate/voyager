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

template <int wdth, bool sgnd>
class Int {
 public:
  typedef ac_int<wdth, sgnd> ac_int_rep;

  static constexpr unsigned int width = wdth;

  static constexpr bool is_floating_point = false;

  // TODO: make this a template parameter
  typedef Int<wdth, sgnd> AccumulationDatatype;

  // Set Int to Fixed type for non linear funcs
  typedef ac_fixed<2 * wdth, wdth, true> ac_int_to_fixed_rep;

  // Set Int to Fixed type for non linear funcs
  typedef ac_fixed<2 * wdth, wdth, false> ac_int_to_fixed_rep_out;

  ac_int_rep int_val;

  Int() {}
  Int(const ac_int_rep &input_float_rep);

#ifndef __SYNTHESIS__
  Int(const float val);
#endif

  template <int wdth2, bool sgnd2>
  Int(const Int<wdth2, sgnd2> input[2]);

  Int(const Int input[2]);

  template <int wdth2, bool sgnd2>
  Int(const Int<wdth2, sgnd2> &input);

  template <int W, bool S>
  Int(const ac_int<W, S> &rhs);

  template <int wdth2, bool sgnd2>
  void storeAsLowerPrecision(Int<wdth2, sgnd2> output[2]);

  ac_int<wdth, false> bits_rep() { return int_val; }

  void negate() { int_val = -int_val; }

  void relu() {
    if (int_val < 0) int_val = 0;
  }

  void masked_relu(const Int &mask) {
    if (mask.int_val == 0) int_val = 0;
  }

  void setbits(int i) { int_val = i; }

  void setZero() { int_val = 0; }

  void custom_converted_reciprocal() { this->reciprocal(); }

  void reciprocal(ac_int<8, false> scale) {
    ac_int_to_fixed_rep converted_to_fixed = int_val;
    ac_int_to_fixed_rep_out reciprocal_in_fixed;
    // Compute with larger width then scale / shift as required
    ac_int<8, false> shift = wdth - scale;
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
    ac_int<8, false> shift = wdth - scale;
    ac_math::ac_inverse_sqrt_pwl(converted_to_fixed, inv_sqrt_in_fixed);
    inv_sqrt_in_fixed = inv_sqrt_in_fixed >> shift;
    return inv_sqrt_in_fixed.to_ac_int();
  }

  Int sqrt(ac_int<8, false> scale) {
    ac_int_to_fixed_rep_out converted_to_fixed = int_val;
    ac_int_to_fixed_rep_out sqrt_in_fixed;
    // Compute with larger width then scale / shift as required
    ac_int<8, false> shift = wdth - scale;
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

  template <int wdth2, bool sgnd2>
  Int<wdth2, sgnd2> fma(Int &b, Int<wdth2, sgnd2> &c);

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> dequantize(
      ac_int<1 + mantissa + exp, false> scale);

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> dequantize(
      StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> scale);

  Int operator+(const Int &rhs);
  Int operator*(const Int &rhs);
  Int operator/(const Int &rhs);
  Int &operator+=(const Int &rhs);
  Int &operator-=(const Int &rhs);
  Int &operator*=(const Int &rhs);
  Int &operator/=(const Int &rhs);

  bool operator<(const Int &rhs) const;
  operator float() const { return float(int_val); }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & int_val;
  }
#endif
};

#ifndef __SYNTHESIS__
template <int wdth, bool sgnd>
Int<wdth, sgnd>::Int(const float val) {
  int_val = Int<wdth, sgnd>::ac_int_rep(val);
}
#endif

template <int wdth, bool sgnd>
template <int wdth2, bool sgnd2>
Int<wdth, sgnd>::Int(const Int<wdth2, sgnd2> input[2]) {
  static_assert(
      (wdth) == 2 * (wdth2),
      "Lower precision type must be half the wdth of higher precision.");
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    int_val.set_slc(0 + i * (wdth2), input[i].int_val);
  }
}

template <int wdth, bool sgnd>
Int<wdth, sgnd>::Int(const Int<wdth, sgnd> input[2]) {
  *this = input[0];
}

template <int wdth, bool sgnd>
Int<wdth, sgnd>::Int(const ac_int_rep &input_float_rep) {
  int_val = input_float_rep;
}

template <int wdth, bool sgnd>
template <int wdth2, bool sgnd2>
Int<wdth, sgnd>::Int(const Int<wdth2, sgnd2> &input) {
  int_val = static_cast<ac_int_rep>(input);
}

template <int wdth, bool sgnd>
template <int W, bool S>
Int<wdth, sgnd>::Int(const ac_int<W, S> &rhs) {
  int_val = ac_int_rep(rhs);
}

template <int wdth, bool sgnd>
template <int wdth2, bool sgnd2>
void Int<wdth, sgnd>::storeAsLowerPrecision(Int<wdth2, sgnd2> output[2]) {
  static_assert(
      (wdth) == 2 * (wdth2),
      "Lower precision type must be half the wdth of higher precision.");
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    output[i].int_val = int_val.template slc<wdth2>(0 + i * (wdth2));
  }
}

template <int wdth, bool sgnd>
Int<wdth, sgnd> exponent(Int<wdth, sgnd> element) {
  // TODO: clean this up
  typedef ac_int<wdth, sgnd> ac_int_rep;

  typedef ac_fixed<2 * wdth, wdth, true, AC_TRN, AC_WRAP> ac_int_to_fixed_rep;
  typedef ac_fixed<2 * wdth, wdth, false> ac_int_to_fixed_out_rep;
  // convert to fixed point
  ac_int_to_fixed_rep converted_to_fixed = element.int_val;

  ac_int_to_fixed_out_rep exponent_in_fixed;
  // take fixed point exponent
  ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
  // convert back to int
  ac_int_rep exponent_in_int = exponent_in_fixed.to_ac_int();

  return static_cast<Int<wdth, sgnd> >(exponent_in_int);
}

template <int wdth, bool sgnd>
inline Int<wdth, sgnd> Int<wdth, sgnd>::operator+(const Int &rhs) {
  return int_val + rhs.int_val;
}

template <int wdth, bool sgnd>
inline Int<wdth, sgnd> Int<wdth, sgnd>::operator*(const Int &rhs) {
  return int_val * rhs.int_val;
}

template <int wdth, bool sgnd>
inline Int<wdth, sgnd> Int<wdth, sgnd>::operator/(const Int &rhs) {
  return int_val / rhs.int_val;
}

template <int wdth, bool sgnd>
inline Int<wdth, sgnd> &Int<wdth, sgnd>::operator+=(const Int &rhs) {
  int_val += rhs.int_val;
  return *this;
}

template <int wdth, bool sgnd>
inline Int<wdth, sgnd> &Int<wdth, sgnd>::operator-=(const Int &rhs) {
  int_val -= rhs.int_val;
  return *this;
}

template <int wdth, bool sgnd>
inline Int<wdth, sgnd> &Int<wdth, sgnd>::operator*=(const Int &rhs) {
  int_val *= rhs.int_val;
  return *this;
}

template <int wdth, bool sgnd>
inline Int<wdth, sgnd> &Int<wdth, sgnd>::operator/=(const Int &rhs) {
  int_val /= rhs.int_val;
  return *this;
}

template <int wdth, bool sgnd>
inline bool Int<wdth, sgnd>::operator<(const Int &rhs) const {
  return int_val < rhs.int_val;
}

template <int wdth, bool sgnd>
template <int wdth2, bool sgnd2>
Int<wdth2, sgnd2> Int<wdth, sgnd>::fma(Int<wdth, sgnd> &b,
                                       Int<wdth2, sgnd2> &c) {
  // Int<wdth2, sgnd2> a_higherprecision(*this);
  // Int<wdth2, sgnd2> b_higherprecision(b);

  return static_cast<Int<wdth2, sgnd2> >(*this) *
             static_cast<Int<wdth2, sgnd2> >(b) +
         c;

  // if (useDWImpl) {
  //   return fp_mac<AC_TRN_ZERO, ieee_compliance, wdth2 + exp2 + 1, exp2>(
  //       a_higherprecision.int_val, b_higherprecision.int_val, c.int_val);
  // } else {
  //   return a_higherprecision.int_val.template fma<AC_TRN_ZERO, true>(
  //       b_higherprecision.int_val, c.int_val);
  // }
}

template <int wdth_i, bool sgnd_i, int wdth_o, bool sgnd_o>
typename Int<wdth_o, sgnd_o>::AccumulationDatatype decomposed_fma(
    const typename Int<wdth_i, sgnd_i>::AccumulationDatatype &a,
    const typename Int<wdth_i, sgnd_i>::AccumulationDatatype &b,
    const typename Int<wdth_o, sgnd_o>::AccumulationDatatype &c) {
  Int<wdth_o, sgnd_o> a_higherprecision(a);
  Int<wdth_o, sgnd_o> b_higherprecision(b);

  return a_higherprecision.int_val.template fma<AC_TRN_ZERO, true>(
      b_higherprecision.int_val, c.int_val);
}

template <int wdth, bool sgnd>
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>
Int<wdth, sgnd>::dequantize(ac_int<1 + mantissa + exp, false> scale) {
  StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> scale_float;
  scale_float.setbits(scale);
  return dequantize(scale_float);
}

template <int wdth, bool sgnd>
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>
Int<wdth, sgnd>::dequantize(
    StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> scale) {
  StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> dequantized_value;
  dequantized_value.float_val.template assign_from<AC_RND_CONV, wdth, sgnd>(
      int_val);
  dequantized_value *= scale;

  return dequantized_value;
}
