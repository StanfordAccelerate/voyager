#pragma once

// clang-format off
#include <ac_std_float.h>
// clang-format on
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ccs_dw_lib.h>

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
class StdFloat {
 public:
  typedef ac_std_float<mantissa + exp + 1, exp> ac_float_rep;
  static constexpr unsigned int width = ac_float_rep::width;

  static constexpr bool is_floating_point = true;

  // TODO: make this a template parameter
  typedef StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>
      AccumulationDatatype;

  // TODO Set Float to Fixed type
  typedef ac_fixed<2 * mantissa, mantissa, true> ac_float_to_fixed_rep;

  // TODO Set Float to Fixed type
  typedef ac_fixed<2 * mantissa, mantissa, false> ac_float_to_fixed_rep_out;

  ac_float_rep float_val;

  StdFloat() {}
  StdFloat(const ac_float_rep &input_float_rep);

#ifndef __SYNTHESIS__
  StdFloat(const float val);
#endif

  template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
            ac_q_mode Q2>
  StdFloat(const StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2>
               input[2]);

  StdFloat(const StdFloat input[2]);

  template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
            ac_q_mode Q2>
  StdFloat(
      const StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> &input);

  template <int W, bool S>
  StdFloat(const ac_int<W, S> &rhs);

  template <int nbits, int es>
  StdFloat(const Posit<nbits, es> &input);

  template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
            ac_q_mode Q2>
  void storeAsLowerPrecision(
      StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> output[2]);

  void storeAsLowerPrecision(StdFloat output[2]);

  ac_int<mantissa + exp + 1, false> bits_rep() { return float_val.d; }

  void negate() { float_val = -float_val; }

  void relu() {
    if (float_val.neg()) float_val = float_val.zero();
  }

  void masked_relu(const StdFloat &mask) {
    if (mask.float_val.d == 0) float_val = float_val.zero();
  }

  void setbits(int i) { float_val.d = i; }

  void setZero() { float_val = float_val.zero(); }

  void custom_converted_reciprocal() { this->reciprocal(); }

  void reciprocal() {
    float_val = ac_math::ac_reciprocal_pwl<ac_float_rep, AC_TRN, ac_float_rep>(
        float_val);
  }

  void exponential() {
    // convert to fixed point
    ac_float_to_fixed_rep converted_to_fixed =
        float_val.template convert_to_ac_fixed<2 * mantissa, mantissa, true,
                                               AC_TRN, AC_WRAP>(true);

    ac_float_to_fixed_rep_out exponent_in_fixed;
    // take fixed point exponent
    ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
    // convert back to float
    float_val = static_cast<ac_float_rep>(exponent_in_fixed);
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

  void expScale(ac_int<8, false> offset) {
    // TODO: fix this to be scale the exponent
    float_val.d += offset;
  }

  template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
            ac_q_mode Q2>
  StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> fma(
      StdFloat &b,
      StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> &c);

  template <int quantized_width, int quantized_sign>
  Int<quantized_width, quantized_sign> quantize(ac_int<width, false> scale);

  template <int quantized_width, int quantized_sign>
  Int<quantized_width, quantized_sign> quantize(StdFloat scale);

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

/* Constructor for building higher precision type from 2 lower precision types
 */
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
          ac_q_mode Q2>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2>
        input[2]) {
  static_assert(
      (mantissa + exp + 1) == 2 * (mantissa2 + exp2 + 1),
      "Lower precision type must be half the width of higher precision.");
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    float_val.d.set_slc(0 + i * (mantissa2 + exp2 + 1), input[i].float_val.d);
  }
}

// TODO: this is a bit of a hack for cases where we expect a double precision
// type, but it's actually the same type. For example, when a BF16 only design,
// where the accumulation type is the same as the input type.
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> input[2]) {
  *this = input[0];
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const ac_float_rep &input_float_rep) {
  float_val = input_float_rep;
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
          ac_q_mode Q2>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> &input) {
  float_val = static_cast<ac_float_rep>(input);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int W, bool S>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const ac_int<W, S> &rhs) {
  float_val = ac_float_rep(rhs);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int nbits, int es>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const Posit<nbits, es> &input) {
  if (input.isZero()) {
    setZero();
  } else {
    bool sign;
    int scale;
    ac_int<ac_float_rep::mant_bits, false> mantissa_bits;
    decode<nbits, es, ac_float_rep::mant_bits>(input.bits, sign, scale,
                                               mantissa_bits);

    ac_int<1, false> sign_bit = sign;
    ac_int<ac_float_rep::e_width, true> exp_bits =
        scale + ac_float_rep::exp_bias;

    float_val.d.set_slc(0, mantissa_bits);
    float_val.d.set_slc(ac_float_rep::mant_bits, exp_bits);
    float_val.d.set_slc(ac_float_rep::width - 1, sign_bit);
  }
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int mantissa2, int exp2, bool useDWImpl2, bool ieee_compliance2,
          ac_q_mode Q2>
void StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::
    storeAsLowerPrecision(
        StdFloat<mantissa2, exp2, useDWImpl2, ieee_compliance2, Q2> output[2]) {
  static_assert(
      (mantissa + exp + 1) == 2 * (mantissa2 + exp2 + 1),
      "Lower precision type must be half the width of higher precision.");
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    output[i].float_val.d = float_val.d.template slc<mantissa2 + exp2 + 1>(
        0 + i * (mantissa2 + exp2 + 1));
  }
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
void StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::
    storeAsLowerPrecision(
        StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> output[2]) {
  // this is called in the BF16 only case, where the accumulation type is the
  // same as the input type. we will only store it in output[0]
  output[0].float_val = float_val;
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> exponent(
    StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> element) {
  // TODO: clean this up
  typedef ac_std_float<mantissa + exp + 1, exp> ac_float_rep;

  typedef ac_fixed<2 * mantissa, mantissa, true, AC_TRN, AC_WRAP>
      ac_float_to_fixed_rep;
  typedef ac_fixed<2 * mantissa, mantissa, false> ac_float_to_fixed_out_rep;
  // convert to fixed point
  ac_float_to_fixed_rep converted_to_fixed =
      element.float_val.template convert_to_ac_fixed<2 * mantissa, mantissa,
                                                     true, AC_TRN, AC_WRAP>();
  // ac_float_to_fixed_rep converted_to_fixed =
  // element.float_val.convert_to_ac_fixed<2 * mantissa, mantissa, true>();
  ac_float_to_fixed_out_rep exponent_in_fixed;
  // take fixed point exponent
  ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
  // convert back to float
  ac_float_rep exponent_in_float = ac_float_rep(exponent_in_fixed);

  return static_cast<StdFloat<mantissa, exp, useDWImpl> >(exponent_in_float);
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

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int quantized_width, int quantized_sign>
Int<quantized_width, quantized_sign>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::quantize(
    ac_int<width, false> scale) {
  StdFloat scale_float;
  scale_float.setbits(scale);
  return quantize<quantized_width, quantized_sign>(scale_float);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
template <int quantized_width, int quantized_sign>
Int<quantized_width, quantized_sign>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::quantize(
    StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> scale) {
  StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> scaledValue =
      *this / scale;

  Int<quantized_width, quantized_sign> quantizedValue;
  quantizedValue.int_val =
      scaledValue.float_val
          .template convert_to_ac_int<quantized_width, quantized_sign>();

  return quantizedValue;
}

/*
template <>
class StdFloat<7, 8> {
 public:
  typedef ac::bfloat16 ac_float_rep;
  static constexpr unsigned int width = ac_float_rep::width;
  static constexpr unsigned int mantissa = 16;
  static constexpr unsigned int exp = 8;

  // TODO: make this a template parameter
  typedef StdFloat<mantissa, exp, useDWImpl> AccumulationDatatype;

  // TODO Set Float to Fixed type
  typedef ac_fixed<2 * mantissa, mantissa, true> ac_float_to_fixed_rep;

  // TODO Set Float to Fixed type
  typedef ac_fixed<2 * mantissa, mantissa, false> ac_float_to_fixed_rep_out;

  typedef ac_std_float<16, 8> ac_std_float_rep;

  ac_float_rep float_val;

  StdFloat() {}
  StdFloat(const ac_float_rep &input_float_rep) { float_val = input_float_rep; }

#ifndef __SYNTHESIS__
  StdFloat(const float val);
#endif

  StdFloat(const StdFloat input[2]);

  template <int mantissa2, int exp2>
  StdFloat(const StdFloat<mantissa2, exp2> input[2]);

  template <int mantissa2, int exp2>
  StdFloat(const StdFloat<mantissa2, exp2> &input);

  template <int W, bool S>
  StdFloat(const ac_int<W, S> &rhs);

  ac_int<mantissa + exp, false> bits_rep() { return float_val.d; }

  void negate() { float_val = -float_val; }

  void relu() {
    if (float_val.to_ac_std_float().neg()) float_val = float_val.zero();
  }

  void masked_relu(const StdFloat &mask) {
    if (mask.float_val.d == 0) float_val = float_val.zero();
  }

  void setbits(int i) { float_val.d = i; }

  void setZero() { float_val = float_val.zero(); }

  void custom_converted_reciprocal() { this->reciprocal(); }

  void reciprocal() {
    typedef ac::bfloat16::ac_std_float_t input_type;
    typedef ac::bfloat16::ac_std_float_t output_type;

    input_type input = float_val.to_ac_std_float();
    output_type output =
        ac_math::ac_reciprocal_pwl<input_type, AC_TRN, output_type>(input);
    float_val = ac::bfloat16(output);
  }

  // void exponent() {
  //   // convert to fixed point
  //   ac_float_to_fixed_rep converted_to_fixed =
  //       float_val.to_ac_std_float().convert_to_ac_fixed();
  //   ac_float_to_fixed_rep_out exponent_in_fixed;
  //   // take fixed point exponent
  //   ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
  //   // convert back to float
  //   float_val = static_cast<ac_float_rep>(exponent_in_fixed);
  // }

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
    ac_fixed<2 * mantissa, mantissa, true, AC_TRN, AC_WRAP> converted_to_fixed =
        float_val.template convert_to_ac_fixed<2 * mantissa, mantissa, true,
                                               AC_TRN, AC_WRAP>();
    ac_fixed<2 * mantissa, mantissa, false> sigmoid_in_fixed;
    ac_math::ac_sigmoid_pwl(converted_to_fixed, sigmoid_in_fixed);
    float_val = ac::bfloat16(static_cast<ac_float_rep>(sigmoid_in_fixed));

    // ac_std_float_rep input = float_val.to_ac_std_float();
    // ac_std_float_rep output;
    // ac_math::ac_sigmoid_pwl<AC_TRN, ac_std_float_rep,
    // ac_std_float_rep>(input,
    // output);
    // float_val = ac::bfloat16(output);
  }

  void expScale(ac_int<8, false> offset) {
    // TODO: fix this to be scale the exponent
    float_val.d += offset;
  }

  template <int mantissa2, int exp2>
  StdFloat<mantissa2, exp2> fma(StdFloat &b, StdFloat<mantissa2, exp2> &c);

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

inline StdFloat<7, 8> StdFloat<7, 8>::operator+(const StdFloat &rhs) {
  return float_val + rhs.float_val;
}

inline StdFloat<7, 8> StdFloat<7, 8>::operator*(const StdFloat &rhs) {
  return float_val * rhs.float_val;
}

inline StdFloat<7, 8> StdFloat<7, 8>::operator/(const StdFloat &rhs) {
  return float_val / rhs.float_val;
}

inline StdFloat<7, 8> &StdFloat<7, 8>::operator+=(const StdFloat &rhs) {
  float_val += rhs.float_val;
  return *this;
}

inline StdFloat<7, 8> &StdFloat<7, 8>::operator-=(const StdFloat &rhs) {
  float_val -= rhs.float_val;
  return *this;
}

inline StdFloat<7, 8> &StdFloat<7, 8>::operator*=(const StdFloat &rhs) {
  float_val *= rhs.float_val;
  return *this;
}

inline StdFloat<7, 8> &StdFloat<7, 8>::operator/=(const StdFloat &rhs) {
  float_val /= rhs.float_val;
  return *this;
}

#ifndef __SYNTHESIS__
inline StdFloat<7, 8>::StdFloat(const float val) {
  float_val = StdFloat<7, 8>::ac_float_rep(val);
}
#endif

template <int mantissa2, int exp2>
StdFloat<7, 8>::StdFloat(const StdFloat<mantissa2, exp2> &input) {
  float_val = ac::bfloat16(input.float_val);
}

template <int W, bool S>
StdFloat<7, 8>::StdFloat(const ac_int<W, S> &rhs) {
  float_val = ac::bfloat16(ac_float_rep(rhs));
}

// template <int mantissa, int exp>
// template <int mantissa2, int exp2>
// StdFloat<mantissa2, exp2> StdFloat<mantissa, exp, useDWImpl>::fma(
//     StdFloat<mantissa, exp, useDWImpl> &b, StdFloat<mantissa2, exp2> &c) {
//   StdFloat<mantissa2, exp2> a_higherprecision(*this);
//   StdFloat<mantissa2, exp2> b_higherprecision(b);

//   return a_higherprecision.float_val.template fma<AC_TRN_ZERO, true>(
//       b_higherprecision.float_val, c.float_val);
// }

// template <int mantissa_o, int exp_o>
// typename StdFloat<mantissa_o, exp_o>::AccumulationDatatype decomposed_fma(
//     const typename StdFloat<23, 8>::AccumulationDatatype &a,
//     const typename StdFloat<23, 8>::AccumulationDatatype &b,
//     const typename StdFloat<mantissa_o, exp_o>::AccumulationDatatype &c) {
//   StdFloat<mantissa_o, exp_o>
//   a_higherprecision(a.float_val.to_ac_std_float()); StdFloat<mantissa_o,
//   exp_o> b_higherprecision(b.float_val.to_ac_std_float());

//   return a_higherprecision.float_val.template fma<AC_TRN_ZERO, true>(
//       b_higherprecision.float_val, c.float_val);
// }

inline StdFloat<7, 8>::StdFloat(const StdFloat<7, 8> input[2]) {
  *this = input[0];
}

template <int mantissa2, int exp2>
StdFloat<7, 8>::StdFloat(const StdFloat<mantissa2, exp2> input[2]) {
  static_assert(
      (7 + 8 + 1) == 2 * (mantissa2 + exp2 + 1),
      "Lower precision type must be half the width of higher precision.");

  ac_std_float_rep temp = float_val.to_ac_std_float();

#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    temp.d.set_slc(0 + i * (mantissa2 + exp2 + 1), input[i].float_val.d);
  }

  float_val = ac::bfloat16(temp);
}

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>::StdFloat(
    const StdFloat<7, 8> input[2]) {
  static_assert(
      (mantissa + exp + 1) == 2 * (7 + 8 + 1),
      "Lower precision type must be half the width of higher precision.");

#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    typename StdFloat<7, 8>::ac_std_float_rep temp =
        input[i].float_val.to_ac_std_float();
    float_val.d.set_slc(0 + i * (7 + 8 + 1), temp.d);
  }
}
*/
