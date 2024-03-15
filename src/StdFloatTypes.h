#pragma once

// clang-format off
#include <ac_std_float.h>
// clang-format on
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>

template <int mantissa, int exp>
class StdFloat {
 public:
  typedef ac_std_float<mantissa + exp + 1, exp> ac_float_rep;
  static constexpr unsigned int width = ac_float_rep::width;

  // TODO: make this a template parameter
  typedef StdFloat<mantissa, exp> AccumulationDatatype;

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

  template <int mantissa2, int exp2>
  StdFloat(const StdFloat<mantissa2, exp2> input[2]);

  StdFloat(const StdFloat<7, 8> input[2]);

  template <int mantissa2, int exp2>
  StdFloat(const StdFloat<mantissa2, exp2> &input);

  template <int W, bool S>
  StdFloat(const ac_int<W, S> &rhs);

  ac_int<mantissa + exp, false> bits_rep() { return float_val.d; }

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

  void exponent() {
    // convert to fixed point
    ac_float_to_fixed_rep converted_to_fixed = float_val.convert_to_ac_fixed();
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

#ifndef __SYNTHESIS__
template <int mantissa, int exp>
StdFloat<mantissa, exp>::StdFloat(const float val) {
  float_val = StdFloat<mantissa, exp>::ac_float_rep(val);
}
#endif

template <int mantissa, int exp>
template <int mantissa2, int exp2>
StdFloat<mantissa, exp>::StdFloat(const StdFloat<mantissa2, exp2> input[2]) {
  static_assert(
      (mantissa + exp + 1) == 2 * (mantissa2 + exp2 + 1),
      "Lower precision type must be half the width of higher precision.");
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    float_val.d.set_slc(0 + i * (mantissa2 + exp2 + 1), input[i].float_val.d);
  }
}

template <int mantissa, int exp>
StdFloat<mantissa, exp>::StdFloat(const ac_float_rep &input_float_rep) {
  float_val = input_float_rep;
}

template <int mantissa, int exp>
template <int mantissa2, int exp2>
StdFloat<mantissa, exp>::StdFloat(const StdFloat<mantissa2, exp2> &input) {
  float_val = static_cast<ac_float_rep>(input);
}

template <int mantissa, int exp>
template <int W, bool S>
StdFloat<mantissa, exp>::StdFloat(const ac_int<W, S> &rhs) {
  float_val = ac_float_rep(rhs);
}

template <int mantissa, int exp>
StdFloat<mantissa, exp> exponent(StdFloat<mantissa, exp> element) {
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

  return static_cast<StdFloat<mantissa, exp> >(exponent_in_float);
}

template <int mantissa, int exp>
inline StdFloat<mantissa, exp> StdFloat<mantissa, exp>::operator+(
    const StdFloat &rhs) {
  return float_val + rhs.float_val;
}

template <int mantissa, int exp>
inline StdFloat<mantissa, exp> StdFloat<mantissa, exp>::operator*(
    const StdFloat &rhs) {
  return float_val * rhs.float_val;
}

template <int mantissa, int exp>
inline StdFloat<mantissa, exp> StdFloat<mantissa, exp>::operator/(
    const StdFloat &rhs) {
  return float_val / rhs.float_val;
}

template <int mantissa, int exp>
inline StdFloat<mantissa, exp> &StdFloat<mantissa, exp>::operator+=(
    const StdFloat &rhs) {
  float_val += rhs.float_val;
  return *this;
}

template <int mantissa, int exp>
inline StdFloat<mantissa, exp> &StdFloat<mantissa, exp>::operator-=(
    const StdFloat &rhs) {
  float_val -= rhs.float_val;
  return *this;
}

template <int mantissa, int exp>
inline StdFloat<mantissa, exp> &StdFloat<mantissa, exp>::operator*=(
    const StdFloat &rhs) {
  float_val *= rhs.float_val;
  return *this;
}

template <int mantissa, int exp>
inline StdFloat<mantissa, exp> &StdFloat<mantissa, exp>::operator/=(
    const StdFloat &rhs) {
  float_val /= rhs.float_val;
  return *this;
}

template <int mantissa, int exp>
inline bool StdFloat<mantissa, exp>::operator<(const StdFloat &rhs) const {
  return float_val < rhs.float_val;
}

template <int mantissa, int exp>
template <int mantissa2, int exp2>
StdFloat<mantissa2, exp2> StdFloat<mantissa, exp>::fma(
    StdFloat<mantissa, exp> &b, StdFloat<mantissa2, exp2> &c) {
  StdFloat<mantissa2, exp2> a_higherprecision(*this);
  StdFloat<mantissa2, exp2> b_higherprecision(b);

  return a_higherprecision.float_val.template fma<AC_TRN_ZERO, true>(
      b_higherprecision.float_val, c.float_val);
}

template <int mantissa_i, int exp_i, int mantissa_o, int exp_o>
typename StdFloat<mantissa_o, exp_o>::AccumulationDatatype decomposed_fma(
    const typename StdFloat<mantissa_i, exp_i>::AccumulationDatatype &a,
    const typename StdFloat<mantissa_i, exp_i>::AccumulationDatatype &b,
    const typename StdFloat<mantissa_o, exp_o>::AccumulationDatatype &c) {
  StdFloat<mantissa_o, exp_o> a_higherprecision(a);
  StdFloat<mantissa_o, exp_o> b_higherprecision(b);

  return a_higherprecision.float_val.template fma<AC_TRN_ZERO, true>(
      b_higherprecision.float_val, c.float_val);
}

template <>
class StdFloat<7, 8> {
 public:
  typedef ac::bfloat16 ac_float_rep;
  static constexpr unsigned int width = ac_float_rep::width;
  static constexpr unsigned int mantissa = 16;
  static constexpr unsigned int exp = 8;

  // TODO: make this a template parameter
  typedef StdFloat<mantissa, exp> AccumulationDatatype;

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
    //                                                                     output);
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
// StdFloat<mantissa2, exp2> StdFloat<mantissa, exp>::fma(
//     StdFloat<mantissa, exp> &b, StdFloat<mantissa2, exp2> &c) {
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

template <int mantissa, int exp>
StdFloat<mantissa, exp>::StdFloat(const StdFloat<7, 8> input[2]) {
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
