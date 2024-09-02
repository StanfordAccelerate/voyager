// This file contains the custom Posting implementation for the accelerator.
// Posits are a new number format, similar to floats, but with more precision
// around 1 and higher dynamic range
// https://www.johndcook.com/blog/2018/04/11/anatomy-of-a-posit-number/
// It's roughly modeled after the Universal Numbers Library posit implementation
// https://github.com/stillwater-sc/universal
// but with more emphasis on a lightweight implementation.
// This implementation mainly consists of two classes: Posit and PositFP
// PositFP is a decoded version of Posit used for the internal calculations
// also referred to as AccumulationDatatype.

// TODO(fpedd): Maybe clean this up a little according to
// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading

#pragma once
#include <stdint.h>
// #include <ac_float.h>
#include <ac_int.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_sqrt_pwl.h>
#include <ac_std_float.h>

inline int max(int a, int b) { return a > b ? a : b; }

template <class T>
void swap_(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <int srcWidth, int dstWidth>
void copy_(ac_int<srcWidth, false> src, ac_int<dstWidth, false> &dst) {
  if (dstWidth > srcWidth) {
    dst = src;
    dst <<= dstWidth - srcWidth;
  } else {
    dst = src.template slc<srcWidth>(srcWidth - dstWidth);
  }
}

template <int nbits>
void twos_complement(ac_int<nbits, false> &value) {
  value = static_cast<ac_int<nbits, false>>(value.bit_complement() + 1);
}

template <int nbits, int es, int fbits>
void convert_(const bool sign, const int scale,
              const ac_int<fbits, false> fraction_in,
              ac_int<nbits, false> &bits) {
  // if ((es > 0 && scale < -(nbits - 1) * (1 << es) + (1 << max(es - 1, 0))) ||
  //     (es == 0 && scale < -(nbits - 1))) {
  //   bits = 0;
  //   return;
  // }

  if (nbits == 8 && es == 1 && scale < -13) {
    bits = 0;
    return;
  }

  if (nbits == 16 && es == 1 && scale < -29) {
    bits = 0;
    return;
  }

  if ((scale >> es) > nbits - 2 || (scale >> es) < -(nbits - 2)) {
    bits = 0;
    if (scale > 0) {
      bits = bits.bit_complement();
      bits[nbits - 1] = 0;
    } else {
      bits[0] = 1;
    }
  } else {
    ac_int<nbits + 1 + es + fbits, false> pt_bits, regime, exponent;

    bool r = (scale >= 0);
    int run = r ? (1 + (scale >> es)) : -(scale >> es);
    regime = r ? (1 << (run + 1)) - 2 : 1;
    exponent = scale % (uint32_t(1) << es);
    pt_bits = (regime << fbits + es) | (exponent << fbits) | fraction_in;

    bool rb = false;
    int len = 2 + run + es + fbits;
    if (len > nbits) {
      bool blast = pt_bits[len - nbits];
      bool bafter = pt_bits[len - nbits - 1];
      bool bsticky = pt_bits << (2 * nbits - run);
      rb = (blast & bafter) | (bafter & bsticky);
    }

    bits = pt_bits >> (len - nbits);
    if (rb) bits++;
  }
  if (sign) twos_complement(bits);
}

template <int nbits, int es, int fbits>
void decode(ac_int<nbits, false> bits, bool &sign, int &scale,
            ac_int<fbits, false> &fraction) {
  if (bits == 0) {
    sign = 0;
    scale = -127;
    fraction = 0;
  } else {
    sign = bits[nbits - 1];
    if (sign) twos_complement(bits);  // convert to positive value
    bits <<= 1;                       // remove sign bit

    bool leadingBit = bits[nbits - 1];
    int run =
        leadingBit ? bits.bit_complement().leading_sign() : bits.leading_sign();
    scale = (leadingBit ? run - 1 : -run) * (1 << es);

    int nrBits = nbits - run - 1;
    if (es > 0) {
      if (nrBits >= es) {
        scale += bits.template slc<es>(nrBits - es);
      } else if (nrBits >= 0) {
        scale += bits & ((1 << nrBits) - 1);
      }
    }

    bits <<= run + 1 + es;
    copy_(bits, fraction);
  }
}

#ifndef __SYNTHESIS__
// Used for float <-> posit conversion
union ufloat {
  float f;
  uint32_t u;
};
#endif

// forward declaration
template <int sbits, int fbits>
class PositFP;

template <int nbits, int es>
class Posit {
 public:
  static constexpr int width = nbits;
  static constexpr int esbits = es;
  static constexpr int max_exp = (nbits - 2) * (1 << es);
  static constexpr int sbits = ac::nbits<max_exp>::val + 1;
  static constexpr int fbits = nbits - 3 - es;
  typedef PositFP<sbits, fbits> AccumulationDatatype;

  ac_int<nbits, false> bits;

  Posit() {}

#ifndef __SYNTHESIS__
  Posit(const float f);
#endif

  template <int W, bool S>
  Posit(const ac_int<W, S> &rhs);

  template <int nbits2, int es2>
  Posit(const Posit<nbits2, es2> &input);

  template <int nbits2, int es2>
  Posit(const Posit<nbits2, es2> input[2]);

  template <int fp_sbits, int fp_fbits>
  Posit(const PositFP<fp_sbits, fp_fbits> &input);

  ac_int<nbits, false> bits_rep() { return bits; }

  bool isZero() const { return bits == 0; }

  void setbits(int i) { bits = i; }
  void setZero() { bits = 0; }

  void relu() {
    if (bits[nbits - 1] == 1) bits = 0;
  }

  void negate() { twos_complement(bits); }

  void reciprocal() {
    // ac_int<nbits, false> sub = (1 << (nbits - 1));
    // bits = sub - bits;
    ac_int<nbits, false> signmask = 1 << (nbits - 1);
    bits = bits ^ ~signmask;
  }

  void sigmoid() {
    bits[nbits - 1] = ~bits[nbits - 1];  // invert MSB
    bits >>= 2;
  }

  Posit operator+(const Posit &rhs);
  Posit operator*(const Posit &rhs);
  Posit operator/(const Posit &rhs);
  Posit log_mult(const Posit &rhs);

  Posit &operator+=(const Posit &rhs);
  Posit &operator-=(const Posit &rhs);
  Posit &operator*=(const Posit &rhs);
  Posit &operator/=(const Posit &rhs);

  bool operator<(const Posit &rhs) const;

#ifndef __SYNTHESIS__
  operator float() const {
    AccumulationDatatype tmp(*this);
    return static_cast<float>(tmp);
  }
#endif

// SystemC is not compatible with C++17
#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & bits;
  }

  inline friend void sc_trace(sc_trace_file *tf, const Posit &posit,
                              const std::string &name) {
    sc_trace(tf, posit.bits, name + ".bits");
  }
#endif
};

template <int nbits, int es>
template <int nbits2, int es2>
Posit<nbits, es>::Posit(const Posit<nbits2, es2> &input) {
  typename Posit<nbits2, es2>::AccumulationDatatype tmp(input);
  *this = tmp;
}

template <int nbits, int es>
template <int nbits2, int es2>
Posit<nbits, es>::Posit(const Posit<nbits2, es2> input[2]) {
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    bits.set_slc(i * nbits2, input[i].bits);
  }
}

template <int nbits, int es>
template <int W, bool S>
Posit<nbits, es>::Posit(const ac_int<W, S> &rhs) {
  bits = rhs;
}

template <int nbits, int es>
template <int fp_sbits, int fp_fbits>
Posit<nbits, es>::Posit(const PositFP<fp_sbits, fp_fbits> &input) {
  if (input.isZero()) {
    bits = 0;
  } else {
    const int e_width = PositFP<fp_sbits, fp_fbits>::e_width;
    const int mant_bits = PositFP<fp_sbits, fp_fbits>::mant_bits;
    const int exp_bias = PositFP<fp_sbits, fp_fbits>::exp_bias;

    // bool sign = input.float_val.signbit();
    // int scale = input.float_val.d.template slc<e_width>(mant_bits) - exp_bias;
    // ac_int<mant_bits, false> fraction = input.float_val.d;
    // FIXME: ac::bfloat16 version
    short d = input.float_val.d;
    bool sign = d < 0;
    int scale = ((d >> mant_bits) & 0xff) - exp_bias;
    ac_int<mant_bits, false> fraction = d & 0x7f;
    convert_<nbits, es, mant_bits>(sign, scale, fraction, bits);
  }
}

#ifndef __SYNTHESIS__
template <int nbits, int es>
Posit<nbits, es>::Posit(const float f) {
  if (f == 0) {
    bits = 0;
  } else {
    union ufloat uf = {f};
    bool sign = f < 0;
    int scale = ((uf.u >> 23) & 0xFF) - 127;
    ac_int<23, false> fraction = uf.u;
    convert_<nbits, es, 23>(sign, scale, fraction, bits);
  }
}
#endif

// Implements an optimized exp that is only valid for inputs smaller than zero
template <int nbits, int es>
Posit<nbits, es> exponential(Posit<nbits, es> val) {
  assert(nbits == 16 && es == 1);

  Posit<nbits, es> min_exp;
  min_exp.setbits(0x9e00);

  if (val < min_exp) {
    Posit<nbits, es> zero;
    zero.bits.template set_val<AC_VAL_0>();
    return zero;
  }

  val.negate();

  Posit<16, 0> es0Posit(val);
  es0Posit.sigmoid();
  es0Posit.reciprocal();

  Posit<16, 0> neg_one;
  neg_one.setbits(0xc000);

  return Posit<nbits, es>(es0Posit + neg_one);
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator+(
    const Posit<nbits, es> &rhs) {
  AccumulationDatatype op1 = *this;
  AccumulationDatatype op2 = rhs;
  return op1 + op2;
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator*(
    const Posit<nbits, es> &rhs) {
  AccumulationDatatype op1 = *this;
  AccumulationDatatype op2 = rhs;
  return op1 * op2;
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator/(
    const Posit<nbits, es> &rhs) {
  Posit<nbits, es> op2 = rhs;
  op2.reciprocal();
  return *this * op2;
}

// TODO(fpedd): Deprecated, no need for log mult anymore?!
template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::log_mult(
    const Posit<nbits, es> &rhs) {
  AccumulationDatatype op1 = *this;
  AccumulationDatatype op2 = rhs;
  return op1.log_mult(op2);
}

template <int nbits, int es>
inline Posit<nbits, es> &Posit<nbits, es>::operator+=(
    const Posit<nbits, es> &rhs) {
  *this = *this + rhs;
  return *this;
}

template <int nbits, int es>
inline Posit<nbits, es> &Posit<nbits, es>::operator-=(
    const Posit<nbits, es> &rhs) {
  Posit<nbits, es> op2 = rhs;
  op2.negate();
  *this = *this + op2;
  return *this;
}

template <int nbits, int es>
inline Posit<nbits, es> &Posit<nbits, es>::operator*=(
    const Posit<nbits, es> &rhs) {
  *this = *this * rhs;
  return *this;
}

template <int nbits, int es>
inline Posit<nbits, es> &Posit<nbits, es>::operator/=(
    const Posit<nbits, es> &rhs) {
  *this = *this / rhs;
  return *this;
}

template <int nbits, int es>
inline bool Posit<nbits, es>::operator<(const Posit<nbits, es> &rhs) const {
  bool lhs_sign = bits[nbits - 1];
  bool rhs_sign = rhs.bits[nbits - 1];
  if (lhs_sign ^ rhs_sign) {
    return lhs_sign;
  }

  ac_int<nbits, false> lhs_fraction = bits;
  ac_int<nbits, false> rhs_fraction = rhs.bits;
  if (lhs_sign) {
    twos_complement(lhs_fraction);
    twos_complement(rhs_fraction);
    return lhs_fraction > rhs_fraction;
  }

  return lhs_fraction < rhs_fraction;
}

template <int nbits, int es>
inline std::ostream &operator<<(std::ostream &os, const Posit<nbits, es> &val) {
#ifndef __SYNTHESIS__
  os << static_cast<float>(val) << " ";
#else
  os << val.bits << " ";
#endif
  return os;
}

template <int sbits, int fbits>
class PositFP {
 public:
  static const int width = 16;
  static const int e_width = 8;
  static const int mant_bits = 7;
  static const int exp_bias = (1 << (e_width - 1)) - 1;
  typedef ac::bfloat16 ac_float_t;
  typedef PositFP<sbits, fbits> AccumulationDatatype;

  ac_float_t float_val;

  PositFP(){};

#ifndef __SYNTHESIS__
  PositFP(const float &fval) : float_val(fval) {}
#endif

  PositFP(const ac::bfloat16 &f) : float_val(f) {}

  template <int W, int E>
  PositFP(const ac_std_float<W, E> &f) : float_val(f) {}

  template <int sbits2, int fbits2>
  PositFP(const PositFP<sbits2, fbits2> &f) : float_val(f.float_val) {}

  template <int nbits, int es>
  PositFP(const Posit<nbits, es> input[2]) {
    static_assert(
        (sbits + fbits + 1) == 2 * nbits,
        "Lower precision type must be half the width of higher precision.");
    ac_std_float<width, sbits> tmp;
#pragma hls_unroll yes
    for (int i = 0; i < 2; i++) {
      tmp.d.set_slc(i * nbits, input[i].bits);
    }
    *this = tmp;
  }

  template <int nbits, int es>
  PositFP(const Posit<nbits, es> &posit) {
    bool sign;
    int scale;
    ac_int<mant_bits, false> mantissa;
    decode<nbits, es, mant_bits>(posit.bits, sign, scale, mantissa);

    float_val.d = ((scale + exp_bias) << 7) | mantissa;
    if (sign) float_val = -float_val;
  }

  ac_int<sbits + fbits + 1, false> bits_rep() { return float_val.d; }

  void setbits(const int &i) { float_val.d = i; }

  void negate() { float_val.set_signbit(!float_val.signbit()); }

  void relu() {
    if (float_val.signbit()) setZero();
  }

  void masked_relu(const PositFP &mask) {
    if (mask.float_val.d == 0) setZero();
  }

  bool isZero() const { return float_val.d == 0; }

  void setZero() { float_val.d = 0; }

  void reciprocal() {
    Posit<16, 0> posit = *this;
    posit.reciprocal();
    *this = posit;
  }

  void exponential() {
    Posit<16, 0> posit16_0(*this);
    posit16_0.negate();
    posit16_0.sigmoid();
    posit16_0.reciprocal();

    *this = posit16_0;
    float_val -= ac_float_t(1.10925f);
    if (float_val.signbit()) {
      float_val.d = 0;
    }
  }

  PositFP inv_sqrt() {
    PositFP result = float_val.template sqrt<AC_RND_CONV, true>();
    result.reciprocal();
    return result;
  }

  PositFP sqrt() { return float_val.template sqrt<AC_RND_CONV, true>(); }

  PositFP max1() {
    ac_float_t one = float_val.one();
    return float_val > one ? float_val : one;
  }

  void expScale(ac_int<8, true> offset) {
    short exponent = ((float_val.d >> 7) & 0xff) + offset;
    float_val.d = (float_val.d & 0x807F) | exponent << 7;
  }

  template <int sbits2, int fbits2>
  PositFP<sbits2, fbits2> fma(const PositFP &b,
                              const PositFP<sbits2, fbits2> &c) {
    typename PositFP<sbits2, fbits2>::ac_float_t op1(float_val);
    typename PositFP<sbits2, fbits2>::ac_float_t op2(b.float_val);
    return op1.template fma<AC_RND_CONV, false>(op2, c.float_val);
  }

  template <int sbits2, int fbits2>
  explicit operator PositFP<sbits2, fbits2>() const {
    return PositFP<sbits2, fbits2>(float_val);
  }

#ifndef __SYNTHESIS__
  operator float() const { return float_val.to_float(); }
#endif

  template <int nbits, int es>
  void storeAsLowerPrecision(Posit<nbits, es> output[2]) {
    short d = float_val.d;

#pragma hls_unroll yes
    for (int i = 0; i < 2; i++) {
      output[i].bits = d & 0xFF;
      d >>= 8;
    }
  }

  PositFP operator+(const PositFP &op2) const {
    return float_val + op2.float_val;
  }

  PositFP operator-(const PositFP &op2) const {
    return float_val - op2.float_val;
  }

  PositFP operator*(const PositFP &op2) const {
    return float_val * op2.float_val;
  }

  PositFP &operator+=(const PositFP &op2) {
    float_val += op2.float_val;
    return *this;
  }

  PositFP &operator-=(const PositFP &op2) {
    float_val -= op2.float_val;
    return *this;
  }

  PositFP &operator*=(const PositFP &op2) {
    float_val *= op2.float_val;
    return *this;
  }

  bool operator<(const PositFP &rhs) const { return float_val < rhs.float_val; }

// SystemC is not compatible with C++17
#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & float_val.d;
  }

  inline friend void sc_trace(sc_trace_file *tf, const PositFP &posit,
                              const std::string &name) {}
#endif
};

template <int sbits, int fbits>
PositFP<sbits, fbits> exponential(const PositFP<sbits, fbits> &val) {
  Posit<16, 0> posit = static_cast<PositFP<sbits, fbits>>(-val.float_val);
  posit.sigmoid();
  posit.reciprocal();

  PositFP<sbits, fbits> result(posit);
  result.float_val -= typename PositFP<sbits, fbits>::ac_float_t(1.10925f);
  if (result.float_val.signbit()) {
    result.float_val.d = 0;
  }
  return result;
}

template <int sbits, int fbits>
inline std::ostream &operator<<(std::ostream &os,
                                const PositFP<sbits, fbits> &val) {
#ifndef __SYNTHESIS__
  os << static_cast<float>(val) << " ";
#endif
  return os;
}

// template <int sbits, int fbits>
// class PositFP {
//  public:
//   static const int width = sbits + fbits + 1;
//   static const int e_width = sbits;
//   static const int mant_bits = fbits;
//   typedef ac_std_float<width, sbits> ac_float_t;
//   typedef PositFP<sbits, fbits> AccumulationDatatype;

//   ac_float_t float_val;

//   PositFP(){};

// #ifndef __SYNTHESIS__
//   PositFP(const float &fval) : float_val(fval) {}
// #endif

//   template <int W, int E>
//   PositFP(const ac_std_float<W, E> &f) : float_val(f) {}

//   template <int sbits2, int fbits2>
//   PositFP(const PositFP<sbits2, fbits2> &f) : float_val(f.float_val) {}

//   template <int nbits, int es>
//   PositFP(const Posit<nbits, es> input[2]) {
//     static_assert(
//         (sbits + fbits + 1) == 2 * nbits,
//         "Lower precision type must be half the width of higher precision.");
// #pragma hls_unroll yes
//     for (int i = 0; i < 2; i++) {
//       float_val.d.set_slc(i * nbits, input[i].bits);
//     }
//   }

//   template <int nbits, int es>
//   PositFP(const Posit<nbits, es> &posit) {
//     bool sign;
//     int scale;
//     ac_int<mant_bits, false> mantissa;
//     decode<nbits, es, mant_bits>(posit.bits, sign, scale, mantissa);

//     ac_int<1, false> sign_bit = sign;
//     ac_int<e_width, true> exp = scale + ac_float_t::exp_bias;

//     float_val.d.set_slc(0, mantissa);
//     float_val.d.set_slc(mant_bits, exp);
//     float_val.d.set_slc(width - 1, sign_bit);
//   }

//   ac_int<sbits + fbits + 1, false> bits_rep() { return float_val.d; }

//   void setbits(const int &i) { float_val.d = i; }

//   void negate() { float_val.set_signbit(!float_val.signbit()); }

//   void relu() {
//     if (float_val.signbit()) setZero();
//   }

//   void masked_relu(const PositFP &mask) {
//     if (mask.float_val.d == 0) setZero();
//   }

//   bool isZero() const { return float_val.d == 0; }

//   void setZero() { float_val.d = 0; }

//   void reciprocal() {
//     Posit<16, 0> posit = *this;
//     posit.reciprocal();
//     *this = posit;
//   }

//   void exponent() {
//     Posit<16, 0> posit16_0 = static_cast<ac_float_t>(-float_val);
//     posit16_0.sigmoid();
//     posit16_0.reciprocal();

//     *this = posit16_0;
//     float_val -= ac_float_t(1.10925f);
//     if (float_val.signbit()) {
//       float_val.d = 0;
//     }
//   }

//   PositFP inv_sqrt() {
//     PositFP result = float_val.template sqrt<AC_RND_CONV, true>();
//     result.reciprocal();
//     return result;
//   }

//   PositFP sqrt() { return float_val.template sqrt<AC_RND_CONV, true>(); }

//   PositFP max1() {
//     ac_float_t one = float_val.one();
//     return float_val > one ? float_val : one;
//   }

//   void expScale(ac_int<8, true> offset) {
//     ac_int<e_width, true> exponent =
//         float_val.d.template slc<e_width>(mant_bits) + offset;
//     float_val.d.set_slc(mant_bits, exponent);
//   }

//   template <int sbits2, int fbits2>
//   PositFP<sbits2, fbits2> fma(const PositFP &b,
//                               const PositFP<sbits2, fbits2> &c) {
//     typename PositFP<sbits2, fbits2>::ac_float_t op1(float_val);
//     typename PositFP<sbits2, fbits2>::ac_float_t op2(b.float_val);
//     return op1.template fma<AC_RND_CONV, true>(op2, c.float_val);
//   }

//   template <int sbits2, int fbits2>
//   explicit operator PositFP<sbits2, fbits2>() const {
//     return PositFP<sbits2, fbits2>(float_val);
//   }

// #ifndef __SYNTHESIS__
//   operator float() const { return float_val.to_float(); }
// #endif

//   PositFP operator+(const PositFP &op2) const {
//     return float_val + op2.float_val;
//   }

//   PositFP operator-(const PositFP &op2) const {
//     return float_val - op2.float_val;
//   }

//   PositFP operator*(const PositFP &op2) const {
//     return float_val * op2.float_val;
//   }

//   PositFP &operator+=(const PositFP &op2) {
//     float_val += op2.float_val;
//     return *this;
//   }

//   PositFP &operator-=(const PositFP &op2) {
//     float_val -= op2.float_val;
//     return *this;
//   }

//   PositFP &operator*=(const PositFP &op2) {
//     float_val *= op2.float_val;
//     return *this;
//   }

//   bool operator<(const PositFP &rhs) const { return float_val <
//   rhs.float_val; }

// // SystemC is not compatible with C++17
// #ifndef NO_SYSC
//   template <unsigned int Size>
//   void Marshall(Marshaller<Size> &m) {
//     m & float_val.d;
//   }

//   inline friend void sc_trace(sc_trace_file *tf, const PositFP &posit,
//                               const std::string &name) {}
// #endif
// };

// template <int sbits, int fbits>
// PositFP<sbits, fbits> exponent(const PositFP<sbits, fbits> &val) {
//   Posit<16, 0> posit = static_cast<PositFP<sbits, fbits>>(-val.float_val);
//   posit.sigmoid();
//   posit.reciprocal();

//   PositFP<sbits, fbits> result(posit);
//   result.float_val -= typename PositFP<sbits, fbits>::ac_float_t(1.10925f);
//   if (result.float_val.signbit()) {
//     result.float_val.d = 0;
//   }
//   return result;
// }

// template <int sbits, int fbits>
// inline std::ostream &operator<<(std::ostream &os,
//                                 const PositFP<sbits, fbits> &val) {
// #ifndef __SYNTHESIS__
//   os << static_cast<float>(val) << " ";
// #endif
//   return os;
// }
