// This file contains the custom Posting implementation for the accelerator.
// Posits are a new number format, similar to floats, but with more precision
// around 1 and higher dynamic range
// https://www.johndcook.com/blog/2018/04/11/anatomy-of-a-posit-number/
// It's roughly modeled after the Universal Numbers Library posit implementation
// https://github.com/stillwater-sc/universal
// but with more emphasis on a lightweight implementation.
// This implementation mainly consists of two classes: Posit and PositFP
// PositFP is a decoded version of Posit used for the internal calculations
// also referred to as DecomposedPosit.

// TODO(fpedd): Maybe clean this up a little according to
// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading

#pragma once

#include <ac_float.h>
#include <ac_int.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>

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
  if (es > 0) {
    if (scale < -(nbits - 1) * (1 << es) + (1 << (es - 1))) {
      bits = 0;
      return;
    }
  } else if (es == 0 && scale < -(nbits - 1)) {
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
    ac_int<nbits + 3 + es, false> pt_bits, regime, exponent, fraction,
        sticky_bit;

    bool r = (scale >= 0);
    int run = r ? (1 + (scale >> es)) : -(scale >> es);
    regime = r ? (1l << (run + 1)) - 1 : 0;
    regime[0] = ~r;

    exponent = scale % (uint32_t(1) << es);

    int nf = max(0, nbits + 1 - (2 + run + es));
    fraction = fraction_in >> max(fbits - nf, 0);
    fraction <<= max(nf - fbits, 0);

    regime <<= es + nf + 1;
    exponent <<= nf + 1;
    fraction <<= 1;
    sticky_bit = fraction_in << nf ? 1 : 0;
    pt_bits = regime | exponent | fraction | sticky_bit;

    int len = 1 + max(nbits + 1, 2 + run + es);
    bool blast = pt_bits[len - nbits];
    bool bafter = pt_bits[len - nbits - 1];
    bool bsticky = pt_bits & ((1 << (len - nbits - 1)) - 1);
    bool rb = (blast & bafter) | (bafter & bsticky);

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
  static constexpr int sbits = ac::log2_ceil<nbits - 2>::val + es + 1;
  static constexpr int fbits = nbits - 3 - es;
  typedef PositFP<8, fbits> DecomposedPosit;

  ac_int<nbits, false> bits;

  Posit() {}
  Posit(const float f);

  template <int nbits2, int es2>
  Posit(const Posit<nbits2, es2> &input);

  template <int fp_sbits, int fp_fbits>
  Posit(const PositFP<fp_sbits, fp_fbits> &input);

  bool isZero() const { return bits == 0; }

  void setbits(int i) { bits = i; }
  void setZero() { bits = 0; }

  void relu() {
    if (bits[nbits - 1] == 1) bits = 0;
  }

  void negate() { twos_complement(bits); }

  void reciprocal() {
    ac_int<nbits, false> sub = (1 << (nbits - 1));
    bits = sub - bits;
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

  operator float() const;

// SystemC is not compatible with C++17
#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &bits;
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
  typename Posit<nbits2, es2>::DecomposedPosit tmp(input);
  *this = tmp;
}

template <int nbits, int es>
template <int fp_sbits, int fp_fbits>
Posit<nbits, es>::Posit(const PositFP<fp_sbits, fp_fbits> &input) {
  if (input.isZero()) {
    bits = 0;
  } else {
    convert_<nbits, es, fp_fbits>(input.sign, input.scale, input.fraction,
                                  bits);
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
Posit<nbits, es> posit_exp(Posit<nbits, es> val) {
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
  DecomposedPosit op1 = *this;
  DecomposedPosit op2 = rhs;
  return op1 + op2;
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator*(
    const Posit<nbits, es> &rhs) {
  DecomposedPosit op1 = *this;
  DecomposedPosit op2 = rhs;
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
  DecomposedPosit op1 = *this;
  DecomposedPosit op2 = rhs;
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

#ifndef __SYNTHESIS__
template <int nbits, int es>
Posit<nbits, es>::operator float() const {
  PositFP<8, 23> fp(*this);
  return (float)fp;
}
#endif

template <int nbits, int es>
inline std::ostream &operator<<(std::ostream &os, const Posit<nbits, es> &val) {
#ifndef __SYNTHESIS__
  os << static_cast<float>(val) << " ";
#else
  os << val.bits << " ";
#endif
  return os;
}

/*
 * Intermediate representation used for MAC
 */
template <int sbits, int fbits>
class PositFP {
 public:
  static constexpr int fhbits = fbits + 1;  // size of fraction + hidden bit
  static constexpr int abits = fhbits + 3;  // size of the addend
  static constexpr int mbits = 2 * fhbits;  // size of the multiplier output
  static const unsigned int width = 1 + sbits + fbits + 1;

  // ac_float<W,I,E>- mantissa: ac_fixed<W,I,true> exp: ac_int<E,true>
  typedef ac_float<fbits + 2, 2, sbits> ac_float_rep;

  // typedef ac_std_float<fbits + sbits, sbits> ac_std_float_rep;

  ac_int<1, false> sign;
  ac_int<sbits, true> scale;
  ac_int<fbits, false> fraction;
  bool _zero;

  PositFP() { _zero = false; }

#pragma hls_design ccore
#pragma ccore_type combinational
  template <int nbits, int es>
  PositFP(const Posit<nbits, es> &input);
  PositFP(ac_float_rep ac_f);
  PositFP(const float f);

  bool isZero() const { return _zero; }
  void setZero() { _zero = true; }

  ac_int<fhbits, false> get_fixed_point() const {
    ac_int<fhbits, false> bits = fraction;
    bits[fhbits - 1] = 1;
    return bits;
  }

  void negate() { sign = !sign; }

  void relu() {
    if (sign == 1) setZero();
  }

  void masked_relu(const PositFP &mask) {
    if (mask.isZero()) setZero();
  }

  PositFP<sbits, abits + 1> operator+(const PositFP &rhs) const;
  PositFP<sbits, abits + 1> operator-(const PositFP &rhs) const;
  PositFP<sbits, mbits> operator*(const PositFP &rhs) const;
  PositFP log_mult(const PositFP &rhs) const;

  PositFP &operator+=(const PositFP &rhs);
  PositFP &operator-=(const PositFP &rhs);
  PositFP &operator*=(const PositFP &rhs);

  bool operator==(const PositFP &rhs) const;
  bool operator!=(const PositFP &rhs) const;
  bool operator<(const PositFP &rhs) const;
  bool operator>(const PositFP &rhs) const;
  bool operator<=(const PositFP &rhs) const;
  bool operator>=(const PositFP &rhs) const;

  template <int sbits2, int fbits2>
  explicit operator PositFP<sbits2, fbits2>() const;
  explicit operator ac_float_rep() const;
  explicit operator float() const;

  PositFP inv_sqrt() {
    ac_float_rep ac_f = static_cast<ac_float_rep>(*this);
    ac_float_rep ac_f_inv_sqrt;

    ac_math::ac_inverse_sqrt_pwl(ac_f, ac_f_inv_sqrt);

    return PositFP(ac_f_inv_sqrt);
  }

  // SystemC is not compatible with C++17
#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &sign;
    m &scale;
    m &fraction;
    m &_zero;
  }

  inline friend void sc_trace(sc_trace_file *tf, const PositFP &posit,
                              const std::string &name) {}
#endif
};

template <int sbits, int fbits>
template <int nbits, int es>
PositFP<sbits, fbits>::PositFP(const Posit<nbits, es> &input) {
  if (input.isZero()) {
    setZero();
  } else {
    bool sign;
    int scale;
    decode<nbits, es, fbits>(input.bits, sign, scale, fraction);

    this->_zero = false;
    this->sign = sign;
    this->scale = scale;
  }
}

template <int sbits, int fbits>
PositFP<sbits, fbits>::PositFP(ac_float_rep ac_f) {
  // only works when ac_f > 1
  _zero = false;
  sign = 0;
  scale = ac_f.exp();

  ac_fixed<fbits + 1, 1, false> mantissa = ac_f.m;
  ac_fixed<fbits, 0, false> mantissa_hidden = mantissa;
  fraction = mantissa_hidden.template slc<fbits>(0);
}

#ifndef __SYNTHESIS__
template <int sbits, int fbits>
PositFP<sbits, fbits>::PositFP(const float f) {
  if (f == 0) {
    setZero();
    return;
  }

  union ufloat uf = {f};
  _zero = false;
  sign = f < 0;
  scale = ((uf.u >> 23) & 0xff) - 127;
  // TODO: Add rounding
  ac_int<23, false> mantissa = uf.u;
  copy_(mantissa, fraction);
}
#endif

template <int sbits, int fbits>
PositFP<sbits, PositFP<sbits, fbits>::abits + 1>
PositFP<sbits, fbits>::operator+(const PositFP<sbits, fbits> &rhs) const {
  PositFP<sbits, fbits> lhs = *this;
  PositFP<sbits, abits + 1> result;

  if (lhs.isZero()) {
    return PositFP<sbits, abits + 1>(rhs);
  }

  if (rhs.isZero()) {
    return PositFP<sbits, abits + 1>(lhs);
  }

  ac_int<sbits, true> lhs_scale = lhs.scale;
  ac_int<sbits, true> rhs_scale = rhs.scale;
  ac_int<sbits, true> result_scale = max(lhs_scale, rhs_scale);

  // align the fraction
  ac_int<abits, false> lhs_fraction = lhs.get_fixed_point();
  ac_int<abits, false> rhs_fraction = rhs.get_fixed_point();
  ac_int<abits, false> r1 = lhs_fraction << (lhs_scale - result_scale + 3);
  r1[0] = (lhs_fraction << (abits + 2 + lhs_scale - result_scale)).or_reduce();
  ac_int<abits, false> r2 = rhs_fraction << (rhs_scale - result_scale + 3);
  r2[0] = (rhs_fraction << (abits + 2 + rhs_scale - result_scale)).or_reduce();

  bool r1_sign = lhs.sign;
  bool r2_sign = rhs.sign;
  bool signs_are_different = r1_sign != r2_sign;
  bool lhs_slt_rhs =
      (lhs_scale < rhs_scale) || (lhs_scale == rhs_scale && r1 < r2);

  if (signs_are_different && lhs_slt_rhs) {
    swap_(r1, r2);
    swap_(r1_sign, r2_sign);
  }

  ac_int<abits + 1, false> sum =
      signs_are_different ? static_cast<ac_int<abits + 1, false>>(r1 - r2)
                          : r1 + r2;
  ac_int<sbits, true> shift = sum.leading_sign() - 1;

  if (shift >= abits) {
    result.setZero();
    return result;
  }

  result._zero = false;
  result.sign = r1_sign;
  result.scale = result_scale - shift;
  result.fraction = sum << (shift + 2);

  return result;
}

template <int sbits, int fbits>
PositFP<sbits, PositFP<sbits, fbits>::abits + 1>
PositFP<sbits, fbits>::operator-(const PositFP<sbits, fbits> &rhs) const {
  if (rhs.isZero()) {
    return PositFP<sbits, abits + 1>(*this);
  }

  PositFP<sbits, fbits> op_neg;
  op_neg._zero = false;
  op_neg.sign = !rhs.sign;
  op_neg.scale = rhs.scale;
  op_neg.fraction = rhs.fraction;
  return *this + op_neg;
}

template <int sbits, int fbits>
PositFP<sbits, PositFP<sbits, fbits>::mbits> PositFP<sbits, fbits>::operator*(
    const PositFP<sbits, fbits> &rhs) const {
  PositFP<sbits, fbits> lhs = *this;
  PositFP<sbits, mbits> result;

  if (lhs.isZero() || rhs.isZero()) {
    result.setZero();
    return result;
  }

  result._zero = false;
  result.sign = lhs.sign ^ rhs.sign;
  result.scale = lhs.scale + rhs.scale;
  result.fraction = lhs.get_fixed_point() * rhs.get_fixed_point();

  if (result.fraction[mbits - 1]) {
    result.scale++;
    result.fraction <<= 1;
  } else {
    result.fraction <<= 2;
  }

  return result;
}

// TODO(fpedd): This is now deprecated ?!
template <int sbits, int fbits>
PositFP<sbits, fbits> PositFP<sbits, fbits>::log_mult(
    const PositFP<sbits, fbits> &rhs) const {
  PositFP<sbits, fbits> lhs = *this;
  PositFP<sbits, fbits> result;

  if (lhs.isZero() || rhs.isZero()) {
    result.setZero();
    return result;
  }

  result.sign = lhs.sign ^ rhs.sign;
  result.scale = lhs.scale + rhs.scale;

  std::cout << (float)lhs << std::endl;

  std::cout << "Scale: " << lhs.scale << " + " << rhs.scale << std::endl;
  std::cout << "Fraction: " << lhs.fraction << " + " << rhs.fraction
            << std::endl;

  ac_int<fbits + 1, false> fractionSum = lhs.fraction + rhs.fraction;
  if (fractionSum[fbits]) {
    result.scale++;
    // fractionSum <<= 1;
  }
  result.fraction.set_slc(0, fractionSum.template slc<fbits>(0));

  ac_int<mbits, false> exactFraction =
      lhs.get_fixed_point() * rhs.get_fixed_point();
  if (exactFraction[mbits - 1]) {
    exactFraction <<= 1;
  } else {
    exactFraction <<= 2;
  }

  std::cout << "approx: " << result.fraction << std::endl;
  std::cout << "exact: " << exactFraction << std::endl;

  // copy_(fractionSum, result.fraction);

  return result;
}

template <int sbits, int fbits>
inline PositFP<sbits, fbits> &PositFP<sbits, fbits>::operator+=(
    const PositFP<sbits, fbits> &rhs) {
  *this = PositFP<sbits, fbits>(*this + rhs);
  return *this;
}

template <int sbits, int fbits>
inline PositFP<sbits, fbits> &PositFP<sbits, fbits>::operator-=(
    const PositFP<sbits, fbits> &rhs) {
  *this = PositFP<sbits, fbits>(*this - rhs);
  return *this;
}

template <int sbits, int fbits>
inline PositFP<sbits, fbits> &PositFP<sbits, fbits>::operator*=(
    const PositFP<sbits, fbits> &rhs) {
  *this = PositFP<sbits, fbits>(*this * rhs);
  return *this;
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator==(const PositFP<sbits, fbits> &rhs) const {
  if (isZero() || rhs.isZero()) {
    return isZero() && rhs.isZero();
  } else {
    return sign == rhs.sign && scale == rhs.scale && fraction == rhs.fraction;
  }
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator!=(const PositFP<sbits, fbits> &rhs) const {
  return !(*this == rhs);
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator<(const PositFP<sbits, fbits> &rhs) const {
  if (_zero) {
    return !rhs._zero && !rhs.sign;
  }

  if (rhs._zero) {
    return sign;
  }

  if (sign ^ rhs.sign) {
    return sign;
  }

  if (scale > rhs.scale) {
    return sign;
  } else if (scale == rhs.scale) {
    if (fraction == rhs.fraction) return false;
    if (fraction > rhs.fraction) {
      return sign;
    }
  }

  return !sign;
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator<=(const PositFP<sbits, fbits> &rhs) const {
  return *this < rhs || *this == rhs;
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator>(const PositFP<sbits, fbits> &rhs) const {
  return !(*this <= rhs);
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator>=(const PositFP<sbits, fbits> &rhs) const {
  return !(*this < rhs);
}

template <int sbits, int fbits>
template <int sbits2, int fbits2>
PositFP<sbits, fbits>::operator PositFP<sbits2, fbits2>() const {
  PositFP<sbits2, fbits2> fp;
  if (isZero()) {
    fp.setZero();
    return fp;
  }

  fp._zero = false;
  fp.sign = sign;
  fp.scale = scale;

  if (fbits <= fbits2) {
    copy_(fraction, fp.fraction);
  } else {
    ac_int<fbits2 + 1, false> frac_in;
    frac_in = fraction.template slc<fbits2>(fbits - fbits2);
    bool last = frac_in[0];
    bool guard = fraction[fbits - fbits2 - 1];
    bool sticky = fraction << (fbits2 + 1);
    if (guard & (last | sticky)) {
      frac_in++;
      if (frac_in[fbits2]) fp.scale++;
    }
    fp.fraction = frac_in;
  }

  return fp;
}

template <int sbits, int fbits>
PositFP<sbits, fbits>::operator ac_float_rep() const {
  // std::cout << "PositFP to ac_float " << std::endl;
  ac_fixed<fbits, 0, false> mantissa_hidden;
  mantissa_hidden.set_slc(0, fraction);

  ac_fixed<fbits + 1, 1, false> mantissa_abs = mantissa_hidden;
  if (!_zero) {
    mantissa_abs = mantissa_abs + 1;
  }
  ac_fixed<fbits + 2, 2, true> mantissa;
  if (sign) {
    mantissa = mantissa_abs * (-1);
  } else {
    mantissa = mantissa_abs;
  }

  // std::cout << (float)(*this ) << " -> " <<
  // ac_float_rep(mantissa,scale).to_float() << std::endl;

  return ac_float_rep(mantissa, scale);
}

#ifndef __SYNTHESIS__
template <int sbits, int fbits>
PositFP<sbits, fbits>::operator float() const {
  if (isZero()) return 0;

  union ufloat uf;
  uf.u = sign ? 1 << 31 : 0;
  uf.u += (scale + 127) << 23;

  // TODO: Add rounding
  ac_int<23, false> mantissa;
  copy_(fraction, mantissa);
  uf.u += mantissa;
  return uf.f;
}
#endif

template <int nbits, int es, int nbits2, int es2>
typename Posit<nbits2, es2>::DecomposedPosit decomposed_fma(
    const typename Posit<nbits, es>::DecomposedPosit &a,
    const typename Posit<nbits, es>::DecomposedPosit &b,
    const typename Posit<nbits2, es2>::DecomposedPosit &c) {
  constexpr size_t fbits = nbits - 3 - es;
  constexpr size_t fhbits = fbits + 1;  // size of fraction + hidden bit
  constexpr size_t mbits = 2 * fhbits;  // size of the multiplier output

  constexpr size_t fbits2 = nbits2 - 3 - es2;
  constexpr size_t abits = fbits2 + 4;  // size of the addend

  if (a.isZero() || b.isZero()) {
    return c;
  }

  PositFP<8, mbits> product = a * b;
  if (c.isZero()) {
    return typename Posit<nbits2, es2>::DecomposedPosit(product);
  } else {
    PositFP<8, abits + 1> sum = PositFP<8, fbits2>(product) + c;
    return typename Posit<nbits2, es2>::DecomposedPosit(sum);
  }
}

template <int sbits, int fbits1, int fbits2>
PositFP<sbits, fbits2> fma(const PositFP<sbits, fbits1> &a,
                           const PositFP<sbits, fbits1> &b,
                           const PositFP<sbits, fbits2> &c) {
  constexpr size_t fhbits = fbits1 + 1;  // size of fraction + hidden bit
  constexpr size_t mbits = 2 * fhbits;   // size of the multiplier output
  constexpr size_t abits = fbits2 + 4;   // size of the addend

  if (a.isZero() || b.isZero()) {
    return c;
  }

  PositFP<sbits, mbits> product = a * b;
  if (c.isZero()) {
    return PositFP<sbits, fbits2>(product);
  } else {
    PositFP<8, abits + 1> sum = PositFP<sbits, fbits2>(product) + c;
    return PositFP<sbits, fbits2>(sum);
  }
}

template <int sbits, int fbits>
inline std::ostream &operator<<(std::ostream &os,
                                const PositFP<sbits, fbits> &val) {
#ifndef __SYNTHESIS__
  os << static_cast<float>(val) << " ";
#endif
  return os;
}

template <int sbits, int fbits>
inline bool operator==(const PositFP<sbits, fbits> &lhs,
                       const PositFP<sbits, fbits> &rhs) {
  return (lhs.sign == rhs.sign) && (lhs.scale == rhs.scale) &&
         (lhs.fraction == rhs.fraction);
}

template <int nbits, int es>
inline bool operator==(const Posit<nbits, es> &lhs,
                       const Posit<nbits, es> &rhs) {
  return lhs.bits == rhs.bits;
}