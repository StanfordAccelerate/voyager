// This file contains the custom Posting implementation for the accelerator.
// Posits are a new number format, similar to floats, but with more precision
// around 1 and higher dynamic range
// https://www.johndcook.com/blog/2018/04/11/anatomy-of-a-posit-number/
// It's roughly modeled after the Universal Numbers Library posit implementation
// https://github.com/stillwater-sc/universal
// but with more emphasis on a lightweight implementation.

#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

#include <ac_int.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_sqrt_pwl.h>
#include <ac_std_float.h>
#include <stdint.h>

#include "StdFloatTypes.h"

inline int max(int a, int b) { return a > b ? a : b; }

template <class T>
void swap_(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <int src_width, int dst_width>
void copy_(ac_int<src_width, false> src, ac_int<dst_width, false>& dst) {
  if (dst_width > src_width) {
    dst = src;
    dst <<= dst_width - src_width;
  } else {
    dst = src.template slc<src_width>(src_width - dst_width);
  }
}

template <int nbits>
void twos_complement(ac_int<nbits, false>& value) {
  value = static_cast<ac_int<nbits, false>>(value.bit_complement() + 1);
}

template <int nbits, int es, int fbits>
void convert_(const bool sign, const int scale,
              const ac_int<fbits, false> fraction_in,
              ac_int<nbits, false>& bits) {
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
void decode(ac_int<nbits, false> bits, bool& sign, int& scale,
            ac_int<fbits, false>& fraction) {
  sign = bits[nbits - 1];
  if (sign) twos_complement(bits);  // convert to positive value
  bits <<= 1;                       // remove sign bit

  bool leading_bit = bits[nbits - 1];
  int run =
      leading_bit ? bits.bit_complement().leading_sign() : bits.leading_sign();
  scale = (leading_bit ? run - 1 : -run) * (1 << es);

  int nr_bits = nbits - run - 1;
  if (es > 0) {
    if (nr_bits >= es) {
      scale += bits.template slc<es>(nr_bits - es);
    } else if (nr_bits >= 0) {
      scale += bits & ((1 << nr_bits) - 1);
    }
  }

  bits <<= run + 1 + es;
  copy_(bits, fraction);
}

#ifndef __SYNTHESIS__
// Used for float <-> posit conversion
union ufloat {
  float f;
  uint32_t u;
};
#endif

template <int nbits, int es>
class Posit {
 public:
  static constexpr unsigned int width = nbits;
  static constexpr unsigned int esbits = es;
  static constexpr int max_exp = (nbits - 2) * (1 << es);
  static constexpr unsigned int sbits = ac::nbits<max_exp>::val + 1;
  static constexpr unsigned int fbits = nbits - 3 - es;
  typedef StdFloat<fbits, sbits> decoded;

  ac_int<nbits, false> bits;

  Posit() {}

#ifndef __SYNTHESIS__
  Posit(const float f);
#endif

  template <int nbits2, int es2>
  Posit(const Posit<nbits2, es2>& other);

  template <int mantissa, int exp>
  Posit(const StdFloat<mantissa, exp>& other);

  ac_int<nbits, false> bits_rep() { return bits; }
  void set_bits(int i) { bits = i; }

  static Posit zero() {
    Posit<nbits, es> r;
    r.bits = 0;
    return r;
  }

  void negate() { twos_complement(bits); }

  void relu() {
    if (bits[nbits - 1] == 1) bits = 0;
  }

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

  Posit operator+(const Posit& rhs) const;
  Posit operator-(const Posit& rhs) const;
  Posit operator*(const Posit& rhs) const;
  Posit operator/(const Posit& rhs) const;

  Posit& operator+=(const Posit& rhs);
  Posit& operator-=(const Posit& rhs);
  Posit& operator*=(const Posit& rhs);
  Posit& operator/=(const Posit& rhs);

  bool operator<(const Posit& rhs) const;

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>() const {
    using std_float_t =
        StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>;
    std_float_t f;
    if (bits == 0) {
      f = std_float_t::zero();
    } else {
      bool sign;
      int scale;
      ac_int<mantissa, false> mantissa_bits;
      decode<nbits, es, mantissa>(bits, sign, scale, mantissa_bits);

      ac_int<1, false> sign_bit = sign;
      ac_int<exp, true> exp_bits = scale + std_float_t::ac_float_rep::exp_bias;

      f.float_val.d.set_slc(0, mantissa_bits);
      f.float_val.d.set_slc(mantissa, exp_bits);
      f.float_val.d.set_slc(mantissa + exp, sign_bit);
    }
    return f;
  }

#ifndef __SYNTHESIS__
  operator float() const {
    decoded tmp(*this);
    return static_cast<float>(tmp);
  }
#endif

// SystemC is not compatible with C++17
#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & bits;
  }

  inline friend void sc_trace(sc_trace_file* tf, const Posit& posit,
                              const std::string& name) {
    sc_trace(tf, posit.bits, name + ".bits");
  }
#endif
};

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

template <int nbits, int es>
template <int nbits2, int es2>
Posit<nbits, es>::Posit(const Posit<nbits2, es2>& other) {
  typename Posit<nbits2, es2>::decoded tmp(other);
  *this = tmp;
}

template <int nbits, int es>
template <int mantissa, int exp>
Posit<nbits, es>::Posit(const StdFloat<mantissa, exp>& other) {
  if (other.float_val.d == 0) {
    bits = 0;
  } else {
    const int e_width = StdFloat<mantissa, exp>::ac_float_rep::e_width;
    const int mant_bits = StdFloat<mantissa, exp>::ac_float_rep::mant_bits;
    const int exp_bias = StdFloat<mantissa, exp>::ac_float_rep::exp_bias;

    bool sign = other.float_val.signbit();
    ac_int<e_width, true> scale =
        other.float_val.d.template slc<e_width>(mant_bits) - exp_bias;
    ac_int<mant_bits, false> fraction = other.float_val.d;
    convert_<nbits, es, mant_bits>(sign, scale, fraction, bits);
  }
}

// Implements an optimized exp that is only valid for inputs smaller than zero
template <int nbits, int es>
Posit<nbits, es> exponential(Posit<nbits, es> val) {
  assert(nbits == 16 && es == 1);

  Posit<nbits, es> min_exp;
  min_exp.set_bits(0x9e00);

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
  neg_one.set_bits(0xc000);

  return Posit<nbits, es>(es0Posit + neg_one);
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator+(
    const Posit<nbits, es>& rhs) const {
  decoded op1 = *this;
  decoded op2 = rhs;
  return op1 + op2;
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator-(
    const Posit<nbits, es>& rhs) const {
  decoded op1 = *this;
  decoded op2 = rhs;
  return op1 - op2;
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator*(
    const Posit<nbits, es>& rhs) const {
  decoded op1 = *this;
  decoded op2 = rhs;
  return op1 * op2;
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator/(
    const Posit<nbits, es>& rhs) const {
  Posit<nbits, es> op2 = rhs;
  op2.reciprocal();
  return *this * op2;
}

template <int nbits, int es>
inline Posit<nbits, es>& Posit<nbits, es>::operator+=(
    const Posit<nbits, es>& rhs) {
  *this = *this + rhs;
  return *this;
}

template <int nbits, int es>
inline Posit<nbits, es>& Posit<nbits, es>::operator-=(
    const Posit<nbits, es>& rhs) {
  *this = *this - rhs;
  return *this;
}

template <int nbits, int es>
inline Posit<nbits, es>& Posit<nbits, es>::operator*=(
    const Posit<nbits, es>& rhs) {
  *this = *this * rhs;
  return *this;
}

template <int nbits, int es>
inline Posit<nbits, es>& Posit<nbits, es>::operator/=(
    const Posit<nbits, es>& rhs) {
  *this = *this / rhs;
  return *this;
}

template <int nbits, int es>
inline bool Posit<nbits, es>::operator<(const Posit<nbits, es>& rhs) const {
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
inline std::ostream& operator<<(std::ostream& os, const Posit<nbits, es>& val) {
#ifndef __SYNTHESIS__
  os << static_cast<float>(val) << " ";
#else
  os << val.bits << " ";
#endif
  return os;
}
