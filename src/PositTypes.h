#pragma once

#include <ac_int.h>

inline int max(int a, int b) { return a > b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }

template <int nbits, int es, int fbits>
void convert_(const bool sign, const int scale, const ac_int<fbits, false> fraction_in, ac_int<nbits, false> &bits) {
  // TODO: handle infinity
  if (scale == -127 || fraction_in == 0) {
    bits = 0;
  } else if ((scale >> es) > nbits - 2 || (scale >> es) < -(nbits - 2)) {
    bits = 0;
    if (scale > 0) {
      bits = bits.bit_complement();
      bits[nbits - 1] = 0;
    } else {
      bits[0] = 1;
    }
  } else {
    ac_int<nbits + 3 + es, false> pt_bits, regime, exponent, fraction, sticky_bit;

    bool r = (scale >= 0);
    int run = r ? (1 + (scale >> es)) : -(scale >> es);
    regime = r ? (1 << (run + 1)) - 1 : 0;
    regime[0] = !r;

    int esval = 1 << es;
    exponent = (scale % esval + esval) % esval;

    int nf = max(0, nbits + 1 - (2 + run + es));
    fraction = (fraction_in << 1) >> (fbits - min(fbits, nf));
    fraction <<= max(nf - fbits, 0);

    regime <<= es + nf + 1;
    exponent <<= nf + 1;
    fraction <<= 1;
    sticky_bit = fraction_in << (nf + 1) ? 0x1 : 0x0;
    pt_bits = regime | exponent | fraction | sticky_bit;

    int len = 1 + max(nbits + 1, 2 + run + es);
    bool blast = pt_bits[len - nbits];
    bool bafter = pt_bits[len - nbits - 1];
    bool bsticky = pt_bits & ((1 << (len - nbits - 1)) - 1);
    bool rb = (blast & bafter) | (bafter & bsticky);

    bits = pt_bits >> (len - nbits);
    if (rb) bits++;
  }
  if (sign) bits = bits.bit_complement() + 1;
}

template <int nbits, int es, int fbits>
void decode(ac_int<nbits, false> bits, bool &sign, int &scale, ac_int<fbits, false> &fraction) {
  if (bits == 0) {
    sign = 0;
    scale = -127;
    fraction = 0;
  } else {
    sign = bits[nbits - 1];
    if (sign) bits = bits.bit_complement() + 1;  // convert to positive value
    bits <<= 1;  // remove sign bit

    bool leadingBit = bits[nbits - 1];
    int run = leadingBit ? bits.bit_complement().leading_sign()
                         : bits.leading_sign();
    scale = leadingBit ? run - 1 : -run;
    scale *= (1 << es);

    int nrBits = nbits - run - 1;
    if (nrBits >= es && es > 0) {
      scale += bits.template slc<es>(nrBits - es);
    } else if (nrBits >= 0 && es > 0) {
      scale += bits & ((1 << nrBits) - 1);
    }

    bits <<= run + 1 + es;
    fraction = bits.template slc<fbits>(fbits >= nbits ? 0 : nbits - fbits);
    fraction <<= max(fbits - 2 - (nbits - 1), -1);
    fraction |= 1 << (fbits - 1);  // add hidden bit
  }
}

#ifndef __SYNTHESIS__

union ufloat {
  float f;
  uint32_t u;
};

template<int fbits>
float to_float(const bool sign, const int scale, ac_int<fbits, false> fraction) {
  union ufloat uf;
  uf.u = sign ? 1 << 31 : 0;
  uf.u += (scale + 127) << 23;

  fraction <<= 1;  // remove hidden bit
  ac_int<23, false> mantissa = fraction.template slc<23>(23 >= fbits ? 0 : fbits - 23);
  uf.u += (mantissa << max(22 - (fbits - 1), 0));
  return uf.f;
}

#endif

// forward declaration
template <int sbits, int fbits>
class PositFP;

template <int nbits, int es, int sbits, int fbits>
class Posit {
 public:
  ac_int<nbits, false> bits;
  Posit() {}
  Posit(int i);
  Posit(const PositFP<sbits, fbits> &input);

  template <int nbits2, int es2>
  Posit(const Posit<nbits2, es2, sbits, fbits> &input);

#ifndef __SYNTHESIS__
  Posit(const float f);
#endif

  bool isZero() const { return bits == 0; }

  void relu() {
    if (bits[nbits - 1] == 1) bits = 0;
  }

  void negate() { bits = bits.bit_complement() + 1; }

  void reciprocal() {
    ac_int<nbits, false> sub = (1 << (nbits - 1));
    bits = sub - bits;
  }

  void sigmoid() {
    // invert MSB
    // bits.set_slc(7, (bits.slc<1>(7).bit_complement()));

    // ac_int<1, false> msb = bits.slc<1>(7);
    bits[nbits - 1] = ~bits[nbits - 1];

    // bits.set_slc(7, bits.slc<1>(7).bit_complement());
    bits = bits >> 2;
  }

  void exp();

  // overridden operators
  template <int nbits2, int es2>
  Posit operator+(const Posit<nbits2, es2, sbits, fbits> &rhs);
  Posit operator*(const Posit &rhs);
  Posit &operator+=(const Posit &rhs);
  Posit &operator-=(const Posit &rhs);
  Posit &operator*=(const Posit &rhs);
  bool operator<(const Posit &rhs) const;

#ifndef __SYNTHESIS__
  operator float() const;
#endif

  static const unsigned int width = nbits;

#ifdef __SYNTHESIS__
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &bits;
  }

  inline friend void sc_trace(sc_trace_file *tf, const Posit &posit,
                              const std::string &name) {
    sc_trace(tf, posit.bits, name + ".bits");
  }
#endif

  // inline friend std::ostream &operator<<(ostream &os, const Posit &posit) {
  //   os << posit.bits << " ";

  //   return os;
  // }
};

template <int nbits, int es, int sbits, int fbits>
Posit<nbits, es, sbits, fbits>::Posit(int i) {
  bits = i;
}

template <int nbits, int es, int sbits, int fbits>
Posit<nbits, es, sbits, fbits>::Posit(
    const PositFP<sbits, fbits> &input) {
  convert_<nbits, es, fbits>(input.sign, input.scale, input.fraction, this->bits);
}

template <int nbits, int es, int sbits, int fbits>
template <int nbits2, int es2>
Posit<nbits, es, sbits, fbits>::Posit(
    const Posit<nbits2, es2, sbits, fbits> &input) {
  PositFP<sbits, fbits> tmp(input);
  *this = tmp;
}

#ifndef __SYNTHESIS__
template <int nbits, int es, int sbits, int fbits>
Posit<nbits, es, sbits, fbits>::Posit(const float f) {
  union ufloat uf;
  uf.f = f;
  bool sign = f < 0;
  int scale = ((uf.u >> 23) & 0xFF) - 127;
  ac_int<24, false> fraction = (uf.u & ((1 << 23) - 1)) | (1 << 23);
  convert_<nbits, es, 24>(sign, scale, fraction, this->bits);
}
#endif

template <int nbits, int es, int sbits, int fbits>
template <int nbits2, int es2>
Posit<nbits, es, sbits, fbits>
Posit<nbits, es, sbits, fbits>::operator+(
    const Posit<nbits2, es2, sbits, fbits> &rhs) {
  PositFP<sbits, fbits> op1 = *this;
  PositFP<sbits, fbits> op2 = rhs;
  PositFP<sbits, fbits> tmpResult = op1 + op2;

  Posit<nbits, es, sbits, fbits> result = tmpResult;

  return result;
}

template <int nbits, int es, int sbits, int fbits>
inline Posit<nbits, es, sbits, fbits>
Posit<nbits, es, sbits, fbits>::operator*(
    const Posit<nbits, es, sbits, fbits> &rhs) {
  PositFP<sbits, fbits> op1 = *this;
  PositFP<sbits, fbits> op2 = rhs;

  return op1 * op2;
}

template <int nbits, int es, int sbits, int fbits>
inline Posit<nbits, es, sbits, fbits>
    &Posit<nbits, es, sbits, fbits>::operator+=(
        const Posit<nbits, es, sbits, fbits> &rhs) {
  PositFP<sbits, fbits> op1 = *this;
  PositFP<sbits, fbits> op2 = rhs;

  *this = op1 + op2;
  return *this;
}

template <int nbits, int es, int sbits, int fbits>
inline Posit<nbits, es, sbits, fbits>
    &Posit<nbits, es, sbits, fbits>::operator-=(
        const Posit<nbits, es, sbits, fbits> &rhs) {
  PositFP<sbits, fbits> op1 = *this;
  PositFP<sbits, fbits> op2 = rhs;

  *this = op1 - op2;
  return *this;
}

template <int nbits, int es, int sbits, int fbits>
inline Posit<nbits, es, sbits, fbits>
    &Posit<nbits, es, sbits, fbits>::operator*=(
        const Posit<nbits, es, sbits, fbits> &rhs) {
  *this = *this * rhs;
  return *this;
}

template <int nbits, int es, int sbits, int fbits>
inline bool Posit<nbits, es, sbits, fbits>::operator<(
    const Posit<nbits, es, sbits, fbits> &rhs) const {
  PositFP<sbits, fbits> op1 = *this;
  PositFP<sbits, fbits> op2 = rhs;

  return op1 < op2;
}

#ifndef __SYNTHESIS__
template <int nbits, int es, int sbits, int fbits>
Posit<nbits, es, sbits, fbits>::operator float() const {
  bool sign;
  int scale;
  ac_int<23, false> fraction;
  decode<nbits, es, 23>(this->bits, sign, scale, fraction);
  return to_float<23>(sign, scale, fraction);
}
#endif

/*
 * Intermediate representation used for MAC
 */
template <int sbits, int fbits>
class PositFP {
 public:
  ac_int<1, false> sign;
  ac_int<sbits, true> scale;
  ac_int<fbits, false> fraction;

  PositFP() {}

#pragma hls_design ccore
#pragma ccore_type combinational
  template <int nbits, int es>
  PositFP(const Posit<nbits, es, sbits, fbits> &input);
  
  PositFP(const float f);

  bool isZero() const { return scale == -127; }
  void setZero() { scale = -127; }

  PositFP operator+(const PositFP &op);
  PositFP operator-(const PositFP &op);
  PositFP operator*(const PositFP &op);
  PositFP &operator+=(const PositFP &rhs);
  bool operator<(const PositFP &rhs) const;

  PositFP fma(const PositFP<sbits, fbits> &op2,
              const PositFP<sbits, fbits> &op3);
  template <int nbits, int es, int nbits2, int es2>
  PositFP fma(const Posit<nbits, es, sbits, fbits> &op2,
              const Posit<nbits2, es2, sbits, fbits> &op3);

#ifndef __SYNTHESIS__
  operator float() const;
#endif
};

template <int sbits, int fbits>
template <int nbits, int es>
PositFP<sbits, fbits>::PositFP(
    const Posit<nbits, es, sbits, fbits> &input) {
  bool sign;
  int scale;
  ac_int<fbits, false> fraction;
  decode<nbits, es, fbits>(input.bits, sign, scale, fraction);
  // printf("sign: %d, scale: %d, fraction: %d\n", sign, scale, fraction);

  this->sign = sign;
  this->scale = scale;
  this->fraction = fraction;
}

#ifndef __SYNTHESIS__
template <int sbits, int fbits>
PositFP<sbits, fbits>::PositFP(const float f) {
  union ufloat uf;
  uf.f = f;
  this->sign = f < 0;
  this->scale = ((uf.u >> 23) & 0xFF) - 127;
  ac_int<23, false> mantissa = uf.u & ((1 << 23) - 1);
  if (f) {
    this->fraction = mantissa.template slc<fbits>(fbits >= 23 ? 0 : 23 - fbits);
    this->fraction <<= max(fbits - 2 - 22, -1);
    this->fraction |= (1 << (fbits - 1));
  } else {
    this->fraction = 0;
  }
}
#endif

template <int sbits, int fbits>
PositFP<sbits, fbits> PositFP<sbits, fbits>::operator+(
    const PositFP<sbits, fbits> &op) {
  PositFP<sbits, fbits> lhs = *this;
  PositFP<sbits, fbits> rhs = op;
  PositFP<sbits, fbits> result;

  // align the fraction
  int result_scale = max(lhs.scale, rhs.scale);
  ac_int<fbits, false> r1 = lhs.fraction >> (result_scale - lhs.scale);
  ac_int<fbits, false> r2 = rhs.fraction >> (result_scale - rhs.scale);
  ac_int<fbits+1, false> sum;

  int shift = 0;
  if (lhs.sign == rhs.sign) {
    result.sign = lhs.sign;
    sum = r1 + r2;
    if (sum[fbits]) shift = -1;
  } else {
    if (r1 > r2) {
      sum = r1 - r2;
      result.sign = lhs.sign;
    } else if (r1 < r2) {
      sum = r2 - r1;
      result.sign = rhs.sign;
    } else {
      result.setZero();
      return result;
    }
    shift = sum.leading_sign() - 1;
  }

  if (shift > fbits) {
    result.setZero();
    return result;
  }

  result.scale = result_scale - shift;
  result.fraction = sum << shift;
  return result;
}

template <int sbits, int fbits>
PositFP<sbits, fbits> PositFP<sbits, fbits>::operator-(
    const PositFP<sbits, fbits> &op) {
  PositFP<sbits, fbits> negOp = op;
  negOp.sign = !negOp.sign;

  return *this + negOp;
}

template <int sbits, int fbits>
PositFP<sbits, fbits> PositFP<sbits, fbits>::operator*(
    const PositFP<sbits, fbits> &op) {
  PositFP<sbits, fbits> lhs = *this;
  PositFP<sbits, fbits> rhs = op;
  PositFP<sbits, fbits> result;

  result.sign = lhs.sign ^ rhs.sign;
  result.scale = lhs.scale + rhs.scale;
  result.fraction = (lhs.fraction >> (fbits/2)) * (rhs.fraction >> (fbits - fbits/2));

  if (result.fraction[fbits-1]) {
    result.scale++;
  } else {
    result.fraction <<= 1;
  }

  return result;
}

template <int sbits, int fbits>
inline PositFP<sbits, fbits> &PositFP<sbits, fbits>::operator+=(
    const PositFP<sbits, fbits> &rhs) {
  *this = *this + rhs;

  return *this;
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator<(
    const PositFP<sbits, fbits> &rhs) const {
  if (this->sign ^ rhs.sign) return sign;
  if (scale != rhs.scale) return scale < rhs.scale;
  return fraction < rhs.fraction;
}

template <int sbits, int fbits>
template <int nbits, int es, int nbits2, int es2>
PositFP<sbits, fbits> PositFP<sbits, fbits>::fma(
    const Posit<nbits, es, sbits, fbits> &op2,
    const Posit<nbits2, es2, sbits, fbits> &op3) {
  if (this->isZero() || op2.isZero()) {
    return op3;
  } else {
    PositFP<sbits, fbits> product = *this * op2;

    if (op3.isZero()) {
      return product;
    } else {
      PositFP<sbits, fbits> sum = product + op3;
      return sum;
    }
  }
}

template <int sbits, int fbits>
PositFP<sbits, fbits> PositFP<sbits, fbits>::fma(
    const PositFP<sbits, fbits> &op2,
    const PositFP<sbits, fbits> &op3) {
  if (this->isZero() || op2.isZero()) {
    return op3;
  } else {
    PositFP<sbits, fbits> product = *this * op2;

    if (op3.isZero()) {
      return product;
    } else {
      PositFP<sbits, fbits> sum = product + op3;
      return sum;
    }
  }
}

template <int sbits, int fbits>
PositFP<sbits, fbits>::operator float() const {
  return to_float<fbits>(this->sign, this->scale, this->fraction);
}

template <int sbits, int fbits>
inline bool operator==(const PositFP<sbits, fbits> &lhs,
                       const PositFP<sbits, fbits> &rhs) {
  return (lhs.sign == rhs.sign) && (lhs.scale == rhs.scale) &&
         (lhs.fraction == rhs.fraction);
}

template <int nbits, int es, int sbits, int fbits>
inline bool operator==(const Posit<nbits, es, sbits, fbits> &lhs,
                       const Posit<nbits, es, sbits, fbits> &rhs) {
  return lhs.bits == rhs.bits;
}

// template <int sbits, int fbits>
// class Wrapped<PositFP<sbits, fbits> > {
//  public:
//   typedef PositFP<sbits, fbits> Type;
//   Type val;
//   Wrapped() {}
//   Wrapped(const Type &v) : val(v) {}
//   static const unsigned int width = Type::width;
//   static const bool is_signed = false;
//   template <unsigned int Size>
//   void Marshall(Marshaller<Size> &m) {
//     m &val.sign;
//     m &val.scale;
//     m &val.fraction;
//   }
// };

// template <unsigned int Size, int sbits, int fbits>
// Marshaller<Size> &operator&(Marshaller<Size> &m, PositFP<sbits, fbits> &rhs) {
//   typedef PositFP<sbits, fbits> Type;
//   m.template AddField<ac_int<1, false>, 1>(rhs.sign);
//   m.template AddField<ac_int<sbits, true>, sbits>(rhs.scale);
//   m.template AddField<ac_int<fbits, false>, fbits>(rhs.fraction);
//   return m;
// }
