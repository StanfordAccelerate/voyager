#pragma once

#include <ac_int.h>

inline int max(int a, int b) { return a > b ? a : b; }

// forward declaration
template <int sbits, int fbits>
class PositFP;

#ifndef __SYNTHESIS__
union ufloat {
  float f;
  uint32_t u;
};
#endif

template <int nbits, int es, int fp_sbits, int fp_fbits>
class Posit {
 public:
  ac_int<nbits, false> bits;
  Posit() {}
  Posit(const PositFP<fp_sbits, fp_fbits> &input);

  template <int nbits2, int es2>
  Posit(const Posit<nbits2, es2, fp_sbits, fp_fbits> &input);

  Posit(const float &f);
  Posit(int i);

  bool isZero() const { return bits == 0; }

  void relu() {
    if (bits[nbits - 1] == 1) {
      bits = 0;
    }
  }

  // overridden operators
  Posit &operator+=(const Posit &rhs);
  Posit &operator-=(const Posit &rhs);
  Posit &operator*=(const Posit &rhs);
  Posit operator*(const Posit &rhs);
  bool operator<(const Posit &rhs) const;

  template <int nbits2, int es2>
  Posit operator+(const Posit<nbits2, es2, fp_sbits, fp_fbits> &rhs);

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

  inline friend std::ostream &operator<<(ostream &os, const Posit &posit) {
    os << posit.bits << " ";

    return os;
  }
#endif
};

template <int nbits, int es, int fp_sbits, int fp_fbits>
Posit<nbits, es, fp_sbits, fp_fbits>::Posit(
    const PositFP<fp_sbits, fp_fbits> &input) {
  // TODO: handle infinity
  if (input.scale == -127) {
    bits = 0;
  } else if ((input.scale >> es) > nbits - 2 ||
             (input.scale >> es) < -(nbits - 2)) {
    bits = 0;
    if (input.scale > 0) {
      bits = bits.bit_complement();
      bits[nbits - 1] = 0;
    } else {
      bits[0] = 1;
    }
  } else {
    ac_int<nbits + 3 + es, false> pt_bits, regime, exponent, fraction;

    int e = input.scale;
    bool r = (input.scale >= 0);

    int run = r ? (1 + (e >> es)) : -(e >> es);
    regime = r ? (1 << (run + 1)) - 1 : 0;
    regime[0] = !r;

    int esval = 1 << es;
    exponent = (e % esval + esval) % esval;
    int nf = max(0, nbits + 1 - (2 + run + es));
    fraction = (input.fraction << 1) >> (fp_fbits - nf);

    pt_bits = regime << es + nf + 1;
    pt_bits |= exponent << nf + 1;
    pt_bits |= fraction << 1;
    pt_bits |= (input.fraction << 1) << nf ? 0x1 : 0x0;  // sticky bit

    int len = 1 + max(nbits + 1, 2 + run + es);
    bool blast = pt_bits[len - nbits];
    bool bafter = pt_bits[len - nbits - 1];
    bool bsticky = pt_bits & ((1 << (len - nbits - 1)) - 1);
    bool rb = (blast & bafter) | (bafter & bsticky);

    bits = pt_bits >> (len - nbits);
    if (rb) bits++;
  }
  if (input.sign) bits = bits.bit_complement() + 1;
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
template <int nbits2, int es2>
Posit<nbits, es, fp_sbits, fp_fbits>::Posit(
    const Posit<nbits2, es2, fp_sbits, fp_fbits> &input) {
  PositFP<fp_sbits, fp_fbits> tmp(input);
  *this = tmp;
}

#ifndef __SYNTHESIS__
template <int nbits, int es, int fp_sbits, int fp_fbits>
Posit<nbits, es, fp_sbits, fp_fbits>::Posit(const float &f) {
  PositFP<fp_sbits, fp_fbits> positFP(f);
  *this = positFP;
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
Posit<nbits, es, fp_sbits, fp_fbits>::operator float() const {
  PositFP<fp_sbits, fp_fbits> positFP = *this;
  return positFP.to_float();
}
#endif

template <int nbits, int es, int fp_sbits, int fp_fbits>
Posit<nbits, es, fp_sbits, fp_fbits>::Posit(int i) {
  bits = i;
}

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

  template <int nbits, int es>
  PositFP(const Posit<nbits, es, sbits, fbits> &input) {
    ac_int<nbits, false> bits;
    bits = input.bits;
    if (bits == 0) {
      this->sign = 0;
      this->fraction = 0;
      this->scale = -127;
    } else {
      this->sign = bits[nbits - 1];
      if (this->sign)
        bits = bits.bit_complement() + 1;  // convert to positive value
      bits <<= 1;                          // remove sign bit

      bool leadingBit = bits[nbits - 1];
      int run = leadingBit ? bits.bit_complement().leading_sign()
                           : bits.leading_sign();
      this->scale = leadingBit ? run - 1 : -run;
      this->scale *= (1 << es);

      int nrBits = nbits - run - 1;
      if (nrBits >= es && es > 0) {
        this->scale += bits.template slc<es>(nrBits - es);
      } else if (nrBits >= 0 && es > 0) {
        this->scale += bits & ((1 << nrBits) - 1);
      }

      // TODO: handle case where bits has longer length than this->fraction
      this->fraction = bits << run + 1 + es;
      this->fraction <<= fbits - 1 - nbits;
      this->fraction |= 0x8000;  // add hidden bit
    }
  }

#ifndef __SYNTHESIS__
  PositFP(const float &f) {
    union ufloat uf;
    uf.f = f;
    this->sign = uf.f < 0;
    this->scale = ((uf.u >> 23) & 0xFF) - 127;
    this->fraction = ((uf.u & ((1 << 23) - 1)) >> 8) | (1 << 15);
  }
#endif

#ifndef __SYNTHESIS__
  float to_float() {
    union ufloat uf;
    uf.u = this->sign ? 1 << 31 : 0;
    uf.u += (this->scale + 127) << 23;
    uf.u += (this->fraction & ((1 << 15) - 1)) << 8;
    return uf.f;
  }
#endif

  bool isZero() const { return scale == -127; }

  void setZero() { scale = -127; }

  template <int nbits, int es, int nbits2, int es2>
  PositFP fma(const Posit<nbits, es, sbits, fbits> &op2,
              const Posit<nbits2, es2, sbits, fbits> &op3);

  PositFP &operator+=(const PositFP &rhs);
  PositFP operator*(const PositFP &op);
  PositFP operator+(const PositFP &op);
  PositFP operator-(const PositFP &op);
  bool operator<(const PositFP &rhs) const;
};

template <int nbits, int es, int fp_sbits, int fp_fbits>
inline Posit<nbits, es, fp_sbits, fp_fbits>
    &Posit<nbits, es, fp_sbits, fp_fbits>::operator+=(
        const Posit<nbits, es, fp_sbits, fp_fbits> &rhs) {
  PositFP<fp_sbits, fp_fbits> op1 = *this;
  PositFP<fp_sbits, fp_fbits> op2 = rhs;

  *this = op1 + op2;
  return *this;
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
inline Posit<nbits, es, fp_sbits, fp_fbits>
    &Posit<nbits, es, fp_sbits, fp_fbits>::operator-=(
        const Posit<nbits, es, fp_sbits, fp_fbits> &rhs) {
  PositFP<fp_sbits, fp_fbits> op1 = *this;
  PositFP<fp_sbits, fp_fbits> op2 = rhs;

  *this = op1 - op2;
  return *this;
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
template <int nbits2, int es2>
Posit<nbits, es, fp_sbits, fp_fbits>
Posit<nbits, es, fp_sbits, fp_fbits>::operator+(
    const Posit<nbits2, es2, fp_sbits, fp_fbits> &rhs) {
  PositFP<fp_sbits, fp_fbits> op1 = *this;
  PositFP<fp_sbits, fp_fbits> op2 = rhs;
  PositFP<fp_sbits, fp_fbits> tmpResult = op1 + op2;

  Posit<nbits, es, fp_sbits, fp_fbits> result = tmpResult;

  return result;
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
inline Posit<nbits, es, fp_sbits, fp_fbits>
    &Posit<nbits, es, fp_sbits, fp_fbits>::operator*=(
        const Posit<nbits, es, fp_sbits, fp_fbits> &rhs) {
  *this = *this * rhs;
  return *this;
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
inline Posit<nbits, es, fp_sbits, fp_fbits>
Posit<nbits, es, fp_sbits, fp_fbits>::operator*(
    const Posit<nbits, es, fp_sbits, fp_fbits> &rhs) {
  PositFP<fp_sbits, fp_fbits> op1 = *this;
  PositFP<fp_sbits, fp_fbits> op2 = rhs;

  return op1 * op2;
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
inline bool Posit<nbits, es, fp_sbits, fp_fbits>::operator<(
    const Posit<nbits, es, fp_sbits, fp_fbits> &rhs) const {
  PositFP<fp_sbits, fp_fbits> op1 = *this;
  PositFP<fp_sbits, fp_fbits> op2 = rhs;

  return op1 < op2;
}

template <int fp_sbits, int fp_fbits>
PositFP<fp_sbits, fp_fbits> PositFP<fp_sbits, fp_fbits>::operator*(
    const PositFP<fp_sbits, fp_fbits> &op) {
  PositFP<fp_sbits, fp_fbits> lhs = *this;
  PositFP<fp_sbits, fp_fbits> rhs = op;
  PositFP<fp_sbits, fp_fbits> result;

  result.sign = lhs.sign ^ rhs.sign;
  result.scale = lhs.scale + rhs.scale;

  lhs.fraction >>= 8;
  rhs.fraction >>= 8;
  result.fraction = lhs.fraction * rhs.fraction;

  if (result.fraction[15]) {
    result.scale++;
  } else {
    result.fraction <<= 1;
  }

  return result;
}

template <int fp_sbits, int fp_fbits>
PositFP<fp_sbits, fp_fbits> PositFP<fp_sbits, fp_fbits>::operator+(
    const PositFP<fp_sbits, fp_fbits> &op) {
  PositFP<fp_sbits, fp_fbits> lhs = *this;
  PositFP<fp_sbits, fp_fbits> rhs = op;
  PositFP<fp_sbits, fp_fbits> result;

  // align the fraction
  int result_scale = max(lhs.scale, rhs.scale);
  ac_int<16, false> r1 = lhs.fraction >> (result_scale - lhs.scale + 1);
  ac_int<16, false> r2 = rhs.fraction >> (result_scale - rhs.scale + 1);
  ac_int<16, false> sum;

  int shift = 0;
  if (lhs.sign == rhs.sign) {
    result.sign = lhs.sign;
    sum = r1 + r2;
    if (sum[15]) shift = -1;
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
  // printf("fraction: %s\n", sum.to_string(AC_BIN).c_str());

  if (shift > 16) {
    result.setZero();
    return result;
  }

  result.scale = result_scale - shift;
  result.fraction = sum << (shift + 1);

  return result;
}

template <int fp_sbits, int fp_fbits>
PositFP<fp_sbits, fp_fbits> PositFP<fp_sbits, fp_fbits>::operator-(
    const PositFP<fp_sbits, fp_fbits> &op) {
  PositFP<fp_sbits, fp_fbits> negOp = op;
  negOp.sign = !negOp.sign;

  return *this + negOp;
}

template <int fp_sbits, int fp_fbits>
template <int nbits, int es, int nbits2, int es2>
PositFP<fp_sbits, fp_fbits> PositFP<fp_sbits, fp_fbits>::fma(
    const Posit<nbits, es, fp_sbits, fp_fbits> &op2,
    const Posit<nbits2, es2, fp_sbits, fp_fbits> &op3) {
  if (this->isZero() || op2.isZero()) {
    return op3;
  } else {
    PositFP<fp_sbits, fp_fbits> product = *this * op2;

    if (op3.isZero()) {
      return product;
    } else {
      PositFP<fp_sbits, fp_fbits> sum = product + op3;
      return sum;
    }
  }
}

template <int fp_sbits, int fp_fbits>
bool PositFP<fp_sbits, fp_fbits>::operator<(
    const PositFP<fp_sbits, fp_fbits> &rhs) const {
  if (this->sign ^ rhs.sign) return sign;
  if (scale != rhs.scale) return scale < rhs.scale;
  return fraction < rhs.fraction;
}

template <int fp_sbits, int fp_fbits>
inline PositFP<fp_sbits, fp_fbits> &PositFP<fp_sbits, fp_fbits>::operator+=(
    const PositFP<fp_sbits, fp_fbits> &rhs) {
  *this = *this + rhs;

  return *this;
}

template <int fp_sbits, int fp_fbits>
inline bool operator==(const PositFP<fp_sbits, fp_fbits> &lhs,
                       const PositFP<fp_sbits, fp_fbits> &rhs) {
  return (lhs.sign == rhs.sign) && (lhs.scale == rhs.scale) &&
         (lhs.fraction == rhs.fraction);
}

template <int nbits, int es, int fp_sbits, int fp_fbits>
inline bool operator==(const Posit<nbits, es, fp_sbits, fp_fbits> &lhs,
                       const Posit<nbits, es, fp_sbits, fp_fbits> &rhs) {
  return lhs.bits == rhs.bits;
}
