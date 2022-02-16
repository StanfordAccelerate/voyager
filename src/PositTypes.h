#pragma once

#include <ac_int.h>

inline int max(int a, int b) { return a > b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }

template <class T>
void swap_(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <int W, int copy_W>
void copy_(ac_int<W, false> src, ac_int<copy_W, false> &dst) {
  dst = src.template slc<copy_W>(copy_W >= W ? 0 : W - copy_W);
  if (copy_W >= W) dst <<= copy_W - W;
}

template <int nbits>
void twos_complement(ac_int<nbits, false> &value) {
  value = static_cast<ac_int<nbits, false>>(value.bit_complement() + 1);
}

template <int nbits, int es, int fbits>
void convert_(const bool sign, const int scale,
              const ac_int<fbits, false> fraction_in,
              ac_int<nbits, false> &bits) {
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
    regime = r ? (1 << (run + 1)) - 1 : 0;
    regime[0] = !r;

    int esval = 1 << es;
    exponent = (scale % esval + esval) % esval;

    int nf = max(0, nbits + 1 - (2 + run + es));
    fraction = fraction_in >> max(fbits - nf, 0);
    fraction <<= max(nf - fbits, 0);

    regime <<= es + nf + 1;
    exponent <<= nf + 1;
    fraction <<= 1;
    sticky_bit = fraction_in << nf ? 0x1 : 0x0;
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
    copy_(bits, fraction);
  }
}

#ifndef __SYNTHESIS__
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
  // static const definitions
  static constexpr int width = nbits;
  static constexpr int esbits = es;
  static constexpr int sbits = ac::log2_ceil<nbits - 2>::val + es + 1;
  static constexpr int fbits = nbits - 3 - es;
  typedef PositFP<sbits, fbits> DecomposedPosit;

  ac_int<nbits, false> bits;


  Posit() {}
#ifndef __SYNTHESIS__
  Posit(const float f);
#endif

  template <int nbits2, int es2>
  Posit(const Posit<nbits2, es2> &input);

  template <int fp_sbits, int fp_fbits>
  Posit(const PositFP<fp_sbits, fp_fbits> &input);

  void setbits(int i) { bits = i; }
  bool isZero() const { return bits == 0; }

  void relu() { if (bits[nbits - 1] == 1) bits = 0; }

  void negate() { twos_complement(bits); }

  void reciprocal() {
    ac_int<nbits, false> sub = (1 << (nbits - 1));
    bits = sub - bits;
  }

  void sigmoid() {
    bits[nbits - 1] = ~bits[nbits - 1];  // invert MSB
    bits = bits >> 2;
  }

  void exp();

  // overridden operators
  template <int nbits2, int es2>
  Posit operator+(const Posit<nbits2, es2> &rhs);
  Posit operator*(const Posit &rhs);
  Posit &operator+=(const Posit &rhs);
  Posit &operator-=(const Posit &rhs);
  Posit &operator*=(const Posit &rhs);

  bool operator<(const Posit &rhs) const;

  // template <int nbits2, int es2, int sbits, int fbits>
  // Posit &fma(const Posit<nbits2, es2> input, const PositFP<sbits, fbits> weight);

#ifndef __SYNTHESIS__
  operator float() const;
#else
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
template <int fp_sbits, int fp_fbits>
Posit<nbits, es>::Posit(const PositFP<fp_sbits, fp_fbits> &input) {
  if (input.isZero()) {
    this->bits = 0;
    return;
  }
  convert_<nbits, es, fp_fbits>(input.sign, input.scale, input.fraction,
                                this->bits);
}

template <int nbits, int es>
template <int nbits2, int es2>
Posit<nbits, es>::Posit(const Posit<nbits2, es2> &input) {
  PositFP<sbits, fbits> tmp(input);
  *this = tmp;
}

#ifndef __SYNTHESIS__
template <int nbits, int es>
Posit<nbits, es>::Posit(const float f) {
  if (f == 0) {
    this->bits = 0;
    return;
  }
  union ufloat uf = {f};
  bool sign = f < 0;
  int scale = ((uf.u >> 23) & 0xFF) - 127;
  ac_int<23, false> fraction = uf.u;  // copy the least significant 23 bits into fraction
  convert_<nbits, es, 23>(sign, scale, fraction, this->bits);
}
#endif

#pragma hls_design ccore
#pragma ccore_type combinational
template <int nbits, int es>
inline void Posit<nbits, es>::exp() {
  // std::cout << "original:\t" << bits.to_string(AC_BIN, false, true)
  //           << std::endl;
  negate();
  // std::cout << "negative:\t" << bits.to_string(AC_BIN, false, true)
  //           << std::endl;

  Posit<nbits, 0> es0Posit = *this;
  es0Posit.sigmoid();
  *this = es0Posit;
  // std::cout << "sigmoid:\t" << bits.to_string(AC_BIN, false, true)
  //           << std::endl;
  reciprocal();
  // std::cout << "reciprocal:\t" << bits.to_string(AC_BIN, false, true)
  //           << std::endl;

  DecomposedPosit op1 = *this;
  DecomposedPosit op2;
  op2.sign = 1;
  op2.scale = 0;
  op2.fraction = 0;
  DecomposedPosit res = (DecomposedPosit) (op1 + op2);
  *this = res;
  // std::cout << "final  :\t" << bits.to_string(AC_BIN, false, true)
  //           << std::endl;
}

template <int nbits, int es>
template <int nbits2, int es2>
Posit<nbits, es> Posit<nbits, es>::operator+(const Posit<nbits2, es2> &rhs) {
  PositFP<8, fbits> op1 = *this;
  PositFP<8, fbits> op2 = rhs;
  return op1 + op2;
}

template <int nbits, int es>
inline Posit<nbits, es> Posit<nbits, es>::operator*(
    const Posit<nbits, es> &rhs) {
  PositFP<8, fbits> op1 = *this;
  PositFP<8, fbits> op2 = rhs;
  return op1 * op2;
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
  op2.negate()
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
inline bool Posit<nbits, es>::operator<(const Posit<nbits, es> &rhs) const {
  bool lhs_sign = this->bits[nbits - 1], rhs_sign = rhs.bits[nbits - 1];
  if (lhs_sign ^ rhs_sign) {
    return lhs_sign;
  } else {
    ac_int<nbits, false> r1 = this->bits, r2 = rhs.bits;
    if (lhs_sign) twos_complement(r1);
    if (rhs_sign) twos_complement(r2);
    return r1 < r2;
  }
}

// template <int nbits, int es>
// template <int nbits2, int es2, int sbits, int fbits>
// Posit<nbits, es> &Posit<nbits, es>::fma(const Posit<nbits2, es2> input, const PositFP<sbits, fbits> weight) {
//   constexpr size_t fbits = nbits - 2 - es;
//   constexpr size_t fbits2 = nbits2 - 2 - es2;
//   PositFP<8, fbits2> a = input;
//   PositFP<8, fbits2> b = (PositFP<8, fbits2>) weight;
//   PositFP<8, 2 * fbits2> product = input * weight;
//   PositFP<8, fbits> c = *this;
//   *this = (PositFP<8, fbits>) product + c;
//   return *this;
// }

#ifndef __SYNTHESIS__
template <int nbits, int es>
Posit<nbits, es>::operator float() const {
  PositFP<8, 23> fp(*this);
  return (float)fp;
}
#endif

/*
 * Intermediate representation used for MAC
 */
template <int sbits, int fbits>
class PositFP {
 public:
	static constexpr int fhbits  = fbits + 1;    // maximum number of fraction + one hidden bit
	static constexpr int abits   = fhbits + 3;   // size of the addend
	static constexpr int mbits   = 2 * fhbits;   // size of the multiplier output

  ac_int<1, false> sign;
  ac_int<sbits, true> scale;
  ac_int<fbits, false> fraction;
  bool _zero = false;

  PositFP() {}

#pragma hls_design ccore
#pragma ccore_type combinational
  template <int nbits, int es>
  PositFP(const Posit<nbits, es> &input);

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

  PositFP<sbits, abits + 1> operator+(const PositFP &op) const;
  PositFP<sbits, mbits> operator*(const PositFP &op) const;
  PositFP &operator+=(const PositFP &rhs);
  bool operator<(const PositFP &rhs) const;

  template <int sbits2, int fbits2>
  PositFP &fma(const PositFP<sbits2, fbits2> &op2,
               const PositFP<sbits2, fbits2> &op3);

#ifndef __SYNTHESIS__
  explicit operator float() const {
    if (isZero()) return 0;

    union ufloat uf;
    uf.u = sign ? 1 << 31 : 0;
    uf.u += (scale + 127) << 23;

    ac_int<23, false> mantissa;
    copy_(fraction, mantissa);
    uf.u += mantissa;
    return uf.f;
  }
#endif

  template <int sbits2, int fbits2>
  explicit operator PositFP<sbits2, fbits2>() const {
    PositFP<sbits2, fbits2> fp;
    if (isZero()) {
      fp.setZero();
      return fp;
    }
    fp.sign = this->sign;
    fp.scale = this->scale;
    copy_(this->fraction, fp.fraction);
    return fp;
  }
};

template <int sbits, int fbits>
template <int nbits, int es>
PositFP<sbits, fbits>::PositFP(const Posit<nbits, es> &input) {
  if (input.isZero()) {
    setZero();
    return;
  }
  bool sign;
  int scale;
  ac_int<fbits, false> fraction;
  decode<nbits, es, fbits>(input.bits, sign, scale, fraction);

  this->sign = sign;
  this->scale = scale;
  this->fraction = fraction;
}

#ifndef __SYNTHESIS__
template <int sbits, int fbits>
PositFP<sbits, fbits>::PositFP(const float f) {
  if (f == 0) {
    setZero();
    return;
  }
  union ufloat uf = {f};
  this->sign = f < 0;
  this->scale = ((uf.u >> 23) & 0xff) - 127;
  ac_int<23, false> mantissa = uf.u;
  copy_(mantissa, this->fraction);
}
#endif

// TODO: scale overflow?
template <int sbits, int fbits>
PositFP<sbits, PositFP<sbits, fbits>::abits + 1> PositFP<sbits, fbits>::operator+(
    const PositFP<sbits, fbits> &op) const {
  PositFP<sbits, fbits> lhs = *this;
  PositFP<sbits, fbits> rhs = op;

  if (lhs.isZero()) {
    return (PositFP<sbits, PositFP<sbits, fbits>::abits + 1>) rhs;
  }
  if (rhs.isZero()) {
    return (PositFP<sbits, PositFP<sbits, fbits>::abits + 1>) lhs;
  }

  int lhs_scale = lhs.scale, rhs_scale = rhs.scale,
      scale_of_result = max(lhs_scale, rhs_scale);

  // align the fraction
  ac_int<abits, false> lhs_fraction = lhs.get_fixed_point(), rhs_fraction = rhs.get_fixed_point();
  ac_int<abits, false> r1 = lhs_fraction << (lhs_scale - scale_of_result + 3);
  r1[0] = lhs_fraction << (abits - 1 + lhs_scale - scale_of_result + 3);  // uncertainty bit
  ac_int<abits, false> r2 = rhs_fraction << (rhs_scale - scale_of_result + 3);
  r2[0] = rhs_fraction << (abits - 1 + rhs_scale - scale_of_result + 3);

  bool r1_sign = lhs.sign, r2_sign = rhs.sign;
  bool signs_are_different = r1_sign != r2_sign;
  bool lhs_slt_rhs = lhs_scale < rhs_scale || (lhs_scale == rhs_scale && r1 < r2);

  if (signs_are_different && lhs_slt_rhs) {
    swap_(r1, r2);
    swap_(r1_sign, r2_sign);
  }

  ac_int<abits + 1, false> sum;
  if (signs_are_different) {
    sum = r1 - r2;
  } else {
    sum = r1 + r2;
  }
  // std::cerr << "r1: " << r1.to_string(AC_BIN) << std::endl;
  // std::cerr << "r2: " << r2.to_string(AC_BIN) << std::endl;
  // std::cerr << "sum: " << sum.to_string(AC_BIN) << std::endl;

  int shift = sum.leading_sign() - 1;

  PositFP<sbits, abits + 1> result;
  result.sign = r1_sign;
  result.scale = scale_of_result - shift;
  result.fraction = sum << (shift + 2);
  // std::cerr << "fraction: " << result.fraction.to_string(AC_BIN) << std::endl;
  // std::cerr << "scale: " << result.scale << std::endl;
  return result;
}

template <int sbits, int fbits>
PositFP<sbits, PositFP<sbits, fbits>::mbits> PositFP<sbits, fbits>::operator*(
    const PositFP<sbits, fbits> &op) const {
  PositFP<sbits, fbits> lhs = *this;
  PositFP<sbits, fbits> rhs = op;
  PositFP<sbits, mbits> result;

  if (lhs.isZero() || rhs.isZero()) {
    result.setZero();
    return result;
  }

  result.sign = lhs.sign ^ rhs.sign;
  result.scale = lhs.scale + rhs.scale;
  result.fraction = lhs.get_fixed_point() * rhs.get_fixed_point();

  if (result.fraction[mbits - 1]) {
    result.scale++;
    result.fraction <<= 1;
  } else {
    result.fraction <<= 2;
  }
  // std::cerr << "fraction: " << result.fraction.to_string(AC_BIN) << std::endl;
  // std::cerr << "scale: " << result.scale << std::endl;

  return result;
}

template <int sbits, int fbits>
inline PositFP<sbits, fbits> &PositFP<sbits, fbits>::operator+=(
    const PositFP<sbits, fbits> &rhs) {
  *this = static_cast<PositFP<sbits, fbits> >(*this + rhs);

  return *this;
}

template <int sbits, int fbits>
bool PositFP<sbits, fbits>::operator<(const PositFP<sbits, fbits> &rhs) const {
  if (this->sign ^ rhs.sign) return sign;
  if (scale != rhs.scale) return scale < rhs.scale;
  return fraction < rhs.fraction;
}

template <int sbits, int fbits>
template <int sbits2, int fbits2>
PositFP<sbits, fbits> &PositFP<sbits, fbits>::fma(
    const PositFP<sbits2, fbits2> &a, const PositFP<sbits2, fbits2> &b) {
  if (!a.isZero() && !b.isZero()) {
    PositFP<8, 2 * fbits2> product = a * b;
    if (!isZero()) {
      *this += (PositFP<8, fbits>) product;
    } else {
      *this = (PositFP<8, fbits>) product;
    }
  }
  return *this;
}

template <int nbits, int es, int nbits2, int es2>
Posit<nbits2, es2> fma(const Posit<nbits, es> &a, const Posit<nbits, es> &b,
                       const Posit<nbits2, es2> &c) {
  constexpr size_t fbits = nbits - 2 - es;  // size of fraction + hidden bit
  constexpr size_t mbits = 2 * fbits;       // size of the multiplier output
  constexpr size_t fbits2 = nbits2 - 2 - es2;
  // TODO: remove hidden bit from fraction. Universal uses 3 more bits for
  // addend. We use 4 because of the extra hidden bit
  constexpr size_t abits = fbits2 + 4;  // size of the addend

  PositFP<8, fbits> va(a), vb(b);
  PositFP<8, mbits> product;
  PositFP<8, fbits2> vc(c);
  PositFP<8, abits + 1> sum;

  if (a.isZero() || b.isZero()) {
    return Posit<nbits2, es2>(c);
  } else {
    product = a * b;
    if (c.isZero()) {
      return Posit<nbits2, es2>(product);
    } else {
      sum = (PositFP<8, fbits2>)product + vc;
      return Posit<nbits2, es2>(sum);
    }
  }
}

template <int sbits, int fbits>
inline bool operator==(const PositFP<sbits, fbits> &lhs,
                       const PositFP<sbits, fbits> &rhs) {
  return (lhs.sign == rhs.sign) && (lhs.scale == rhs.scale) &&
         (lhs.fraction == rhs.fraction);
}

template <int nbits, int es, int sbits, int fbits>
inline bool operator==(const Posit<nbits, es> &lhs,
                       const Posit<nbits, es> &rhs) {
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
// Marshaller<Size> &operator&(Marshaller<Size> &m, PositFP<sbits, fbits> &rhs)
// {
//   typedef PositFP<sbits, fbits> Type;
//   m.template AddField<ac_int<1, false>, 1>(rhs.sign);
//   m.template AddField<ac_int<sbits, true>, sbits>(rhs.scale);
//   m.template AddField<ac_int<fbits, false>, fbits>(rhs.fraction);
//   return m;
// }
