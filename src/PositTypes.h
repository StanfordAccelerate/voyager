#pragma once

#include <ac_int.h>

inline int max(int a, int b) { return a > b ? a : b; }

class PositFP;

class Posit {
 public:
  Posit() {}
  Posit(const PositFP &input);
  Posit(int i) { bits = i; }

  // overridden operators
  Posit &operator+=(const Posit &rhs);
  Posit &operator-=(const Posit &rhs);
  Posit &operator*=(const Posit &rhs);
  Posit operator*(const Posit &rhs);
  bool operator<(const Posit &rhs) const;
  operator float() const;

  ac_int<8, false> bits;
  static const unsigned int width = 8;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &bits;
  }

  inline friend void sc_trace(sc_trace_file *tf, const Posit &posit,
                              const std::string &name) {
    // TODO
  }

  inline friend std::ostream &operator<<(ostream &os, const Posit &posit) {
    os << float(posit);
    return os;
  }

 private:
  static const int nbits = 8;
  static const int es = 1;
};

class PositFP {
 public:
  PositFP() {}
  PositFP(const Posit &input);
  PositFP(int i) { setZero(); }

  bool isZero() const { return scale == -127; }
  void setZero() {
    fraction = 0;
    scale = -127;
  }

  PositFP &operator+=(const PositFP &rhs);
  PositFP &operator+=(const Posit &rhs);

  PositFP operator*(const PositFP &op) {
    PositFP result;
    result.sign = sign ^ op.sign;
    result.scale = scale + op.scale;

    result.fraction = (fraction >> 8) * (op.fraction >> 8);

    if (result.fraction[15]) {
      result.scale++;
    } else {
      result.fraction <<= 1;
    }

    return result;
  }

  PositFP operator+(const PositFP &op) {
    PositFP result;
    // align the fraction
    int result_scale = max(scale, op.scale);
    ac_int<16, false> r1 = fraction >> (result_scale - scale + 1);
    ac_int<16, false> r2 = op.fraction >> (result_scale - op.scale + 1);
    ac_int<16, false> sum;

    int shift = 0;
    if (sign == op.sign) {
      result.sign = sign;
      sum = r1 + r2;
      if (sum[15]) shift = -1;
    } else {
      if (r1 > r2) {
        sum = r1 - r2;
        result.sign = sign;
      } else if (r1 < r2) {
        sum = r2 - r1;
        result.sign = op.sign;
      } else {
        result.setZero();
        return result;
      }
      shift = 14 - fsb(sum);
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

  PositFP operator-(const PositFP &op) {
    PositFP negOp = op;
    negOp.sign = !negOp.sign;

    return *this + negOp;
  }

  PositFP fma(const PositFP &op2, const PositFP &op3) {
    if (isZero() || op2.isZero()) {
      return op3;
    } else {
      PositFP product = *this * op2;

      if (op3.isZero()) {
        return product;
      } else {
        PositFP sum = product + op3;
        return sum;
      }
    }
  }

  bool operator<(const PositFP &rhs) const {
    if (sign ^ rhs.sign) return sign;
    if (scale != rhs.scale) return scale < rhs.scale;
    return fraction < rhs.fraction;
  }

  ac_int<1, false> sign;
  ac_int<8, true> scale;       // regime + exponent
  ac_int<16, false> fraction;  // 1 hidden bit + max 5 bit

  static const unsigned int width = 1 + 8 + 16;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &sign;
    m &scale;
    m &fraction;
  }

  inline friend void sc_trace(sc_trace_file *tf, const PositFP &positFP,
                              const std::string &name) {
    // TODO
  }

  inline friend std::ostream &operator<<(ostream &os, const PositFP &positFP) {
    Posit p = positFP;
    os << p;
    return os;
  }

 private:
  static const int nbits = 8;
  static const int es = 1;

  int trailingZeros(unsigned int v) {
    static const int
        Mod37BitPosition[] =  // map a bit value mod 37 to its position
        {32, 0, 1,  26, 2,  23, 27, 0,  3,  16, 24, 30, 28,
         11, 0, 13, 4,  7,  17, 0,  25, 22, 31, 15, 29, 10,
         12, 6, 0,  21, 14, 9,  5,  20, 8,  19, 18};
    return Mod37BitPosition[(-v & v) % 37];
  }

  int fsb(ac_int<8, false> x) {
    x |= x >> 4;
    x |= x >> 2;
    x |= x >> 1;
    x ^= x >> 1;
    return trailingZeros(x);
  }
};

// Encoder
inline Posit::Posit(const PositFP &input) {
  if (input.isZero()) {
    bits = 0;
  } else {
    // TODO: handle infinity
    if ((input.scale >> es) > 6 or (input.scale >> es) < -6) {
      bits = input.scale > 0 ? 0xEF : 0x1;
    } else {
      int pt_len = nbits + 3 + es;
      ac_int<12, false> pt_bits, regime, exponent, fraction, sticky_bit;

      bool s = input.sign;
      int e = input.scale;
      bool r = (e >= 0);

      int run = r ? (1 + (e >> es)) : -(e >> es);
      regime = r ? (1 << (run + 1)) - 1 : 0;
      regime[0] = !r;

      exponent = (e % 2 + 2) % 2;
      int nf = max(0, nbits + 1 - (2 + run + es));
      fraction = (input.fraction << 1) >> (16 - nf);

      pt_bits = regime << es + nf + 1;
      pt_bits |= exponent << nf + 1;
      pt_bits |= fraction << 1;
      pt_bits |= input.fraction << nf ? 0x1 : 0x0;  // any after fbits - 1 - nf
      // std::cerr << pt_bits.to_string(AC_BIN) << std::endl;

      int len = 1 + max(nbits + 1, 2 + run + es);
      bool blast = pt_bits[len - nbits];
      bool bafter = pt_bits[len - nbits - 1];
      bool bsticky = pt_bits << (pt_len - (len - nbits - 1));
      bool rb = (blast & bafter) | (bafter & bsticky);

      bits = pt_bits >> (len - nbits);
      if (rb) bits++;
    }
    if (input.sign) bits = bits.bit_complement() + 1;
  }
}

// decoder
inline PositFP::PositFP(const Posit &input) {
  ac_int<8, false> bits = input.bits;
  if (bits == 0) {
    fraction = 0;
    scale = -127;
  } else {
    sign = bits[nbits - 1];
    if (sign) bits = (bits - 1).bit_complement();  // convert to positive posit

    bits <<= 1;  // remove sign bit
    int leadingBit = bits[nbits - 1];
    // std::cerr << "leading bit: " << leadingBit << std::endl;
    int index = leadingBit ? fsb(bits.bit_complement()) : fsb(bits);

    // std::cerr << "bit pattern: " << bits.to_string(AC_BIN) << std::endl;
    // std::cerr << "index: " << index << std::endl;
    scale = leadingBit ? 6 - index : index - 7;
    scale *= 2;
    // std::cerr << "scale: " << scale << std::endl;
    if (index > 0) scale += bits[index - 1];  // exponent bit
    // std::cerr << "scale: " << scale << std::endl;

    // std::cerr << "bit pattern: " << bits.to_string(AC_BIN) << std::endl;
    fraction = bits << (nbits - index + es);
    // printf("fraction: %s\n", fraction.to_string(AC_BIN).c_str());
    fraction = (fraction << 7) | 0x8000;
    // printf("fraction: %s\n", fraction.to_string(AC_BIN).c_str());
  }
}

inline Posit &Posit::operator+=(const Posit &rhs) {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  *this = op1 + op2;
  return *this;
}

inline Posit &Posit::operator-=(const Posit &rhs) {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  *this = op1 - op2;
  return *this;
}

inline Posit &Posit::operator*=(const Posit &rhs) {
  *this = *this * rhs;
  return *this;
}

inline Posit Posit::operator*(const Posit &rhs) {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  return op1 * op2;
}

inline bool Posit::operator<(const Posit &rhs) const {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  return op1 < op2;
}

inline PositFP &PositFP::operator+=(const PositFP &rhs) {
  *this = *this + rhs;

  return *this;
}

inline PositFP &PositFP::operator+=(const Posit &rhs) {
  PositFP op = rhs;
  *this = *this + op;

  return *this;
}

inline Posit::operator float() const {
  union ufloat {
    float f;
    uint32_t u;
  };
  union ufloat uf;

  PositFP p = *this;
  uf.u = p.sign ? 1 << 31 : 0;
  uf.u += (p.scale + 127) << 23;
  uf.u += (p.fraction & ((1 << 15) - 1)) << 8;
  return uf.f;
}

inline bool operator==(const PositFP &lhs, const PositFP &rhs) {
  return (lhs.sign == rhs.sign) && (lhs.scale == rhs.scale) &&
         (lhs.fraction == rhs.fraction);
}