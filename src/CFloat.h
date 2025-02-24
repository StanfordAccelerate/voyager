#pragma once

#include <ac_int.h>

#include <cmath>

/*
 * Wrapper class for C/C++ float used to provide a common
 * interface across different data types.
 * This is only meant to be used in the gold model, not in HLS.
 */
class CFloat {
 public:
  static constexpr unsigned int width = 32;
  float float_val;

  typedef CFloat Decoded;

  CFloat() : float_val(0) {}
  CFloat(const float val) : float_val(val) {}

  template <int W, int I, bool S, ac_q_mode Q2, ac_o_mode O>
  CFloat(const ac_fixed<W, I, S, Q2, O> &rhs) {
    float_val = rhs.to_double();
  }

  template <int WFX, int IFX, bool SFX, ac_q_mode QFX, ac_o_mode OFX>
  ac_fixed<WFX, IFX, SFX, QFX, OFX> to_ac_fixed() {
    return ac_fixed<WFX, IFX, SFX, QFX, OFX>(float_val);
  }

  ac_int<width, false> bits_rep() {
    uint32_t float_bits = *reinterpret_cast<uint32_t *>(&float_val);
    ac_int<width, false> bits = float_bits;
    return bits;
  }

  void negate() { float_val = -float_val; }

  void relu() {
    if (float_val < 0) float_val = 0;
  }

  void masked_relu(const CFloat &mask) {
    if (mask.float_val == 0) float_val = 0;
  }

  void set_bits(int i) {
    uint32_t float_bits = i;
    float_val = *reinterpret_cast<float *>(&float_bits);
  }

  void set_zero() { float_val = 0; }

  void custom_converted_reciprocal() { this->reciprocal(); }

  void reciprocal() { float_val = 1.0 / float_val; }

  void exponential() { float_val = exp(float_val); }

  CFloat inv_sqrt() { return CFloat(1.0 / std::sqrt(float_val)); }

  CFloat sqrt() { return CFloat(std::sqrt(float_val)); }

  CFloat max1() {
    if (float_val > 1) {
      float_val = 1;
    }
    return *this;
  }

  void expScale(ac_int<8, false> offset) {
    // TODO: fix this to be scale the exponent
  }

  CFloat fma(CFloat &b, CFloat &c) {
    return CFloat(float_val * b.float_val + c.float_val);
  }

  CFloat operator+(const CFloat &rhs) {
    return CFloat(float_val + rhs.float_val);
  }
  CFloat operator*(const CFloat &rhs) {
    return CFloat(float_val * rhs.float_val);
  }
  CFloat operator/(const CFloat &rhs) {
    return CFloat(float_val / rhs.float_val);
  }
  CFloat &operator+=(const CFloat &rhs) {
    float_val += rhs.float_val;
    return *this;
  }
  CFloat &operator-=(const CFloat &rhs) {
    float_val -= rhs.float_val;
    return *this;
  }
  CFloat &operator*=(const CFloat &rhs) {
    float_val *= rhs.float_val;
    return *this;
  }
  CFloat &operator/=(const CFloat &rhs) {
    float_val /= rhs.float_val;
    return *this;
  }

  bool operator<(const CFloat &rhs) const { return float_val < rhs.float_val; }
  operator float() const { return float_val; }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    assert(false && "Marshall not implemented for CFloat");
  }
#endif
};
