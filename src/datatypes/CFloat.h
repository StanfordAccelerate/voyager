#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

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

  typedef CFloat decoded;

  CFloat() : float_val(0) {}
  CFloat(const float val) : float_val(val) {}

  template <int W, int I, bool S, ac_q_mode Q2, ac_o_mode O>
  CFloat(const ac_fixed<W, I, S, Q2, O>& rhs) {
    float_val = rhs.to_double();
  }

  template <int WFX, int IFX, bool SFX, ac_q_mode QFX, ac_o_mode OFX>
  ac_fixed<WFX, IFX, SFX, QFX, OFX> to_ac_fixed() const {
    return ac_fixed<WFX, IFX, SFX, QFX, OFX>(float_val);
  }

  ac_int<width, false> bits_rep() {
    uint32_t float_bits;
    std::memcpy(&float_bits, &float_val, sizeof(float_bits));
    ac_int<width, false> bits = float_bits;
    return bits;
  }

  void set_bits(uint32_t bits) { std::memcpy(&float_val, &bits, sizeof(bits)); }

  static CFloat zero() { return CFloat(0.0f); }

  CFloat abs() const { return std::abs(float_val); }
  CFloat negate() { return -float_val; }
  CFloat exponential() const { return std::exp(float_val); }
  CFloat reciprocal() const { return 1.0 / float_val; }
  CFloat sqrt() const { return std::sqrt(float_val); }
  CFloat inv_sqrt() const { return 1.0 / std::sqrt(float_val); }
  CFloat relu() const { return float_val < 0 ? 0 : float_val; }
  CFloat gelu() {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    return 0.5f * float_val *
           (1.0f + std::tanh(sqrt_2_over_pi *
                             (float_val + 0.044715f * pow(float_val, 3))));
  }
  CFloat silu() { return float_val / (1.0f + std::exp(-float_val)); }
  CFloat tanh() { return std::tanh(float_val); }

  CFloat fma(CFloat& b, CFloat& c) {
    return CFloat(float_val * b.float_val + c.float_val);
  }

  CFloat operator+(const CFloat& rhs) {
    return CFloat(float_val + rhs.float_val);
  }
  CFloat operator*(const CFloat& rhs) {
    return CFloat(float_val * rhs.float_val);
  }
  CFloat operator/(const CFloat& rhs) {
    return CFloat(float_val / rhs.float_val);
  }
  CFloat& operator+=(const CFloat& rhs) {
    float_val += rhs.float_val;
    return *this;
  }
  CFloat& operator-=(const CFloat& rhs) {
    float_val -= rhs.float_val;
    return *this;
  }
  CFloat& operator*=(const CFloat& rhs) {
    float_val *= rhs.float_val;
    return *this;
  }
  CFloat& operator/=(const CFloat& rhs) {
    float_val /= rhs.float_val;
    return *this;
  }

  bool operator<(const CFloat& rhs) const { return float_val < rhs.float_val; }
  operator float() const { return float_val; }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    assert(false && "Marshall not implemented for CFloat");
  }
#endif
};
