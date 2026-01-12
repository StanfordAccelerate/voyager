#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

#include <ac_int.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl_vha.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_sqrt_pwl.h>

#include "StdFloatTypes.h"

template <int W, bool S>
class Int {
 public:
  static constexpr unsigned int width = W;
  static constexpr int emax = W - 2;  // max normal exponent

  typedef ac_int<W, S> ac_int_rep;
  typedef ac_fixed<2 * W, W, true> ac_int_to_fixed_rep;
  typedef ac_fixed<2 * W, W, false> ac_int_to_fixed_rep_out;
  typedef Int<W, S> decoded;

  ac_int_rep int_val;

  Int() {}

#ifndef __SYNTHESIS__
  Int(const float val);
#endif

  template <int W2, bool S2>
  Int(const ac_int<W2, S2>& other);

  template <int W2, bool S2>
  Int(const Int<W2, S2>& other);

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  Int(const StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>& other);

  ac_int<W, false> bits_rep() { return int_val; }

  void set_bits(int i) { int_val = i; }

  template <int W2, bool S2>
  void set_bits(ac_int<W2, S2> bits) {
    int_val = bits;
  }

  static Int zero() {
    Int<W, S> r;
    r.int_val = 0;
    return r;
  }

  template <int W2, bool S2>
  Int<W2, S2> fma(Int& b, Int<W2, S2>& c);

  Int operator+(const Int& rhs) const;
  Int operator-(const Int& rhs) const;
  Int operator*(const Int& rhs) const;
  Int operator/(const Int& rhs) const;

  Int& operator+=(const Int& rhs);
  Int& operator-=(const Int& rhs);
  Int& operator*=(const Int& rhs);
  Int& operator/=(const Int& rhs);

  bool operator==(const Int& rhs) const;
  bool operator!=(const Int& rhs) const;
  bool operator>(const Int& rhs) const;
  bool operator>=(const Int& rhs) const;
  bool operator<(const Int& rhs) const;
  bool operator<=(const Int& rhs) const;

#ifndef __SYNTHESIS__
  operator float() const { return float(int_val); }
#endif

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>() const {
    using std_float_t =
        StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>;
    using rep = typename std_float_t::ac_float_rep;

    std_float_t f;
    // 1-bit ac_int conversion to ac_std_float will cause an error during
    // synthesis: 'ac_int' is declared with non-positive width of '0'
    if constexpr (W == 1) {
      if (int_val == 0) {
        f.float_val = rep::zero();
      } else {
        f.float_val = S ? -rep::one() : rep::one();
      }
    } else {
      f.float_val = rep(int_val);
    }
    return f;
  }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & int_val;
  }
#endif
};

#ifndef __SYNTHESIS__
template <int W, bool S>
Int<W, S>::Int(const float val) {
  int_val = Int<W, S>::ac_int_rep(val);
}
#endif

template <int W, bool S>
template <int W2, bool S2>
Int<W, S>::Int(const ac_int<W2, S2>& other) {
  int_val = ac_int_rep(other);
}

template <int W, bool S>
template <int W2, bool S2>
Int<W, S>::Int(const Int<W2, S2>& other) {
  int_val = static_cast<ac_int_rep>(other.int_val);
}

template <int W, bool S>
template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
Int<W, S>::Int(
    const StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>& other) {
  int_val = other.float_val
                .template convert_to_ac_fixed<W, W, S, AC_RND_CONV, AC_SAT>()
                .to_ac_int();
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator+(const Int& rhs) const {
  return int_val + rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator-(const Int& rhs) const {
  return int_val - rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator*(const Int& rhs) const {
  return int_val * rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator/(const Int& rhs) const {
  return int_val / rhs.int_val;
}

template <int W, bool S>
inline Int<W, S>& Int<W, S>::operator+=(const Int& rhs) {
  int_val += rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S>& Int<W, S>::operator-=(const Int& rhs) {
  int_val -= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S>& Int<W, S>::operator*=(const Int& rhs) {
  int_val *= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S>& Int<W, S>::operator/=(const Int& rhs) {
  int_val /= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline bool Int<W, S>::operator==(const Int& rhs) const {
  return int_val == rhs.int_val;
}

template <int W, bool S>
inline bool Int<W, S>::operator!=(const Int& rhs) const {
  return int_val != rhs.int_val;
}

template <int W, bool S>
inline bool Int<W, S>::operator>(const Int& rhs) const {
  return int_val > rhs.int_val;
}

template <int W, bool S>
inline bool Int<W, S>::operator>=(const Int& rhs) const {
  return int_val >= rhs.int_val;
}

template <int W, bool S>
inline bool Int<W, S>::operator<(const Int& rhs) const {
  return int_val < rhs.int_val;
}

template <int W, bool S>
inline bool Int<W, S>::operator<=(const Int& rhs) const {
  return int_val <= rhs.int_val;
}

template <int W, bool S>
template <int W2, bool S2>
Int<W2, S2> Int<W, S>::fma(Int<W, S>& b, Int<W2, S2>& c) {
  return static_cast<Int<W2, S2>>(*this) * static_cast<Int<W2, S2>>(b) + c;
}
