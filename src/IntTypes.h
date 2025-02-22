#pragma once

// clang-format off
#include <ac_int.h>
// clang-format on
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_reciprocal_pwl_vha.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_sqrt_pwl.h>

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
  Int(const ac_int<W2, S2> &other);

  template <int W2, bool S2>
  Int(const Int<W2, S2> &other);

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  Int(const StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q> &other);

  ac_int<W, false> bits_rep() { return int_val; }

  void set_bits(int i) { int_val = i; }

  void set_zero() { int_val = 0; }

  static decoded max() {
    return ac_int_rep(S ? (1 << (W - 1)) - 1 : (1 << W) - 1);
  }

  template <int W2, bool S2>
  Int<W2, S2> fma(Int &b, Int<W2, S2> &c);

  Int operator+(const Int &rhs) const;
  Int operator*(const Int &rhs) const;
  Int operator/(const Int &rhs) const;
  Int &operator+=(const Int &rhs);
  Int &operator-=(const Int &rhs);
  Int &operator*=(const Int &rhs);
  Int &operator/=(const Int &rhs);

  bool operator<(const Int &rhs) const;
  operator float() const { return float(int_val); }

  template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>() const {
    using std_float_t =
        StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q>;
    std_float_t f;
    f.float_val = typename std_float_t::ac_float_rep(int_val);
    return f;
  }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
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
Int<W, S>::Int(const ac_int<W2, S2> &other) {
  int_val = ac_int_rep(other);
}

template <int W, bool S>
template <int W2, bool S2>
Int<W, S>::Int(const Int<W2, S2> &other) {
  int_val = static_cast<ac_int_rep>(other.int_val);
}

template <int W, bool S>
template <int mantissa, int exp, bool use_dw_impl, bool ieee_compliance,
          ac_q_mode Q>
Int<W, S>::Int(
    const StdFloat<mantissa, exp, use_dw_impl, ieee_compliance, Q> &other) {
  int_val = other.float_val.template convert_to_ac_int<W, S>();
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator+(const Int &rhs) const {
  return int_val + rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator*(const Int &rhs) const {
  return int_val * rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> Int<W, S>::operator/(const Int &rhs) const {
  return int_val / rhs.int_val;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator+=(const Int &rhs) {
  int_val += rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator-=(const Int &rhs) {
  int_val -= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator*=(const Int &rhs) {
  int_val *= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline Int<W, S> &Int<W, S>::operator/=(const Int &rhs) {
  int_val /= rhs.int_val;
  return *this;
}

template <int W, bool S>
inline bool Int<W, S>::operator<(const Int &rhs) const {
  return int_val < rhs.int_val;
}

template <int W, bool S>
template <int W2, bool S2>
Int<W2, S2> Int<W, S>::fma(Int<W, S> &b, Int<W2, S2> &c) {
  return static_cast<Int<W2, S2> >(*this) * static_cast<Int<W2, S2> >(b) + c;
}
