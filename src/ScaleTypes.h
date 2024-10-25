#pragma once

#include <ac_int.h>

// Represents a microscaling type used for scaling, e.g., E8M0
// Currently in this implementation, we don't actually store it according to the
// E8M0 standard as specified in the OCP MX specification, but instead as a
// signed unbiased exponent.
template <int W>
class Scale {
 public:
  typedef ac_int<W, true> ac_int_rep;

  static constexpr unsigned int width = W;
  static constexpr bool is_floating_point = false;
  static constexpr int exp_bias = (1 << (W - 1)) - 1;

  ac_int_rep int_val;

  Scale() {}

  template <int W2, bool S2>
  Scale(const ac_int<W2, S2> &rhs);

#ifndef __SYNTHESIS__
  Scale(const float val);
#endif

  ac_int<W, true> bits_rep() { return int_val; }

  void setbits(const ac_int<W, true> &rhs) { int_val = rhs; }

  template <int W2, bool S2>
  void set_exponent(const ac_int<W2, S2> &exp) {
    int_val = exp;
    // + exp_bias;
  }

  Scale operator+(const Scale &rhs) const {
    Scale result;
    result.int_val = this->int_val + rhs.int_val;
    return result;
  }

#ifndef __SYNTHESIS__
  operator float() const { return int_val; }
#endif

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & int_val;
  }
#endif
};

template <int W>
template <int W2, bool S2>
Scale<W>::Scale(const ac_int<W2, S2> &rhs) {
  int_val = rhs;
}

#ifndef __SYNTHESIS__
template <int W>
Scale<W>::Scale(const float val) {
  int_val = log2(val);
  //  + exp_bias;
}
#endif