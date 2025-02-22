#pragma once

#include <ac_math/ac_gelu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>

using namespace ac_math;

#pragma hls_design ccore
template <typename T, int Width>
void vadd(const Pack1D<T, Width>& op0, const Pack1D<T, Width>& op1,
          Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = static_cast<T>(op0[i] + op1[i]);
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vmult(const Pack1D<T, Width>& op0, const Pack1D<T, Width>& op1,
           Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = static_cast<T>(op0[i] * op1[i]);
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vexp(const Pack1D<T, Width>& op0, Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].exponential();
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vscaleexp(const Pack1D<T, Width>& op0, ac_int<8, false> scale,
               Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i].scale_exp(scale);
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vrelu(const Pack1D<T, Width>& op0, const Pack1D<T, Width>& mask,
           bool is_masked, Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    if (is_masked) {
      res[i] = op0[i].masked_relu(mask[i]);
    } else {
      res[i] = op0[i].relu();
    }
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vgelu(const Pack1D<T, Width>& op0, Pack1D<T, Width>& res) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> output_type;
  Pack1D<input_type, Width> tmp;

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    tmp[i] = op0[i].template to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = ac_gelu_pwl<output_type>(tmp[i]);
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vsilu(const Pack1D<T, Width>& op0, Pack1D<T, Width>& res) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<30, 3, false, AC_RND, AC_SAT> output_type;
  Pack1D<input_type, Width> tmp;

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    tmp[i] = op0[i].template to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = tmp[i] * ac_sigmoid_pwl<output_type>(tmp[i]);
  }
}

#pragma hls_design ccore
template <typename Input, typename Output, typename Scale, int Width>
void vquantize(const Pack1D<Input, Width>& op0, Pack1D<Output, Width>& res,
               const Scale scale) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i] / static_cast<Input>(scale);
  }
}

#pragma hls_design ccore
template <typename Input, typename Output, int Width>
void vdequantize(const Pack1D<Input, Width>& op0, Pack1D<Output, Width>& res,
                 ac_int<Output::width, false> scale_bits) {
  Output scale;
  scale.set_bits(scale_bits);

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = static_cast<Output>(op0[i]) * scale;
  }
}

constexpr int constexpr_log2(unsigned int n, int result = 0) {
  return (n < 2) ? result : constexpr_log2(n / 2, result + 1);
}

#pragma hls_design ccore
template <typename T, size_t Width>
T treeadd(const Pack1D<T, Width>& op) {
  constexpr int num_stage = constexpr_log2(Width);
  Pack1D<T, Width> temp[num_stage + 1];

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    temp[0][i] = op[i];
  }

#pragma hls_unroll yes
  for (int stage = 1; stage <= num_stage; stage++) {
#pragma hls_unroll yes
    for (int i = 0; i < Width / (1 << stage); i++) {
      temp[stage][i] = temp[stage - 1][i * 2] + temp[stage - 1][i * 2 + 1];
    }
  }

  return temp[num_stage][0];
}

#pragma hls_design ccore
template <typename T, size_t Width>
T treemax(const Pack1D<T, Width>& op) {
  constexpr int num_stage = constexpr_log2(Width);
  Pack1D<T, Width> temp[num_stage + 1];

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    temp[0][i] = op[i];
  }

#pragma hls_unroll yes
  for (int stage = 1; stage <= num_stage; stage++) {
#pragma hls_unroll yes
    for (int i = 0; i < Width / (1 << stage); i++) {
      temp[stage][i] =
          std::max(temp[stage - 1][i * 2], temp[stage - 1][i * 2 + 1]);
    }
  }

  return temp[num_stage][0];
}

#pragma hls_design ccore
template <typename InputType, typename OutputType, typename ScaleType,
          int Width>
void vquantize_mx(const Pack1D<InputType, Width>& op0,
                  Pack1D<OutputType, Width>& res, ScaleType& scale) {
  if constexpr (ScaleType::width == ScaleType::e_width) {
    using exp_t = ac_int<InputType::e_width, false>;

    Pack1D<exp_t, Width> exponents;
#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      exponents[i] = op0[i].unbiased_exponent();
    }

    exp_t max_exp = treemax(exponents);

    ac_int<InputType::e_width, true> scaled_exp;
    if (max_exp == 0) {
      scaled_exp = 127;
    } else {
      scaled_exp = max_exp - OutputType::emax;
    }

    scale.set_bits(scaled_exp);
  } else {
    //     Pack1D<InputType, Width> temp;
    // #pragma hls_unroll yes
    //     for (int i = 0; i < Width; i++) {
    //       temp[i] = op0[i].abs();
    //     }

    //     InputType amax = treemax(temp);

    // TODO: Catapult HLS exhibits an issue where using the treemax function in
    // both the vector unit reduction and this location causes incorrect outputs
    // from the vector unit max reduction. The root cause is unclear, but as a
    // workaround, explicitly implement the logic here instead of using treemax.
    constexpr int num_stage = constexpr_log2(Width);
    Pack1D<InputType, Width> temp[num_stage + 1];

#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      temp[0][i] = op0[i].abs();
    }

#pragma hls_unroll yes
    for (int stage = 1; stage <= num_stage; stage++) {
#pragma hls_unroll yes
      for (int i = 0; i < Width / (1 << stage); i++) {
        temp[stage][i] =
            std::max(temp[stage - 1][i * 2], temp[stage - 1][i * 2 + 1]);
      }
    }

    InputType amax = temp[num_stage][0];

    InputType max_value = static_cast<InputType>(OutputType::max());
    scale = amax * max_value.reciprocal();

    if (scale.to_ac_float() == ScaleType::ac_float_rep::zero()) {
      scale.set_one();
    }
  }

  vquantize<InputType, OutputType, ScaleType, Width>(op0, res, scale);
}
