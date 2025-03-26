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
    res[i] = op0[i] + op1[i];
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vmult(const Pack1D<T, Width>& op0, const Pack1D<T, Width>& op1,
           Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i] * op1[i];
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vdiv(const Pack1D<T, Width>& op0, const Pack1D<T, Width>& op1,
          Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i] / op1[i];
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
void vabs(const Pack1D<T, Width>& op0, Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].abs();
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vrelu(const Pack1D<T, Width>& op0, Pack1D<T, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].relu();
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vgelu(const Pack1D<T, Width>& op0, Pack1D<T, Width>& res) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> output_type;
  Pack1D<input_type, Width> temp;

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    temp[i] = op0[i].template to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = ac_gelu_pwl<output_type>(temp[i]);
  }
}

#pragma hls_design ccore
template <typename T, int Width>
void vsilu(const Pack1D<T, Width>& op0, Pack1D<T, Width>& res) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<30, 3, false, AC_RND, AC_SAT> output_type;
  Pack1D<input_type, Width> temp;

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    temp[i] = op0[i].template to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = temp[i] * ac_sigmoid_pwl<output_type>(temp[i]);
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
template <typename Input, typename Output, int Width>
void vdequantize(const Pack1D<Input, Width>& op0, Pack1D<Output, Width>& res,
                 ac_int<Output::width, false> scale_bits) {
  Output scale;
  scale.set_bits(scale_bits);

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    Output temp = op0[i];
    res[i] = scale.is_zero() ? temp : temp * scale;
  }
}

template <typename VectorType, typename ScaleType, int Width>
void vquantize_mx(const Pack1D<VectorType, Width>& op0, ScaleType& scale,
                  ac_int<16> qparam) {
  if constexpr (ScaleType::width == ScaleType::e_width) {
    using exp_t = ac_int<VectorType::e_width, false>;

    Pack1D<exp_t, Width> exponents;
#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      exponents[i] = op0[i].unbiased_exponent();
    }

    exp_t max_exp = treemax(exponents);

    ac_int<VectorType::e_width, true> scaled_exp;
    if (max_exp == 0) {
      scaled_exp = 127;
    } else {
      scaled_exp = max_exp - qparam;
    }

    scale.set_bits(scaled_exp);
  } else {
    constexpr int num_stage = constexpr_log2(Width);
    Pack1D<VectorType, Width> temp[num_stage + 1];

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

    VectorType quant_max;
    quant_max.set_bits(qparam);

    scale = temp[num_stage][0] / quant_max;
  }

  if (scale.is_zero()) {
    scale = ScaleType::one();
  }
}

template <typename T, size_t Width>
void send_request(ac_int<32, false> address,
                  ac_int<ADDRESS_WIDTH, false> offset,
                  Connections::Out<MemoryRequest>& channel) {
  MemoryRequest request = {offset + address * T::width / 8,
                           Width * T::width / 8};
  channel.Push(request);
}

template <typename FetchType, typename VectorType, size_t Width>
void feed_response(Connections::In<ac_int<OC_PORT_WIDTH, false>>& input_channel,
                   Pack1D<VectorType, Width>& outputs) {
  constexpr int num_words = FetchType::width * Width / OC_PORT_WIDTH;

  static_assert(
      num_words > 0,
      "Width of input type must be greater than or equal to the width "
      "of the input channel");

  ac_int<FetchType::width * Width, false> bits;

  for (int i = 0; i < num_words; i++) {
    bits.set_slc(i * OC_PORT_WIDTH, input_channel.Pop());
  }

  Pack1D<FetchType, Width> typed =
      BitsToType<Pack1D<FetchType, Width>>(TypeToBits(bits));

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    outputs[i] = typed[i];
  }
}

template <typename VectorType, typename OutputType, size_t Width>
void vwrite_out(
    Pack1D<VectorType, Width> inputs, ac_int<32, false> address,
    ac_int<ADDRESS_WIDTH, false> offset,
    Connections::Out<ac_int<OC_PORT_WIDTH, false>>& output_channel,
    Connections::Out<ac_int<ADDRESS_WIDTH, false>>& address_channel) {
  constexpr int num_words = OutputType::width * Width / OC_PORT_WIDTH;

  static_assert(
      num_words > 0,
      "Width of output type must be greater than or equal to the width "
      "of the output channel");

  Pack1D<OutputType, Width> outputs;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    outputs[i] = inputs[i];
  }

  ac_int<OutputType::width * Width, false> bits;
  bits = BitsToType<decltype(bits)>(TypeToBits(outputs));

  for (int i = 0; i < num_words; i++) {
    output_channel.Push(bits.template slc<OC_PORT_WIDTH>(i * OC_PORT_WIDTH));
    address_channel.Push(offset + address * OutputType::width / 8 +
                         i * OC_PORT_WIDTH / 8);
  }
}
