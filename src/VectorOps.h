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
template <typename Input, typename Output, typename Scale, int Width>
void vquantize(const Pack1D<Input, Width>& op0, Pack1D<Output, Width>& res,
               const Scale scale) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = scale.is_zero() ? op0[i] : op0[i] / static_cast<Input>(scale);
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
    Output temp = op0[i];
    res[i] = scale.is_zero() ? temp : temp * scale;
  }
}

#pragma hls_design ccore
template <typename InputType, typename QuantizedType, typename ScaleType,
          int Width>
void vquantize_mx(const Pack1D<InputType, Width>& op0,
                  Pack1D<InputType, Width>& res, ScaleType& scale) {
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
      scaled_exp = max_exp - QuantizedType::emax;
    }

    scale.set_bits(scaled_exp);
  } else {
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
    scale = amax / static_cast<InputType>(QuantizedType::max());
  }

  if (scale.is_zero()) {
    scale.set_one();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i] / static_cast<InputType>(scale);
  }
}

template <typename T, size_t Width>
void send_request(ac_int<32, false> address, ac_int<64, false> offset,
                  Connections::Out<MemoryRequest>& channel) {
  MemoryRequest request = {offset + address * T::width / 8,
                           Width * T::width / 8};
  channel.Push(request);
}

template <typename VectorType, typename IOType, typename OutputType,
          size_t Width>
void feed_response(Connections::In<Pack1D<IOType, Width>>& input_channel,
                   Pack1D<VectorType, Width>& outputs) {
  if constexpr (OutputType::width >= IOType::width) {
    constexpr int num_words = OutputType::width / IOType::width;
    Pack1D<IOType, Width> response[num_words];

    for (int i = 0; i < num_words; i++) {
      response[i] = input_channel.Pop();
    }

    Pack1D<OutputType, Width> full_response;
    convertPack1D<IOType, OutputType, Width>(response, full_response);

#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      outputs[i] = full_response[i];
    }
  } else {
    throw std::runtime_error("Not implemented");
  }
}

template <typename VectorType, typename IOType, typename OutputType,
          size_t Width>
void vwrite_out(Pack1D<VectorType, Width> inputs, ac_int<32, false> address,
                ac_int<64, false> offset,
                Connections::Out<Pack1D<IOType, Width>>& output_channel,
                Connections::Out<ac_int<64, false>>& address_channel) {
  Pack1D<OutputType, Width> outputs;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    outputs[i] = inputs[i];
  }

  if constexpr (OutputType::width >= IOType::width) {
    constexpr int num_words = OutputType::width / IOType::width;
    Pack1D<IOType, Width> converted_outputs[num_words];

    convertPack1D<IOType, OutputType, Width>(outputs, converted_outputs);

    for (int i = 0; i < num_words; i++) {
      output_channel.Push(converted_outputs[i]);
      address_channel.Push(offset + address * OutputType::width / 8 +
                           i * Width * IOType::width / 8);
    }
  } else {
    throw std::runtime_error("Not implemented");
  }
}
