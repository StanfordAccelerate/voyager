#pragma once

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "../TypeToBits.h"

using namespace ac_math;

#pragma hls_design ccore
template <typename T>
T add(T op0, T op1) {
  return op0 + op1;
}

// #pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vadd(const Pack1D<T, Width> op0, const Pack1D<T, Width> op1) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = add(op0[i], op1[i]);
  }
  return res;
}

#pragma hls_design ccore
template <typename T>
T mul(T op0, T op1) {
  return op0 * op1;
}

// #pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vmul(const Pack1D<T, Width> op0, const Pack1D<T, Width> op1) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = mul(op0[i], op1[i]);
  }
  return res;
}

#pragma hls_design ccore
template <typename T>
T div(T op0, T op1) {
  // Handle division by zero
  if (op1.is_zero()) {
    return T::zero();  // or some other error handling
  }
  return op0 / op1;
}

// #pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vdiv(const Pack1D<T, Width> op0, const Pack1D<T, Width> op1) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = div(op0[i], op1[i]);
  }
  return res;
}

// #pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vexp(const Pack1D<T, Width> op0) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].exponential();
  }
  return res;
}

#pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vabs(Pack1D<T, Width> op0) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].abs();
  }
  return res;
}

#pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vrelu(Pack1D<T, Width> op0) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].relu();
  }
  return res;
}

// #pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vgelu(Pack1D<T, Width> op0) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].gelu();
  }
  return res;
}

// #pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vsilu(Pack1D<T, Width> op0) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i].silu();
  }
  return res;
}

#pragma hls_design ccore
template <typename T, size_t Width>
Pack1D<T, Width> vmax(const Pack1D<T, Width> op0, const Pack1D<T, Width> op1) {
  Pack1D<T, Width> res;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = std::max(op0[i], op1[i]);
  }
  return res;
}

template <int N>
struct sum_s {
  template <typename T>
  static T sum(Pack1D<T, N> a) {
    Pack1D<T, N / 2> a0;
    Pack1D<T, N - N / 2> a1;

#pragma hls_unroll yes
    for (int i = 0; i < N / 2; i++) {
      a0[i] = a[i];
    }

#pragma hls_unroll yes
    for (int i = 0; i < N - N / 2; i++) {
      a1[i] = a[i + N / 2];
    }

    T m0 = sum_s<N / 2>::sum(a0);
    T m1 = sum_s<N - N / 2>::sum(a1);
    return m0 + m1;
  }
};

// terminate template recursion
template <>
struct sum_s<1> {
  template <typename T>
  static T sum(Pack1D<T, 1> a) {
    return a[0];
  }
};

#pragma hls_design ccore
template <typename T, size_t N>
T tree_sum(Pack1D<T, N> a) {
  return sum_s<N>::sum(a);
}

template <int N>
struct max_s {
  template <typename T>
  static T max(Pack1D<T, N> a) {
    Pack1D<T, N / 2> a0;
    Pack1D<T, N - N / 2> a1;

#pragma hls_unroll yes
    for (int i = 0; i < N / 2; i++) {
      a0[i] = a[i];
    }

#pragma hls_unroll yes
    for (int i = 0; i < N - N / 2; i++) {
      a1[i] = a[i + N / 2];
    }

    T m0 = max_s<N / 2>::max(a0);
    T m1 = max_s<N - N / 2>::max(a1);
    return std::max(m0, m1);
  }
};

// terminate template recursion
template <>
struct max_s<1> {
  template <typename T>
  static T max(Pack1D<T, 1> a) {
    return a[0];
  }
};

#pragma hls_design ccore
template <typename T, size_t N>
T tree_max(Pack1D<T, N> a) {
  return max_s<N>::max(a);
}

#pragma hls_design ccore
template <typename Input, typename Output>
Output dequantize(Input input, Output scale) {
  return (Output)input * scale;
}

#pragma hls_design ccore
template <typename Input, typename Output, int Width>
void vdequantize(const Pack1D<Input, Width>& op0, Pack1D<Output, Width>& res,
                 ac_int<Output::width, false> scale_bits) {
  Output scale;
  scale.set_bits(scale_bits);
  if (scale.is_zero()) {
    scale = Output::one();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = dequantize(op0[i], scale);
  }
}

template <typename VectorType, typename ScaleType, int Width>
ScaleType calculate_mx_scale(const Pack1D<VectorType, Width>& op0,
                             ac_int<16> qparam) {
  ScaleType scale;

  if constexpr (ScaleType::width == ScaleType::e_width) {
    using exp_t = ac_int<VectorType::e_width, false>;

    Pack1D<exp_t, Width> exponents;
#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      exponents[i] = op0[i].unbiased_exponent();
    }

    exp_t max_exp = tree_max(exponents);

    ac_int<VectorType::e_width, true> scaled_exp;
    if (max_exp == 0) {
      scaled_exp = 127;
    } else {
      scaled_exp = max_exp - qparam;
    }

    scale.set_bits(scaled_exp);
  } else {
    Pack1D<VectorType, Width> temp;
#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      temp[i] = op0[i].abs();
    }

    VectorType max_val = tree_max(temp);

    VectorType quant_max;
    quant_max.set_bits(qparam);

    scale = max_val / quant_max;
  }

  return scale.is_zero() ? ScaleType::one() : scale;
}

template <typename T, size_t Width, typename... Ts>
bool fetch_vector_input(ac_int<4, false> dtype, ac_int<32, false> address,
                        ac_int<ADDRESS_WIDTH, false> offset,
                        Connections::Out<MemoryRequest>& channel) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  MemoryRequest request = {offset + address * T::width / 8,
                           Width * T::width / 8};
  channel.Push(request);
  return true;
}

template <typename T, size_t Width, typename VectorType, typename... Ts>
bool process_vector_input(
    ac_int<4, false> dtype,
    Connections::In<ac_int<OC_PORT_WIDTH, false>>& input_channel,
    Pack1D<VectorType, Width>& outputs) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int num_words =
      (T::width * Width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;

  ac_int<num_words * OC_PORT_WIDTH, false> bits;

  for (int i = 0; i < num_words; i++) {
    bits.set_slc(i * OC_PORT_WIDTH, input_channel.Pop());
  }

  ac_int<Width * T::width, false> bits_rep =
      bits.template slc<Width * T::width>(0);

  Pack1D<T, Width> typed = BitsToType<Pack1D<T, Width>>(TypeToBits(bits_rep));

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    outputs[i] = typed[i];
  }

  return true;
}

#pragma hls_design ccore
template <typename T1, typename T2>
T2 cast_output(T1 input, bool is_codebook_quant) {
  T2 output;
  if (is_codebook_quant) {
    output.set_bits(input.bits_rep());
  } else {
    output = input;
  }
  return output;
}

template <typename T, size_t N, typename VectorType, unsigned PortWidth,
          typename... Ts>
bool send_output_data(
    ac_int<4, false> dtype, bool is_codebook_quant,
    Pack1D<VectorType, N> inputs,
    Connections::Out<ac_int<PortWidth, false>>& output_channel) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int num_words = (T::width * N + PortWidth - 1) / PortWidth;

  Pack1D<T, N> outputs;

#pragma hls_unroll yes
  for (unsigned i = 0; i < N; i++) {
    // if (is_codebook_quant) {
    //   outputs[i].set_bits(inputs[i].bits_rep());
    // } else {
    //   outputs[i] = inputs[i];
    // }
    outputs[i] = cast_output<VectorType, T>(inputs[i], is_codebook_quant);
  }

  ac_int<T::width * N, false> bits;
  bits = BitsToType<decltype(bits)>(TypeToBits(outputs));

  for (unsigned i = 0; i < num_words; i++) {
    output_channel.Push(bits.template slc<PortWidth>(i * PortWidth));
  }

  return true;
}

template <typename T, size_t N, unsigned PortWidth, typename... Ts>
bool send_output_address(
    ac_int<4, false> dtype, ac_int<ADDRESS_WIDTH, false> offset,
    ac_int<32, false> address,
    Connections::Out<ac_int<ADDRESS_WIDTH, false>>& address_channel) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int num_words = (T::width * N + PortWidth - 1) / PortWidth;

  for (int i = 0; i < num_words; i++) {
    address_channel.Push(offset + address * T::width / 8 + i * PortWidth / 8);
  }

  return true;
}
