#pragma once

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "../TypeToBits.h"
#include "ApproximationUnit.h"

using namespace ac_math;

#pragma hls_design ccore
template <typename T>
T add(T op0, T op1) {
  return op0 + op1;
}

template <typename T, size_t width>
Pack1D<T, width> vadd(const Pack1D<T, width> op0, const Pack1D<T, width> op1) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = add(op0[i], op1[i]);
  }
  return res;
}

#pragma hls_design ccore
template <typename T>
T mul(T op0, T op1) {
  return op0 * op1;
}

template <typename T, size_t width>
Pack1D<T, width> vmul(const Pack1D<T, width> op0, const Pack1D<T, width> op1) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
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

template <typename T, size_t width>
Pack1D<T, width> vdiv(const Pack1D<T, width> op0, const Pack1D<T, width> op1) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = div(op0[i], op1[i]);
  }
  return res;
}

template <typename T, size_t width>
Pack1D<T, width> vexp(const Pack1D<T, width> op0) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = op0[i].exponential();
  }
  return res;
}

template <typename T, size_t width>
Pack1D<T, width> vabs(Pack1D<T, width> op0) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = op0[i].abs();
  }
  return res;
}

template <typename T, size_t width>
Pack1D<T, width> vrelu(Pack1D<T, width> op0) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = op0[i].relu();
  }
  return res;
}

template <typename T, size_t width>
Pack1D<T, width> vgelu(Pack1D<T, width> op0) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = op0[i].gelu();
  }
  return res;
}

template <typename T, size_t width>
Pack1D<T, width> vsilu(Pack1D<T, width> op0) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = op0[i].silu();
  }
  return res;
}

#pragma hls_design ccore
template <typename T, size_t width>
Pack1D<T, width> vmax(const Pack1D<T, width> op0, const Pack1D<T, width> op1) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = std::max(op0[i], op1[i]);
  }
  return res;
}

// For approximated functions (e.g. exp, gelu, silu, etc.)
#pragma hls_design ccore
template <typename T, size_t width>
Pack1D<T, width> vpoly(const Pack1D<T, width>& op0, const T maxes[NUM_MAXES],
                       const T ranges[NUM_RANGES][NUM_COEFFS],
                       const ac_int<1, false> clamp_min,
                       const ac_int<1, false> clamp_max) {
  Pack1D<T, width> res;
#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = vepoly<T>(op0[i], maxes, ranges, clamp_min, clamp_max);
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

template <typename Input, typename Output, int width>
void vdequantize(const Pack1D<Input, width>& op0, Pack1D<Output, width>& res,
                 ac_int<Output::width, false> scale_bits) {
  Output scale;
  scale.set_bits(scale_bits);
  if (scale.is_zero()) {
    scale = Output::one();
  }

#pragma hls_unroll yes
  for (int i = 0; i < width; i++) {
    res[i] = dequantize(op0[i], scale);
  }
}

template <typename VectorType, typename ScaleType>
ScaleType compute_scale(const VectorType amax, ac_int<16> qparam) {
  ScaleType scale;

  if constexpr (ScaleType::width == ScaleType::e_width) {
    auto max_exp = amax.unbiased_exponent();

    ac_int<VectorType::e_width, true> scaled_exp;
    if (max_exp == 0) {
      scaled_exp = 127;
    } else {
      scaled_exp = max_exp - qparam;
    }

    scale.set_bits(scaled_exp);
  } else {
    VectorType quant_max;
    quant_max.set_bits(qparam);
    scale = amax / quant_max;
  }

  return scale.is_zero() ? ScaleType::one() : scale;
}

template <typename VectorType, typename ScaleType, int width>
ScaleType calculate_mx_scale(const Pack1D<VectorType, width>& op0,
                             ac_int<16> qparam) {
  ScaleType scale;

  if constexpr (ScaleType::width == ScaleType::e_width) {
    using exp_t = ac_int<VectorType::e_width, false>;

    Pack1D<exp_t, width> exponents;
#pragma hls_unroll yes
    for (int i = 0; i < width; i++) {
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
    Pack1D<VectorType, width> temp;
#pragma hls_unroll yes
    for (int i = 0; i < width; i++) {
      temp[i] = op0[i].abs();
    }

    VectorType max_val = tree_max(temp);

    VectorType quant_max;
    quant_max.set_bits(qparam);

    scale = max_val / quant_max;
  }

  return scale.is_zero() ? ScaleType::one() : scale;
}

template <typename... Ts>
void fetch_vector_input(ac_int<4, false> dtype,
                        ac_int<ADDRESS_WIDTH, false> offset,
                        ac_int<32, false> address, ac_int<16, false> burst_size,
                        Connections::Out<MemoryRequest>& channel) {
  ac_int<6, false> width = get_type_width<Ts...>(dtype);
  MemoryRequest request = {offset + address * width / 8, burst_size};
  channel.Push(request);
}

template <typename T, typename VectorType, size_t N, int width, typename... Ts>
bool unpack_vector_data(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                        const ac_int<width, false> bits,
                        Pack1D<VectorType, N>& outputs,
                        ac_int<4, false> packing_index) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  ac_int<16, false> offset = packing_index * T::width * N;

#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    T data;
    data.set_bits(bits.template slc<T::width>(offset + i * T::width));
    outputs[i] = data;
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

template <typename T, size_t N, typename VectorType, unsigned port_width,
          typename... Ts>
bool send_output_data(
    ac_int<4, false> dtype, bool is_codebook_quant,
    Pack1D<VectorType, N> inputs,
    Connections::Out<ac_int<port_width, false>>& output_channel) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int num_words = (T::width * N + port_width - 1) / port_width;

  Pack1D<T, N> outputs;
#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs[i] = cast_output<VectorType, T>(inputs[i], is_codebook_quant);
  }

  ac_int<T::width * N, false> bits;
  bits = BitsToType<decltype(bits)>(TypeToBits(outputs));

  for (int i = 0; i < num_words; i++) {
    output_channel.Push(bits.template slc<port_width>(i * port_width));
  }

  return true;
}

template <typename T, size_t N, unsigned port_width, typename... Ts>
bool send_output_address(
    ac_int<4, false> dtype, ac_int<ADDRESS_WIDTH, false> offset,
    ac_int<32, false> address,
    Connections::Out<ac_int<ADDRESS_WIDTH, false>>& address_channel) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int num_words = (T::width * N + port_width - 1) / port_width;

  for (int i = 0; i < num_words; i++) {
    address_channel.Push(offset + address * T::width / 8 + i * port_width / 8);
  }

  return true;
}

#pragma hls_design ccore
template <typename T>
ac_int<4, false> find_codebook_index(T x, const T B[16]) {
  ac_int<4, false> low = 0;
  ac_int<5, false> high = 16;

  // Stage 1
  ac_int<4, false> mid1 = 8;
  if (x <= B[mid1]) {
    high = mid1;
  } else {
    low = mid1;
  }

  // Stage 2
  ac_int<4, false> mid2 = (low + high) >> 1;
  if (x <= B[mid2]) {
    high = mid2;
  } else {
    low = mid2;
  }

  // Stage 3
  ac_int<4, false> mid3 = (low + high) >> 1;
  if (x <= B[mid3]) {
    high = mid3;
  } else {
    low = mid3;
  }

  // Stage 4
  ac_int<4, false> mid4 = (low + high) >> 1;
  if (x <= B[mid4]) {
    high = mid4;
  } else {
    low = mid4;
  }

  return low;  // 0..15
}
