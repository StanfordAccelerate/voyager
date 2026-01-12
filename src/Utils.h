#pragma once

#include "datatypes/DataTypes.h"

template <typename T, size_t N>
bool send_input_request(ac_int<ADDRESS_WIDTH, false> offset,
                        ac_int<32, false> address,
                        Connections::Out<MemoryRequest>& channel) {
  MemoryRequest request = {offset + address * T::width / 8, N * T::width / 8};
  channel.Push(request);
  return true;
}

template <typename T, size_t N, typename... Ts>
bool fetch_matrix_input(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                        ac_int<ADDRESS_WIDTH, false> offset,
                        ac_int<32, false> address,
                        Connections::Out<MemoryRequest>& channel) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  send_input_request<T, N>(offset, address, channel);
  return true;
}

template <typename T, size_t N, int port_width, int buffer_width>
void process_matrix_input(Connections::In<ac_int<port_width, false>>& response,
                          ac_int<buffer_width, false>& outputs) {
  constexpr int num_words = (T::width * N + port_width - 1) / port_width;

  ac_int<num_words * port_width, false> bits;

  for (int i = 0; i < num_words; i++) {
    bits.set_slc(i * port_width, response.Pop());
  }

  constexpr int data_width = buffer_width / N;

#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs.set_slc(i * data_width, bits.template slc<T::width>(i * T::width));
  }
}

template <typename T, size_t N, int port_width, int buffer_width,
          typename... Ts>
bool process_matrix_input(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                          Connections::In<ac_int<port_width, false>>& response,
                          ac_int<buffer_width, false>& outputs) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  process_matrix_input<T, N, port_width, buffer_width>(response, outputs);
  return true;
}

template <typename... Ts>
void send_packed_request(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                         ac_int<ADDRESS_WIDTH, false> offset,
                         ac_int<32, false> address,
                         ac_int<16, false> burst_bytes,
                         Connections::Out<MemoryRequest>& channel) {
  ac_int<6, false> width = get_type_width<Ts...>(dtype);
  MemoryRequest request = {offset + address * width / 8, burst_bytes};
  channel.Push(request);
}

// Unpack bits into outputs for a specific type
template <typename T, size_t N, int buffer_width, int bits_width,
          typename... Ts>
bool unpack_bits(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                 const ac_int<bits_width, false> bits,
                 ac_int<buffer_width, false>& outputs,
                 ac_int<4, false> packing_index) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int data_width = buffer_width / N;
  ac_int<10, false> offset = packing_index * T::width * N;

#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs.set_slc(i * data_width,
                    bits.template slc<T::width>(offset + i * T::width));
  }

  return true;
}

template <typename T, int width, typename... Ts>
bool get_zero(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
              ac_int<width, false>& bits) {
  if (get_type_index<T, Ts...>() == dtype) {
    bits = T::zero().bits_rep();
    return true;
  }
  return false;
}

template <size_t N, int buffer_width, typename... Ts>
void set_zero(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
              ac_int<buffer_width, false>& outputs, bool codebook,
              ac_int<4, false> zero_idx) {
  constexpr int width = buffer_width / N;
  ac_int<width, false> zero_bits;

  if (codebook) {
    zero_bits = zero_idx;
  } else {
    bool handled = (get_zero<Ts, width, Ts...>(dtype, zero_bits) || ...);
#ifndef __SYNTHESIS__
    if (!handled) {
      throw std::runtime_error("Unsupported dtype for matrix input: " +
                               std::to_string(dtype));
    }
#endif
  }

#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs.set_slc(i * width, zero_bits);
  }
}

template <typename T1, typename T2, int width, typename... Ts>
bool decode_type(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                 ac_int<width, false> bits, T2& output) {
  if (get_type_index<T1, Ts...>() != dtype) {
    return false;
  }

  T1 temp;
  temp.set_bits(bits);
  output = typename T1::decoded(temp);

  return true;
}

#pragma hls_design ccore
template <typename Psum, typename Scale, typename Buffer>
Buffer dequantize_mx_op(const Psum psum, const Scale input_scale,
                        const Scale weight_scale, const Buffer prev_accum) {
  return prev_accum + (Buffer)input_scale * (Buffer)weight_scale * (Buffer)psum;
}

#pragma hls_design ccore
template <typename T, int num_entries, typename... Ts>
T decode_input(const ac_int<T::width, false> bits, const ac_int<8, false> dtype,
               const bool use_codebook,
               const ac_int<T::width, false> code[num_entries]) {
  T output;
  if (use_codebook) {
    output.set_bits(code[bits]);
  } else {
    bool success =
        (decode_type<Ts, T, T::width, Ts...>(dtype, bits, output) || ...);
#ifndef __SYNTHESIS__
    if (!success) {
      std::cerr << "Error: dtype '" << dtype << "' is not valid" << std::endl;
    }
#endif
  }
  return output;
}

#pragma hls_design ccore
template <typename T, typename Scale, typename Vector, int num_entries,
          typename... Ts>
T decode_and_dequantize(const ac_int<T::width, false> bits,
                        const ac_int<8, false> dtype, const bool use_codebook,
                        const ac_int<T::width, false> code[num_entries],
                        const bool dequant, const Scale scale,
                        const Scale zero_point) {
  T output;
  if (use_codebook) {
    output.set_bits(code[bits]);
  } else {
    bool success =
        (decode_type<Ts, T, T::width, Ts...>(dtype, bits, output) || ...);
#ifndef __SYNTHESIS__
    if (!success) {
      std::cerr << "Error: dtype '" << dtype << "' is not valid" << std::endl;
    }
#endif
  }

  if (dequant) {
    output = ((Vector)output - (Vector)zero_point) * (Vector)scale;
  }
  return output;
}
