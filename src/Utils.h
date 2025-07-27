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

template <typename T, size_t N, int port_width, int buf_width>
void process_matrix_input(Connections::In<ac_int<port_width, false>>& response,
                          ac_int<buf_width, false>& outputs) {
  constexpr int num_words = (T::width * N + port_width - 1) / port_width;

  ac_int<num_words * port_width, false> bits;

  for (int i = 0; i < num_words; i++) {
    bits.set_slc(i * port_width, response.Pop());
  }

  constexpr int data_width = buf_width / N;

#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs.set_slc(i * data_width, bits.template slc<T::width>(i * T::width));
  }
}

template <typename T, size_t N, int port_width, int buf_width, typename... Ts>
bool process_matrix_input(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                          Connections::In<ac_int<port_width, false>>& response,
                          ac_int<buf_width, false>& outputs) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  process_matrix_input<T, N, port_width, buf_width>(response, outputs);
  return true;
}

template <typename... Ts>
void send_packed_request(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                         ac_int<ADDRESS_WIDTH, false> offset,
                         ac_int<32, false> address,
                         ac_int<32, false> fetch_width,
                         Connections::Out<MemoryRequest>& channel) {
  ac_int<6, false> width = get_type_width<Ts...>(dtype);
  MemoryRequest request = {offset + address * width / 8, fetch_width};
  channel.Push(request);
}

// Unpack bits into outputs for a specific type
template <typename T, size_t N, int buf_width, int bits_width, typename... Ts>
bool unpack_bits(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                 const ac_int<bits_width, false> bits,
                 ac_int<buf_width, false>& outputs,
                 ac_int<4, false> packing_index) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  constexpr int data_width = buf_width / N;
  ac_int<10, false> offset = packing_index * T::width * N;

#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs.set_slc(i * data_width,
                    bits.template slc<T::width>(offset + i * T::width));
  }

  return true;
}

template <size_t N, int port_width, int buf_width, typename... Ts>
void process_packed_response(
    ac_int<DTYPE_INDEX_WIDTH, false> dtype, ac_int<4, false> num_fetches,
    ac_int<4, false> packing_factor,
    Connections::In<ac_int<port_width, false>>& response,
    Connections::Combinational<ac_int<buf_width, false>>& output) {
  constexpr int max_fetch_width =
      std::max({dtype_fetch_config<Ts, N, port_width>::max_fetch_width...});
  ac_int<max_fetch_width, false> bits;

  for (ac_int<4, false> i = 0;; i++) {
    bits.set_slc(i * port_width, response.Pop());
    if (i == num_fetches - 1) {
      break;
    }
  }

  // Unpack bits into outputs based on dtype
  for (ac_int<4, false> i = 0;; i++) {
    ac_int<buf_width, false> outputs = 0;
    bool handled = (unpack_bits<Ts, N, buf_width, max_fetch_width, Ts...>(
                        dtype, bits, outputs, i) ||
                    ...);

#ifndef __SYNTHESIS__
    if (!handled) {
      throw std::runtime_error("Unsupported dtype for matrix input: " +
                               std::to_string(dtype));
    }
#endif
    output.Push(outputs);

    if (i == packing_factor - 1) {
      break;
    }
  }
}

template <typename T, size_t N, int width, typename... Ts>
bool set_zero(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
              ac_int<width, false>& outputs) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }
  constexpr int output_width = width / N;
#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs.set_slc(i * output_width, T::zero().bits_rep());
  }
  return true;
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
