#pragma once

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
template <typename Psum, typename Scale, typename Buffer, int N>
Pack1D<Buffer, N> dequantize_mx(const Pack1D<Psum, N>& psums,
                                const Scale input_scale,
                                const Pack1D<Scale, N>& weight_scales,
                                const Pack1D<Buffer, N>& prev_accum,
                                bool is_mx_op) {
  Pack1D<Buffer, N> outputs;
#pragma hls_unroll yes
  for (int i = 0; i < N; i++) {
    outputs[i] = static_cast<Buffer>(psums[i]);
    if (is_mx_op) {
      outputs[i] *= static_cast<Buffer>(input_scale) *
                    static_cast<Buffer>(weight_scales[i]);
    }
    outputs[i] += prev_accum[i];
  }
  return outputs;
}