#pragma once

#include "test/common/operations/Common.h"

template <typename INPUT_T, typename OUTPUT_T>
OUTPUT_T* calculate_mx_qparam(std::any input_tensor,
                              const codegen::ReduceParam& param) {
  INPUT_T* inputs = std::any_cast<INPUT_T*>(input_tensor);

  int mx_axis = param.dim(0);
  if (mx_axis == -1) {
    mx_axis = param.input().shape().size() - 1;
  }
  int mx_axis_size = param.input().shape(mx_axis);

  int tensor_size = 1;
  for (int i = 0; i < param.input().shape().size(); i++) {
    tensor_size *= param.input().shape(i);
  }

  int outer_size = tensor_size / mx_axis_size;

  // assume block size is 32
  int block_size = std::min(mx_axis_size, 32);

  int num_blocks = (mx_axis_size + block_size - 1) / block_size;

  OUTPUT_T* outputs = new OUTPUT_T[num_blocks * outer_size];

  for (int i = 0; i < outer_size; i++) {
    for (int c = 0; c < num_blocks; c++) {
      ac_int<INPUT_T::exponent_width, true> max_exponent = 0;

      int index = i * num_blocks + c;

      for (int block = 0; block < block_size; block++) {
        int input_index = i * mx_axis_size + c * block_size + block;

        ac_int<INPUT_T::exponent_width, true> exponent =
            inputs[input_index].unbiased_exponent();

        max_exponent = std::max(max_exponent, exponent);
      }

      ac_int<INPUT_T::exponent_width, true> scaled_exponent =
          max_exponent - INPUT_T::ac_float_rep::exp_bias -
          (OUTPUT_T::width - 2);

      outputs[index].setbits(scaled_exponent);
    }
  }

  delete[] inputs;
  return outputs;
}