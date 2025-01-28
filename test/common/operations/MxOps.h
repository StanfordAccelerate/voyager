#pragma once

#include "test/common/operations/Common.h"

template <typename Input, typename Scale>
Scale* calculate_mx_qparam(std::any input_tensor,
                           const codegen::ReduceOp& param) {
  Input* inputs = std::any_cast<Input*>(input_tensor);

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

  Scale* outputs = new Scale[num_blocks * outer_size];

  for (int i = 0; i < outer_size; i++) {
    for (int c = 0; c < num_blocks; c++) {
      ac_int<Input::exponent_width, false> max_exponent = 0;

      int index = i * num_blocks + c;

      for (int block = 0; block < block_size; block++) {
        int input_index = i * mx_axis_size + c * block_size + block;

        ac_int<Input::exponent_width, false> exponent =
            inputs[input_index].unbiased_exponent();

        max_exponent = std::max(max_exponent, exponent);
      }

      // FIXME: 6 is hardcoded for INT8
      ac_int<Input::exponent_width, true> scaled_exponent = max_exponent - 6;

      outputs[index].set_bits(scaled_exponent);
    }
  }

  delete[] inputs;
  return outputs;
}