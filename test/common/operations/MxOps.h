#pragma once

#include "test/common/operations/Common.h"

template <typename INPUT_T, typename OUTPUT_T>
OUTPUT_T* calculate_mx_qparam(std::any input_tensor,
                              const codegen::ReduceParam& param) {
  INPUT_T* inputs = std::any_cast<INPUT_T*>(input_tensor);

  // assume block size is 32

  int N = param.input().shape(0);
  int C = param.input().shape(1);
  int H = param.input().shape(2);
  int W = param.input().shape(3);

  int block_size = std::min(C, 32);
  int num_blocks = (C + block_size - 1) / block_size;

  OUTPUT_T* outputs = new OUTPUT_T[num_blocks * H * W];

  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      for (int c = 0; c < num_blocks; c++) {
        ac_int<INPUT_T::exponent_width, true> max_exponent = 0;

        int index = h * W * num_blocks + w * num_blocks + c;

        for (int block = 0; block < block_size; block++) {
          int input_index = h * W * C + w * C + c * block_size + block;

          ac_int<INPUT_T::exponent_width, true> exponent =
              inputs[input_index].unbiased_exponent();

          max_exponent = std::max(max_exponent, exponent);

          std::cout << inputs[input_index] << " ";
          // std::cout << "input_index: " << input_index << " ";
        }
        std::cout << std::endl;

        ac_int<INPUT_T::exponent_width, true> scaled_exponent =
            max_exponent - INPUT_T::ac_float_rep::exp_bias -
            (OUTPUT_T::width - 2);

        std::cout << "scaled_exponent: " << scaled_exponent << std::endl;

        // outputs[index].set_exponent(scaled_exponent);
        outputs[index].setbits(scaled_exponent);
      }
    }
  }

  delete[] inputs;
  return outputs;
}