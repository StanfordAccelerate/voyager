#pragma once

#include "test/common/operations/Common.h"

template <typename INPUT_T, typename ACCUMULATE_T>
ACCUMULATE_T *pooling(const ACCUMULATE_T *inputs,
                      const codegen::AcceleratorParam &param) {
  const auto &pooling_param = param.pooling_param();
  int input_height = pooling_param.input().shape(2);
  int input_width = pooling_param.input().shape(3);
  int input_depth = pooling_param.input().shape(1);

  int stride;
  int kernel_size;
  int padding;
  int output_height;
  int output_width;
  // Adaptive pooling has output_size set
  if (pooling_param.output_size_size() > 0) {
    output_height = pooling_param.output_size(0);
    output_width = pooling_param.output_size(1);
    stride = input_height / output_height;
    kernel_size = input_height - (output_height - 1) * stride;
    padding = 0;
  } else {
    stride = pooling_param.stride(0);
    kernel_size = pooling_param.kernel_size(0);
    padding = pooling_param.padding(0);
    output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  }

  bool is_max_pool = pooling_param.opcode().find("max") != std::string::npos;

  ACCUMULATE_T *output =
      new ACCUMULATE_T[output_height * output_width * input_depth];

  for (int y = 0; y < output_height; ++y) {
    for (int x = 0; x < output_width; ++x) {
      for (int d = 0; d < input_depth; ++d) {
        ACCUMULATE_T value = is_max_pool ? -9999 : 0;

        for (int y_window = 0; y_window < kernel_size; ++y_window) {
          for (int x_window = 0; x_window < kernel_size; ++x_window) {
            int input_x = x * stride + x_window - padding;
            int input_y = y * stride + y_window - padding;

            if (input_x >= 0 && input_x < input_width && input_y >= 0 &&
                input_y < input_height) {
              ACCUMULATE_T input = inputs[input_y * input_width * input_depth +
                                          input_x * input_depth + d];
              if (is_max_pool) {
                value = std::max(value, input);
              } else {
                value += input;
              }
            }
          }
        }
        if (!is_max_pool) {
          value *= INPUT_T(1.0 / (kernel_size * kernel_size));
        }
        output[y * output_width * input_depth + x * input_depth + d] = value;
      }
    }
  }

  return output;
}
