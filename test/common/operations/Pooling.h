#pragma once

#include "test/common/operations/Common.h"

template <typename T>
T* pooling(std::any input_ptr, const std::vector<int>& input_shape,
           const std::vector<int>& output_shape, int stride, int kernel_size,
           int padding, const bool is_max_pool) {
  int input_height = input_shape[1];
  int input_width = input_shape[2];
  int input_depth = input_shape[3];

  int output_height = output_shape[0];
  int output_width = output_shape[1];

  spdlog::debug("Performing {} pooling with kernel size {} and stride {}\n",
                is_max_pool ? "max" : "average", kernel_size, stride);
  spdlog::debug("Input shape: {}x{}x{}\n", input_height, input_width,
                input_depth);
  spdlog::debug("Output shape: {}x{}x{}\n", output_height, output_width,
                input_depth);
  spdlog::debug("Padding: {}\n", padding);

  T* inputs = std::any_cast<T*>(input_ptr);

  T* output = new T[output_height * output_width * input_depth];

  for (int y = 0; y < output_height; ++y) {
    for (int x = 0; x < output_width; ++x) {
      for (int d = 0; d < input_depth; ++d) {
        T value = is_max_pool ? -9999 : 0;

        for (int y_window = 0; y_window < kernel_size; ++y_window) {
          for (int x_window = 0; x_window < kernel_size; ++x_window) {
            int input_x = x * stride + x_window - padding;
            int input_y = y * stride + y_window - padding;

            if (input_x >= 0 && input_x < input_width && input_y >= 0 &&
                input_y < input_height) {
              T input = inputs[input_y * input_width * input_depth +
                               input_x * input_depth + d];
              if (is_max_pool) {
                value = std::max(value, input);
              } else {
                value += input * T(1.0 / (kernel_size * kernel_size));
              }
            }
          }
        }

        output[y * output_width * input_depth + x * input_depth + d] = value;
      }
    }
  }

  delete[] inputs;

  return output;
}

template <typename T>
T* adaptive_avg_pool2d(std::map<std::string, std::any>& kwargs,
                       const codegen::OpOverload op) {
  assert(op.target() == "adaptive_avg_pool2d");

  const auto input = op.kwargs().at("input").tensor();
  std::any input_ptr = kwargs[input.node()];
  const auto input_shape = get_shape(input);

  const auto output_size = op.kwargs().at("output_size").int_list().values();

  int input_height = input_shape[1];
  int output_height = output_size[0];
  int output_width = output_size[1];

  std::vector<int> output_shape{output_height, output_width};

  int stride = input_height / output_height;
  int kernel_size = input_height - (output_height - 1) * stride;
  int padding = 0;

  return pooling<T>(input_ptr, input_shape, output_shape, stride, kernel_size,
                    padding, false);
}

template <typename T>
T* max_pool2d(std::map<std::string, std::any>& kwargs,
              const codegen::OpOverload op) {
  assert(op.target() == "max_pool2d");

  const auto op_kwargs = op.kwargs();

  const auto input = op_kwargs.at("input").tensor();
  std::any input_ptr = kwargs[input.node()];
  const auto input_shape = get_shape(input);

  const auto stride = op_kwargs.at("stride").int_list().values()[0];
  const auto kernel_size = op_kwargs.at("kernel_size").int_list().values()[0];
  const auto padding = op_kwargs.at("padding").int_list().values()[0];

  int input_height = input_shape[1];
  int input_width = input_shape[2];

  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  std::vector<int> output_shape = {output_height, output_width};

  return pooling<T>(input_ptr, input_shape, output_shape, stride, kernel_size,
                    padding, true);
}
