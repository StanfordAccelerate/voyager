#pragma once

#include "test/common/operations/Common.h"

// The vector accumulator sums a window into this many interleaved registers
// and combines them with one fused add tree at the end
// (VectorAccumulator::SUM_N in src/vector_unit/Accumulator.h).
#ifdef CLOCK_PERIOD
constexpr double accumulator_clock_period = CLOCK_PERIOD;
#else
constexpr double accumulator_clock_period = 5.0;
#endif
constexpr int ACCUMULATOR_SUM_N = (accumulator_clock_period < 5) ? 4 : 2;

template <typename T>
std::shared_ptr<T[]> pooling(std::any input_ptr,
                             const std::vector<int>& input_shape,
                             const std::vector<int>& output_shape, int stride,
                             int kernel_size, int padding,
                             const bool is_max_pool) {
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

  T* inputs = std::any_cast<std::shared_ptr<T[]>&>(input_ptr).get();

  std::shared_ptr<T[]> output(
      new T[output_height * output_width * input_depth]);

  for (int y = 0; y < output_height; ++y) {
    for (int x = 0; x < output_width; ++x) {
      for (int d = 0; d < input_depth; ++d) {
        T value = is_max_pool ? -9999 : 0;
        T sums[ACCUMULATOR_SUM_N];
        for (int i = 0; i < ACCUMULATOR_SUM_N; ++i) {
          sums[i] = 0;
        }
        int index = 0;

        for (int y_window = 0; y_window < kernel_size; ++y_window) {
          for (int x_window = 0; x_window < kernel_size; ++x_window) {
            int input_x = x * stride + x_window - padding;
            int input_y = y * stride + y_window - padding;

            const bool in_bounds = input_x >= 0 && input_x < input_width &&
                                   input_y >= 0 && input_y < input_height;

            if (is_max_pool) {
              if (in_bounds) {
                value =
                    std::max(value, inputs[input_y * input_width * input_depth +
                                           input_x * input_depth + d]);
              }
              continue;
            }

            // The accumulator consumes one element per window position, so a
            // padded position still advances the interleave.
            T scaled = 0;
            if (in_bounds) {
              scaled = inputs[input_y * input_width * input_depth +
                              input_x * input_depth + d] *
                       T(1.0 / (kernel_size * kernel_size));
            }
            sums[index++ % ACCUMULATOR_SUM_N] += scaled;
          }
        }

        if (!is_max_pool) {
          value = fused_tree_reduce<ACCUMULATOR_SUM_N>(sums);
        }

        output[y * output_width * input_depth + x * input_depth + d] = value;
      }
    }
  }

  return output;
}

template <typename T>
std::shared_ptr<T[]> adaptive_avg_pool2d(
    std::map<std::string, std::any>& kwargs, const voyager::PrimOp& op,
    const ScalarEnv& env) {
  assert(strip_namespace(op.target()) == "adaptive_avg_pool2d");

  const auto input = resolve(op, "input", env);
  std::any input_ptr = kwargs[input.node];
  const auto input_shape = get_shape(input);

  const auto output_size = arg_ints(op, "output_size", env);

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
std::shared_ptr<T[]> max_pool2d(std::map<std::string, std::any>& kwargs,
                                const voyager::PrimOp& op,
                                const ScalarEnv& env) {
  assert(strip_namespace(op.target()) == "max_pool2d");
  const auto input = resolve(op, "input", env);
  std::any input_ptr = kwargs[input.node];
  const auto input_shape = get_shape(input);

  const auto stride = arg_ints(op, "stride", env)[0];
  const auto kernel_size = arg_ints(op, "kernel_size", env)[0];
  const auto padding = arg_ints(op, "padding", env)[0];

  int input_height = input_shape[1];
  int input_width = input_shape[2];

  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  std::vector<int> output_shape = {output_height, output_width};

  return pooling<T>(input_ptr, input_shape, output_shape, stride, kernel_size,
                    padding, true);
}
