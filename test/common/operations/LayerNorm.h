#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T *layer_norm(std::any input_tensor, std::any weight_tensor,
                     std::any bias_tensor, const codegen::MatrixParam &param) {
  T *inputs = std::any_cast<T *>(input_tensor);
  T *weights = param.has_weight() ? std::any_cast<T *>(weight_tensor) : nullptr;
  T *bias = param.has_bias() ? std::any_cast<T *>(bias_tensor) : nullptr;
  T *output = new T[get_size(param.input())];

  const auto &input = param.input();
  const auto input_shape = get_shape(input);
  const int last_dim = input_shape.back();
  const int num_rows = get_size(input_shape) / last_dim;

  T size_inv(1.0 / last_dim);

  for (int i = 0; i < num_rows; i++) {
    // In the first pass, scale inputs by 1 / last_dim
    T scaled_inputs[last_dim];
    for (int j = 0; j < last_dim; j++) {
      T normalized_input = inputs[i * last_dim + j] * size_inv;
      scaled_inputs[j] = normalized_input;
    }

    // Perform a tree addition to compute the mean
    T mean = 0.0;
    for (int j = 0; j < last_dim; j += OC_DIMENSION) {
      T buffer[OC_DIMENSION];
      for (int k = 0; k < OC_DIMENSION; k++) {
        buffer[k] = scaled_inputs[j + k];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int k = 0; k < depth; k += 2) {
          buffer[k / 2] = static_cast<T>(buffer[k] + buffer[k + 1]);
        }
        depth = depth / 2;
      }
      mean += buffer[0];
    }

    // Variance = sum_of_squares - mean * mean
    T squares[last_dim];
    for (int j = 0; j < last_dim; j++) {
      T input = inputs[i * last_dim + j];
      squares[j] = static_cast<T>(input * input);
    }

    // Perform a tree addition to compute the sum of squares
    T true_sum_of_squares = 0.0;
    for (int j = 0; j < last_dim; j += OC_DIMENSION) {
      T buffer[OC_DIMENSION];
      for (int k = 0; k < OC_DIMENSION; k++) {
        buffer[k] = squares[j + k];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int k = 0; k < depth; k += 2) {
          buffer[k / 2] = static_cast<T>(buffer[k] + buffer[k + 1]);
        }
        depth = depth / 2;
      }
      true_sum_of_squares += buffer[0];
    }

    T norm_dim(last_dim);
    T eps(1e-05);

    true_sum_of_squares -= mean * mean;
    true_sum_of_squares /= norm_dim;
    true_sum_of_squares += eps;
    T stddev_inv = true_sum_of_squares.inv_sqrt();

    // Normalize by variance and perform an affine transformation
    for (int j = 0; j < last_dim; j++) {
      T input = inputs[i * last_dim + j];
      input -= mean;
      input *= stddev_inv;

      // perform affine transformation
      if (weights) input *= weights[j];
      if (bias) input += bias[j];

      output[i * last_dim + j] = input;
    }
  }

  return output;
}
