#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* layer_norm(std::any input_ptr, std::any weight_ptr, std::any bias_ptr,
                     const std::vector<int> input_shape,
                     const std::vector<int> normalized_shape) {
  T* inputs = std::any_cast<T*>(input_ptr);
  T* weights = std::any_cast<T*>(weight_ptr);
  T* bias = std::any_cast<T*>(bias_ptr);

  T* output = new T[get_size(input_shape)];

  const int outer_dim = get_size(normalized_shape);
  const int inner_dim = get_size(input_shape) / outer_dim;

  T size_inv(1.0 / outer_dim);

  for (int i = 0; i < inner_dim; i++) {
    // In the first pass, scale inputs by 1 / outer_dim
    T normalized_inputs[outer_dim];
    for (int j = 0; j < outer_dim; j++) {
      normalized_inputs[j] = inputs[i * outer_dim + j] * size_inv;
    }

    // Compute the mean
    T sums[2] = {0, 0};
    int index = 0;

    for (int j = 0; j < outer_dim; j += VECTOR_UNIT_WIDTH) {
      T buffer[VECTOR_UNIT_WIDTH];
      for (int k = 0; k < VECTOR_UNIT_WIDTH; k++) {
        buffer[k] = normalized_inputs[j + k];
      }
      sums[index++ % 2] += tree_reduce(buffer, VECTOR_UNIT_WIDTH);
    }

    T mean = tree_reduce(sums, 2);

    // In the second pass, subtract the mean from the tensor and square the
    // result
    T squares[outer_dim];
    for (int j = 0; j < outer_dim; j++) {
      T input = inputs[i * outer_dim + j] - mean;
      squares[j] = static_cast<T>(input * input);
    }

    // Compute the variance
    sums[0] = 0;
    sums[1] = 0;
    index = 0;

    for (int j = 0; j < outer_dim; j += VECTOR_UNIT_WIDTH) {
      T buffer[VECTOR_UNIT_WIDTH];
      for (int k = 0; k < VECTOR_UNIT_WIDTH; k++) {
        buffer[k] = squares[j + k];
      }
      sums[index++ % 2] += tree_reduce(buffer, VECTOR_UNIT_WIDTH);
    }

    T variance = tree_reduce(sums, 2);
    T divisor = sqrt(outer_dim);
    T stddev_inv = variance.inv_sqrt() * divisor;

    if (variance == T::zero()) {
      stddev_inv = 1.0;
    }

    // Normalize by variance and perform an affine transformation
    for (int j = 0; j < outer_dim; j++) {
      T input = inputs[i * outer_dim + j];
      input -= mean;
      input *= stddev_inv;

      // perform affine transformation
      if (weights) input *= weights[j];
      if (bias) input += bias[j];

      output[i * outer_dim + j] = input;
    }
  }

  return output;
}

template <typename T>
inline T* layer_norm(std::map<std::string, std::any>& kwargs,
                     const codegen::OpOverload op) {
  assert(op.target() == "layer_norm");

  const auto input = op.kwargs().at("input").tensor();
  std::any input_ptr = kwargs[input.node()];

  std::any weight_ptr = static_cast<T*>(nullptr);

  if (op.kwargs().contains("weight")) {
    const auto weight = op.kwargs().at("weight").tensor();
    weight_ptr = kwargs[weight.node()];
  }

  std::any bias_ptr = static_cast<T*>(nullptr);

  if (op.kwargs().contains("bias")) {
    const auto bias = op.kwargs().at("bias").tensor();
    bias_ptr = kwargs[bias.node()];
  }

  const auto input_shape = get_shape(input);

  const auto normalized_shape = op.kwargs().at("normalized_shape").int_list();
  std::vector<int> norm_shape(normalized_shape.values().begin(),
                              normalized_shape.values().end());

  return layer_norm<T>(input_ptr, weight_ptr, bias_ptr, input_shape,
                       norm_shape);
}
