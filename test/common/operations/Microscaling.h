#pragma once

#include "test/common/operations/Common.h"

template <typename Vector, typename Scale>
Vector* calculate_mx_qparam(std::any input_tensor, std::vector<int> shape,
                            float quant_max, int block_size, int axis,
                            bool force_scale_power_of_two) {
  Vector* inputs = std::any_cast<Vector*>(input_tensor);

  // Handle the case of convolutional layers
  if (axis == 1 && shape.size() == 4) {
    shape = {shape[0], shape[2], shape[3], shape[1]};
    axis = 3;
  }

  if (axis < 0) {
    axis += shape.size();
  }

  int input_size = get_size(shape);
  int result_size = (input_size + block_size - 1) / block_size;

  std::vector<int> output_shape(shape);
  output_shape[axis] = (shape[axis] + block_size - 1) / block_size;

  Vector* amax_arr = new Vector[result_size];
  std::fill(amax_arr, amax_arr + result_size, 0);

  for (int i = 0; i < input_size; i++) {
    auto indices = get_indices(i, shape);
    indices[axis] = indices[axis] / block_size;

    int index = get_flat_index(indices, output_shape);
    amax_arr[index] = std::max(amax_arr[index], Vector(abs(inputs[i])));
  }

  Vector* scales = new Vector[result_size];

  for (int i = 0; i < result_size; i++) {
    Vector scale;
    if (force_scale_power_of_two) {
      static const int emax = floor(log2(quant_max));
      int power_of_two = floor(log2(amax_arr[i])) - emax;
      scale = pow(2, power_of_two);
    } else {
      scale = amax_arr[i] * Vector(quant_max).reciprocal();
    }

    // A contiguous (last-axis) block is quantized single-pass and a strided
    // block two-pass, and the paths differ in what the quantizer divides by.
    // The single-pass quantizer keeps the unrounded scale and substitutes one
    // when it is zero (calculate_mx_scale in src/vector_unit/VectorOps.h).
    // The two-pass scale pass stores the scale and the quantize pass divides
    // by what it reads back, so that divisor is rounded to the scale type --
    // including the flush to zero below the smallest exponent it can hold.
    // Converting an exact zero would substitute one, so leave it alone.
    const bool strided_block = axis != shape.size() - 1;
    if (strided_block && !scale.is_zero()) {
      scale = static_cast<Vector>(static_cast<Scale>(scale));
    }
    scales[i] = (scale.is_zero() && !strided_block) ? Vector::one() : scale;
  }

  return scales;
}

template <typename Vector, typename Scale>
Vector* calculate_mx_qparam(std::map<std::string, std::any>& kwargs,
                            const voyager::PrimOp& op, const ScalarEnv& env) {
  const auto input = resolve(op, "input", env);
  std::any input_ptr = kwargs[input.node];
  const auto input_shape = get_shape(input);
  const float quant_max = arg_float(op, "quant_max", env);
  const int block_size = arg_int(op, "block_size", env);
  const int axis = arg_ints(op, "axes", env)[0];
  const bool force_scale_power_of_two =
      arg_bool(op, "force_scale_power_of_two", env);
  return calculate_mx_qparam<Vector, Scale>(input_ptr, input_shape, quant_max,
                                            block_size, axis,
                                            force_scale_power_of_two);
}
