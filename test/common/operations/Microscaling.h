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
    Scale scale;
    if (force_scale_power_of_two) {
      static const int emax = floor(log2(quant_max));
      int power_of_two = floor(log2(amax_arr[i])) - emax;
      scale = pow(2, power_of_two);
    } else {
      scale = amax_arr[i] / Vector(quant_max);
    }

    scales[i] = scale.is_zero() ? Scale::one() : scale;
  }

  return scales;
}

template <typename Vector, typename Scale>
Vector* calculate_mx_qparam(std::map<std::string, std::any>& kwargs,
                            const codegen::OpOverload op) {
  const auto input = op.kwargs().at("input").tensor();
  std::any input_ptr = kwargs[input.node()];
  const auto input_shape = get_shape(input);
  const float quant_max = op.kwargs().at("quant_max").float_value();
  const int block_size = op.kwargs().at("block_size").int_value();
  const int axis = op.kwargs().at("axes").int_list().values()[0];
  const bool force_scale_power_of_two =
      op.kwargs().at("force_scale_power_of_two").bool_value();
  return calculate_mx_qparam<Vector, Scale>(input_ptr, input_shape, quant_max,
                                            block_size, axis,
                                            force_scale_power_of_two);
}
