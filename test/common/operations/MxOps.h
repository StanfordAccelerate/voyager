#pragma once

#include "test/common/operations/Common.h"

template <typename Input, typename Scale, typename Output>
Scale* calculate_mx_qparam(std::any input_tensor,
                           const codegen::ReduceOp& param) {
  Input* inputs = std::any_cast<Input*>(input_tensor);

  const auto& shape = get_shape(param.input());

  int mx_axis = param.dim(0);
  if (mx_axis < 0) {
    mx_axis += shape.size();
  }
  int mx_axis_size = shape[mx_axis];

  int tensor_size = get_size(param.input());
  int outer_size = tensor_size / mx_axis_size;
  int block_size = std::min(mx_axis_size, 32);
  int num_blocks = (mx_axis_size + block_size - 1) / block_size;

  std::vector<int> output_shape(shape);
  output_shape[mx_axis] = num_blocks;

  float* amax_arr = new float[num_blocks * outer_size];
  std::fill(amax_arr, amax_arr + num_blocks * outer_size, 0);

  for (int i = 0; i < tensor_size; i++) {
    auto indices = get_indices(i, shape);
    indices[mx_axis] = indices[mx_axis] / block_size;

    int index = get_flat_index(indices, output_shape);
    amax_arr[index] = std::max(amax_arr[index], abs(inputs[i]));
  }

  Scale* outputs = new Scale[num_blocks * outer_size];

  for (int i = 0; i < outer_size * num_blocks; i++) {
    outputs[i] = amax_arr[i] / Output::max_value;
  }

  delete[] inputs;
  return outputs;
}