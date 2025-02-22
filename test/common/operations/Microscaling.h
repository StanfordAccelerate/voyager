#pragma once

#include "test/common/operations/Common.h"

template <typename Vector, typename Scale, typename Input>
Scale* calculate_mx_qparam(std::any input_tensor, const std::vector<int>& shape,
                           const int block_size) {
  Vector* inputs = std::any_cast<Vector*>(input_tensor);

  int input_size = get_size(shape);
  int result_size = (input_size + block_size - 1) / block_size;

  Scale* scales = new Scale[result_size];

  for (int i = 0; i < result_size; i++) {
    Vector max_value = 0;

    for (int block = 0; block < block_size; block++) {
      int index = i * block_size + block;
      max_value = std::max(max_value, Vector(abs(inputs[index])));
    }

    if constexpr (Scale::width == Scale::e_width) {
      scales[i] = pow(2, floor(log2(max_value)) - floor(log2(Input::max())));
    } else {
      scales[i] = max_value / Input::max();
    }

    if (scales[i].to_ac_float() == Scale::ac_float_rep::zero()) {
      scales[i].set_one();
    }
  }

  return scales;
}

template <typename Vector, typename Scale, typename Input>
Scale* calculate_mx_qparam(std::map<std::string, std::any>& kwargs,
                           const codegen::OpOverload op) {
  const auto input = op.kwargs().at("input").tensor();
  std::any input_ptr = kwargs[input.node()];
  const auto input_shape = get_shape(input);
  const int block_size = op.kwargs().at("block_size").int_value();
  return calculate_mx_qparam<Vector, Scale, Input>(input_ptr, input_shape,
                                                   block_size);
}

// template <typename Vector, typename Scale, typename Input>
// Scale* calculate_mx_qparam(std::any input_tensor,
//                            const codegen::ReduceOp& param) {
//   Vector* inputs = std::any_cast<Vector*>(input_tensor);

//   const auto& shape = get_shape(param.input());

//   int mx_axis = param.dim(0);
//   if (mx_axis < 0) {
//     mx_axis += shape.size();
//   }

//   int block_size = 32;
//   int input_size = get_size(param.input());
//   int result_size = (input_size + block_size - 1) / block_size;

//   std::vector<int> output_shape(shape);
//   output_shape[mx_axis] = (shape[mx_axis] + block_size - 1) / block_size;

//   float* amax_arr = new float[result_size];
//   std::fill(amax_arr, amax_arr + result_size, 0);

//   for (int i = 0; i < input_size; i++) {
//     auto indices = get_indices(i, shape);
//     indices[mx_axis] = indices[mx_axis] / block_size;

//     int index = get_flat_index(indices, output_shape);
//     amax_arr[index] = std::max(amax_arr[index], abs(inputs[i]));
//   }

//   Scale* scales = new Scale[*result_size];

//   for (int i = 0; i < result_size; i++) {
//     if constexpr (Scale::width == Scale::e_width) {
//       int power_of_two = floor(log2(amax_arr[i])) -
//       floor(log2(Input::max())); scales[i] = pow(2, power_of_two);
//     } else {
//       scales[i] = amax_arr[i] / static_cast<float>(Input::max());
//     }
//   }

//   delete[] inputs;
//   return scales;
// }