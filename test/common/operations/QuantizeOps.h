#pragma once
#define NO_SYSC

#include <vector>

#include "src/datatypes/DataTypes.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

template <typename Input, typename Scale>
Input* quantize(std::any input, std::any scale, std::vector<int> shape) {
  const int size = get_size(shape);

  Input* inputs = std::any_cast<Input*>(input);
  Scale* scales = std::any_cast<Scale*>(scale);

  Input* outputs = new Input[size];

  for (int i = 0; i < size; i++) {
    outputs[i] = inputs[i] / (*scales);
  }

  delete[] inputs;
  delete[] scales;

  return outputs;
}

template <typename Input, typename Scale>
Input* quantize_mx(std::any input, std::any scale, std::vector<int> input_shape,
                   int block_size, int axis) {
  spdlog::debug("Performing microscaling quantization operation\n");

  // Handle the case of convolutional layers
  if (axis == 1 && input_shape.size() == 4) {
    input_shape = {input_shape[0], input_shape[2], input_shape[3],
                   input_shape[1]};
    axis = 3;
  }

  if (axis < 0) {
    axis += input_shape.size();
  }

  Input* inputs = std::any_cast<Input*>(input);
  Scale* scales = std::any_cast<Scale*>(scale);

  std::vector<int> scale_shape = input_shape;
  scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;
  const int num_dims = scale_shape.size();

  const int output_size = get_size(input_shape);
  Input* outputs = new Input[output_size];

  // Perform elementwise division with broadcasting
  for (int i = 0; i < output_size; ++i) {
    std::vector<int> indices_a = get_indices(i, input_shape);

    // Map indices_a to indices_b with broadcasting
    std::vector<int> indices_b(num_dims);
    for (int d = 0; d < num_dims; ++d) {
      if (scale_shape[d] == input_shape[d]) {
        indices_b[d] = indices_a[d];
      } else if (input_shape[d] % scale_shape[d] == 0) {
        indices_b[d] = indices_a[d] / block_size;
      } else {
        throw std::runtime_error("Invalid shape for broadcasting!");
      }
    }

    int flat_idx_b = get_flat_index(indices_b, scale_shape);
    outputs[i] = inputs[i] / scales[flat_idx_b];
  }

  delete[] inputs;

  return outputs;
}

template <typename Input, typename Output>
bool dequantize(std::any input, std::any scale, Output*& output,
                codegen::Tensor tensor) {
  if (tensor.dtype() != DataTypes::TypeName<Input>::name()) {
    return false;
  }

  Input* casted_input = std::any_cast<Input*>(input);
  Output* casted_scale = std::any_cast<Output*>(scale);

  const int size = get_size(tensor);
  output = new Output[size];

  for (int i = 0; i < size; i++) {
    output[i] = static_cast<Output>(casted_input[i]) * (*casted_scale);
  }

  delete[] casted_input;
  delete[] casted_scale;

  return true;
}

template <typename Output, typename... Ts>
Output* dequantize_helper(std::any input, std::any scale,
                          codegen::Tensor tensor) {
  Output* output = nullptr;
  bool matched = (dequantize<Ts, Output>(input, scale, output, tensor) || ...);
  if (!matched) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
  }
  return output;
}

template <typename Output>
Output* dequantize_tensor(std::any input, std::any scale,
                          codegen::Tensor tensor) {
  return dequantize_helper<Output, SUPPORTED_TYPES>(input, scale, tensor);
}
