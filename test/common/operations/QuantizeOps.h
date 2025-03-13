#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

template <typename Input, typename Output, typename Scale>
Output* quantize(std::any input, std::any scale, std::vector<int> shape) {
  const int size = get_size(shape);

  Input* inputs = std::any_cast<Input*>(input);
  Scale* scales = std::any_cast<Scale*>(scale);

  Output* outputs = new Output[size];

  for (int i = 0; i < size; i++) {
    outputs[i] = inputs[i] / (*scales);
  }

  delete[] inputs;
  delete[] scales;

  return outputs;
}

template <typename Input, typename Output, typename Scale>
Output* quantize_mx(std::any input, std::any scale,
                    std::vector<int> input_shape, int block_size, int axis) {
  spdlog::debug("Performing microscaling quantization operation");

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
  Output* outputs = new Output[output_size];

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
Output* dequantize(std::any input, std::any scale, int size) {
  Input* inputs = std::any_cast<Input*>(input);
  Output* scales = std::any_cast<Output*>(scale);

  Output* outputs = new Output[size];

  for (int i = 0; i < size; i++) {
    outputs[i] = static_cast<Output>(inputs[i]) * (*scales);
  }

  delete[] inputs;
  delete[] scales;

  return outputs;
}

template <typename Output>
Output* dequantize_tensor(std::any input, std::any scale,
                          codegen::Tensor tensor) {
  if (tensor.dtype() == "int32") {
    return dequantize<DataTypes::int32, Output>(input, scale, get_size(tensor));
  } else if (tensor.dtype() == "int24") {
    return dequantize<DataTypes::int24, Output>(input, scale, get_size(tensor));
  } else if (tensor.dtype() == "int8") {
    return dequantize<DataTypes::int8, Output>(input, scale, get_size(tensor));
  } else if (tensor.dtype() == "fp8_e4m3") {
    return dequantize<DataTypes::e4m3, Output>(input, scale, get_size(tensor));
  } else if (tensor.dtype() == "posit8_1") {
    return dequantize<DataTypes::posit8, Output>(input, scale,
                                                 get_size(tensor));
  } else if (tensor.dtype() == "bfloat16") {
    return dequantize<DataTypes::bfloat16, Output>(input, scale,
                                                   get_size(tensor));
  } else {
    spdlog::error("No dequantization operation for dtype: {}\n",
                  tensor.dtype());
    std::abort();
  }
}
