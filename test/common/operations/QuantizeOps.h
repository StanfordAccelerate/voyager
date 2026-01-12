#pragma once
#define NO_SYSC

#include <vector>

#include "src/datatypes/DataTypes.h"
#include "test/common/Utils.h"
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

template <typename Input, typename Meta>
std::tuple<Input*, Meta*, Meta*, Input*> filter_outlier(
    std::any input, std::vector<int> input_shape, const int data_size,
    Input threshold) {
  spdlog::debug("Performing outlier filtering operation\n");
  Input* inputs = std::any_cast<Input*>(input);

  // Expect 2D tensor
  const int ndim = input_shape.size();
  const int K = input_shape[ndim - 1];
  const int X = get_size(input_shape) / K;

  Input* data = new Input[data_size];
  Meta* indices = new Meta[data_size];
  Meta* indptr = new Meta[X + 1];
  Input* filtered = new Input[X * K];

  // Initialize indices and data
  for (int i = 0; i < data_size; i++) {
    data[i] = 0;
    indices[i] = -1;
  }

  indptr[0] = 0;
  int nnz = 0;

  for (int x = 0; x < X; ++x) {
    for (int k = 0; k < K; ++k) {
      Input v = inputs[x * K + k];
      if (v.abs() > threshold) {
        indices[nnz] = k;
        data[nnz] = v;
        nnz++;
        filtered[x * K + k] = 0;
      } else {
        filtered[x * K + k] = v;
      }
    }
    indptr[x + 1] = nnz;
  }

  delete[] inputs;

  spdlog::debug("Filtered {} outliers\n", nnz);

  return {data, indices, indptr, filtered};
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

template <typename Input, typename Scale, typename Vector, typename Output>
bool dequantize_group(std::any input, std::any scale, std::any zero_point,
                      Output*& output, codegen::Tensor tensor, int block_size,
                      int axis) {
  if (tensor.dtype() != DataTypes::TypeName<Input>::name()) {
    return false;
  }

  const auto input_shape = get_shape(tensor);

  Input* casted_input = std::any_cast<Input*>(input);
  Scale* casted_scale = std::any_cast<Scale*>(scale);
  Scale* casted_zero_point = std::any_cast<Scale*>(zero_point);

  const int num_dims = input_shape.size();
  if (axis < 0) {
    axis += num_dims;
  }

  // Compute broadcast shape for scales and zero_points
  std::vector<int> scale_shape = input_shape;
  scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;

  const int output_size = get_size(input_shape);
  output = new Output[output_size];

  // Elementwise dequantization with broadcasting
  for (int i = 0; i < output_size; ++i) {
    std::vector<int> indices_a = get_indices(i, input_shape);
    std::vector<int> indices_b(num_dims);

    // Map input indices to scale/zero_point indices
    for (int d = 0; d < num_dims; ++d) {
      if (scale_shape[d] == input_shape[d]) {
        indices_b[d] = indices_a[d];
      } else if (input_shape[d] % scale_shape[d] == 0) {
        indices_b[d] = indices_a[d] / block_size;
      } else {
        throw std::runtime_error(
            "Invalid shape for broadcasting in dequantize!");
      }
    }

    int flat_idx_b = get_flat_index(indices_b, scale_shape);

    Vector s_val = casted_scale[flat_idx_b];
    Vector zp_val =
        casted_zero_point ? casted_zero_point[flat_idx_b] : Scale::zero();
    output[i] = (static_cast<Vector>(casted_input[i]) - zp_val) * s_val;
  }

  delete[] casted_input;
  delete[] casted_scale;
  if (casted_zero_point) {
    delete[] casted_zero_point;
  }

  return true;
}

template <typename Output, typename Scale, typename Vector, typename... Ts>
Output* dequantize_group_helper(std::any input, std::any scale,
                                std::any zero_point, codegen::Tensor tensor,
                                int block_size, int axis) {
  Output* output = nullptr;
  bool matched =
      (dequantize_group<Ts, Scale, Vector, Output>(
           input, scale, zero_point, output, tensor, block_size, axis) ||
       ...);
  if (!matched) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
  }
  return output;
}

template <typename Output, typename Scale, typename Vector>
Output* dequantize_tensor_group(std::any input, std::any scale,
                                std::any zero_point, codegen::Tensor tensor,
                                int block_size, int axis) {
  return dequantize_group_helper<Output, Scale, Vector, SUPPORTED_TYPES>(
      input, scale, zero_point, tensor, block_size, axis);
}
