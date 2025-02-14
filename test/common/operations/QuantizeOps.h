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
                    const std::vector<int> input_shape,
                    const std::vector<int> scale_shape) {
  LOG("Performing microscaling quantization operation");

  Input* inputs = std::any_cast<Input*>(input);
  Scale* scales = std::any_cast<Scale*>(scale);

  const int input_size = get_size(input_shape);
  const int scale_size = get_size(scale_shape);

  Output* outputs = new Output[input_size];

  int block_size = input_size / scale_size;

  for (int i = 0; i < scale_size; i++) {
    Scale scale = scales[i];
    for (int j = 0; j < block_size; j++) {
      outputs[i * block_size + j] = inputs[i * block_size + j] / scale;
    }
  }

  delete[] inputs;
  delete[] scales;

  return outputs;
}

// template <typename Input, typename Output, typename Scale>
// Output* quantize_mx(std::any input, std::any scale,
//                     const codegen::VectorOp& op) {
//   LOG("Performing microscaling quantization operation");

//   Input* inputs = std::any_cast<Input*>(input);
//   Scale* scales = std::any_cast<Scale*>(scale);

//   const auto& shape_a = get_shape(op.input());
//   const auto& shape_b = get_shape(op.other());

//   Output* outputs = new Output[get_size(op.input())];

//   // Ensure shape_a and shape_b have the same number of dimensions
//   int num_dims = shape_a.size();
//   if (shape_b.size() != num_dims) {
//     throw std::runtime_error("Shapes must have the same number of
//     dimensions!");
//   }

//   // Compute the total number of elements in a
//   int total_elements_a = 1;
//   for (int dim : shape_a) total_elements_a *= dim;

//   // Perform elementwise division with broadcasting
//   for (int i = 0; i < total_elements_a; ++i) {
//     std::vector<int> indices_a = get_indices(i, shape_a);

//     // Map indices_a to indices_b with broadcasting
//     std::vector<int> indices_b(num_dims);
//     for (int d = 0; d < num_dims; ++d) {
//       if (shape_b[d] == shape_a[d]) {
//         indices_b[d] = indices_a[d];  // Match
//       } else if (shape_a[d] % shape_b[d] == 0) {
//         indices_b[d] = indices_a[d] / (shape_a[d] / shape_b[d]);
//       } else {
//         throw std::runtime_error("Invalid shape for broadcasting!");
//       }
//     }

//     int flat_idx_b = get_flat_index(indices_b, shape_b);
//     outputs[i] = inputs[i] / scales[flat_idx_b];
//   }

//   delete[] inputs;
//   delete[] scales;

//   return outputs;
// }

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
    std::cerr << "No dequantization operation for dtype: " << tensor.dtype()
              << std::endl;
    std::abort();
  }
}
