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
Output* quantize(std::any input, std::any scale, int size) {
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
Output* quantizeMX(std::any input, std::any scale, int tensor_size,
                   int scale_size) {
  Input* inputs = std::any_cast<Input*>(input);
  Scale* scales = std::any_cast<Scale*>(scale);
  Output* outputs = new Output[tensor_size];

  int block_size = tensor_size / scale_size;

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
