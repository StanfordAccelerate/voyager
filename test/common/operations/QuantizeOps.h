#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

template <typename TYPE, typename QUANTIZED_TYPE>
QUANTIZED_TYPE* quantize(std::any input, std::any scale, int size) {
  TYPE* input_tensor = std::any_cast<TYPE*>(input);
  TYPE* scale_val = std::any_cast<TYPE*>(scale);
  QUANTIZED_TYPE* quantized_output = new QUANTIZED_TYPE[size];

  if constexpr (!QUANTIZED_TYPE::is_floating_point) {
    // perform quantization operation
    for (int i = 0; i < size; i++) {
      quantized_output[i] =
          input_tensor[i]
              .template quantize<QUANTIZED_TYPE::ac_int_rep::width,
                                 QUANTIZED_TYPE::ac_int_rep::sign>(*scale_val);
    }
  } else {
    // cast to the other type
    for (int i = 0; i < size; i++) {
      quantized_output[i] = static_cast<QUANTIZED_TYPE>(input_tensor[i]);
    }
  }

  delete[] input_tensor;
  delete[] scale_val;

  return quantized_output;
}

template <typename TYPE, typename SCALE_TYPE, typename QUANTIZED_TYPE>
QUANTIZED_TYPE* quantizeMX(std::any input, std::any scale, int tensor_size,
                           int scale_size) {
  TYPE* input_tensor = std::any_cast<TYPE*>(input);
  SCALE_TYPE* scale_val = std::any_cast<SCALE_TYPE*>(scale);
  QUANTIZED_TYPE* quantized_output = new QUANTIZED_TYPE[tensor_size];

  int block_size = tensor_size / scale_size;

  if constexpr (!QUANTIZED_TYPE::is_floating_point) {
    // perform quantization operation
    for (int i = 0; i < scale_size; i++) {
      SCALE_TYPE scale = scale_val[i];
      for (int j = 0; j < block_size; j++) {
        quantized_output[i * block_size + j] =
            input_tensor[i * block_size + j]
                .template quantize<QUANTIZED_TYPE::ac_int_rep::width,
                                   QUANTIZED_TYPE::ac_int_rep::sign>(scale);
      }
    }
  } else {
    std::cerr << "Unsupported quantization operation for floating point type"
              << std::endl;
    std::abort();
  }

  delete[] input_tensor;
  delete[] scale_val;

  return quantized_output;
}

template <typename TYPE, typename DEQUANTIZED_TYPE>
DEQUANTIZED_TYPE* dequantize(std::any input, std::any scale, int size) {
  TYPE* input_tensor = std::any_cast<TYPE*>(input);
  DEQUANTIZED_TYPE* scale_val = std::any_cast<DEQUANTIZED_TYPE*>(scale);
  DEQUANTIZED_TYPE* dequantized_output = new DEQUANTIZED_TYPE[size];

  if constexpr (std::is_same_v<TYPE, DEQUANTIZED_TYPE>) {
    // dequantization is just going to be a multiplication by the scale
    for (int i = 0; i < size; i++) {
      dequantized_output[i] = input_tensor[i] * (*scale_val);
    }
  } else if constexpr (!TYPE::is_floating_point) {
    // perform dequantization operation
    for (int i = 0; i < size; i++) {
      dequantized_output[i] = input_tensor[i].template dequantize(*scale_val);
    }
  } else {
    // cast to the other type
    for (int i = 0; i < size; i++) {
      dequantized_output[i] = static_cast<DEQUANTIZED_TYPE>(input_tensor[i]);
    }
  }

  delete[] input_tensor;
  delete[] scale_val;

  return dequantized_output;
}