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

  for (int i = 0; i < size; i++) {
    quantized_output[i] =
        input_tensor[i]
            .template quantize<QUANTIZED_TYPE::ac_int_rep::width,
                               QUANTIZED_TYPE::ac_int_rep::sign>(*scale_val);
  }

  return quantized_output;
}

template <typename TYPE, typename DEQUANTIZED_TYPE>
DEQUANTIZED_TYPE* dequantize(std::any input, std::any scale, int size) {
  TYPE* input_tensor = std::any_cast<TYPE*>(input);
  DEQUANTIZED_TYPE* scale_val = std::any_cast<DEQUANTIZED_TYPE*>(scale);

  DEQUANTIZED_TYPE* dequantized_output = new DEQUANTIZED_TYPE[size];

  for (int i = 0; i < size; i++) {
    dequantized_output[i] = input_tensor[i].template dequantize(*scale_val);
  }

  return dequantized_output;
}