#include "GoldModel.h"

#include <vector>

#include "test/common/VerificationTypes.h"
#include "test/common/operations/MatrixOps.h"
#include "test/common/operations/Pooling.h"
#include "test/common/operations/QuantizeOps.h"
#include "test/common/operations/ReduceOps.h"
#include "test/common/operations/ReshapeOps.h"
#include "test/common/operations/VectorOps.h"

template <typename T>
void save_tensor(char *output_bytes, std::any output_tensor, int size) {
  T *output_tensor_casted = std::any_cast<T *>(output_tensor);

  for (int i = 0; i < size; i++) {
    auto bits = static_cast<T>(output_tensor_casted[i]).bits_rep();
    constexpr int num_bytes = T::width / 8;
    for (int j = 0; j < num_bytes; j++) {
      output_bytes[i * num_bytes + j] = bits.template slc<8>(j * 8);
    }
  }

  delete[] output_tensor_casted;
}

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T,
          typename VECTOR_T>
void run_operation(const codegen::AcceleratorParam param,
                   std::vector<std::any> args) {
  int arg_index = 0;
  std::any output_tensor;

  if (param.has_reduce_param()) {
    const auto &reduce_param = param.reduce_param();
    if (reduce_param.opcode() == "softmax") {
      const auto &input = reduce_param.input();
      const auto input_shape = get_shape(input);
      output_tensor = softmax<VECTOR_T>(args[arg_index++], input_shape);
    } else if (reduce_param.opcode() == "sum") {
      const auto &input = reduce_param.input();
      const auto input_shape = get_shape(input);

      std::vector<int> dims;
      for (int dim : reduce_param.dim()) {
	  dims.push_back(dim);
      }

      output_tensor = sum<VECTOR_T>(args[arg_index++], input_shape, dims);
    } else {
      std::cerr << "Unsupported reduce instruction: " << reduce_param.opcode()
                << std::endl;
      exit(1);
    }
  }

  if (param.has_pooling_param()) {
    const auto input = param.pooling_param().input();
    output_tensor = pooling<VECTOR_T>(args[arg_index++], param);
  }

  if (param.has_reshape_param()) {
    const auto &reshape_param = param.reshape_param();
    const auto &input = reshape_param.input();
    output_tensor = permute<INPUT_T>(args[arg_index++], reshape_param);
  }

  if (param.has_matrix_param()) {
    const auto &matrix_param = param.matrix_param();

    if (matrix_param.opcode() == "conv2d" && matrix_param.groups() != 1) {
      std::cerr << "Grouped convolution is not supported" << std::endl;
      std::abort();
    }

    // Permute input tensor
    const auto &input = matrix_param.input();
    std::any input_tensor = args[0];
    if (input.has_permutation()) {
      input_tensor = permute<INPUT_T>(input_tensor, input);
    }

    // Permute weight tensor
    const auto &weight = matrix_param.weight();
    std::any weight_tensor = args[1];
    if (weight.has_permutation()) {
      weight_tensor = permute<INPUT_T>(weight_tensor, weight);
    }

    int dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      dim *= input.shape(i);
    }

    if (dim == 1) {
      output_tensor = matrix_vector_multiply<INPUT_T, ACCUMULATE_T,
                                             INTERMEDIATE_T, VECTOR_T>(
          input_tensor, weight_tensor, args[2], matrix_param);
    } else {
      output_tensor = gemm<INPUT_T, ACCUMULATE_T, INTERMEDIATE_T>(
          input_tensor, weight_tensor, args[2], param);
    }
    arg_index = 3;
  } else if (param.vector_params_size() > 0) {
    // fetch the input of the first vector instruction
    output_tensor = args[arg_index++];
  }

  for (const auto &vector_param : param.vector_params()) {
    if (vector_param.opcode().rfind("sqrt", 0) == 0) {
      VECTOR_T *input_tensor = std::any_cast<VECTOR_T *>(output_tensor);
      output_tensor = sqrt(input_tensor, get_shape(vector_param.input()));
    } else if (activations.find(vector_param.opcode()) != activations.end()) {
      if (vector_param.opcode() != "relu" && vector_param.opcode() != "relu_") {
        std::cerr << "Unsupported activation function: "
                  << vector_param.opcode() << std::endl;
        std::abort();
      }
      VECTOR_T *tensor = std::any_cast<VECTOR_T *>(output_tensor);
      int input_size = get_size(vector_param.input());
      for (int i = 0; i < input_size; i++) {
        relu(tensor[i]);
      }
    } else if (arithmetics.find(vector_param.opcode()) != arithmetics.end()) {
      const auto &input = vector_param.other().has_memory()
                              ? vector_param.input()
                              : vector_param.other();
      const auto &other = vector_param.other().has_memory()
                              ? vector_param.other()
                              : vector_param.input();

      VECTOR_T *input_tensor = std::any_cast<VECTOR_T *>(output_tensor);
      const auto input_shape = get_shape(input);

      VECTOR_T *other_tensor;
      if (other.dtype() != input.dtype()) {
        if constexpr (std::is_same<VECTOR_T, CFloat>::value) {
          std::cerr << "No quantization operations should be emitted for CFloat"
                    << std::endl;
          std::abort();
        } else {
          // A dequantize is needed
          VECTOR_T *scale = new VECTOR_T[1];
          scale[0] = other.scale() != 0 ? other.scale() : 1.0;
          if (other.dtype() == "int32") {
            other_tensor = dequantize<DataTypes::int32, VECTOR_T>(
                args[arg_index++], scale, get_size(other));
          } else if (other.dtype() == "int24") {
            other_tensor = dequantize<DataTypes::int24, VECTOR_T>(
                args[arg_index++], scale, get_size(other));
          } else if (other.dtype() == "int8") {
            other_tensor = dequantize<DataTypes::int8, VECTOR_T>(
                args[arg_index++], scale, get_size(other));
          } else if (other.dtype() == "fp8_e4m3") {
            other_tensor = dequantize<DataTypes::e4m3, VECTOR_T>(
                args[arg_index++], scale, get_size(other));
          } else if (other.dtype() == "posit8_1") {
            other_tensor = dequantize<DataTypes::posit8, VECTOR_T>(
                args[arg_index++], scale, get_size(other));
          } else {
            std::cerr << "No dequantization operation for dtype: "
                      << other.dtype() << std::endl;
            std::abort();
          }
        }
      } else {
        other_tensor = std::any_cast<VECTOR_T *>(args[arg_index++]);
      }
      const auto other_shape = get_shape(other);

      output_tensor =
          perform_elwise_operation(input_tensor, input_shape, other_tensor,
                                   other_shape, vector_param.opcode());

    } else if (vector_param.opcode().rfind("quantize", 0) == 0) {
      if constexpr (std::is_same<VECTOR_T, CFloat>::value) {
        std::cerr << "No quantization operations should be emitted for CFloat"
                  << std::endl;
        std::abort();
      } else {
        // perform quantization operation
        output_tensor = quantize<VECTOR_T, INPUT_T>(
            output_tensor, args[arg_index++], get_size(vector_param.input()));
      }
    } else if (vector_param.opcode().rfind("dequantize", 0) == 0) {
      // perform dequantization operation
      if constexpr (std::is_same<VECTOR_T, CFloat>::value) {
        std::cerr << "No quantization operations should be emitted for CFloat"
                  << std::endl;
        std::abort();
      } else {
        if (vector_param.input().dtype() == "int32") {
          output_tensor = dequantize<DataTypes::int32, VECTOR_T>(
              output_tensor, args[arg_index++], get_size(vector_param.input()));
        } else if (vector_param.input().dtype() == "int24") {
          output_tensor = dequantize<DataTypes::int24, VECTOR_T>(
              output_tensor, args[arg_index++], get_size(vector_param.input()));
        } else if (vector_param.input().dtype() == "int8") {
          output_tensor = dequantize<DataTypes::int8, VECTOR_T>(
              output_tensor, args[arg_index++], get_size(vector_param.input()));
        } else if (vector_param.input().dtype() == "fp8_e4m3") {
          output_tensor = dequantize<DataTypes::e4m3, VECTOR_T>(
              output_tensor, args[arg_index++], get_size(vector_param.input()));
        } else {
          std::cerr << "No dequantization operation for dtype: "
                    << vector_param.input().dtype() << std::endl;
        }
      }
    } else {
      std::cerr << "Unsupported vector instruction: " << vector_param.opcode()
                << std::endl;
      std::abort();
    }
  }

  int output_size = get_size(param.output());
  if (param.output().has_permutation()) {
    if (param.output().dtype() == "bfloat16") {
      output_tensor = permute<DataTypes::bfloat16>(
          std::any_cast<DataTypes::bfloat16 *>(output_tensor), param.output());
    } else {
      // assume INPUT_T if the output tensor is not bfloat16
      output_tensor = permute<INPUT_T>(std::any_cast<INPUT_T *>(output_tensor),
                                       param.output());
    }
  }

  // save the output tensor
  char *output_bytes = std::any_cast<char *>(args.back());

  if (param.output().dtype() == "bfloat16") {
    save_tensor<DataTypes::bfloat16>(output_bytes, output_tensor, output_size);
  } else if (param.output().dtype() == "int8") {
    save_tensor<DataTypes::int8>(output_bytes, output_tensor, output_size);
  } else {
    // assume INPUT_T if the output tensor is not bfloat16 or int8
    save_tensor<INPUT_T>(output_bytes, output_tensor, output_size);
  }
}

void run_gold_model(const codegen::AcceleratorParam &param,
                    std::vector<std::any> args) {
  run_operation<INPUT_DATATYPE, INTERMEDIATE_DTYPE, ACCUM_DATATYPE,
                VECTOR_DATATYPE>(param, args);
}
