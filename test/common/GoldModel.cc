#include "GoldModel.h"

#include <vector>

#include "test/common/VerificationTypes.h"
#include "test/common/operations/LayerNorm.h"
#include "test/common/operations/MatrixOps.h"
#include "test/common/operations/MxOps.h"
#include "test/common/operations/Pooling.h"
#include "test/common/operations/QuantizeOps.h"
#include "test/common/operations/ReduceOps.h"
#include "test/common/operations/ReshapeOps.h"
#include "test/common/operations/SlicingOp.h"
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

template <typename Input, typename Psum, typename AccumBuffer, typename Scale,
          typename Vector>
std::any run_operation(const codegen::Operator param,
                       std::vector<std::any> args) {
  int arg_index = 0;
  std::any output_tensor;

  if (param.has_reduce_op()) {
    const auto &reduce_op = param.reduce_op();
    if (reduce_op.opcode() == "softmax") {
      const auto &input = reduce_op.input();
      const auto input_shape = get_shape(input);
      output_tensor = softmax<Vector>(args[arg_index++], input_shape);
    } else if (reduce_op.opcode() == "sum") {
      const auto &input = reduce_op.input();
      const auto input_shape = get_shape(input);

      std::vector<int> dims;
      for (int dim : reduce_op.dim()) {
        dims.push_back(dim);
      }

      output_tensor = sum<Vector>(args[arg_index++], input_shape, dims);
    } else if (reduce_op.opcode() == "calculate_mx_qparam") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        std::cerr
            << "No calculate_mx_param operation should be emitted for CFloat"
            << std::endl;
        std::abort();
      } else {
        output_tensor = calculate_mx_qparam<Vector, Scale, Input>(
            args[arg_index++], reduce_op);
      }
    } else {
      std::cerr << "Unsupported reduce instruction: " << reduce_op.opcode()
                << std::endl;
      exit(1);
    }
  }

  if (param.has_pooling_op()) {
    const auto input = param.pooling_op().input();
    output_tensor = pooling<Vector>(args[arg_index++], param);
  }

  if (param.has_reshape_op()) {
    const auto &reshape_op = param.reshape_op();
    if (reshape_op.input().dtype() ==
        DataTypes::TypeName<INPUT_DATATYPE>::name()) {
      output_tensor = permute<INPUT_DATATYPE>(args[arg_index++], reshape_op);
    } else {
      output_tensor = permute<Vector>(args[arg_index++], reshape_op);
    }
  }

  if (param.has_slicing_op()) {
    const auto &slicing_op = param.slicing_op();
    if (DataTypes::TypeName<INPUT_DATATYPE>::name() ==
        slicing_op.input().dtype()) {
      output_tensor = slice<INPUT_DATATYPE>(args[arg_index++], slicing_op);
    } else {
      output_tensor = slice<Vector>(args[arg_index++], slicing_op);
    }
  }

  if (param.matrix_op().opcode() == "layer_norm") {
    const auto &matrix_op = param.matrix_op();
    output_tensor = layer_norm<Vector>(args[0], args[1], args[2], matrix_op);
  } else if (param.has_matrix_op()) {
    const auto &matrix_op = param.matrix_op();

    if (matrix_op.opcode() == "conv2d" && matrix_op.groups() != 1) {
      std::cerr << "Grouped convolution is not supported" << std::endl;
      std::abort();
    }

    // Permute input tensor
    const auto &input = matrix_op.has_mx_input() ? matrix_op.mx_input().input()
                                                 : matrix_op.input();

    std::any input_tensor = args[arg_index++];
    std::any input_scale = nullptr;
    if (matrix_op.has_mx_input()) {
      input_scale = args[arg_index++];
    }
    if (input.has_reshape()) {
      input_tensor = permute<Input>(input_tensor, input);
    } else if (input.has_slicing()) {
      input_tensor = slice<Input>(input_tensor, input);
    }

    // Permute weight tensor
    const auto &weight = matrix_op.has_mx_weight()
                             ? matrix_op.mx_weight().input()
                             : matrix_op.weight();

    std::any weight_tensor = args[arg_index++];
    std::any weight_scale = nullptr;
    if (matrix_op.has_mx_weight()) {
      weight_scale = args[arg_index++];
    }
    if (weight.has_reshape()) {
      weight_tensor = permute<Input>(weight_tensor, weight);
    } else if (weight.has_slicing()) {
      weight_tensor = slice<Input>(weight_tensor, weight);
    }

    int dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      dim *= input.shape(i);
    }

    if (dim == 1) {
      output_tensor = matrix_vector_multiply<Input, Psum, Vector>(
          input_tensor, weight_tensor, args[arg_index++], matrix_op);
    } else {
      AccumBuffer *outputs = gemm<Input, Psum, AccumBuffer, Scale>(
          input_tensor, input_scale, weight_tensor, weight_scale,
          args[arg_index++], param);

      // If accumulation buffer type is different from vector unit type, cast
      // the output of GEMM to vector unit type
      if (DataTypes::TypeName<AccumBuffer>::name() !=
          DataTypes::TypeName<Vector>::name()) {
        int size = get_size(param.output());
        Vector *casted_outputs = new Vector[size];
        for (int i = 0; i < size; i++) {
          casted_outputs[i] = static_cast<Vector>(outputs[i]);
        }

        output_tensor = casted_outputs;
        delete[] outputs;
      } else {
        output_tensor = outputs;
      }
    }
  } else if (param.vector_ops_size() > 0) {
    // fetch the input of the first vector instruction
    output_tensor = args[arg_index++];

    const auto vector_input = param.vector_ops(0).input();
    if (vector_input.has_reshape()) {
      output_tensor = permute<Vector>(output_tensor, vector_input);
    } else if (vector_input.has_slicing()) {
      output_tensor = slice<Vector>(output_tensor, vector_input);
    }
  }

  for (const auto &vector_param : param.vector_ops()) {
    Vector *input_tensor = std::any_cast<Vector *>(output_tensor);

    if (vector_param.opcode().rfind("sqrt", 0) == 0) {
      output_tensor = sqrt(input_tensor, get_shape(vector_param.input()));
    } else if (activations.find(vector_param.opcode()) != activations.end()) {
      int input_size = get_size(vector_param.input());
      for (int i = 0; i < input_size; i++) {
        if (vector_param.opcode() == "relu" ||
            vector_param.opcode() == "relu_") {
          relu(input_tensor[i]);
        } else if (vector_param.opcode() == "gelu" ||
                   vector_param.opcode() == "gelu_") {
          gelu(input_tensor[i]);
        } else if (vector_param.opcode() == "silu" ||
                   vector_param.opcode() == "silu_") {
          silu(input_tensor[i]);
        }
      }
    } else if (arithmetics.find(vector_param.opcode()) != arithmetics.end()) {
      const bool use_input =
          !vector_param.has_other() || vector_param.other().has_memory();
      const auto &input =
          use_input ? vector_param.input() : vector_param.other();
      const auto &other =
          use_input ? vector_param.other() : vector_param.input();

      const auto input_shape = get_shape(input);

      Vector *other_tensor;
      if (vector_param.has_other_scalar()) {
        other_tensor = new Vector[1];
        other_tensor[0] = vector_param.other_scalar();
      } else if (other.dtype() != input.dtype()) {
        if constexpr (std::is_same<Vector, CFloat>::value) {
          std::cerr << "No quantization operations should be emitted for CFloat"
                    << std::endl;
          std::abort();
        } else {
          Vector *scale = new Vector[1];
          scale[0] = other.scale() != 0 ? other.scale() : 1.0;
          other_tensor =
              dequantize_tensor<Vector>(args[arg_index++], scale, other);
        }
      } else {
        other_tensor = std::any_cast<Vector *>(args[arg_index++]);
      }

      if (other.has_reshape()) {
        other_tensor = permute<Vector>(other_tensor, other);
      } else if (other.has_slicing()) {
        other_tensor = slice<Vector>(other_tensor, other);
      }

      auto other_shape = get_shape(other);
      if (vector_param.has_other_scalar()) {
        other_shape = {1};
      }

      output_tensor =
          perform_vector_operation(input_tensor, input_shape, other_tensor,
                                   other_shape, vector_param.opcode());

    } else if (vector_param.opcode().rfind("quantize", 0) == 0) {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        std::cerr << "No quantization operations should be emitted for CFloat"
                  << std::endl;
        std::abort();
      } else {
        if (vector_param.other().shape_size() == 1) {
          // perform quantization operation
          output_tensor = quantize<Vector, Input, Vector>(
              input_tensor, args[arg_index++], get_size(vector_param.input()));
        } else if (vector_param.other().shape_size() > 1) {
          // perform microscaling quantization operation
          output_tensor = quantize_mx<Vector, Input, Scale>(
              input_tensor, args[arg_index++], vector_param);
        } else {
          std::cerr << "No quantization operation for dtype: "
                    << vector_param.other().dtype() << std::endl;
        }
      }
    } else if (vector_param.opcode().rfind("dequantize", 0) == 0) {
      // perform dequantization operation
      if constexpr (std::is_same<Vector, CFloat>::value) {
        std::cerr << "No quantization operations should be emitted for CFloat"
                  << std::endl;
        std::abort();
      } else {
        output_tensor = dequantize_tensor<Vector>(
            input_tensor, args[arg_index++], vector_param.input());
      }
    } else if (vector_param.opcode().rfind("vmap", 0) == 0) {
      const auto &input = vector_param.input();
      const int size = get_size(input);

      DataTypes::bfloat16 *value_map =
          std::any_cast<DataTypes::bfloat16 *>(args[arg_index++]);

      for (int i = 0; i < size; i++) {
        DataTypes::bfloat16 value =
            static_cast<DataTypes::bfloat16>(input_tensor[i]);
        unsigned int index = value.bits_rep().to_uint();
        input_tensor[i] = static_cast<Vector>(value_map[index]);
      }

      output_tensor = input_tensor;
    } else {
      std::cerr << "Unsupported vector instruction: " << vector_param.opcode()
                << std::endl;
      std::abort();
    }
  }

  int output_size = get_size(param.output());
  if (param.output().has_reshape()) {
    if (param.output().dtype() == "bfloat16") {
      output_tensor = permute<DataTypes::bfloat16>(
          std::any_cast<DataTypes::bfloat16 *>(output_tensor), param.output());
    } else {
      // assume Input if the output tensor is not bfloat16
      output_tensor = permute<Input>(output_tensor, param.output());
    }
  }

  return output_tensor;
}

std::any run_gold_model(const codegen::Operator &param,
                        std::vector<std::any> args) {
  return run_operation<INPUT_DATATYPE, ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE,
                       SCALE_DATATYPE, VECTOR_DATATYPE>(param, args);
}
