#include "GoldModel.h"

#include <vector>

#include "spdlog/spdlog.h"
#include "test/common/VerificationTypes.h"
#include "test/common/operations/LayerNorm.h"
#include "test/common/operations/MatrixOps.h"
#include "test/common/operations/Microscaling.h"
#include "test/common/operations/Pooling.h"
#include "test/common/operations/QuantizeOps.h"
#include "test/common/operations/ReshapeOps.h"
#include "test/common/operations/SlicingOp.h"
#include "test/common/operations/Softmax.h"
#include "test/common/operations/VectorOps.h"

template <typename Input, typename Psum, typename AccumBuffer, typename Scale,
          typename Vector>
std::vector<std::any> run_operation(const Operation &operation,
                                    std::map<std::string, std::any> kwargs) {
  std::any output_ptr;
  std::vector<std::any> outputs;

  const auto param = operation.param;
  auto op_list = get_op_list(param);
  const auto first_op = op_list.front();
  spdlog::debug("Running operation: {}\n", first_op.target());

  if (GEMM_OPS.find(first_op.target()) != GEMM_OPS.end()) {
    if (first_op.target() == "conv2d" &&
        first_op.kwargs().at("groups").int_value() > 1) {
      spdlog::error("Grouped convolution is not supported\n");
      std::abort();
    }

    const auto input = first_op.kwargs().at("input").tensor();
    std::any input_ptr = kwargs[input.node()];

    bool is_matmul = first_op.target().find("matmul") != std::string::npos;
    std::string weight_key = is_matmul ? "other" : "weight";
    const auto weight = first_op.kwargs().at(weight_key).tensor();
    std::any weight_ptr = kwargs[weight.node()];

    std::any input_scale_ptr = static_cast<Scale *>(nullptr);
    std::any weight_scale_ptr = static_cast<Scale *>(nullptr);

    bool is_mx_op = first_op.target().find("mx") != std::string::npos;
    if (is_mx_op) {
      const auto input_scale = first_op.kwargs().at("input_scale").tensor();
      input_scale_ptr = kwargs[input_scale.node()];

      const auto weight_scale = first_op.kwargs().at("weight_scale").tensor();
      weight_scale_ptr = kwargs[weight_scale.node()];
    }

    std::any bias_ptr = static_cast<AccumBuffer *>(nullptr);

    if (first_op.kwargs().contains("bias")) {
      const auto bias = first_op.kwargs().at("bias").tensor();
      bias_ptr = kwargs[bias.node()];
    }

    if (input.has_reshape()) {
      input_ptr = transpose<Input>(input_ptr, input.reshape());
      input_ptr = permute<Input>(input_ptr, input.reshape());
    }

    if (weight.has_reshape()) {
      weight_ptr = transpose<Input>(weight_ptr, weight.reshape());
      weight_ptr = permute<Input>(weight_ptr, weight.reshape());
    }

    int dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      dim *= input.shape(i);
    }

    if (dim == 1) {
      output_ptr = matrix_vector_multiply<Input, Psum, Vector>(
          input_ptr, weight_ptr, bias_ptr, get_shape(weight));
    } else {
      output_ptr = gemm<Input, Psum, AccumBuffer, Scale>(
          input_ptr, input_scale_ptr, weight_ptr, weight_scale_ptr, bias_ptr,
          operation);
    }
  }

  if (first_op.target() == "layer_norm") {
    output_ptr = layer_norm<Vector>(kwargs, first_op);
  }

  if (first_op.target() == "softmax") {
    output_ptr = softmax<Vector>(kwargs, first_op);
  }

  if (first_op.target() == "calculate_mx_qparam") {
    if constexpr (std::is_same<Vector, CFloat>::value) {
      spdlog::error(
          "No calculate_mx_param operation should be emitted for CFloat\n");
      std::abort();
    } else {
      output_ptr = calculate_mx_qparam<Vector, Scale, Input>(kwargs, first_op);
    }
  }

  if (first_op.target() == "max_pool2d") {
    output_ptr = max_pool2d<Vector>(kwargs, first_op);
  }

  if (first_op.target() == "adaptive_avg_pool2d") {
    output_ptr = adaptive_avg_pool2d<Vector>(kwargs, first_op);
  }

  if (first_op.target() == "transpose") {
    const auto input = first_op.kwargs().at("input").tensor();
    if (input.dtype() == DataTypes::TypeName<Input>::name()) {
      output_ptr = transpose<Input>(kwargs[input.node()], first_op);
    } else {
      output_ptr = transpose<Vector>(kwargs[input.node()], first_op);
    }
  }

  if (first_op.target() == "permute") {
    const auto input = first_op.kwargs().at("input").tensor();
    if (input.dtype() == DataTypes::TypeName<Input>::name()) {
      output_ptr = permute<Input>(kwargs[input.node()], first_op);
    } else {
      output_ptr = permute<Vector>(kwargs[input.node()], first_op);
    }
  }

  if (first_op.target() == "slice") {
    const auto input = first_op.kwargs().at("input").tensor();
    if (input.dtype() == DataTypes::TypeName<Input>::name()) {
      output_ptr = slice<Input>(kwargs[input.node()], first_op);
    } else {
      output_ptr = slice<Vector>(kwargs[input.node()], first_op);
    }
  }

  if (output_ptr.has_value()) {
    op_list.erase(op_list.begin());
  } else {
    // fetch the input of the first vector instruction
    const auto input = first_op.kwargs().at("input").tensor();
    output_ptr = kwargs[input.node()];

    if (input.has_reshape()) {
      const auto reshape_op = input.reshape();
      output_ptr = permute<Vector>(output_ptr, reshape_op);
      output_ptr = slice<Vector>(output_ptr, reshape_op);
    }
  }

  for (const auto &op : op_list) {
    spdlog::debug("Performing fused operation: {}\n", op.target());

    if (unary_ops.find(op.target()) != unary_ops.end()) {
      const auto input = op.kwargs().at("input").tensor();
      const auto input_shape = get_shape(input);

      Vector *input_ptr = std::any_cast<Vector *>(output_ptr);
      output_ptr = perform_unary_operation(input_ptr, input_shape, op.target());
    } else if (arithmetics.find(op.target()) != arithmetics.end()) {
      Vector *input_ptr = std::any_cast<Vector *>(output_ptr);

      Vector *other_ptr;
      std::vector<int> input_shape;
      std::vector<int> other_shape;

      const auto other = op.kwargs().at("other");

      if (other.has_float_value() || other.has_int_value()) {
        input_shape = get_shape(op.kwargs().at("input").tensor());
        other_shape = {1};

        other_ptr = new Vector[1];
        other_ptr[0] =
            other.has_float_value() ? other.float_value() : other.int_value();
      } else {
        auto operand1 = op.kwargs().at("input").tensor();
        auto operand2 = op.kwargs().at("other").tensor();

        // input comes from outputs of previous operations
        if (!operand2.has_memory()) {
          std::swap(operand1, operand2);
        }

        input_shape = get_shape(operand1);
        other_shape = get_shape(operand2);

        if (operand1.dtype() == operand2.dtype()) {
          other_ptr = std::any_cast<Vector *>(kwargs[operand2.node()]);
        } else {
          if constexpr (std::is_same<Vector, CFloat>::value) {
            spdlog::error(
                "No quantization operations should be emitted for CFloat\n");
            std::abort();
          } else {
            Vector *scale = new Vector[1];
            scale[0] = operand2.scale() != 0 ? operand2.scale() : 1.0;
            other_ptr = dequantize_tensor<Vector>(kwargs[operand2.node()],
                                                  scale, operand2);
          }
        }

        if (operand2.has_reshape()) {
          other_ptr = permute<Vector>(other_ptr, operand2.reshape());
          other_ptr = slice<Vector>(other_ptr, operand2.reshape());
        }
      }

      output_ptr = perform_vector_operation(input_ptr, input_shape, other_ptr,
                                            other_shape, op.target());
    } else if (op.target() == "quantize") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto input = op.kwargs().at("input").tensor();
        const auto input_shape = get_shape(input);

        const auto scale = op.kwargs().at("scale").tensor();
        std::any scale_ptr = kwargs[scale.node()];
        const auto scale_shape = get_shape(scale);

        if (scale.shape_size() == 1) {
          output_ptr = quantize<Vector, Input, Vector>(output_ptr, scale_ptr,
                                                       input_shape);
        } else {
          const int block_size = op.kwargs().at("block_size").int_value();

          assert(input_shape.size() == scale_shape.size());

          int axis = 0;
          for (; axis < input_shape.size(); axis++) {
            if (input_shape[axis] != scale_shape[axis]) break;
          }

          output_ptr = quantize_mx<Vector, Input, Scale>(
              output_ptr, scale_ptr, input_shape, block_size, axis);
        }
      }
    } else if (op.target() == "quantize_mx") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto input = op.kwargs().at("input").tensor();
        const auto input_shape = get_shape(input);

        const int block_size = op.kwargs().at("block_size").int_value();
        const int axis = op.kwargs().at("axis").int_value();

        Scale *mx_scale = calculate_mx_qparam<Vector, Scale, Input>(
            output_ptr, input_shape, block_size, axis);
        output_ptr = quantize_mx<Vector, Input, Scale>(
            output_ptr, mx_scale, input_shape, block_size, axis);

        outputs.push_back(mx_scale);
      }
    } else if (op.target() == "dequantize") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto input = op.kwargs().at("input").tensor();
        const auto scale = op.kwargs().at("scale").tensor();
        std::any scale_ptr = kwargs[scale.node()];
        output_ptr = dequantize_tensor<Vector>(output_ptr, scale_ptr, input);
      }
    } else if (op.target().rfind("vmap", 0) == 0) {
      const auto input = op.kwargs().at("input").tensor();
      const int size = get_size(input);

      Vector *input_ptr = std::any_cast<Vector *>(output_ptr);

      const auto other = op.kwargs().at("other").tensor();
      DataTypes::bfloat16 *value_map =
          std::any_cast<DataTypes::bfloat16 *>(kwargs[other.node()]);

      for (int i = 0; i < size; i++) {
        DataTypes::bfloat16 value =
            static_cast<DataTypes::bfloat16>(input_ptr[i]);
        unsigned int index = value.bits_rep().to_uint();
        input_ptr[i] = static_cast<Vector>(value_map[index]);
      }

      output_ptr = input_ptr;
    } else {
      spdlog::error("Unsupported instruction: {}\n", op.target());
      std::abort();
    }
  }

  if (param.output().has_reshape()) {
    const auto reshape_op = param.output().reshape();
    if (param.output().dtype() == DataTypes::TypeName<Vector>::name()) {
      output_ptr = transpose<Vector>(output_ptr, reshape_op);
      output_ptr = permute<Vector>(output_ptr, reshape_op);
    } else {
      output_ptr = transpose<Input>(output_ptr, reshape_op);
      output_ptr = permute<Input>(output_ptr, reshape_op);
    }
  }

  outputs.push_back(output_ptr);

  return outputs;
}

std::vector<std::any> run_gold_model(const Operation &operation,
                                     std::map<std::string, std::any> kwargs) {
  return run_operation<INPUT_DATATYPE, ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE,
                       SCALE_DATATYPE, VECTOR_DATATYPE>(operation, kwargs);
}
