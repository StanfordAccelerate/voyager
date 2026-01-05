#include "GoldModel.h"

#include <vector>

#include "spdlog/spdlog.h"
#include "test/common/MemoryInterface.h"
#include "test/common/Utils.h"
#include "test/common/operations/LayerNorm.h"
#include "test/common/operations/MatrixOps.h"
#include "test/common/operations/Microscaling.h"
#include "test/common/operations/Pooling.h"
#include "test/common/operations/QuantizeOps.h"
#include "test/common/operations/ReshapeOps.h"
#include "test/common/operations/SlicingOp.h"
#include "test/common/operations/Softmax.h"
#include "test/common/operations/SpMM.h"
#include "test/common/operations/VectorOps.h"

template <typename T, typename U, bool is_input>
bool type_cast_helper(std::any& input_ptr, float* codebook,
                      codegen::Tensor tensor) {
  if ((is_input && tensor.dtype() != DataTypes::TypeName<T>::name()) ||
      (!is_input && tensor.dtype() != DataTypes::TypeName<U>::name())) {
    return false;
  }

  if (DataTypes::TypeName<T>::name() == DataTypes::TypeName<U>::name()) {
    return true;
  }

  int size = get_size(tensor);

  T* inputs = std::any_cast<T*>(input_ptr);
  U* outputs = new U[size];

  for (int i = 0; i < size; i++) {
    if (codebook != nullptr) {
      if constexpr (is_input) {
        outputs[i] = codebook[inputs[i].bits_rep().to_uint()];
      } else {
        unsigned index = 0;
        for (; index < NUM_CODEBOOK_ENTRIES - 1; index++) {
          if (static_cast<float>(inputs[i]) <= codebook[index]) {
            break;
          }
        }
        outputs[i] = index;
      }
    } else {
      outputs[i] = static_cast<U>(inputs[i]);
    }
  }

  delete[] inputs;

  input_ptr = outputs;

  return true;
}

template <typename U, typename... Ts>
void cast_input(std::any& input_ptr, codegen::Tensor tensor,
                float* codebook = nullptr) {
  if (!(type_cast_helper<Ts, U, true>(input_ptr, codebook, tensor) || ...)) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
  }
}

template <typename U, typename... Ts>
void cast_output(std::any& input_ptr, codegen::Tensor tensor,
                 float* codebook = nullptr) {
  if (!(type_cast_helper<U, Ts, false>(input_ptr, codebook, tensor) || ...)) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
  }
}

template <typename T>
std::any process_gemm_inputs(const codegen::OpOverload& first_op,
                             const std::string& code_key,
                             const codegen::Tensor& tensor, std::any& ptr) {
  float* codebook = nullptr;

  if (first_op.kwargs().contains(code_key)) {
    const auto code = first_op.kwargs().at(code_key).tensor();
    codebook = read_constant_param(code);
  }

  // Cast tensor to systolic array compute type
  cast_input<T, SUPPORTED_TYPES>(ptr, tensor, codebook);

  if (codebook != nullptr) {
    delete[] codebook;
  }

  // Perform reshape if necessary
  if (tensor.has_reshape()) {
    ptr = reshape_if_needed<T>(ptr, tensor.reshape());
  }

  return ptr;
}

template <typename SaInput, typename SaWeight, typename Psum,
          typename AccumBuffer, typename Scale, typename Vector>
std::vector<std::any> run_operation(const Operation& operation,
                                    std::map<std::string, std::any> kwargs) {
  std::any output_ptr;
  std::vector<std::any> outputs;
  float* output_code = nullptr;

  const auto param = operation.param;
  auto op_list = get_op_list(param);
  const auto first_op = op_list.front();
  spdlog::debug("Running operation: {}\n", first_op.target());

  const auto& output_tensors = get_op_outputs(param);
  const int num_outputs = output_tensors.size();

  if (is_gemm_op(first_op.target())) {
    const auto input = first_op.kwargs().at("input").tensor();
    std::any input_ptr = kwargs[input.node()];

    bool is_matmul = first_op.target().find("matmul") != std::string::npos;
    std::string weight_key = is_matmul ? "other" : "weight";
    const auto weight = first_op.kwargs().at(weight_key).tensor();
    std::any weight_ptr = kwargs[weight.node()];

    std::any bias_ptr = static_cast<AccumBuffer*>(nullptr);

    if (first_op.kwargs().contains("bias")) {
      const auto bias = first_op.kwargs().at("bias").tensor();
      bias_ptr = kwargs[bias.node()];
    }

    bool is_dwc = first_op.target().find("conv2d") != std::string::npos &&
                  first_op.kwargs().at("groups").int_value() > 1;

    const auto input_shape = get_shape(input);
    int input_dim = get_size(input_shape) / input_shape.back();
    bool is_fc = input_dim == 1;

    if (!is_dwc && input.dtype() != "bfloat16") {
      input_ptr = process_gemm_inputs<SaInput>(first_op, "input_code", input,
                                               input_ptr);

      if (weight.has_dequant()) {
        const auto& dequantized_op = weight.dequant();
        const auto scale_tensor = dequantized_op.kwargs().at("scale").tensor();
        const auto scale = kwargs[scale_tensor.node()];

        std::any zero_point = static_cast<Scale*>(nullptr);
        if (dequantized_op.kwargs().contains("zero_point")) {
          const auto zero_point_tensor =
              dequantized_op.kwargs().at("zero_point").tensor();
          zero_point = kwargs[zero_point_tensor.node()];
        }

        const int block_size =
            dequantized_op.kwargs().at("block_size").int_value();
        const auto axes =
            dequantized_op.kwargs().at("axes").int_list().values();

        if constexpr (std::is_same<Vector, CFloat>::value) {
          spdlog::error(
              "No quantization operations should be emitted for CFloat\n");
          std::abort();
        } else {
          weight_ptr = dequantize_tensor_group<SaWeight, Scale, Vector>(
              weight_ptr, scale, zero_point, weight, block_size, axes[0]);
        }
      } else {
        weight_ptr = process_gemm_inputs<SaWeight>(first_op, "weight_code",
                                                   weight, weight_ptr);
      }
    }

    // Fetch microscaling scales
    std::any input_scale_ptr = static_cast<Scale*>(nullptr);
    std::any weight_scale_ptr = static_cast<Scale*>(nullptr);

    bool is_mx_op = first_op.target().find("mx") != std::string::npos;
    if (is_mx_op) {
      const auto input_scale = first_op.kwargs().at("input_scale").tensor();
      input_scale_ptr = kwargs[input_scale.node()];

      const auto weight_scale = first_op.kwargs().at("weight_scale").tensor();
      weight_scale_ptr = kwargs[weight_scale.node()];
    }

    if (is_dwc) {
#if SUPPORT_DWC
      output_ptr = DwC<DWC_DATATYPE, DWC_PSUM, ACCUM_BUFFER_DATATYPE, Scale>(
          input_ptr, input_scale_ptr, weight_ptr, weight_scale_ptr, bias_ptr,
          operation);
#else
      throw std::runtime_error("DWC not supported in this build");
#endif
    } else if (is_fc) {
      if (input.dtype() == "bfloat16" || input.dtype() == "float32") {
        output_ptr = gemv_bfloat16<Vector>(input_ptr, weight_ptr, bias_ptr,
                                           get_shape(weight));
      } else {
#if !SUPPORT_MVM
        throw std::runtime_error(
            "Matrix-vector multiply not supported in this build.");
#endif
        output_ptr = gemv_quantized<SaInput, SaWeight, Psum, AccumBuffer, Scale,
                                    MV_UNIT_WIDTH>(input_ptr, input_scale_ptr,
                                                   weight_ptr, weight_scale_ptr,
                                                   bias_ptr, operation);
      }
    } else {
      output_ptr = gemm<SaInput, SaWeight, Psum, AccumBuffer, Scale>(
          input_ptr, input_scale_ptr, weight_ptr, weight_scale_ptr, bias_ptr,
          operation);
    }
  } else if (first_op.target() == "spmm_csr") {
#if !SUPPORT_SPMM
    throw std::runtime_error("SpMM not supported in this build.");
#endif
    const auto weight = first_op.kwargs().at("B").tensor();
    std::any weight_ptr = kwargs[weight.node()];
    const auto weight_scale = first_op.kwargs().at("B_scale").tensor();
    std::any weight_scale_ptr = kwargs[weight_scale.node()];
    const auto input_data = first_op.kwargs().at("data").tensor();
    std::any input_data_ptr = kwargs[input_data.node()];
    const auto input_indices = first_op.kwargs().at("indices").tensor();
    std::any input_indices_ptr = kwargs[input_indices.node()];
    const auto input_indptr = first_op.kwargs().at("indptr").tensor();
    std::any input_indptr_ptr = kwargs[input_indptr.node()];

    float* weight_code = nullptr;
    if (first_op.kwargs().contains("B_code")) {
      const auto code = first_op.kwargs().at("B_code").tensor();
      weight_code = read_constant_param(code);
    }

    cast_input<SaWeight, SUPPORTED_TYPES>(weight_ptr, weight, weight_code);

    if (weight_code != nullptr) {
      delete[] weight_code;
    }

    output_ptr = spmm_csr<SPMM_META_DATATYPE, Vector, SaWeight, Vector, Scale>(
        input_data_ptr, input_indices_ptr, input_indptr_ptr, weight_ptr,
        weight_scale_ptr, first_op);
  } else if (first_op.kwargs().contains("input")) {
    const auto input = first_op.kwargs().at("input").tensor();
    std::any input_ptr = kwargs[input.node()];

    if (input.has_dequant()) {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto dequant_op = input.dequant();
        const auto scale = dequant_op.kwargs().at("scale").tensor();
        float* param = read_constant_param(scale);
        Vector* scale_ptr = new Vector[1];
        scale_ptr[0] = param[0];
        kwargs[input.node()] =
            dequantize_tensor<Vector>(input_ptr, scale_ptr, input);
      }
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
      output_ptr = calculate_mx_qparam<Vector, Scale>(kwargs, first_op);
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
    std::any input_ptr = kwargs[input.node()];
    cast_input<Vector, SUPPORTED_TYPES>(input_ptr, input);
    output_ptr = transpose<Vector>(input_ptr, first_op);
  }

  if (first_op.target() == "permute") {
    const auto input = first_op.kwargs().at("input").tensor();
    std::any input_ptr = kwargs[input.node()];
    cast_input<Vector, SUPPORTED_TYPES>(input_ptr, input);
    output_ptr = permute<Vector>(input_ptr, first_op);
  }

  if (first_op.target() == "slice") {
    const auto input = first_op.kwargs().at("input").tensor();
    std::any input_ptr = kwargs[input.node()];
    cast_input<Vector, SUPPORTED_TYPES>(input_ptr, input);
    output_ptr = slice<Vector>(input_ptr, first_op);
  }

  if (first_op.target() == "pad") {
    const auto input = first_op.kwargs().at("input").tensor();
    std::any input_ptr = kwargs[input.node()];
    cast_input<Vector, SUPPORTED_TYPES>(input_ptr, input);
    output_ptr = pad_tensor<Vector>(input_ptr, first_op);
  }

  if (output_ptr.has_value()) {
    op_list.erase(op_list.begin());
  } else if (first_op.kwargs().contains("input")) {
    // fetch the input of the first vector instruction
    const auto input = first_op.kwargs().at("input").tensor();
    output_ptr = kwargs[input.node()];

    if (input.has_reshape()) {
      const auto reshape_op = input.reshape();
      cast_input<Vector, SUPPORTED_TYPES>(output_ptr, input);
      output_ptr = permute<Vector>(output_ptr, reshape_op);
      output_ptr = slice<Vector>(output_ptr, reshape_op);
      cast_output<Vector, SUPPORTED_TYPES>(output_ptr, input);
    }
  }

  for (const auto& op : op_list) {
    spdlog::debug("Performing fused operation: {}\n", op.target());

    if (unary_ops.find(op.target()) != unary_ops.end()) {
      const auto input = op.kwargs().at("input").tensor();
      const auto input_shape = get_shape(input);

      Vector* input_ptr = std::any_cast<Vector*>(output_ptr);

      // Grab kwargs that are relevant for some activation functions
      std::map<std::string, Vector> filtered_kwargs;
      if (unary_ops_with_kwargs.find(op.target()) !=
          unary_ops_with_kwargs.end()) {
        for (const auto& [key, value] : op.kwargs()) {
          if (key != "input") {
            filtered_kwargs[key] = Vector(value.float_value());
          }
        }
      }
      output_ptr = perform_unary_operation(input_ptr, input_shape, op.target(),
                                           filtered_kwargs);
    } else if (arithmetics.find(op.target()) != arithmetics.end()) {
      Vector* input_ptr = std::any_cast<Vector*>(output_ptr);

      Vector* other_ptr;
      std::vector<int> input_shape;
      std::vector<int> other_shape;

      const auto other = op.kwargs().at("other");

      if (other.has_float_value() || other.has_int_value() ||
          (other.has_tensor() && get_size(other.tensor()) == 1)) {
        input_shape = get_shape(op.kwargs().at("input").tensor());
        other_shape = {1};

        other_ptr = new Vector[1];
        if (other.has_float_value()) {
          other_ptr[0] = other.float_value();
        } else if (other.has_int_value()) {
          other_ptr[0] = other.int_value();
        } else if (other.has_tensor()) {
          float* array = read_constant_param(other.tensor());
          other_ptr[0] = array[0];
          delete[] array;
        }
      } else {
        auto operand1 = op.kwargs().at("input").tensor();
        auto operand2 = op.kwargs().at("other").tensor();

        // input comes from outputs of previous operations
        if (!operand2.has_memory() && get_size(operand2) != 1) {
          std::swap(operand1, operand2);
        }

        input_shape = get_shape(operand1);
        other_shape = get_shape(operand2);

        if (operand1.dtype() == operand2.dtype() ||
            operand2.dtype() == DataTypes::TypeName<Vector>::name()) {
          other_ptr = std::any_cast<Vector*>(kwargs[operand2.node()]);
        } else {
          if constexpr (std::is_same<Vector, CFloat>::value) {
            spdlog::error(
                "No quantization operations should be emitted for CFloat\n");
            std::abort();
          } else {
            float scale_val = get_tensor_scalar_scale(operand2);
            Vector* scale = new Vector[1];
            scale[0] = scale_val != 0 ? scale_val : 1.0;
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

        if (op.kwargs().contains("output_code")) {
          const auto code = op.kwargs().at("output_code").tensor();
          output_code = read_constant_param(code);
        }

        if (get_size(scale) == 1) {
          output_ptr =
              quantize<Vector, Vector>(output_ptr, scale_ptr, input_shape);
        } else {
          const int block_size = op.kwargs().at("block_size").int_value();

          assert(input_shape.size() == scale_shape.size());

          int axis = 0;
          for (; axis < input_shape.size(); axis++) {
            if (input_shape[axis] != scale_shape[axis]) break;
          }

          output_ptr = quantize_mx<Vector, Scale>(
              output_ptr, scale_ptr, input_shape, block_size, axis);
        }
      }
    } else if (op.target() == "quantize_mx" ||
               op.target() == "quantize_mx_outlier") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto input = op.kwargs().at("input").tensor();
        const auto input_shape = get_shape(input);
        const float quant_max = op.kwargs().at("quant_max").float_value();
        const int block_size = op.kwargs().at("block_size").int_value();
        const int axis = op.kwargs().at("axes").int_list().values()[0];
        const bool force_scale_power_of_two =
            op.kwargs().at("force_scale_power_of_two").bool_value();

        if (op.kwargs().contains("output_code")) {
          const auto code = op.kwargs().at("output_code").tensor();
          output_code = read_constant_param(code);
        }

        if (op.target() == "quantize_mx_outlier") {
          const float threshold = op.kwargs().at("threshold").float_value();
          const auto& csr_data = output_tensors[0];
          const int data_size = get_size(csr_data);
          auto [data, indices, indptr, filtered] =
              filter_outlier<Vector, SPMM_META_DATATYPE>(
                  output_ptr, input_shape, data_size, threshold);
          outputs.insert(outputs.end(), {data, indices, indptr});
          output_ptr = filtered;
        }

        std::any mx_scale = calculate_mx_qparam<Vector, Scale>(
            output_ptr, input_shape, quant_max, block_size, axis,
            force_scale_power_of_two);

        output_ptr = quantize_mx<Vector, Vector>(output_ptr, mx_scale,
                                                 input_shape, block_size, axis);

        cast_output<Vector, SUPPORTED_TYPES>(mx_scale,
                                             output_tensors[num_outputs - 2]);
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

      Vector* input_ptr = std::any_cast<Vector*>(output_ptr);

      const auto other = op.kwargs().at("other").tensor();
      DataTypes::bfloat16* value_map =
          std::any_cast<DataTypes::bfloat16*>(kwargs[other.node()]);

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

  const auto output_tensor = output_tensors[num_outputs - 1];

  if (output_tensor.has_reshape()) {
    const auto reshape_op = output_tensor.reshape();
    output_ptr = reshape_if_needed<Vector>(output_ptr, reshape_op);
  }

  cast_output<Vector, SUPPORTED_TYPES>(output_ptr, output_tensor, output_code);
  outputs.push_back(output_ptr);

  if (output_code != nullptr) {
    delete[] output_code;
  }

  spdlog::debug("Operation {} finished\n", first_op.target());

  return outputs;
}

std::vector<std::any> run_gold_model(const Operation& operation,
                                     std::map<std::string, std::any> kwargs) {
  return run_operation<SA_INPUT_TYPE, SA_WEIGHT_TYPE, ACCUM_DATATYPE,
                       ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE, VECTOR_DATATYPE>(
      operation, kwargs);
}
