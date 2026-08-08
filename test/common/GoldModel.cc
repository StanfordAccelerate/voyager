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
#include "test/common/operations/Reduction.h"
#include "test/common/operations/ReshapeOps.h"
#include "test/common/operations/SlicingOp.h"
#include "test/common/operations/Softmax.h"
#include "test/common/operations/SpMM.h"
#include "test/common/operations/VectorOps.h"

template <typename T, typename U, bool is_input>
bool type_cast_helper(std::any& input_ptr, float* codebook, Tensor tensor) {
  if ((is_input && tensor.dtype != DataTypes::TypeName<T>::name()) ||
      (!is_input && tensor.dtype != DataTypes::TypeName<U>::name())) {
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
    } else if (static_cast<float>(inputs[i]) == 0.0f) {
      // UFloat's float constructor substitutes one for zero, which would
      // mangle the raw zero scale of an all-zero strided block on its way
      // into the stored scale dtype. A zero always casts to a zero.
      outputs[i] = U::zero();
    } else {
      outputs[i] = static_cast<U>(inputs[i]);
    }
  }

  delete[] inputs;

  input_ptr = outputs;

  return true;
}

template <typename U, typename... Ts>
void cast_input(std::any& input_ptr, Tensor tensor, float* codebook = nullptr) {
  if (!(type_cast_helper<Ts, U, true>(input_ptr, codebook, tensor) || ...)) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype);
  }
}

template <typename U, typename... Ts>
void cast_output(std::any& input_ptr, Tensor tensor,
                 float* codebook = nullptr) {
  if (!(type_cast_helper<U, Ts, false>(input_ptr, codebook, tensor) || ...)) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype);
  }
}

template <typename T>
std::any process_gemm_inputs(const voyager::PrimOp& first_op,
                             const ScalarEnv& env, const std::string& code_key,
                             const Tensor& tensor, std::any& ptr) {
  float* codebook = nullptr;

  if (has_arg(first_op, code_key)) {
    const auto code = resolve(first_op, code_key, env);
    codebook = read_constant_param(code);
  }

  // Cast tensor to systolic array compute type
  cast_input<T, SUPPORTED_TYPES>(ptr, tensor, codebook);

  if (codebook != nullptr) {
    delete[] codebook;
  }

  return ptr;
}

template <typename SaInput, typename SaWeight, typename Psum,
          typename AccumBuffer, typename Scale, typename Vector>
std::vector<std::any> run_operation(const voyager::Operation& operation,
                                    const ScalarEnv& env,
                                    std::map<std::string, std::any> kwargs) {
  std::vector<std::any> outputs;
  float* output_code = nullptr;

  const auto op_list = get_prim_ops(operation);
  const auto output_tensors = resolve_outputs(operation, env);
  const int num_outputs = output_tensors.size();

  spdlog::debug("Running operation: {}\n",
                strip_namespace(op_list.front()->target()));

  // Every operand is in kwargs under its node name: a buffer preloaded from
  // memory, a scalar immediate read off disk, or a value an earlier prim of
  // this fusion produced and stored under its own name. So the op_list needs
  // no ordering beyond the topological order the compiler emits it in, and a
  // fusion may be a DAG rather than a chain.
  const auto operand = [&](const voyager::PrimOp& op, const std::string& key) {
    return kwargs[resolve(op, key, env).node];
  };

  // The vector unit computes in one type, so a buffer stored more narrowly --
  // a residual kept in fp8 -- is converted as it is fetched. A value an
  // earlier prim of this fusion produced is already in that type whatever
  // dtype the IR declares for it; the chain narrows once, at cast_output.
  const auto vector_operand = [&](const voyager::PrimOp& op,
                                  const std::string& key) {
    const Tensor tensor = resolve(op, key, env);
    std::any ptr = kwargs[tensor.node];
    if (ptr.type() != typeid(Vector*)) {
      cast_input<Vector, SUPPORTED_TYPES>(ptr, tensor);
    }
    return ptr;
  };

  for (const voyager::PrimOp* op_ptr : op_list) {
    const voyager::PrimOp& op = *op_ptr;
    const std::string target = strip_namespace(op.target());
    spdlog::debug("Performing fused operation: {}\n", target);

    std::any output_ptr;

    if (is_gemm_op(target)) {
      const auto input = resolve(op, "input", env);
      std::any input_ptr = operand(op, "input");

      bool is_matmul = target.find("matmul") != std::string::npos;
      std::string weight_key = is_matmul ? "other" : "weight";
      const auto weight = resolve(op, weight_key, env);
      std::any weight_ptr = operand(op, weight_key);

      std::any bias_ptr = static_cast<AccumBuffer*>(nullptr);
      if (has_arg(op, "bias")) {
        bias_ptr = operand(op, "bias");
      }

      bool is_dwc = target.find("conv2d") != std::string::npos &&
                    arg_int(op, "groups", env) > 1;

      const auto input_shape = get_shape(input);
      int input_dim = get_size(input_shape) / input_shape.back();
      bool is_fc = input_dim == 1;

      if (!is_dwc && input.dtype != "bfloat16") {
        input_ptr = process_gemm_inputs<SaInput>(op, env, "input_code", input,
                                                 input_ptr);
        weight_ptr = process_gemm_inputs<SaWeight>(op, env, "weight_code",
                                                   weight, weight_ptr);
      }

      // Fetch microscaling scales
      std::any input_scale_ptr = static_cast<Scale*>(nullptr);
      std::any weight_scale_ptr = static_cast<Scale*>(nullptr);

      bool is_mx_op = target.find("mx") != std::string::npos;
      if (is_mx_op) {
        input_scale_ptr = operand(op, "input_scale");
        weight_scale_ptr = operand(op, "weight_scale");
      }

      if (is_dwc) {
#if SUPPORT_DWC
        output_ptr = DwC<DWC_DATATYPE, DWC_PSUM, ACCUM_BUFFER_DATATYPE, Scale>(
            input_ptr, input_scale_ptr, weight_ptr, weight_scale_ptr, bias_ptr,
            operation, env);
#else
        throw std::runtime_error("DWC not supported in this build");
#endif
      } else if (is_fc) {
        if (input.dtype == "bfloat16" || input.dtype == "float32") {
          output_ptr = gemv_bfloat16<Vector>(input_ptr, weight_ptr, bias_ptr,
                                             get_shape(weight));
        } else {
#if !SUPPORT_MVM
          throw std::runtime_error(
              "Matrix-vector multiply not supported in this build.");
#endif
          output_ptr = gemv_quantized<SaInput, SaWeight, Psum, AccumBuffer,
                                      Scale, MV_UNIT_WIDTH>(
              input_ptr, input_scale_ptr, weight_ptr, weight_scale_ptr,
              bias_ptr, operation, env);
        }
      } else {
        output_ptr = gemm<SaInput, SaWeight, Psum, AccumBuffer, Scale>(
            input_ptr, input_scale_ptr, weight_ptr, weight_scale_ptr, bias_ptr,
            operation, env);
        if (has_fused_spmm(op)) {
#if !SUPPORT_SPMM
          throw std::runtime_error(
              "Sparse matrix operations not supported in this build.");
#endif
          std::any spmm_output_ptr =
              spmm_csr<SPMM_META_DATATYPE, Vector, SaWeight, Vector, Scale>(
                  operand(op, "A_data"), operand(op, "A_indices"),
                  operand(op, "A_indptr"), weight_ptr, weight_scale_ptr, op,
                  env);

          // Combine GEMM output with SpMM output
          int output_size = get_size(get_shape(output_tensors[0]));
          for (int i = 0; i < output_size; i++) {
            std::any_cast<Vector*>(output_ptr)[i] +=
                std::any_cast<Vector*>(spmm_output_ptr)[i];
          }
        }
      }
    } else if (target == "spmm_csr") {
#if !SUPPORT_SPMM
      throw std::runtime_error("SpMM not supported in this build.");
#endif
      const auto weight = resolve(op, "weight", env);
      std::any weight_ptr = operand(op, "weight");

      float* weight_code = nullptr;
      if (has_arg(op, "weight_code")) {
        weight_code = read_constant_param(resolve(op, "weight_code", env));
      }

      cast_input<SaWeight, SUPPORTED_TYPES>(weight_ptr, weight, weight_code);

      if (weight_code != nullptr) {
        delete[] weight_code;
      }

      output_ptr =
          spmm_csr<SPMM_META_DATATYPE, Vector, SaWeight, Vector, Scale>(
              operand(op, "A_data"), operand(op, "A_indices"),
              operand(op, "A_indptr"), weight_ptr, operand(op, "weight_scale"),
              op, env);
    } else if (target == "layer_norm") {
      output_ptr = layer_norm<Vector>(kwargs, op, env);
    } else if (target == "softmax") {
      output_ptr = softmax<Vector>(kwargs, op, env);
    } else if (target == "calculate_mx_qparam") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No calculate_mx_param operation should be emitted for CFloat\n");
        std::abort();
      } else {
        output_ptr = calculate_mx_qparam<Vector, Scale>(kwargs, op, env);
      }
    } else if (target == "max_pool2d") {
      output_ptr = max_pool2d<Vector>(kwargs, op, env);
    } else if (target == "adaptive_avg_pool2d") {
      output_ptr = adaptive_avg_pool2d<Vector>(kwargs, op, env);
    } else if (target == "transpose") {
      output_ptr = transpose<Vector>(vector_operand(op, "input"), op, env);
    } else if (target == "permute") {
      output_ptr = permute<Vector>(vector_operand(op, "input"), op, env);
    } else if (target == "expand") {
      output_ptr = expand<Vector>(vector_operand(op, "input"), op, env);
    } else if (target == "slice") {
      output_ptr = slice<Vector>(vector_operand(op, "input"), op, env);
    } else if (target == "pad") {
      output_ptr = pad_tensor<Vector>(vector_operand(op, "input"), op, env);
    } else if (unary_ops.find(target) != unary_ops.end()) {
      const auto input_shape = get_shape(resolve(op, "input", env));
      Vector* input_ptr = std::any_cast<Vector*>(vector_operand(op, "input"));

      // Grab kwargs that are relevant for some activation functions
      std::map<std::string, Vector> filtered_kwargs;
      if (unary_ops_with_kwargs.find(target) != unary_ops_with_kwargs.end()) {
        for (const auto& [key, value] : arg_scalars(op, env)) {
          filtered_kwargs[key] = Vector(value);
        }
      }
      output_ptr = perform_unary_operation(input_ptr, input_shape, target,
                                           filtered_kwargs);
    } else if (arithmetics.find(target) != arithmetics.end()) {
      const auto input_shape = get_shape(resolve(op, "input", env));
      Vector* input_ptr = std::any_cast<Vector*>(vector_operand(op, "input"));

      // The addend is either an immediate, a scalar constant read off disk, or
      // a whole buffer.
      Vector* other_ptr;
      std::vector<int> other_shape;
      const bool other_is_tensor = has_tensor(op, "other");
      const Tensor other =
          other_is_tensor ? resolve(op, "other", env) : Tensor();

      if (!other_is_tensor || other.is_constant) {
        other_shape = {1};
        other_ptr = new Vector[1];
        if (other_is_tensor) {
          float* array = read_constant_param(other);
          other_ptr[0] = array[0];
          delete[] array;
        } else {
          other_ptr[0] = arg_float(op, "other", env);
        }
      } else {
        other_shape = get_shape(other);
        other_ptr = std::any_cast<Vector*>(vector_operand(op, "other"));
      }

      output_ptr = perform_vector_operation(input_ptr, input_shape, other_ptr,
                                            other_shape, target);
    } else if (target == "quantize") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto input = resolve(op, "input", env);
        const auto input_shape = get_shape(input);
        const auto scale = resolve(op, "scale", env);
        std::any input_ptr = vector_operand(op, "input");

        if (has_arg(op, "output_code")) {
          output_code = read_constant_param(resolve(op, "output_code", env));
        }

        if (get_size(scale) == 1) {
          output_ptr = quantize<Vector, Vector>(input_ptr, operand(op, "scale"),
                                                input_shape);
        } else {
          const auto scale_shape = get_shape(scale);
          const int block_size = arg_int(op, "block_size", env);

          assert(input_shape.size() == scale_shape.size());

          int axis = 0;
          for (; axis < input_shape.size(); axis++) {
            if (input_shape[axis] != scale_shape[axis]) break;
          }

          output_ptr = quantize_mx<Vector, Scale>(
              input_ptr, operand(op, "scale"), input_shape, block_size, axis);
        }
      }
    } else if (target == "quantize_mx" || target == "quantize_mx_outlier") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto input_shape = get_shape(resolve(op, "input", env));
        const float quant_max = arg_float(op, "quant_max", env);
        const int block_size = arg_int(op, "block_size", env);
        const int axis = arg_ints(op, "axes", env)[0];
        const bool force_scale_power_of_two =
            arg_bool(op, "force_scale_power_of_two", env);
        std::any input_ptr = vector_operand(op, "input");

        if (has_arg(op, "output_code")) {
          output_code = read_constant_param(resolve(op, "output_code", env));
        }

        if (target == "quantize_mx_outlier") {
          const float threshold = arg_float(op, "threshold", env);
          const int data_size = get_size(output_tensors[0]);
          // The running CSR base, so this tile's entries continue the
          // packed stream where the previous tile ended. The IR carries it as
          // a scalar kwarg, fed by the _local_scalar_dense that read the
          // previous tile's indptr entry.
          const int indptr_offset = has_arg(op, "indptr_offset")
                                        ? arg_int(op, "indptr_offset", env)
                                        : 0;
          auto [data, indices, indptr, filtered] =
              filter_outlier<Vector, SPMM_META_DATATYPE>(
                  input_ptr, input_shape, data_size, threshold, indptr_offset);
          outputs.insert(outputs.end(), {data, indices, indptr});
          input_ptr = filtered;
        }

        std::any mx_scale = calculate_mx_qparam<Vector, Scale>(
            input_ptr, input_shape, quant_max, block_size, axis,
            force_scale_power_of_two);

        output_ptr = quantize_mx<Vector, Vector>(input_ptr, mx_scale,
                                                 input_shape, block_size, axis);

        cast_output<Vector, SUPPORTED_TYPES>(mx_scale,
                                             output_tensors[num_outputs - 2]);
        outputs.push_back(mx_scale);
      }
    } else if (target == "dequantize") {
      if constexpr (std::is_same<Vector, CFloat>::value) {
        spdlog::error(
            "No quantization operations should be emitted for CFloat\n");
        std::abort();
      } else {
        const auto input = resolve(op, "input", env);
        // A grouped dequantization carries one scale per block along an axis,
        // and optionally a zero point. It stands in for the codebook
        // conversion the gemm would otherwise apply to this operand, so it
        // produces the systolic array's weight type directly -- which is the
        // dtype the compiler declares for its result.
        if (has_arg(op, "block_size")) {
          std::any zero_point = static_cast<Scale*>(nullptr);
          if (has_arg(op, "zero_point")) {
            zero_point = operand(op, "zero_point");
          }
          output_ptr = dequantize_tensor_group<SaWeight, Scale, Vector>(
              operand(op, "input"), operand(op, "scale"), zero_point, input,
              arg_int(op, "block_size", env), arg_ints(op, "axes", env)[0]);
        } else {
          output_ptr = dequantize_tensor<Vector>(operand(op, "input"),
                                                 operand(op, "scale"), input);
        }
      }
    } else if (ReductionRegistry<Vector>::is_supported(target)) {
      output_ptr = reduce<Vector>(kwargs, op, env);
    } else {
      spdlog::error("Unsupported instruction: {}\n", target);
      std::abort();
    }

    if (!output_ptr.has_value()) {
      throw std::runtime_error("Operation " + op.name() + " (" + target +
                               ") produced no value.");
    }
    kwargs[op.name()] = output_ptr;
  }

  std::any output_ptr = kwargs[op_list.back()->name()];
  const auto output_tensor = output_tensors[num_outputs - 1];

  cast_output<Vector, SUPPORTED_TYPES>(output_ptr, output_tensor, output_code);
  outputs.push_back(output_ptr);

  if (output_code != nullptr) {
    delete[] output_code;
  }

  spdlog::debug("Operation {} finished\n",
                strip_namespace(op_list.front()->target()));

  return outputs;
}

std::vector<std::any> run_gold_model(const voyager::Operation& operation,
                                     const ScalarEnv& env,
                                     std::map<std::string, std::any> kwargs) {
  return run_operation<SA_INPUT_TYPE, SA_WEIGHT_TYPE, ACCUM_DATATYPE,
                       ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE, VECTOR_DATATYPE>(
      operation, env, kwargs);
}

void run_host_operation(const voyager::Operation& operation,
                        const ScalarEnv& env, MemoryInterface* memory) {
  spdlog::debug("Executing {} ({})\n", operation.name(),
                get_prim_ops(operation).front()->target());

  // CSR bookkeeping the control processor runs in place -- a clone of an
  // index cell, or the integer accumulator maintaining a running base. Both
  // work in the box's own dtype: an int32 index round-tripped through the
  // bfloat16 vector path would come back corrupted.
  const auto& prims = get_prim_ops(operation);
  if (prims.size() == 1 && is_host_bookkeeping(*prims.front())) {
    const voyager::PrimOp& prim = *prims.front();
    const Tensor src = resolve(prim, "input", env);
    const Tensor dst = resolve_outputs(operation, env).front();
    if (src.dtype != dst.dtype || get_size(src) != get_size(dst)) {
      throw std::runtime_error(operation.name() +
                               " must preserve dtype and size.");
    }

    std::any data = memory->read_tensor(src);

    if (strip_namespace(prim.target()) == "add") {
      const int64_t alpha =
          has_arg(prim, "alpha") ? arg_int(prim, "alpha", env) : 1;
      const int64_t addend = alpha * arg_int(prim, "other", env);
      if (DataTypes::int32** values = std::any_cast<DataTypes::int32*>(&data)) {
        for (int i = 0; i < get_size(src); i++) {
          (*values)[i].int_val = (*values)[i].int_val.to_int64() + addend;
        }
      } else if (DataTypes::int64** values =
                     std::any_cast<DataTypes::int64*>(&data)) {
        for (int i = 0; i < get_size(src); i++) {
          (*values)[i].int_val = (*values)[i].int_val.to_int64() + addend;
        }
      } else {
        throw std::runtime_error(operation.name() +
                                 " accumulates a non-index dtype.");
      }
    }

    memory->write_tensor(dst, data);
    return;
  }

  // Fetch every operand the kernels look up by name.
  //
  // A fusion intermediate has no memory -- it only ever exists inside the
  // datapath. A config constant is not in memory either; the ones that are a
  // single value (a dequantization scale) come off disk here, while the larger
  // ones (codebooks, quantization maps) are immediates the kernels load
  // themselves and never fetch as tensors.
  std::map<std::string, std::any> kwargs;
  for (const auto* prim : get_prim_ops(operation)) {
    for (const auto& [key, value] : prim->kwargs()) {
      if (!value.has_tensor_box()) continue;
      const Tensor tensor = resolve(value.tensor_box(), env, prim->name());
      if (kwargs.count(tensor.node)) continue;

      // A constant is fetchable only when it is the single value read off
      // disk; anything else has to be in memory. read_tensor covers both, and
      // builds the operand in the box's own dtype -- an int64 index must not
      // be materialized as a bfloat16, since that dtype is what the consumer
      // casts from.
      const bool fetchable =
          tensor.is_constant ? get_size(tensor) == 1 : tensor.materialized;
      if (!fetchable) continue;

      kwargs[tensor.node] = memory->read_tensor(tensor);
    }
  }

  const std::vector<std::any> results = run_gold_model(operation, env, kwargs);

  const std::vector<Tensor> destinations = resolve_outputs(operation, env);
  if (results.size() != destinations.size()) {
    throw std::runtime_error(
        "Operation " + operation.name() + " produced " +
        std::to_string(results.size()) + " results but declares " +
        std::to_string(destinations.size()) + " destinations.");
  }

  // Destination-passing: the operation writes into buffers that already exist.
  for (size_t i = 0; i < results.size(); i++) {
    memory->write_tensor(destinations[i], results[i]);
  }
}

void GoldBackend::execute(const voyager::Operation& op, const ScalarEnv& env) {
  run_host_operation(op, env, memory_);
}
