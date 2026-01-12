#pragma once

#include <map>
#include <set>

#include "src/Params.h"
#include "test/common/Utils.h"
#include "test/compiler/proto/param.pb.h"

constexpr int MAX_LOOP_VALUE = 65535;

std::vector<std::set<std::string>> vector_unit_stages = {
    {"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", "neg"},
    {"exp",          "abs",          "relu",      "relu_",      "gelu",
     "gelu_",        "silu",         "silu_",     "elu",        "elu_",
     "hardsigmoid",  "hardsigmoid_", "hardswish", "hardswish_", "log_sigmoid",
     "log_sigmoid_", "selu",         "selu_",     "celu",       "celu_",
     "sigmoid",      "sigmoid_",     "mish",      "mish_",      "softplus",
     "softplus_",    "tanh",         "tanh_",     "tanh_1",     "tanh_1_",
     "hardshrink",   "hardshrink_",  "hardtanh",  "hardtanh_",  "leaky_relu",
     "leaky_relu_",  "rrelu",        "rrelu_",    "softshrink", "softshrink_",
     "threshold",    "threshold_"},
    {"add", "add_", "mul", "mul_", "square", "div", "div_"},
    {"div", "div_", "quantize", "quantize_mx", "quantize_mx_outlier"},
};

std::map<std::string, unsigned int> get_vector_instruction_mapping() {
  std::map<std::string, unsigned int> mapping;
  mapping["add"] = VectorInstructions::vadd;
  mapping["add_"] = VectorInstructions::vadd;
  mapping["sub"] = VectorInstructions::vsub;
  mapping["sub_"] = VectorInstructions::vsub;
  mapping["mul"] = VectorInstructions::vmult;
  mapping["mul_"] = VectorInstructions::vmult;
  mapping["div"] = VectorInstructions::vdiv;
  mapping["div_"] = VectorInstructions::vdiv;
  mapping["square"] = VectorInstructions::vsquare;
  mapping["neg"] = VectorInstructions::vsub;
  mapping["exp"] = VectorInstructions::vpoly;
  mapping["abs"] = VectorInstructions::vabs;
  mapping["relu"] = VectorInstructions::vrelu;
  mapping["relu_"] = VectorInstructions::vrelu;
  mapping["gelu"] = VectorInstructions::vpoly;
  mapping["gelu_"] = VectorInstructions::vpoly;
  mapping["silu"] = VectorInstructions::vpoly;
  mapping["silu_"] = VectorInstructions::vpoly;
  mapping["elu"] = VectorInstructions::vpoly;
  mapping["elu_"] = VectorInstructions::vpoly;
  mapping["hardsigmoid"] = VectorInstructions::vpoly;
  mapping["hardsigmoid_"] = VectorInstructions::vpoly;
  mapping["hardswish"] = VectorInstructions::vpoly;
  mapping["hardswish_"] = VectorInstructions::vpoly;
  mapping["log_sigmoid"] = VectorInstructions::vpoly;
  mapping["log_sigmoid_"] = VectorInstructions::vpoly;
  mapping["selu"] = VectorInstructions::vpoly;
  mapping["selu_"] = VectorInstructions::vpoly;
  mapping["celu"] = VectorInstructions::vpoly;
  mapping["celu_"] = VectorInstructions::vpoly;
  mapping["sigmoid"] = VectorInstructions::vpoly;
  mapping["sigmoid_"] = VectorInstructions::vpoly;
  mapping["mish"] = VectorInstructions::vpoly;
  mapping["mish_"] = VectorInstructions::vpoly;
  mapping["softplus"] = VectorInstructions::vpoly;
  mapping["softplus_"] = VectorInstructions::vpoly;
  mapping["tanh"] = VectorInstructions::vpoly;
  mapping["tanh_"] = VectorInstructions::vpoly;
  mapping["tanh_1"] = VectorInstructions::vpoly;
  mapping["tanh_1_"] = VectorInstructions::vpoly;
  mapping["hardshrink"] = VectorInstructions::vpoly;
  mapping["hardshrink_"] = VectorInstructions::vpoly;
  mapping["hardtanh"] = VectorInstructions::vpoly;
  mapping["hardtanh_"] = VectorInstructions::vpoly;
  mapping["leaky_relu"] = VectorInstructions::vpoly;
  mapping["leaky_relu_"] = VectorInstructions::vpoly;
  mapping["rrelu"] = VectorInstructions::vpoly;
  mapping["rrelu_"] = VectorInstructions::vpoly;
  mapping["softshrink"] = VectorInstructions::vpoly;
  mapping["softshrink_"] = VectorInstructions::vpoly;
  mapping["threshold"] = VectorInstructions::vpoly;
  mapping["threshold_"] = VectorInstructions::vpoly;
  mapping["quantize"] = VectorInstructions::vdiv;
  mapping["quantize_mx"] = VectorInstructions::vquantize_mx;
  mapping["quantize_mx_outlier"] = VectorInstructions::vquantize_mx_outlier;
  return mapping;
}

inline std::vector<int> squeeze_shape(const std::vector<int>& input) {
  std::vector<int> result;
  for (int value : input) {
    if (value > 1) {
      result.push_back(value);
    }
  }
  return result;
}

inline void squeeze_front_ones(std::vector<int>& vec) {
  vec.erase(vec.begin(),
            std::find_if(vec.begin(), vec.end(), [](int x) { return x != 1; }));
}

std::vector<int> split_loops(const std::vector<int> loops, int max_value) {
  if (max_value <= 1) {
    throw std::invalid_argument("max_value must be greater than 1.");
  }

  std::vector<int> result;

  for (int value : loops) {
    if (value <= max_value) {
      result.push_back(value);
    } else {
      // Find two factors of value such that both are <= max_value
      bool split_found = false;
      for (int factor = std::sqrt(value); factor > 1; --factor) {
        if (value % factor == 0) {
          int other_factor = value / factor;
          if (factor <= max_value && other_factor <= max_value) {
            result.push_back(factor);
            result.push_back(other_factor);
            split_found = true;
            break;
          }
        }
      }

      if (!split_found) {
        throw std::runtime_error("Unable to split value " +
                                 std::to_string(value) +
                                 " into two factors <= max_value.");
      }
    }
  }

  return result;
}

// Function to compute the prime factors of a given number
std::vector<int> prime_factors(int n) {
  std::vector<int> factors;
  for (int i = 2; i <= n / i; ++i) {
    while (n % i == 0) {
      factors.push_back(i);
      n /= i;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

/**
 * Adjusts loop indices such that the prime factors of the target factor are
 * distributed among the indices, starting from the rightmost index.
 *
 * @param loops A vector of integers representing the loop indices.
 * @param target_factor An integer whose prime factors are to be distributed.
 * @return A vector of adjusted loop indices.
 * @throws std::runtime_error if the prime factors cannot be fully distributed.
 */
std::vector<int> adjust_loop_indices(const std::vector<int>& loops,
                                     int target_factor) {
  // Prime factorization of the target factor
  std::vector<int> remaining_factors = prime_factors(target_factor);

  // Result vector
  std::vector<int> result = loops;

  // Process from right to left
  for (int i = result.size() - 1; i >= 0; --i) {
    for (auto it = remaining_factors.begin(); it != remaining_factors.end();) {
      int factor = *it;
      if (result[i] % factor == 0) {
        result[i] /= factor;  // Divide the current index by the factor
        it = remaining_factors.erase(it);  // Remove the used factor
      } else {
        ++it;  // Move to the next factor
      }
    }
  }

  // If any factors remain unused, error out
  if (!remaining_factors.empty()) {
    throw std::runtime_error("Unused prime factors remain.");
  }

  // Multiply the rightmost index by the target factor
  result[result.size() - 1] *= target_factor;

  return result;
}

bool is_transpose(const std::vector<int>& dims) {
  int n = dims.size();
  // If there are fewer than 2 axes, there's nothing to swap.
  if (n < 2) return false;

  // Check that all axes except the last two are in their natural order.
  for (int i = 0; i < n - 2; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }

  // Check that the last two axes are swapped.
  return (dims[n - 2] == n - 1 && dims[n - 1] == n - 2);
}

std::pair<std::vector<int>, std::vector<int>> factor_out_non_broadcastable_dim(
    std::vector<int> shape1, std::vector<int> shape2) {
  if (shape1.size() != shape2.size()) {
    throw std::invalid_argument(
        "Shapes must have the same number of dimensions");
  }

  int ndim = shape1.size();
  int non_broadcast_dim = -1;
  int min_dim;
  int max_dim;

  // Find the non-broadcastable dimension where one shape is bs times larger
  // than the other
  for (int i = 0; i < ndim; ++i) {
    min_dim = std::min(shape1[i], shape2[i]);
    max_dim = std::max(shape1[i], shape2[i]);
    if (shape1[i] != shape2[i] && max_dim % min_dim == 0) {
      non_broadcast_dim = i;
      break;
    }
  }

  if (non_broadcast_dim == -1) {
    throw std::runtime_error("No non-broadcastable dimension found");
  }

  // Merge the non-broadcastable dimension into the previous dimension
  int merge_dim = non_broadcast_dim - 1;
  if (merge_dim < 0) {
    throw std::runtime_error(
        "Cannot merge into a non-existent previous dimension");
  }

  shape1[merge_dim] *= min_dim;
  shape2[merge_dim] *= min_dim;

  shape1[non_broadcast_dim] /= min_dim;
  shape2[non_broadcast_dim] /= min_dim;

  return {shape1, shape2};
}

void update_tensor_shape(codegen::Tensor& tensor,
                         const std::vector<int>& new_shape) {
  if (is_soc_sim()) {
    tensor.clear_tiled_shape();
    for (int dim : new_shape) {
      tensor.add_tiled_shape(dim);
    }
  } else {
    tensor.clear_shape();
    for (int dim : new_shape) {
      tensor.add_shape(dim);
    }
  }
}

void set_dequantize_scale(const codegen::Tensor& tensor,
                          VectorParams* vector_params) {
  if (tensor.has_dequant()) {
    const auto& dequant_op = tensor.dequant();
    const auto scale = dequant_op.kwargs().at("scale").tensor();
    assert(get_size(scale) == 1);

    float* array = read_constant_param(scale);
    VECTOR_DATATYPE immediate = array[0];
    vector_params->vector_fetch_0_dq_scale = immediate.bits_rep();
  }
}

void set_quantize_params(const codegen::Operation& param,
                         VectorParams* vector_params, VectorInstructions& inst,
                         VectorInstructionConfig* vector_instruction_config) {
  const auto op_list = get_op_list(param);
  const auto op = op_list.back();
  const auto kwargs = op.kwargs();
  const std::string opcode = op.target();

  if (opcode == "quantize") {
    const auto scale = kwargs.at("scale").tensor();
    assert(get_size(scale) == 1);

    inst.vector_op3 = VectorInstructions::vdiv;
    inst.vector_op3_src1 = VectorInstructions::from_immediate_2;

    // scalar scale factor
    float* array = read_constant_param(scale);
    VECTOR_DATATYPE immediate = array[0];
    inst.immediate2 = immediate.bits_rep();

    delete[] array;
  } else if (opcode == "quantize_mx" || opcode == "quantize_mx_outlier") {
    int block_size = kwargs.at("block_size").int_value();
    float quant_max = kwargs.at("quant_max").float_value();
    bool force_scale_power_of_two =
        kwargs.at("force_scale_power_of_two").bool_value();

    assert(block_size == IC_DIMENSION);

    inst.vector_op3 = VectorInstructions::vquantize_mx;

    if (force_scale_power_of_two) {
      inst.immediate2 = floor(log2(quant_max));
    } else {
      VECTOR_DATATYPE scale = quant_max;
      inst.immediate2 = scale.bits_rep();
    }

    const auto outputs = get_op_outputs(param);
    const int num_outputs = outputs.size();

    vector_params->quantize_output_mx = true;
    vector_params->mx_scale_offset = get_address(outputs[num_outputs - 2]);

    if (opcode == "quantize_mx_outlier") {
      vector_params->has_sparse_output = true;
      vector_params->csr_data_offset = get_address(outputs[0]);
      vector_params->csr_indices_offset = get_address(outputs[1]);
      vector_params->csr_indptr_offset = get_address(outputs[2]);

      auto& config = vector_instruction_config->outlier_filter;

      VECTOR_DATATYPE threshold = kwargs.at("threshold").float_value();
      config.outlier_threshold = threshold.bits_rep();

      const auto quantize_input = kwargs.at("input").tensor();
      const auto quantize_shape = get_shape(quantize_input);
      config.dense_input_shape[1] = quantize_shape.back() / VECTOR_UNIT_WIDTH;
      config.dense_input_shape[0] =
          get_size(quantize_input) / quantize_shape.back();
    }

    if (kwargs.contains("output_code")) {
      const auto code = kwargs.at("output_code").tensor();
      const int size = get_size(code);

      float* array = read_constant_param(code);

      for (int i = 0; i < size; i++) {
        vector_params->output_code[i] = array[i] * 2;
      }

      delete[] array;

      vector_params->use_output_codebook = true;
    }
  }
}

template <typename T, size_t N, int port_width, typename... Ts>
bool try_fetch_params_for_type(ac_int<DTYPE_INDEX_WIDTH, false> dtype,
                               int loop_bound, int& out_pf, int& out_fw) {
  if (get_type_index<T, Ts...>() != dtype) {
    return false;
  }

  using cfg = dtype_fetch_config<T, N, port_width>;
  constexpr int pf = cfg::packing_factor;

  // decide if we can do a full pack
  out_pf = (loop_bound % pf == 0 ? pf : 1);

  // Only allow powers of two packing factors
  if (out_pf <= 0 || (out_pf & (out_pf - 1)) != 0) {
    out_pf = 1;
  }

  // choose the matching width
  out_fw = (out_pf > 1 ? cfg::packed_fetch_width : cfg::fetch_width);

  return true;
}

template <size_t N, int port_width, typename... Ts>
int get_packing_factor(ac_int<DTYPE_INDEX_WIDTH, false> dtype, int loop_bound,
                       int& effective_fetch_width) {
  int pf = 1;

  // fold: try each Ts in turn. ||-fold stops updating pf once one returns true.
  bool found = (try_fetch_params_for_type<Ts, N, port_width, Ts...>(
                    dtype, loop_bound, pf, effective_fetch_width) ||
                ...);

#ifndef __SYNTHESIS__
  if (!found) {
    throw std::runtime_error("Unsupported dtype for packing‐factor: " +
                             std::to_string(dtype));
  }
#endif

  return pf;
}
