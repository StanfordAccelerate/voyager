#pragma once

#include <deque>
#include <map>
#include <set>

#include "spdlog/spdlog.h"
#include "src/Params.h"
#include "test/common/GraphUtils.h"
#include "test/common/Utils.h"
#include "test/toolchain/ApproximationConstants.h"

constexpr int MAX_LOOP_VALUE = 65535;

// The vector fetch unit walks its nest with counters narrower than the params
// carry (VectorFetchUnit::LOOP_WIDTH), so its loops split shorter: an extent it
// cannot count to is fetched short, and the pipeline then waits forever for the
// rest of the tile.
constexpr int MAX_VECTOR_LOOP_VALUE = (1 << 11) - 1;

const voyager::PrimOp* get_fused_producer(const voyager::Operation& operation,
                                          const ScalarEnv& env,
                                          const Tensor& tensor) {
  for (const auto* op_ptr : get_prim_ops(operation)) {
    const voyager::PrimOp& op = *op_ptr;
    if (op.name() == tensor.node) return &op;
  }
  return nullptr;
}

// Whether this fusion computes `tensor` in the datapath rather than reading it
// from memory, looking through a fused dequantize to the codes it reads.
bool is_datapath_value(const voyager::Operation& operation,
                       const ScalarEnv& env, const Tensor& tensor) {
  const voyager::PrimOp* producer = get_fused_producer(operation, env, tensor);
  if (producer != nullptr &&
      strip_namespace(producer->target()) == "dequantize") {
    producer =
        get_fused_producer(operation, env, resolve(*producer, "input", env));
  }
  return producer != nullptr;
}

bool is_commutative(const std::string& opcode) {
  return opcode == "add" || opcode == "add_" || opcode == "mul" ||
         opcode == "mul_";
}

void set_dequantize_scale(const voyager::PrimOp* dequantize_op,
                          const ScalarEnv& env, VectorInstructions& inst) {
  if (dequantize_op == nullptr ||
      strip_namespace(dequantize_op->target()) != "dequantize") {
    return;
  }

  const Tensor& scale = resolve(*dequantize_op, "scale", env);
  if (get_size(scale) != 1) {
    throw std::runtime_error(
        "Reduction dequantization requires a scalar scale.");
  }

  float* scale_value = read_constant_param(scale);
  VECTOR_DATATYPE immediate = scale_value[0];
  delete[] scale_value;

  inst.vdequantize = true;
  inst.vector_dq_scale = immediate.bits_rep();
}

const std::set<std::string> poly_ops = {
    "gelu",        "gelu_",        "silu",        "silu_",        "elu",
    "elu_",        "tanh",         "tanh_",       "tanh_1",       "tanh_1_",
    "sigmoid",     "sigmoid_",     "hardsigmoid", "hardsigmoid_", "hardswish",
    "hardswish_",  "mish",         "mish_",       "softplus",     "softplus_",
    "log_sigmoid", "log_sigmoid_", "selu",        "selu_",        "celu",
    "celu_",       "hardshrink",   "hardshrink_", "hardtanh",     "hardtanh_",
    "leaky_relu",  "leaky_relu_",  "rrelu",       "rrelu_",       "softshrink",
    "softshrink_", "threshold",    "threshold_"};

const std::vector<std::set<std::string>> vector_unit_ops = {
    {"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", "neg",
     "quantize"},
    {"exp", "abs", "relu", "relu_"},
    {"add", "add_", "mul", "mul_", "div", "div_", "square", "quantize"},
    {"mul", "mul_", "div", "div_", "quantize", "quantize_mx",
     "quantize_mx_outlier"},
};

// --------------------------------------------------------------------------
// Stage 0: Basic Arithmetic (Add, Sub, Mult)
// --------------------------------------------------------------------------
unsigned int get_stage0_op(const std::string& opcode) {
  if (opcode == "add" || opcode == "add_") return VectorInstructions::op0_add;
  if (opcode == "sub" || opcode == "sub_" || opcode == "neg")
    return VectorInstructions::op0_sub;
  if (opcode == "mul" || opcode == "mul_") return VectorInstructions::op0_mul;

  // Division/Quantization in Stage 0 is implemented as Multiplication by
  // reciprocal
  if (opcode == "div" || opcode == "div_" || opcode == "quantize")
    return VectorInstructions::op0_mul;

  return VectorInstructions::op0_nop;
}

// --------------------------------------------------------------------------
// Stage 1: Abs, Exp, and Activations Functions
// --------------------------------------------------------------------------
unsigned int get_stage1_op(const std::string& opcode) {
  if (opcode == "abs") return VectorInstructions::op1_abs;
  if (opcode == "exp") return VectorInstructions::op1_exp;
  if (opcode == "relu" || opcode == "relu_")
    return VectorInstructions::op1_relu;

  return VectorInstructions::op1_nop;
}

// --------------------------------------------------------------------------
// Stage 2: Advanced Arithmetic (Add, Mult, Square)
// --------------------------------------------------------------------------
unsigned int get_stage2_op(const std::string& opcode) {
  if (opcode == "add" || opcode == "add_") return VectorInstructions::op2_add;
  if (opcode == "mul" || opcode == "mul_") return VectorInstructions::op2_mul;
  if (opcode == "square") return VectorInstructions::op2_sqr;

  if (opcode == "div" || opcode == "div_" || opcode == "quantize")
    return VectorInstructions::op2_mul;

  return VectorInstructions::op2_nop;
}

// --------------------------------------------------------------------------
// Stage 3: Quantization & True Division
// --------------------------------------------------------------------------
unsigned int get_stage3_op(const std::string& opcode) {
  if (opcode == "mul" || opcode == "mul_") return VectorInstructions::op3_mul;
  if (opcode == "div" || opcode == "div_" || opcode == "quantize")
    return VectorInstructions::op3_div;
  if (opcode == "quantize_mx") return VectorInstructions::op3_quantize_mx;
  if (opcode == "quantize_mx_outlier")
    return VectorInstructions::op3_quantize_mx_outlier;

  return VectorInstructions::op3_nop;
}

// --------------------------------------------------------------------------
// Approximation Constants Loader
// --------------------------------------------------------------------------
void load_approx_params(const std::string& opcode,
                        VectorInstructionConfig* vector_instruction_config,
                        const std::map<std::string, float>& kwargs) {
  auto& config = vector_instruction_config->approx_config;

// Macro updated to use braces for all loops
#define LOAD_ACTIVATION_CONSTANTS(NAME)            \
  do {                                             \
    for (int i = 0; i < NUM_MAXES; i++) {          \
      config.maxes[i] = NAME##_MAXES[i];           \
    }                                              \
    for (int i = 0; i < NUM_RANGES; i++) {         \
      for (int j = 0; j < NUM_COEFFS; j++) {       \
        config.ranges[i][j] = NAME##_RANGES[i][j]; \
      }                                            \
    }                                              \
    config.clamp_min = NAME##_CLAMP_MIN;           \
    config.clamp_max = NAME##_CLAMP_MAX;           \
  } while (0)

  // Standard Activations
  if (opcode == "gelu" || opcode == "gelu_") {
    LOAD_ACTIVATION_CONSTANTS(GELU);
  } else if (opcode == "silu" || opcode == "silu_") {
    LOAD_ACTIVATION_CONSTANTS(SILU);
  } else if (opcode == "elu" || opcode == "elu_") {
    LOAD_ACTIVATION_CONSTANTS(ELU);
  } else if (opcode == "log_sigmoid" || opcode == "log_sigmoid_") {
    LOAD_ACTIVATION_CONSTANTS(LOGSIGMOID);
  } else if (opcode == "tanh" || opcode == "tanh_") {
    LOAD_ACTIVATION_CONSTANTS(TANH);
  } else if (opcode == "tanh_1" || opcode == "tanh_1_") {
    LOAD_ACTIVATION_CONSTANTS(TANHSHRINK);
  } else if (opcode == "softplus" || opcode == "softplus_") {
    LOAD_ACTIVATION_CONSTANTS(SOFTPLUS);
  } else if (opcode == "mish" || opcode == "mish_") {
    LOAD_ACTIVATION_CONSTANTS(MISH);
  } else if (opcode == "sigmoid" || opcode == "sigmoid_") {
    LOAD_ACTIVATION_CONSTANTS(SIGMOID);
  } else if (opcode == "selu" || opcode == "selu_") {
    LOAD_ACTIVATION_CONSTANTS(SELU);
  } else if (opcode == "celu" || opcode == "celu_") {
    LOAD_ACTIVATION_CONSTANTS(CELU);
  } else if (opcode == "hardsigmoid" || opcode == "hardsigmoid_") {
    LOAD_ACTIVATION_CONSTANTS(HARDSIGMOID);
  } else if (opcode == "hardswish" || opcode == "hardswish_") {
    LOAD_ACTIVATION_CONSTANTS(HARDSWISH);
  }

  // Parametric Activations
  else if (opcode == "hardshrink" || opcode == "hardshrink_") {
    const VECTOR_DATATYPE lambda = kwargs.at("lambd");
    const VECTOR_DATATYPE HARDSHRINK_MAXES[NUM_MAXES] = {-lambda, 0.0, 0.0,
                                                         0.0,     0.0, lambda};
    LOAD_ACTIVATION_CONSTANTS(HARDSHRINK);
  } else if (opcode == "hardtanh" || opcode == "hardtanh_") {
    const VECTOR_DATATYPE min_val = kwargs.at("min_val");
    const VECTOR_DATATYPE max_val = kwargs.at("max_val");
    const VECTOR_DATATYPE HARDTANH_MAXES[NUM_MAXES] = {
        min_val, max_val, max_val, max_val, max_val, max_val};
    const VECTOR_DATATYPE HARDTANH_RANGES[NUM_RANGES][NUM_COEFFS] = {
        {min_val, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},     {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
        {max_val, 0.0, 0.0}};
    LOAD_ACTIVATION_CONSTANTS(HARDTANH);
  } else if (opcode == "leaky_relu" || opcode == "leaky_relu_") {
    const VECTOR_DATATYPE negative_slope = kwargs.at("negative_slope");
    const VECTOR_DATATYPE LEAKYRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
        {0.0, negative_slope, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}};
    LOAD_ACTIVATION_CONSTANTS(LEAKYRELU);
  } else if (opcode == "rrelu" || opcode == "rrelu_") {
    const VECTOR_DATATYPE lower = kwargs.at("lower");
    const VECTOR_DATATYPE upper = kwargs.at("upper");
    const VECTOR_DATATYPE alpha = (lower + upper) / VECTOR_DATATYPE(2);
    const VECTOR_DATATYPE RRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
        {0.0, alpha, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},   {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    LOAD_ACTIVATION_CONSTANTS(RRELU);
  } else if (opcode == "softshrink" || opcode == "softshrink_") {
    const VECTOR_DATATYPE lambda = kwargs.at("lambd");
    const VECTOR_DATATYPE SOFTSHRINK_MAXES[NUM_MAXES] = {-lambda, 0.0, 0.0,
                                                         0.0,     0.0, lambda};
    const VECTOR_DATATYPE SOFTSHRINK_RANGES[NUM_RANGES][NUM_COEFFS] = {
        {lambda, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},    {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
        {-lambda, 1.0, 0.0}};
    LOAD_ACTIVATION_CONSTANTS(SOFTSHRINK);
  } else if (opcode == "threshold" || opcode == "threshold_") {
    const VECTOR_DATATYPE threshold = kwargs.at("threshold");
    const VECTOR_DATATYPE value = kwargs.at("value");
    const VECTOR_DATATYPE THRESHOLD_MAXES[NUM_MAXES] = {0.0, 0.0, 0.0,
                                                        0.0, 0.0, threshold};
    const VECTOR_DATATYPE THRESHOLD_RANGES[NUM_RANGES][NUM_COEFFS] = {
        {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
        {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
        {0.0, 1.0, 0.0}};
    LOAD_ACTIVATION_CONSTANTS(THRESHOLD);
  }

#undef LOAD_ACTIVATION_CONSTANTS
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

// A fused operand that names an earlier PrimOp owns no storage: its value only
// ever exists in the datapath. A pass that has to read that value out of
// memory starts from the buffer the fusion staged it in, which is what the
// relayouts in front of it read.
Tensor get_materialized_operand(const voyager::Operation& operation,
                                const ScalarEnv& env, Tensor tensor) {
  while (!tensor.materialized) {
    const voyager::PrimOp* producer =
        get_fused_producer(operation, env, tensor);

    if (producer == nullptr ||
        !is_dma_op(strip_namespace(producer->target()))) {
      throw std::invalid_argument(
          "Fused value " + tensor.node +
          " is computed in the datapath and cannot be read from memory.");
    }

    tensor = resolve(*producer, "input", env);
  }

  return tensor;
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

void update_tensor_shape(Tensor& tensor, const std::vector<int>& new_shape) {
  tensor.shape = new_shape;
}

void set_codebook_config(const voyager::PrimOp& op, const ScalarEnv& env,
                         VectorParams* vector_params,
                         VectorInstructionConfig* vector_instruction_config) {
  if (!has_arg(op, "output_code")) {
    return;
  }

  const auto code = resolve(op, "output_code", env);

  int size;
  float* array = read_constant_param(code, &size);

  auto& config = vector_instruction_config->codebook_config;

  config.enable = true;
  for (int i = 0; i < size; i++) {
    config.output_code[i] = array[i];
  }

  vector_params->is_codebook_quantization = true;

  delete[] array;
}

// True when a microscaled quantize blocks along the second to last dimension.
// Such a block is strided, so the output controller cannot fold it while the
// data streams past; the scale has to be accumulated in a pass of its own.
// Blocking along the last dimension is contiguous and needs no extra pass.
bool needs_mx_scale_pass(const voyager::PrimOp& op, const ScalarEnv& env,
                         const Tensor& input) {
  const int axis = arg_ints(op, "axes", env)[0];
  return axis == -2 || axis == get_shape(input).size() - 2;
}

// How a microscaled quantize's blocks sit in the buffer the scale pass reads.
//
// The pass re-reads its input out of memory, so it sees the buffer's layout --
// which is the one *before* any relayout the quantize sits behind. The block
// axis is named in the quantize's own view, so the two have to be related
// before the walk can be described: `outer`, `blocked` and `inner` are the
// view's dims around the block axis, and `outer_hoisted` says the relayout
// moved `outer` in front of the blocked axis in memory (a transpose of the
// two), which is what makes the block stride the outer one rather than the
// inner one.
struct MxScaleLayout {
  int outer = 1;
  int blocked = 1;
  int inner = 1;
  bool outer_hoisted = false;
};

// The quantize's block axis and the dims around it, mapped onto the buffer the
// scale pass reads. `relayout` is the op standing between them, if any.
MxScaleLayout mx_scale_layout(const voyager::PrimOp& op, const ScalarEnv& env,
                              const voyager::PrimOp* relayout) {
  const std::vector<int> view_shape = get_shape(resolve(op, "input", env));
  const int ndim = static_cast<int>(view_shape.size());

  int axis = arg_ints(op, "axes", env)[0];
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) {
    throw std::invalid_argument("quantize_mx block axis is out of range.");
  }

  MxScaleLayout layout;
  for (int d = 0; d < axis; d++) layout.outer *= view_shape[d];
  layout.blocked = view_shape[axis];
  for (int d = axis + 1; d < ndim; d++) layout.inner *= view_shape[d];

  if (relayout == nullptr) return layout;

  // Where each view dim lives in the buffer. Only a transpose of the block
  // axis with a dim in front of it is expressible below; anything else has to
  // fail here rather than reach the address generator as a degenerate walk.
  const std::string target = strip_namespace(relayout->target());
  std::vector<int> perm(ndim);
  for (int d = 0; d < ndim; d++) perm[d] = d;

  if (target == "transpose") {
    int dim0 = arg_int(*relayout, "dim0", env);
    int dim1 = arg_int(*relayout, "dim1", env);
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    std::swap(perm[dim0], perm[dim1]);
  } else if (target == "permute") {
    const std::vector<int> dims = arg_ints(*relayout, "dims", env);
    if (static_cast<int>(dims.size()) != ndim) {
      throw std::invalid_argument(
          "quantize_mx scale pass: permute rank does not match its input.");
    }
    for (int d = 0; d < ndim; d++) perm[d] = dims[d];
  } else {
    throw std::invalid_argument("quantize_mx scale pass cannot read behind a " +
                                target + ".");
  }

  // The dims the lanes stream -- everything after the block axis -- have to be
  // the innermost run of the buffer, in order. They need not sit next to the
  // block axis: a hoisted dim may come between, which is the transposed case
  // handled below.
  std::vector<int> tail;
  for (int d = axis + 1; d < ndim; d++) {
    if (view_shape[d] != 1) tail.push_back(perm[d]);
  }
  for (size_t i = 1; i < tail.size(); i++) {
    if (tail[i] != tail[i - 1] + 1) {
      throw std::invalid_argument(
          "quantize_mx scale pass needs a contiguous tail after the block "
          "axis.");
    }
  }
  if (!tail.empty() && tail.back() != ndim - 1) {
    throw std::invalid_argument(
        "quantize_mx scale pass needs the block axis's tail to be the "
        "buffer's innermost dims.");
  }

  // Every dim outside the block axis has to end up on the same side of it in
  // memory, since one address role carries all of them.
  int hoisted = 0;
  int kept = 0;
  for (int d = 0; d < axis; d++) {
    if (view_shape[d] == 1) continue;
    if (perm[d] > perm[axis]) {
      hoisted++;
    } else {
      kept++;
    }
  }
  if (hoisted > 0 && kept > 0) {
    throw std::invalid_argument(
        "quantize_mx scale pass cannot express this relayout: the dims "
        "outside the block axis straddle it in memory.");
  }
  layout.outer_hoisted = hoisted > 0;

  return layout;
}

// The scale pass of a microscaled quantize: reduce |input| / quant_max to one
// maximum per block and write it where the quantize pass will read it. The
// compiler used to emit this as a calculate_mx_qparam operation of its own; a
// quantize_mx now carries both halves, so the mapper raises this pass itself.
void push_mx_scale_pass(const voyager::PrimOp& op, const ScalarEnv& env,
                        const Tensor& input, const Tensor& scale,
                        const MxScaleLayout& layout,
                        std::deque<BaseParams*>& mapped_params) {
  const int block_size = arg_int(op, "block_size", env);
  const float quant_max = arg_float(op, "quant_max", env);
  const bool force_scale_power_of_two =
      arg_bool(op, "force_scale_power_of_two", env);

  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  int input_dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype);
  int input_dtype_width = get_type_width<VU_INPUT_TYPES>(input_dtype);
  int input_fetch_width = ACCUMULATOR_WIDTH * input_dtype_width;

  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 1;
  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = ACCUMULATOR_WIDTH;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = 1;

  // The block count, not a shape index: deriving it from the buffer's own
  // dims picks whichever dim happens to sit there, and a relayout puts a
  // different one there. Truncating to zero here is what used to reach the
  // address generator as an unbounded walk.
  if (layout.blocked % block_size != 0) {
    throw std::invalid_argument("quantize_mx block axis " +
                                std::to_string(layout.blocked) +
                                " is not a multiple of the block size " +
                                std::to_string(block_size) + ".");
  }
  const int blocks = layout.blocked / block_size;

  if (layout.inner % ACCUMULATOR_WIDTH != 0 ||
      layout.inner % OC_DIMENSION != 0) {
    throw std::invalid_argument(
        "quantize_mx scale pass needs a contiguous tail that is a multiple of "
        "the datapath width.");
  }

  vector_params->vector_fetch_0_loops[0][0] = layout.outer;
  vector_params->vector_fetch_0_loops[0][1] = blocks;
  vector_params->vector_fetch_0_loops[0][2] = layout.inner / ACCUMULATOR_WIDTH;

  // The address generator computes y * X * K + x * K + k, and the loop
  // indices choose which counter drives which of those three roles. That
  // decoupling is what lets one walk order serve both memory layouts: the
  // walk always runs (outer, block, tail chunk, ... block element) so the
  // maxima leave in the quantize's own view order, and only the roles change.
  if (!layout.outer_hoisted) {
    // Memory is [outer, blocked, inner]: the block axis strides by the tail.
    vector_params->vector_fetch_0_loops[1][0] = 1;
    vector_params->vector_fetch_0_loops[1][1] = block_size;
    vector_params->vector_fetch_0_loops[1][2] = 1;

    for (int i = 0; i < 2; i++) {
      vector_params->vector_fetch_0_y_loop_idx[i] = 0;
      vector_params->vector_fetch_0_x_loop_idx[i] = 1;
      vector_params->vector_fetch_0_k_loop_idx[i] = 2;
    }
  } else {
    // A relayout hoisted `outer` in front of the block axis, so memory is
    // [blocked, outer, inner] and the block axis strides by outer * inner --
    // the y role -- while `outer` takes the x role.
    vector_params->vector_fetch_0_loops[1][0] = 1;
    vector_params->vector_fetch_0_loops[1][1] = 1;
    vector_params->vector_fetch_0_loops[1][2] = block_size;

    vector_params->vector_fetch_0_y_loop_idx[0] = 1;
    vector_params->vector_fetch_0_y_loop_idx[1] = 2;
    vector_params->vector_fetch_0_x_loop_idx[0] = 0;
    vector_params->vector_fetch_0_x_loop_idx[1] = 0;
    vector_params->vector_fetch_0_k_loop_idx[0] = 2;
    vector_params->vector_fetch_0_k_loop_idx[1] = 1;
  }

  vector_params->vector_output_offset = get_address(scale);
  vector_params->output_mode = 1;

  // The scale always lands in the quantize's view order -- one maximum per
  // (outer, block, lane) -- which is the layout the quantize pass reads back.
  vector_params->output_loops[0][0] = layout.outer;
  vector_params->output_loops[0][1] = blocks;
  vector_params->output_loops[0][2] = layout.inner / OC_DIMENSION;
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = 1;

  for (int i = 0; i < 2; i++) {
    vector_params->output_y_loop_idx[i] = 0;
    vector_params->output_x_loop_idx[i] = 1;
    vector_params->output_k_loop_idx[i] = 2;
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(scale.dtype);

  int loop_count = get_size(input) / ACCUMULATOR_WIDTH;

  // perform max accumulation
  VectorInstructions inst0;
  inst0.op_type = VectorInstructions::accumulation;
  inst0.inst_loop_count = loop_count / block_size;
  inst0.reduce_op = VectorInstructions::rmax;
  inst0.reduce_count = block_size;
  inst0.rdest = VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst0;

  // feed accumulator
  VectorInstructions inst1;
  inst1.op_type = VectorInstructions::vector;
  inst1.inst_loop_count = loop_count;
  inst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst1.vector_op0_src1 = VectorInstructions::from_immediate_0;
  VECTOR_DATATYPE immediate = force_scale_power_of_two
                                  ? 1.0 / pow(2, floor(log2(quant_max)))
                                  : 1.0 / quant_max;
  inst1.immediate0 = immediate.bits_rep();
  inst1.vector_op0 = VectorInstructions::op0_mul;
  inst1.vector_op1 = VectorInstructions::op1_abs;
  inst1.vdest = VectorInstructions::to_accumulate;
  vector_instruction_config->inst[1] = inst1;

  vector_instruction_config->num_inst = 2;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}

void set_quantize_mx_params(const voyager::Operation& operation,
                            const ScalarEnv& env, VectorParams* vector_params,
                            VectorInstructions& inst,
                            VectorInstructionConfig* vector_instruction_config,
                            std::deque<BaseParams*>& mapped_params,
                            Tensor* mx_scale_fetch = nullptr) {
  const auto& op_list = get_prim_ops(operation);
  const voyager::PrimOp& op = *op_list.back();
  const std::string opcode = strip_namespace(op.target());

  if (opcode != "quantize_mx" && opcode != "quantize_mx_outlier") {
    return;
  }

  int block_size = arg_int(op, "block_size", env);
  float quant_max = arg_float(op, "quant_max", env);
  bool force_scale_power_of_two = arg_bool(op, "force_scale_power_of_two", env);

  assert(block_size == IC_DIMENSION);

  const auto outputs = resolve_outputs(operation, env);
  const int num_outputs = outputs.size();
  const Tensor mx_scale = outputs[num_outputs - 2];
  const Tensor quantize_input = resolve(op, "input", env);

  // op3_quantize_mx derives the scale from the lanes streaming past stage 3,
  // which is the block only when the block axis is the last one. A strided
  // block has to be accumulated in a pass of its own, and this pass then
  // divides by what that pass wrote -- the caller binds it to fetch 2.
  const bool two_pass = opcode == "quantize_mx" && mx_scale_fetch != nullptr &&
                        needs_mx_scale_pass(op, env, quantize_input);

  if (two_pass) {
    // The scale pass reads the data a second time, out of memory, so it needs
    // the buffer that value was staged in and not the fusion intermediate --
    // and the relayout standing between the two, since the block axis is
    // named in the quantize's view while the pass walks the buffer.
    const Tensor scale_pass_input =
        get_materialized_operand(operation, env, quantize_input);
    const voyager::PrimOp* relayout =
        quantize_input.materialized
            ? nullptr
            : get_fused_producer(operation, env, quantize_input);

    push_mx_scale_pass(op, env, scale_pass_input, mx_scale,
                       mx_scale_layout(op, env, relayout), mapped_params);
    *mx_scale_fetch = mx_scale;
    inst.vector_op3 = VectorInstructions::op3_div;
    inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
  } else {
    inst.vector_op3 = VectorInstructions::op3_quantize_mx;

    if (force_scale_power_of_two) {
      inst.immediate2 = floor(log2(quant_max));
    } else {
      VECTOR_DATATYPE scale = quant_max;
      inst.immediate2 = scale.bits_rep();
    }

    vector_params->quantize_output_mx = true;
    vector_params->mx_scale_offset = get_address(mx_scale);
  }

  if (opcode == "quantize_mx_outlier") {
    inst.vector_op3 = VectorInstructions::op3_quantize_mx_outlier;
    vector_params->has_sparse_output = true;
    vector_params->csr_data_offset = get_address(outputs[0]);
    vector_params->csr_indices_offset = get_address(outputs[1]);
    vector_params->csr_indptr_offset = get_address(outputs[2]);

    auto& config = vector_instruction_config->outlier_filter;

    VECTOR_DATATYPE threshold = arg_float(op, "threshold", env);
    config.outlier_threshold = threshold.bits_rep();

    // The running CSR base: entries this tile appends continue the packed
    // stream where the previous tile ended.
    config.indptr_offset =
        has_arg(op, "indptr_offset") ? arg_int(op, "indptr_offset", env) : 0;

    const auto quantize_input = resolve(op, "input", env);
    const auto quantize_shape = get_shape(quantize_input);
    config.dense_input_shape[1] = quantize_shape.back() / VECTOR_UNIT_WIDTH;
    config.dense_input_shape[0] =
        get_size(quantize_input) / quantize_shape.back();
  }

  set_codebook_config(op, env, vector_params, vector_instruction_config);
}

void set_quantize_params(const voyager::Operation& operation,
                         const ScalarEnv& env, VectorParams* vector_params,
                         VectorInstructions& inst,
                         VectorInstructionConfig* vector_instruction_config,
                         std::deque<BaseParams*>& mapped_params) {
  const auto& op_list = get_prim_ops(operation);
  const voyager::PrimOp& op = *op_list.back();
  const std::string opcode = strip_namespace(op.target());

  if (opcode == "quantize") {
    const auto scale = resolve(op, "scale", env);
    assert(get_size(scale) == 1);

    inst.vector_op3 = VectorInstructions::op3_mul;
    inst.vector_op3_src1 = VectorInstructions::from_immediate_2;

    // The reciprocal is taken on the host: op3_div would route the scale
    // through the pipeline's pwl reciprocal, which gold does not mirror
    // (set_vector_immediate lowers scalar quantize the same way).
    float* array = read_constant_param(scale);
    VECTOR_DATATYPE immediate = 1.0 / array[0];
    inst.immediate2 = immediate.bits_rep();

    delete[] array;
  } else if (opcode == "quantize_mx" || opcode == "quantize_mx_outlier") {
    set_quantize_mx_params(operation, env, vector_params, inst,
                           vector_instruction_config, mapped_params);
  }
}

void set_vector_immediate(const float scalar, const int stage,
                          const std::string& opcode, VectorInstructions& inst) {
  VECTOR_DATATYPE immediate = scalar;

  if (opcode == "div" || opcode == "div_" || opcode == "quantize") {
    immediate = 1.0 / scalar;
  }

  if (stage == 0) {
    inst.vector_op0_src1 = VectorInstructions::from_immediate_0;
    inst.immediate0 = immediate.bits_rep();
  } else if (stage == 2) {
    inst.vector_op2_src1 = VectorInstructions::from_immediate_1;
    inst.immediate1 = immediate.bits_rep();
  } else {
    inst.vector_op3_src1 = VectorInstructions::from_immediate_2;
    inst.vector_op3 = VectorInstructions::op3_mul;
    inst.immediate2 = immediate.bits_rep();
  }
}

// True when the chain consumes this dequantize's result as a side operand
// rather than as the value flowing down the pipeline. The hardware's
// vdequantize scales only the pipeline input; a side operand is scaled by
// its consumer through the stage mac instead.
bool is_side_operand_dequantize(
    const voyager::Operation& operation, const ScalarEnv& env,
    const voyager::PrimOp& dequantize,
    const std::vector<const voyager::PrimOp*>& op_list) {
  for (const voyager::PrimOp* op_ptr : op_list) {
    const voyager::PrimOp& op = *op_ptr;
    if (op.name() == dequantize.name()) continue;
    for (const auto& [key, value] : op.kwargs()) {
      if (value.has_tensor_box() &&
          value.tensor_box().box().node() == dequantize.name()) {
        if (is_commutative(strip_namespace(op.target()))) {
          return !is_datapath_value(operation, env,
                                    resolve(dequantize, "input", env));
        }
        return key != "input";
      }
    }
  }
  return false;
}

// Map an ordered chain of operations onto the vector pipeline. Compute-engine
// mappers provide only the address-generation details for a non-scalar side
// operand and any engine-specific restrictions on pipeline stages.
template <typename MapTensorOperand, typename IsStageAvailable>
void map_vector_pipeline_ops(const voyager::Operation& operation,
                             const ScalarEnv& env,
                             const std::vector<const voyager::PrimOp*>& op_list,
                             VectorParams* vector_params,
                             VectorInstructionConfig* vector_instruction_config,
                             VectorInstructions& inst,
                             std::deque<BaseParams*>& mapped_params,
                             MapTensorOperand&& map_tensor_operand,
                             IsStageAvailable&& is_stage_available,
                             int stage = 0, Tensor* mx_scale_fetch = nullptr) {
  const auto pipeline_input = inst.vector_op0_src0;

  for (const voyager::PrimOp* op_ptr : op_list) {
    const voyager::PrimOp& op = *op_ptr;
    const std::string& opcode = strip_namespace(op.target());

    // Dequantization is a modifier on the pipeline input, not a stage. A
    // side-operand dequantize is applied by its consumer below instead.
    if (opcode == "dequantize") {
      if (is_side_operand_dequantize(operation, env, op, op_list)) continue;

      const Tensor& scale = resolve(op, "scale", env);
      if (get_size(scale) != 1) {
        throw std::runtime_error(
            "Vector dequantization requires a scalar scale.");
      }

      float* array = read_constant_param(scale);
      VECTOR_DATATYPE immediate = array[0];
      delete[] array;

      inst.vdequantize = true;
      inst.vector_dq_scale = immediate.bits_rep();
      continue;
    }

    if (poly_ops.count(opcode)) {
      if (stage != 0 || !is_stage_available(op, env, 0) ||
          !is_stage_available(op, env, 2)) {
        throw std::runtime_error(
            "Polynomial approximation must be the first vector operation "
            "and requires stages 0 and 2.");
      }

      std::map<std::string, float> kwargs;
      for (const auto& [key, value] : arg_scalars(op, env)) {
        kwargs[key] = value;
      }

      load_approx_params(opcode, vector_instruction_config, kwargs);
      inst.vector_op0 = VectorInstructions::op0_poly;
      inst.vector_op2 = VectorInstructions::op2_poly;
      stage = 3;
      set_codebook_config(op, env, vector_params, vector_instruction_config);
      continue;
    }

    for (; stage < vector_unit_ops.size(); stage++) {
      if (!is_stage_available(op, env, stage)) continue;

      // Stages 0 and 2 support only per-tensor quantization.
      if (opcode == "quantize") {
        const Tensor& scale = resolve(op, "scale", env);
        if (stage != 3 && get_size(scale) > 1) continue;
      }

      if (vector_unit_ops[stage].count(opcode)) break;
    }

    if (stage == vector_unit_ops.size()) {
      throw std::runtime_error("Vector operation " + opcode +
                               " cannot be mapped to the available pipeline "
                               "stages.");
    }

    spdlog::debug("stage {} target: {}\n", stage, opcode);

    switch (stage) {
      case 0:
        inst.vector_op0 = get_stage0_op(opcode);
        break;
      case 1:
        inst.vector_op1 = get_stage1_op(opcode);
        break;
      case 2:
        inst.vector_op2 = get_stage2_op(opcode);
        break;
      case 3:
        inst.vector_op3 = get_stage3_op(opcode);
        break;
    }

    if (opcode == "neg") {
      VECTOR_DATATYPE immediate = 0;
      inst.immediate0 = immediate.bits_rep();
      inst.vector_op0_src0 = VectorInstructions::from_immediate_0;
      inst.vector_op0_src1 = pipeline_input;
    } else if (opcode == "quantize_mx" || opcode == "quantize_mx_outlier") {
      set_quantize_mx_params(operation, env, vector_params, inst,
                             vector_instruction_config, mapped_params,
                             mx_scale_fetch);
    } else if (has_arg(op, "other") || opcode == "quantize") {
      std::string other_key = opcode == "quantize" ? "scale" : "other";

      // A commutative op may name the pipeline's own value in either operand;
      // the side operand is the one the datapath does not produce.
      if (is_commutative(opcode) && has_tensor(op, other_key) &&
          is_datapath_value(operation, env, resolve(op, other_key, env)) &&
          !is_datapath_value(operation, env, resolve(op, "input", env))) {
        other_key = "input";
      }

      if (!has_tensor(op, other_key)) {
        set_vector_immediate(arg_float(op, other_key, env), stage, opcode,
                             inst);
      } else if (opcode == "quantize" &&
                 !resolve(op, "scale", env).materialized) {
        int size;
        float* array = read_constant_param(resolve(op, "scale", env), &size);
        if (size != 1) {
          delete[] array;
          throw std::runtime_error(
              "quantize scale constant \"" + resolve(op, "scale", env).node +
              "\" has " + std::to_string(size) +
              " elements; only per-tensor (scalar) scales are supported");
        }
        set_vector_immediate(array[0], stage, opcode, inst);
        delete[] array;
      } else if (get_size(resolve(op, other_key, env)) == 1) {
        float* array = read_constant_param(resolve(op, other_key, env));
        set_vector_immediate(array[0], stage, opcode, inst);
        delete[] array;
      } else {
        // A side operand a fused dequantize produced arrives as its raw
        // codes; the stage mac multiplies them by the dequantize scale on
        // the way into the add.
        const voyager::PrimOp* producer =
            get_fused_producer(operation, env, resolve(op, other_key, env));
        if (producer != nullptr &&
            strip_namespace(producer->target()) == "dequantize") {
          const Tensor& scale = resolve(*producer, "scale", env);
          if (get_size(scale) != 1) {
            throw std::runtime_error(
                "Vector dequantization requires a scalar scale.");
          }

          float* array = read_constant_param(scale);
          VECTOR_DATATYPE dq_scale = array[0];
          delete[] array;

          if (stage == 0 && (inst.vector_op0 == VectorInstructions::op0_add ||
                             inst.vector_op0 == VectorInstructions::op0_sub)) {
            inst.immediate0 = dq_scale.bits_rep();
            inst.vector_op0 = VectorInstructions::op0_mac;
          } else if (stage == 2 &&
                     inst.vector_op2 == VectorInstructions::op2_add) {
            inst.immediate1 = dq_scale.bits_rep();
            inst.vector_op2 = VectorInstructions::op2_mac;
          } else {
            throw std::runtime_error(
                "A dequantized side operand requires the stage 0 or stage 2 "
                "mac.");
          }
        }

        map_tensor_operand(op, env, other_key, stage, inst);
      }
    }

    set_codebook_config(op, env, vector_params, vector_instruction_config);
    stage++;
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
