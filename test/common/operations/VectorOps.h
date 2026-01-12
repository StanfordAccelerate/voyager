#pragma once

#include "src/vector_unit/ApproximationUnit.h"
#include "test/common/operations/Common.h"
#include "test/toolchain/ApproximationConstants.h"

const std::set<std::string> unary_ops = {
    "sqrt",       "sqrt_",       "neg",          "neg_",         "relu",
    "relu_",      "gelu",        "gelu_",        "silu",         "silu_",
    "elu",        "elu_",        "hardsigmoid",  "hardsigmoid_", "hardswish",
    "hardswish_", "log_sigmoid", "log_sigmoid_", "selu",         "selu_",
    "celu",       "celu_",       "sigmoid",      "sigmoid_",     "mish",
    "mish_",      "softplus",    "softplus_",    "tanh",         "tanh_",
    "tanh_1",     "tanh_1_",     "hardshrink",   "hardshrink_",  "hardtanh",
    "hardtanh_",  "leaky_relu",  "leaky_relu_",  "rrelu",        "rrelu_",
    "softshrink", "softshrink_", "threshold",    "threshold_"};

const std::set<std::string> arithmetics = {"add", "add_", "sub", "sub_",
                                           "mul", "mul_", "div", "div_"};

/** \brief Gold model for `vepoly`. */
template <typename T>
inline T poly_approx(T x, const T maxes[NUM_MAXES],
                     const T ranges[NUM_RANGES][NUM_COEFFS],
                     const bool clamp_min, const bool clamp_max) {
  // Clamp x to the range of the maxes.
  x = (clamp_min && x < maxes[0])               ? maxes[0]
      : (clamp_max && x > maxes[NUM_MAXES - 1]) ? maxes[NUM_MAXES - 1]
                                                : x;

  // Find the bin that x falls into.
  T c0, c1, c2, c3;
  for (size_t i = 0; i < NUM_RANGES; i++) {
    if (i >= NUM_MAXES || x <= maxes[i]) {
      c0 = ranges[i][0];
      c1 = ranges[i][1];
      c2 = ranges[i][2];
      break;
    }
  }

  // Evaluate the polynomial. We have to use the same calculation as the
  // accelerator to avoid floating point errors.
  return c0 + x * (c1 + x * c2);
}

template <typename T>
inline T* perform_unary_operation(T* input, const std::vector<int> shape,
                                  const std::string opcode,
                                  const std::map<std::string, T> kwargs) {
  int result_size = get_size(shape);
  T* result = new T[result_size];

  for (int i = 0; i < result_size; ++i) {
    if (opcode == "relu" || opcode == "relu_") {
      result[i] = input[i].relu();
    } else if (opcode == "gelu" || opcode == "gelu_") {
      result[i] = poly_approx(input[i], GELU_MAXES, GELU_RANGES, GELU_CLAMP_MIN,
                              GELU_CLAMP_MAX);
    } else if (opcode == "silu" || opcode == "silu_") {
      result[i] = poly_approx(input[i], SILU_MAXES, SILU_RANGES, SILU_CLAMP_MIN,
                              SILU_CLAMP_MAX);
    } else if (opcode == "elu" || opcode == "elu_") {
      result[i] = poly_approx(input[i], ELU_MAXES, ELU_RANGES, ELU_CLAMP_MIN,
                              ELU_CLAMP_MAX);
    } else if (opcode == "hardswish" || opcode == "hardswish_") {
      result[i] = poly_approx(input[i], HARDSWISH_MAXES, HARDSWISH_RANGES,
                              HARDSWISH_CLAMP_MIN, HARDSWISH_CLAMP_MAX);
    } else if (opcode == "hardsigmoid" || opcode == "hardsigmoid_") {
      result[i] = poly_approx(input[i], HARDSIGMOID_MAXES, HARDSIGMOID_RANGES,
                              HARDSIGMOID_CLAMP_MIN, HARDSIGMOID_CLAMP_MAX);
    } else if (opcode == "log_sigmoid" || opcode == "log_sigmoid_") {
      result[i] = poly_approx(input[i], LOGSIGMOID_MAXES, LOGSIGMOID_RANGES,
                              LOGSIGMOID_CLAMP_MIN, LOGSIGMOID_CLAMP_MAX);
    } else if (opcode == "selu" || opcode == "selu_") {
      result[i] = poly_approx(input[i], SELU_MAXES, SELU_RANGES, SELU_CLAMP_MIN,
                              SELU_CLAMP_MAX);
    } else if (opcode == "celu" || opcode == "celu_") {
      result[i] = poly_approx(input[i], CELU_MAXES, CELU_RANGES, CELU_CLAMP_MIN,
                              CELU_CLAMP_MAX);
    } else if (opcode == "sigmoid" || opcode == "sigmoid_") {
      result[i] = poly_approx(input[i], SIGMOID_MAXES, SIGMOID_RANGES,
                              SIGMOID_CLAMP_MIN, SIGMOID_CLAMP_MAX);
    } else if (opcode == "mish" || opcode == "mish_") {
      result[i] = poly_approx(input[i], MISH_MAXES, MISH_RANGES, MISH_CLAMP_MIN,
                              MISH_CLAMP_MAX);
    } else if (opcode == "softplus" || opcode == "softplus_") {
      result[i] = poly_approx(input[i], SOFTPLUS_MAXES, SOFTPLUS_RANGES,
                              SOFTPLUS_CLAMP_MIN, SOFTPLUS_CLAMP_MAX);
    } else if (opcode == "tanh" || opcode == "tanh_") {
      result[i] = poly_approx(input[i], TANH_MAXES, TANH_RANGES, TANH_CLAMP_MIN,
                              TANH_CLAMP_MAX);
    } else if (opcode == "tanh_1" || opcode == "tanh_1_") {
      result[i] = poly_approx(input[i], TANHSHRINK_MAXES, TANHSHRINK_RANGES,
                              TANHSHRINK_CLAMP_MIN, TANHSHRINK_CLAMP_MAX);

      // Start of activation functions with kwargs
    } else if (opcode == "hardshrink" || opcode == "hardshrink_") {
      const T lambda = kwargs.at("lambd");
      const T HARDSHRINK_MAXES[NUM_MAXES] = {-lambda, 0.0, 0.0,
                                             0.0,     0.0, lambda};
      result[i] = poly_approx(input[i], HARDSHRINK_MAXES, HARDSHRINK_RANGES,
                              HARDSHRINK_CLAMP_MIN, HARDSHRINK_CLAMP_MAX);
    } else if (opcode == "hardtanh" || opcode == "hardtanh_") {
      const T min_val = kwargs.at("min_val");
      const T max_val = kwargs.at("max_val");
      const T HARDTANH_MAXES[NUM_MAXES] = {min_val, max_val, max_val,
                                           max_val, max_val, max_val};
      const T HARDTANH_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {min_val, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
          {0.0, 1.0, 0.0},     {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
          {max_val, 0.0, 0.0}};
      result[i] = poly_approx(input[i], HARDTANH_MAXES, HARDTANH_RANGES,
                              HARDSHRINK_CLAMP_MIN, HARDSHRINK_CLAMP_MAX);
    } else if (opcode == "leaky_relu" || opcode == "leaky_relu_") {
      const T negative_slope = kwargs.at("negative_slope");
      const T LEAKYRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {0.0, negative_slope, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 1.0, 0.0}};
      result[i] = poly_approx(input[i], LEAKYRELU_MAXES, LEAKYRELU_RANGES,
                              LEAKYRELU_CLAMP_MIN, LEAKYRELU_CLAMP_MAX);
    } else if (opcode == "rrelu" || opcode == "rrelu_") {
      const T lower = kwargs.at("lower");
      const T upper = kwargs.at("upper");
      const T alpha = (lower + upper) / T(2);
      const T RRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {0.0, alpha, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},   {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
      result[i] = poly_approx(input[i], RRELU_MAXES, RRELU_RANGES,
                              RRELU_CLAMP_MIN, RRELU_CLAMP_MAX);
    } else if (opcode == "softshrink" || opcode == "softshrink_") {
      const T lambda = kwargs.at("lambd");
      const T SOFTSHRINK_MAXES[NUM_MAXES] = {-lambda, 0.0, 0.0,
                                             0.0,     0.0, lambda};
      const T SOFTSHRINK_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {lambda, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},    {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {-lambda, 1.0, 0.0}};
      result[i] = poly_approx(input[i], SOFTSHRINK_MAXES, SOFTSHRINK_RANGES,
                              SOFTSHRINK_CLAMP_MIN, SOFTSHRINK_CLAMP_MAX);
    } else if (opcode == "threshold" || opcode == "threshold_") {
      const T threshold = kwargs.at("threshold");
      const T value = kwargs.at("value");
      const T THRESHOLD_MAXES[NUM_MAXES] = {0.0, 0.0, 0.0, 0.0, 0.0, threshold};
      const T THRESHOLD_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
          {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
          {0.0, 1.0, 0.0}};
      result[i] = poly_approx(input[i], THRESHOLD_MAXES, THRESHOLD_RANGES,
                              THRESHOLD_CLAMP_MIN, THRESHOLD_CLAMP_MAX);
    } else if (opcode == "sqrt" || opcode == "sqrt_") {
      result[i] = input[i].sqrt();
    } else if (opcode == "neg" || opcode == "neg_") {
      result[i] = -input[i];
    } else {
      spdlog::error("Unsupported vector operation: {}\n", opcode);
      std::abort();
    }
  }

  delete[] input;

  return result;
}

inline bool are_broadcastable(const std::vector<int>& shape1,
                              const std::vector<int>& shape2) {
  size_t len1 = shape1.size();
  size_t len2 = shape2.size();
  size_t min_len = std::min(len1, len2);

  for (size_t i = 0; i < min_len; ++i) {
    int dim1 = shape1[len1 - 1 - i];
    int dim2 = shape2[len2 - 1 - i];
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return false;
    }
  }
  return true;
}

inline std::vector<int> broadcast_shape(const std::vector<int>& shape1,
                                        const std::vector<int>& shape2) {
  if (!are_broadcastable(shape1, shape2)) {
    std::ostringstream error_message;
    error_message << "Shapes are not broadcastable: [";
    for (size_t i = 0; i < shape1.size(); ++i) {
      error_message << shape1[i] << (i + 1 < shape1.size() ? ", " : "");
    }
    error_message << "] vs. [";
    for (size_t i = 0; i < shape2.size(); ++i) {
      error_message << shape2[i] << (i + 1 < shape2.size() ? ", " : "");
    }
    error_message << "]";
    throw std::invalid_argument(error_message.str());
  }

  int n1 = shape1.size();
  int n2 = shape2.size();
  int max_size = std::max(n1, n2);
  std::vector<int> result_shape(max_size);

  for (int i = 1; i <= max_size; i++) {
    int dim1 = n1 - i >= 0 ? shape1[n1 - i] : 1;
    int dim2 = n2 - i >= 0 ? shape2[n2 - i] : 1;
    result_shape[max_size - i] = std::max(dim1, dim2);
  }

  return result_shape;
}

inline std::vector<int> pad_shape(const std::vector<int>& shape, size_t size) {
  std::vector<int> padded_shape(size, 1);
  size_t len = shape.size();
  for (size_t i = 0; i < len; ++i) {
    padded_shape[size - 1 - i] = shape[len - 1 - i];
  }
  return padded_shape;
}

template <typename T>
inline T* perform_vector_operation(const T* input1,
                                   const std::vector<int>& shape1,
                                   const T* input2,
                                   const std::vector<int>& shape2,
                                   std::string opcode) {
  auto result_shape = broadcast_shape(shape1, shape2);

  auto shape_a = pad_shape(shape1, result_shape.size());
  auto shape_b = pad_shape(shape2, result_shape.size());

  int num_dims = result_shape.size();

  int total_elements = get_size(result_shape);
  T* result = new T[total_elements];

  for (int i = 0; i < total_elements; ++i) {
    std::vector<int> indices = get_indices(i, result_shape);

    // Map indices to indices_a with broadcasting
    std::vector<int> indices_a(num_dims);
    for (size_t d = 0; d < num_dims; ++d) {
      if (shape_a[d] == result_shape[d]) {
        indices_a[d] = indices[d];  // Match
      } else if (result_shape[d] % shape_a[d] == 0) {
        indices_a[d] = indices[d] / (result_shape[d] / shape_a[d]);
      } else {
        throw std::runtime_error("Invalid shape for broadcasting!");
      }
    }

    // Map indices_a to indices_b with broadcasting
    std::vector<int> indices_b(num_dims);
    for (int d = 0; d < num_dims; ++d) {
      if (shape_b[d] == result_shape[d]) {
        indices_b[d] = indices[d];  // Match
      } else if (result_shape[d] % shape_b[d] == 0) {
        indices_b[d] = indices[d] / (result_shape[d] / shape_b[d]);
      } else {
        throw std::runtime_error("Invalid shape for broadcasting!");
      }
    }

    int flat_idx_a = get_flat_index(indices_a, shape_a);
    int flat_idx_b = get_flat_index(indices_b, shape_b);

    if (opcode == "add" || opcode == "add_") {
      result[i] = input1[flat_idx_a] + input2[flat_idx_b];
    } else if (opcode == "sub" || opcode == "sub_") {
      result[i] = input1[flat_idx_a] - input2[flat_idx_b];
    } else if (opcode == "mul" || opcode == "mul_") {
      result[i] = input1[flat_idx_a] * input2[flat_idx_b];
    } else if (opcode == "div" || opcode == "div_") {
      T immediate = 1.0 / input2[flat_idx_b];
      if (input2[flat_idx_b] == T::zero()) {
        result[i] = T::zero();
      } else {
        result[i] = input1[flat_idx_a] * immediate;
      }
    } else {
      throw std::invalid_argument("Invalid opcode: " + opcode);
    }
  }

  delete[] input1;
  delete[] input2;

  return result;
}
