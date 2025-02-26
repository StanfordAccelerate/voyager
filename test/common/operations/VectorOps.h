#pragma once

#include <ac_math/ac_gelu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>

#include "test/common/operations/Common.h"

using namespace ac_math;

const std::set<std::string> unary_ops = {"relu", "relu_", "gelu", "gelu_",
                                         "silu", "silu_", "sqrt", "sqrt_",
                                         "neg",  "neg_"};
const std::set<std::string> arithmetics = {"add", "add_", "sub", "sub_",
                                           "mul", "mul_", "div", "div_"};

template <typename T>
inline T gelu(T i) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> output_type;
  input_type x = i.template to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
  return ac_gelu_pwl<output_type>(x);
}

template <typename T>
inline T silu(T i) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<30, 3, false, AC_RND, AC_SAT> output_type;
  input_type x = i.template to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
  output_type y = ac_sigmoid_pwl<output_type>(x);
  return x * y;
}

template <typename T>
inline T *perform_unary_operation(T *input, const std::vector<int> shape,
                                  const std::string opcode) {
  int result_size = get_size(shape);
  T *result = new T[result_size];

  for (int i = 0; i < result_size; ++i) {
    if (opcode == "relu" || opcode == "relu_") {
      T zero = T(0);
      result[i] = input[i] < zero ? zero : input[i];
    } else if (opcode == "gelu" || opcode == "gelu_") {
      result[i] = gelu(input[i]);
    } else if (opcode == "silu" || opcode == "silu_") {
      result[i] = silu(input[i]);
    } else if (opcode == "sqrt" || opcode == "sqrt_") {
      result[i] = input[i].sqrt();
    } else if (opcode == "neg" || opcode == "neg_") {
      result[i] = -input[i];
    } else {
      std::cerr << "Unsupported vector instruction: " << opcode << std::endl;
      std::abort();
    }
  }

  delete[] input;

  return result;
}

inline bool are_broadcastable(const std::vector<int> &shape1,
                              const std::vector<int> &shape2) {
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

inline std::vector<int> broadcast_shape(const std::vector<int> &shape1,
                                        const std::vector<int> &shape2) {
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

inline std::vector<int> pad_shape(const std::vector<int> &shape, size_t size) {
  std::vector<int> padded_shape(size, 1);
  size_t len = shape.size();
  for (size_t i = 0; i < len; ++i) {
    padded_shape[size - 1 - i] = shape[len - 1 - i];
  }
  return padded_shape;
}

template <typename T>
inline T *perform_vector_operation(const T *input1,
                                   const std::vector<int> &shape1,
                                   const T *input2,
                                   const std::vector<int> &shape2,
                                   std::string opcode) {
  auto result_shape = broadcast_shape(shape1, shape2);

  auto shape_a = pad_shape(shape1, result_shape.size());
  auto shape_b = pad_shape(shape2, result_shape.size());

  int num_dims = result_shape.size();

  int total_elements = get_size(result_shape);
  T *result = new T[total_elements];

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
      result[i] = input1[flat_idx_a] * immediate;
    } else {
      throw std::invalid_argument("Invalid opcode: " + opcode);
    }
  }

  delete[] input1;
  delete[] input2;

  return result;
}
