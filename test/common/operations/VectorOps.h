#pragma once

#include <ac_math/ac_gelu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>

#include "test/common/operations/Common.h"

using namespace ac_math;

const std::set<std::string> activations = {"relu",  "relu_", "gelu",
                                           "gelu_", "silu",  "silu_"};
const std::set<std::string> arithmetics = {"add",  "add_", "sub",
                                           "sub_", "mul",  "mul_"};

template <typename T>
inline void relu(T &x) {
  x.relu();
}

template <typename T>
inline void gelu(T &i) {
  typedef ac_fixed<9, 4, false, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<64, 32, false, AC_RND, AC_SAT> output_type;
  input_type x = static_cast<float>(i);
  output_type y = ac_gelu_pwl<output_type>(x);
  i = y.to_double();
}

template <typename T>
inline void silu(T &i) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<30, 3, false, AC_RND, AC_SAT> output_type;
  input_type x = static_cast<float>(i);
  output_type y = ac_sigmoid_pwl<output_type>(x);
  i = (x * y).to_double();
}

template <typename T>
inline T *sqrt(T *tensor, const std::vector<int> &shape) {
  int result_size = get_size(shape);
  T *result = new T[result_size];

  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        result[i * shape[1] * shape[2] + j * shape[2] + k] =
            tensor[i * shape[1] * shape[2] + j * shape[2] + k].sqrt();
      }
    }
  }

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

// Recursive function to add tensors with broadcasting
template <typename T>
inline void perform_elwise_op_recursivly(
    const T *tensor1, const std::vector<int> &shape1, const T *tensor2,
    const std::vector<int> &shape2, T *result,
    const std::vector<int> &result_shape, std::string operation, int dim,
    int offset1, int offset2, int offset_res) {
  if (dim == result_shape.size()) {
    if (operation == "add" || operation == "add_") {
      result[offset_res] = tensor1[offset1] + tensor2[offset2];
    } else if (operation == "sub" || operation == "sub_") {
      result[offset_res] = tensor1[offset1] - tensor2[offset2];
    } else if (operation == "mul" || operation == "mul_") {
      result[offset_res] = tensor1[offset1] * tensor2[offset2];
    } else if (operation == "div" || operation == "div_") {
      T immediate = 1.0 / tensor2[offset2];
      result[offset_res] = tensor1[offset1] * immediate;
    } else {
      throw std::invalid_argument("Invalid operation: " + operation);
    }
    return;
  }

  int size1 = 1;
  int size2 = 1;
  int size_res = 1;
  for (int i = dim + 1; i < result_shape.size(); i++) {
    size1 *= shape1[i];
    size2 *= shape2[i];
    size_res *= result_shape[i];
  }

  int stride_res = result_shape[dim];

  for (int i = 0; i < stride_res; i++) {
    perform_elwise_op_recursivly(tensor1, shape1, tensor2, shape2, result,
                                 result_shape, operation, dim + 1,
                                 offset1 + (shape1[dim] == 1 ? 0 : i * size1),
                                 offset2 + (shape2[dim] == 1 ? 0 : i * size2),
                                 offset_res + i * size_res);
  }
}

// Function to add two tensors with broadcasting
template <typename T>
inline T *perform_elwise_operation(const T *tensor1,
                                   const std::vector<int> &shape1,
                                   const T *tensor2,
                                   const std::vector<int> &shape2,
                                   std::string operation) {
  auto result_shape = broadcast_shape(shape1, shape2);
  auto padded_shape1 = pad_shape(shape1, result_shape.size());
  auto padded_shape2 = pad_shape(shape2, result_shape.size());

  int result_size = get_size(result_shape);
  T *result = new T[result_size];

  perform_elwise_op_recursivly(tensor1, padded_shape1, tensor2, padded_shape2,
                               result, result_shape, operation, 0, 0, 0, 0);

  delete[] tensor1;
  delete[] tensor2;

  return result;
}
