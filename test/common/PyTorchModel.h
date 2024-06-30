#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/PytorchMemoryModel.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

using INTERMEDIATE_DTYPE = ACCUM_DATATYPE::AccumulationDatatype;

/***************************************************************************
 * fused_multiply_add Functions
 *
 * These inline functions performs the multiply-accumulate operation.
 ***************************************************************************/

inline void fused_multiply_add(float a, float b, float &c) { c += a * b; }

inline void fused_multiply_add(INPUT_DATATYPE a, INPUT_DATATYPE b,
                               INTERMEDIATE_DTYPE &c) {
#ifdef HYBRID_FP8
  HYBRID_TYPE hybrid_a(a);
  HYBRID_TYPE hybrid_b(b);
  c = hybrid_a.fma(hybrid_b, c);
#else
  INPUT_DATATYPE::AccumulationDatatype v1 = a;
  INPUT_DATATYPE::AccumulationDatatype v2 = b;
  c = v1.fma(v2, c);
#endif
}

#ifndef NO_UNIVERSAL
inline void fused_multiply_add(UniversalPosit a, UniversalPosit b,
                               UniversalPositAccum &c) {
  UniversalPositAccum product;
  sw::universal::value<15> internal = sw::universal::fma<8, 1>(a, b, 0);
  sw::universal::convert<16, 1, 15>(internal, product);
  c += product;
}
#endif

/***************************************************************************
 * Element-wise Operations
 *
 * These inline functions handle element-wise operations like activation
 * function, exp, and reciprocal.
 ***************************************************************************/

inline void relu(float &x) { x = x > 0 ? x : 0; }

inline void relu(INTERMEDIATE_DTYPE &x) { x.relu(); }

#ifndef NO_UNIVERSAL
inline void relu(UniversalPositAccum &x) { x = x > 0 ? x : 0; }
#endif

inline float exponent(const float &x) { return exp(x); }

inline INTERMEDIATE_DTYPE exponent(const INTERMEDIATE_DTYPE &x) {
  ACCUM_DATATYPE tmp = static_cast<ACCUM_DATATYPE>(x);
  tmp.exponential();
  return static_cast<INTERMEDIATE_DTYPE>(tmp);
}

#ifndef NO_UNIVERSAL
inline UniversalPositAccum exponent(const UniversalPositAccum &x) {
  return sw::universal::exp(x);
}
#endif

inline float reciprocal(const float &a) { return 1.0f / a; }

inline INTERMEDIATE_DTYPE reciprocal(const INTERMEDIATE_DTYPE &a) {
  ACCUM_DATATYPE tmp = static_cast<ACCUM_DATATYPE>(a);
  tmp.reciprocal();
  return static_cast<INTERMEDIATE_DTYPE>(tmp);
}

#ifndef NO_UNIVERSAL
inline UniversalPositAccum reciprocal(const UniversalPositAccum &a) {
  return a.reciprocate();
}
#endif

/***************************************************************************
 * read_tensor Functions
 *
 * These inline functions handle reading tensor values from different types
 * of matrices. The functions manage whether the data should be read in
 * single or double precision formats. There are different overloads to
 * support various data types including float, INPUT_DATATYPE, and
 * UniversalPosit (if enabled).
 ***************************************************************************/

inline float read_tensor(const float *matrix, int index,
                         bool double_precision) {
  return double_precision ? matrix[2 * index] : matrix[index];
}

inline ACCUM_DATATYPE read_tensor(const INPUT_DATATYPE *matrix, int index,
                                  bool double_precision) {
  if (!double_precision) {
    return static_cast<ACCUM_DATATYPE>(matrix[index]);
  } else {
    return ACCUM_DATATYPE(&matrix[2 * index]);
  }
}

#ifndef NO_UNIVERSAL
inline UniversalPositAccum read_tensor(const UniversalPosit *matrix, int index,
                                       bool double_precision) {
  if (!double_precision) {
    return static_cast<UniversalPositAccum>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].encoding();
  int encoding2 = matrix[2 * index + 1].encoding();
  UniversalPositAccum p16;
  p16.setbits((encoding2 << 8) + encoding1);
  return p16;
}
#endif

/***************************************************************************
 * save_tensor Functions
 *
 * These inline functions handle saving tensor values into different types
 * of matrices. The functions manage whether the data should be stored in
 * single or double precision formats. There are different overloads to
 * support various data types including float, INPUT_DATATYPE, and
 * UniversalPosit (if enabled).
 ***************************************************************************/

inline void save_tensor(float *matrix, int index, float value,
                        bool double_precision) {
  if (!double_precision) {
    matrix[index] = value;
  } else {
    matrix[2 * index] = value;
    matrix[2 * index + 1] = 0;
  }
}

inline void save_tensor(INPUT_DATATYPE *matrix, int index,
                        INTERMEDIATE_DTYPE value, bool double_precision) {
  if (!double_precision) {
    matrix[index] = static_cast<INPUT_DATATYPE>(value);
  } else {
    ACCUM_DATATYPE p16 = value;
    p16.storeAsLowerPrecision(&matrix[2 * index]);
  }
}

#ifndef NO_UNIVERSAL
inline void save_tensor(UniversalPosit *matrix, int index,
                        UniversalPositAccum value, bool double_precision) {
  if (!double_precision) {
    matrix[index] = static_cast<UniversalPosit>(value);
  } else {
    int bits = value.encoding();
    matrix[2 * index].setbits(bits & 0xFF);
    matrix[2 * index + 1].setbits((bits >> 8) & 0xFF);
  }
}
#endif

inline std::vector<int> get_shape(const codegen::Tensor &tensor) {
  auto repeated_field = tensor.shape();
  return std::vector<int>(repeated_field.begin(), repeated_field.end());
}

inline int get_size(const std::vector<int> &shape) {
  int size = 1;
  for (int dim : shape) {
    size *= dim;
  }
  return size;
}

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T>
inline ACCUMULATE_T *gemm(const INPUT_T *inputs, const INPUT_T *weights,
                          const INPUT_T *bias, Tiling tiling,
                          const codegen::MatrixParam &param) {
  std::cerr << "gemm" << std::endl;
  std::cerr << tiling << std::endl;

  const int X = tiling.loops[0][tiling.x_loop_index[0]] *
                tiling.loops[1][tiling.x_loop_index[1]];
  const int Y = tiling.loops[0][tiling.y_loop_index[0]] *
                tiling.loops[1][tiling.y_loop_index[1]];
  const int C = tiling.replication
                    ? 3
                    : tiling.loops[1][tiling.reduction_loop_index[1]] * 16;
  const int K = tiling.loops[0][tiling.weight_loop_index[0]] *
                tiling.loops[1][tiling.weight_loop_index[1]] * 16;
  const int FX = tiling.replication ? 7 : tiling.loops[1][tiling.fx_index];
  const int FY = tiling.loops[1][tiling.fy_index];
  const int stride = tiling.stride;

  // adjust loop counters for dimension != 16
  if (IC_DIMENSION < 16) {
    tiling.loops[1][tiling.reduction_loop_index[1]] *= (16 / IC_DIMENSION);
  } else if (IC_DIMENSION > 16) {
    tiling.loops[1][tiling.reduction_loop_index[1]] /= (IC_DIMENSION / 16);
  }

  if (OC_DIMENSION < 16) {
    tiling.loops[0][tiling.weight_loop_index[0]] *= (16 / OC_DIMENSION);
  } else if (OC_DIMENSION > 16) {
    // if the inner weight loop is >=4, we should reduce the inner loop
    // (otherwise, we violate the weight buffer constraint) otherwise, we
    // reduce the outer loop
    if ((tiling.loops[1][tiling.weight_loop_index[1]] >= 4 &&
         tiling.loops[1][tiling.fx_index] > 1 &&
         tiling.loops[1][tiling.fy_index] > 1)) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= (OC_DIMENSION / 16);
    } else if (tiling.loops[0][tiling.weight_loop_index[0]] <
                   (OC_DIMENSION / 16) &&
               tiling.loops[0][tiling.weight_loop_index[0]] != 1) {
      const int reduction_factor =
          OC_DIMENSION / 16 / tiling.loops[0][tiling.weight_loop_index[0]];
      tiling.loops[0][tiling.weight_loop_index[0]] = 1;
      tiling.loops[1][tiling.weight_loop_index[1]] /= reduction_factor;
    } else if (tiling.loops[0][tiling.weight_loop_index[0]] == 1) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= (OC_DIMENSION / 16);
    } else {
      tiling.loops[0][tiling.weight_loop_index[0]] /= (OC_DIMENSION / 16);
    }
  }

  int X0 = tiling.loops[1][tiling.x_loop_index[1]];
  int Y0 = tiling.loops[1][tiling.y_loop_index[1]];
  int K0 = tiling.loops[1][tiling.weight_loop_index[1]];
  int IC_unroll = IC_DIMENSION;

  if (tiling.replication) {
    tiling.loops[1][tiling.fx_index] = 7;
    IC_unroll = 3;
    tiling.loops[1][tiling.reduction_loop_index[1]] = 1;
  }

  // assert that none of tiling.loops are 0
  for (int j = 0; j < 3; j++) {
    assert(tiling.loops[0][j] != 0);
  }
  for (int j = 0; j < 6; j++) {
    assert(tiling.loops[1][j] != 0);
  }

  ACCUMULATE_T *output_tensor = new ACCUMULATE_T[X * Y * K];

  bool input_double_precision = is_double_precision(param.input());
  bool weight_double_precision = is_double_precision(param.weight());
  bool bias_double_precision = is_double_precision(param.bias());

  for (int xy = 0; xy < X * Y; xy++) {
    for (int k = 0; k < K; k++) {
      // FIXME: hardcode bias to double precision for now
      output_tensor[xy * K + k] =
          param.has_bias() ? read_tensor(bias, k, true) : ACCUMULATE_T(0.0);
    }
  }

  int counters[2][6] = {0};
  for (counters[0][0] = 0; counters[0][0] < tiling.loops[0][0];
       counters[0][0]++) {
    for (counters[0][1] = 0; counters[0][1] < tiling.loops[0][1];
         counters[0][1]++) {
      for (counters[0][2] = 0; counters[0][2] < tiling.loops[0][2];
           counters[0][2]++) {
        int x1 = counters[0][tiling.x_loop_index[0]];
        int y1 = counters[0][tiling.y_loop_index[0]];
        int k1 = counters[0][tiling.weight_loop_index[0]];

        for (counters[1][0] = 0; counters[1][0] < tiling.loops[1][0];
             counters[1][0]++) {
          for (counters[1][1] = 0; counters[1][1] < tiling.loops[1][1];
               counters[1][1]++) {
            for (counters[1][2] = 0; counters[1][2] < tiling.loops[1][2];
                 counters[1][2]++) {
              for (counters[1][3] = 0; counters[1][3] < tiling.loops[1][3];
                   counters[1][3]++) {
                for (counters[1][4] = 0; counters[1][4] < tiling.loops[1][4];
                     counters[1][4]++) {
                  for (counters[1][5] = 0; counters[1][5] < tiling.loops[1][5];
                       counters[1][5]++) {
                    int x0 = counters[1][tiling.x_loop_index[1]];
                    int y0 = counters[1][tiling.y_loop_index[1]];
                    int c0 = counters[1][tiling.reduction_loop_index[1]];
                    int k0 = counters[1][tiling.weight_loop_index[1]];
                    int fx = counters[1][tiling.fx_index] - (FX - 1) / 2;
                    int fy = counters[1][tiling.fy_index] - (FY - 1) / 2;

                    int x = x1 * X0 + x0;
                    int y = y1 * Y0 + y0;

                    for (int oc0 = 0; oc0 < OC_DIMENSION; oc0++) {
                      int k = (k1 * K0 + k0) * OC_DIMENSION + oc0;
                      int output_addr = y * X * K + x * K + k;

                      for (int ic0 = 0; ic0 < IC_unroll; ic0++) {
                        int c = c0 * IC_unroll + ic0;
                        int input_addr = (stride * y + fy) * stride * X * C +
                                         (stride * x + fx) * C + c;
                        int weight_addr = (fy + (FY - 1) / 2) * FX * C * K +
                                          (fx + (FX - 1) / 2) * C * K + c * K +
                                          k;
                        if (stride * x + fx >= 0 &&
                            stride * x + fx < stride * X &&
                            stride * y + fy >= 0 &&
                            stride * y + fy < stride * Y) {
                          INTERMEDIATE_T input = read_tensor(
                              inputs, input_addr, input_double_precision);
                          INTERMEDIATE_T weight = read_tensor(
                              weights, weight_addr, weight_double_precision);
                          // std::cerr << "input[" << input_addr << "] = " <<
                          // input
                          //           << std::endl;
                          // std::cerr << "weight[" << weight_addr
                          //           << "] = " << weight << std::endl;
                          // std::cerr << "output[" << output_addr
                          //           << "] = " << output_tensor[output_addr]
                          //           << std::endl;
                          fused_multiply_add(input, weight,
                                             output_tensor[output_addr]);
                        }
                      }
                      if (tiling.replication) {
                        if (IC_DIMENSION == 16) {
                          if (counters[1][tiling.fx_index] == 3 ||
                              counters[1][tiling.fx_index] == 6) {
                            output_tensor[output_addr] =
                                static_cast<INTERMEDIATE_T>(
                                    output_tensor[output_addr]);
                          }
                        } else if (IC_DIMENSION == 32) {
                          if (counters[1][tiling.fx_index] == 6) {
                            output_tensor[output_addr] =
                                static_cast<INTERMEDIATE_T>(
                                    output_tensor[output_addr]);
                          }
                        }
                      } else {
                        output_tensor[output_addr] =
                            static_cast<INTERMEDIATE_T>(
                                output_tensor[output_addr]);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return output_tensor;
}

const std::set<std::string> activations = {"relu", "relu_", "gelu", "gelu_"};
const std::set<std::string> arithmetics = {"add", "add_", "sub", "sub_",
                                           "mul", "mul_", "div", "div_"};

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
    throw std::invalid_argument("Shapes are not broadcastable");
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
      result[offset_res] = tensor1[offset1] / tensor2[offset2];
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
  std::cerr << "perform_elwise_operation: " << operation << std::endl;
  auto result_shape = broadcast_shape(shape1, shape2);
  auto padded_shape1 = pad_shape(shape1, result_shape.size());
  auto padded_shape2 = pad_shape(shape2, result_shape.size());

  int result_size = get_size(result_shape);
  T *result = new T[result_size];

  perform_elwise_op_recursivly(tensor1, padded_shape1, tensor2, padded_shape2,
                               result, result_shape, operation, 0, 0, 0, 0);

  return result;
}

template <typename T>
inline T *softmax(const T *inputs, const std::vector<int> shape) {
  int num_rows = 1;
  for (int i = 0; i < shape.size() - 1; i++) {
    num_rows *= shape[i];
  }
  int num_cols = shape[shape.size() - 1];

  T *outputs = new T[num_rows * num_cols];

  for (int i = 0; i < num_rows; i++) {
    int offset = i * num_cols;
    T max = -32768;
    for (int j = 0; j < num_cols; j++) {
      max = inputs[offset + j] > max ? inputs[offset + j] : max;
    }

    for (int j = 0; j < num_cols; j++) {
      T normalized = static_cast<T>(inputs[offset + j] - max);
      outputs[offset + j] = exponent(normalized);
    }

    // perform a tree addition
    T sum = 0.0;
    for (int j = 0; j < num_cols; j += OC_DIMENSION) {
      T buffer[OC_DIMENSION];
      for (int k = 0; k < OC_DIMENSION; k++) {
        buffer[k] = outputs[offset + j + k];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int k = 0; k < depth; k += 2) {
          buffer[k / 2] = static_cast<T>(buffer[k] + buffer[k + 1]);
        }
        depth = depth / 2;
      }
      sum = static_cast<T>(sum + buffer[0]);
    }

    T divisor = reciprocal(sum);
    for (int j = 0; j < num_cols; j++) {
      outputs[offset + j] *= divisor;
    }
  }
  return outputs;
}

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T>
void run_pytorch_op(const codegen::AcceleratorParam param,
                    std::vector<INPUT_T *> args) {
  int arg_index = 0;
  ACCUMULATE_T *output_tensor;

  if (param.has_reduce_param()) {
    const auto &reduce_param = param.reduce_param();
    if (reduce_param.opcode() == "softmax") {
      auto input_shape = get_shape(reduce_param.input());
      int input_size = get_size(input_shape);
      bool double_precision = is_double_precision(reduce_param.input());

      ACCUMULATE_T *input_tensor = new ACCUMULATE_T[input_size];
      for (int j = 0; j < input_size; j++) {
        input_tensor[j] = read_tensor(args[arg_index], j, double_precision);
      }
      arg_index++;

      output_tensor = softmax(input_tensor, input_shape);

      delete[] input_tensor;
    } else {
      std::cerr << "Unsupported reduce instruction: " << reduce_param.opcode()
                << std::endl;
      exit(1);
    }
  }

  if (param.has_matrix_param()) {
    const auto &matrix_param = param.matrix_param();
    Tiling tiling;
    if (matrix_param.opcode() == "conv2d") {
      tiling = get_conv_tiling(matrix_param);
    } else if (matrix_param.opcode() == "linear") {
      tiling = get_linear_tiling(matrix_param);
    } else if (matrix_param.opcode() == "matmul") {
      tiling = get_matmul_tiling(matrix_param);
    } else {
      std::cerr << "Unsupported matrix instruction: " << matrix_param.opcode()
                << std::endl;
      exit(1);
    }
    output_tensor = gemm<INPUT_T, ACCUMULATE_T, INTERMEDIATE_T>(
        args[0], args[1], args[2], tiling, matrix_param);
    arg_index = 3;
  } else if (param.vector_params_size() > 0) {
    // fetch the input of the first vector instruction
    auto input_shape = get_shape(param.vector_params(0).input());
    int input_size = get_size(input_shape);
    bool double_precision = is_double_precision(param.vector_params(0).input());

    output_tensor = new ACCUMULATE_T[input_size];
    for (int i = 0; i < input_size; i++) {
      output_tensor[i] = read_tensor(args[arg_index], i, double_precision);
    }
    arg_index++;
  }

  // std::cerr << "inputs:" << std::endl;
  // for (int i = 0; i < 256; i++) {
  //   std::cerr << "inputs[" << i << "]: " << (float)args[0][i] << std::endl;
  // }
  // std::cerr << "====================" << std::endl;

  // std::cerr << "weights:" << std::endl;
  // for (int i = 0; i < 256; i++) {
  //   std::cerr << "weights[" << i << "]: " << (float)args[1][i] << std::endl;
  // }
  // std::cerr << "====================" << std::endl;

  // std::cerr << "output_tensor:" << std::endl;
  // for (int i = 0; i < 256; i++) {
  //   std::cerr << "output_tensor[" << i << "]: " << (float)output_tensor[i]
  //             << std::endl;
  // }
  // std::cerr << "====================" << std::endl;

  for (const auto &vector_param : param.vector_params()) {
    std::cerr << "vector_param: " << vector_param.opcode() << std::endl;
    if (activations.find(vector_param.opcode()) != activations.end()) {
      auto input_shape = get_shape(vector_param.input());
      int input_size = get_size(input_shape);
      for (int i = 0; i < input_size; i++) {
        relu(output_tensor[i]);
      }
    } else if (arithmetics.find(vector_param.opcode()) != arithmetics.end()) {
      auto input_shape = get_shape(vector_param.input());
      ACCUMULATE_T *input_tensor = output_tensor;

      auto other_shape = get_shape(vector_param.other());
      int other_size = get_size(other_shape);
      bool double_precision = is_double_precision(vector_param.other());

      ACCUMULATE_T *other_tensor = new ACCUMULATE_T[other_size];
      for (int i = 0; i < other_size; i++) {
        other_tensor[i] = read_tensor(args[arg_index], i, double_precision);
      }
      arg_index++;

      // TODO: swap input and other tensor if the other tensor is the output

      output_tensor =
          perform_elwise_operation(input_tensor, input_shape, other_tensor,
                                   other_shape, vector_param.opcode());

      // std::cerr << "output_tensor:" << std::endl;
      // for (int i = 0; i < 256; i++) {
      //   std::cerr << "output_tensor[" << i << "]: " << output_tensor[i]
      //             << std::endl;
      // }

      delete[] input_tensor;
      delete[] other_tensor;
    } else {
      std::cerr << "Unsupported vector instruction: " << vector_param.opcode()
                << std::endl;
      exit(1);
    }
  }

  int output_size = get_size(get_shape(param.output()));
  bool double_precision = is_double_precision(param.output());
  for (int i = 0; i < output_size; i++) {
    save_tensor(args.back(), i, output_tensor[i], double_precision);
  }

  delete[] output_tensor;
}

inline void run_pytorch_model(const codegen::AcceleratorParam &param,
                              std::vector<float *> args) {
  run_pytorch_op<float, float, float>(param, args);
}

inline void run_pytorch_model(const codegen::AcceleratorParam &param,
                              std::vector<INPUT_DATATYPE *> args) {
  run_pytorch_op<INPUT_DATATYPE, INTERMEDIATE_DTYPE, ACCUM_DATATYPE>(param,
                                                                     args);
}

#ifndef NO_UNIVERSAL
inline void run_pytorch_model(const codegen::AcceleratorParam &param,
                              std::vector<UniversalPosit *> args) {
  run_pytorch_op<UniversalPosit, UniversalPositAccum, UniversalPositAccum>(
      param, args);
}
#endif
