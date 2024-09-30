#pragma once

#include "test/common/operations/Common.h"

#ifdef CHECK_PE
#include "test/checker/PEChecker.h"
#endif

inline bool is_double_precision(const codegen::Tensor &tensor) {
  // FIXME: replace with proper check
  // return tensor.dtype().find("8") == std::string::npos;
  return false;
}

// inline void fused_multiply_add(float a, float b, float &c) { c += a * b; }

template <typename T, typename T2>
inline void fused_multiply_add(T a, T b, T2 &c) {
#ifdef HYBRID_FP8
  HYBRID_TYPE hybrid_a(a);
  HYBRID_TYPE hybrid_b(b);
  c = hybrid_a.fma(hybrid_b, c);
#else
  typename T::AccumulationDatatype v1 = a;
  typename T::AccumulationDatatype v2 = b;
  c = v1.fma(v2, c);
#endif
}

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T>
inline ACCUMULATE_T *gemm(std::any input_tensor, std::any weight_tensor,
                          std::any bias_tensor,
                          const codegen::AcceleratorParam &param) {
  const auto matrix_param = param.matrix_param();

  INPUT_T *inputs = std::any_cast<INPUT_T *>(input_tensor);
  INPUT_T *weights = std::any_cast<INPUT_T *>(weight_tensor);
  // bias is assumed to be in ACCUMULATE_T
  ACCUMULATE_T *bias = nullptr;
  if (matrix_param.has_bias()) {
    bias = std::any_cast<ACCUMULATE_T *>(bias_tensor);
  }

  Tiling tiling;
  if (matrix_param.opcode() == "conv2d") {
    tiling = get_conv2d_tiling(param);
  } else {
    tiling = get_linear_tiling(param);
  }

  int X = tiling.loops[0][tiling.x_loop_index[0]] *
          tiling.loops[1][tiling.x_loop_index[1]];
  int Y = tiling.loops[0][tiling.y_loop_index[0]] *
          tiling.loops[1][tiling.y_loop_index[1]];
  int C = tiling.loops[1][tiling.reduction_loop_index[1]] * 16;
  int K = tiling.loops[0][tiling.weight_loop_index[0]] *
          tiling.loops[1][tiling.weight_loop_index[1]] * 16;
  int FX = tiling.loops[1][tiling.fx_index];
  int FY = tiling.loops[1][tiling.fy_index];
  int STRIDE = tiling.stride;

  if (tiling.replication) {
    FX = 7;
    C = 3;
  }

  adjust_tiling_for_dimension(tiling);

  int X0 = tiling.loops[1][tiling.x_loop_index[1]];
  int Y0 = tiling.loops[1][tiling.y_loop_index[1]];
  int K0 = tiling.loops[1][tiling.weight_loop_index[1]];
  int IC_unroll = IC_DIMENSION;
  int FX_UNROLL = 1;

  if (tiling.replication) {
    tiling.loops[1][tiling.fx_index] = 7;
    IC_unroll = 3;
    tiling.loops[1][tiling.reduction_loop_index[1]] = 1;

    if (IC_DIMENSION == 4) {
      FX_UNROLL = 1;
    } else if (IC_DIMENSION == 8) {
      FX_UNROLL = 2;
    } else if (IC_DIMENSION == 16) {
      FX_UNROLL = 4;
    } else if (IC_DIMENSION == 32) {
      FX_UNROLL = 7;
    }
  }

  // assert that none of tiling.loops are 0
  for (int j = 0; j < 3; j++) {
    assert(tiling.loops[0][j] != 0);
  }
  for (int j = 0; j < 6; j++) {
    assert(tiling.loops[1][j] != 0);
  }

  bool input_double_precision = is_double_precision(matrix_param.input());
  bool weight_double_precision = is_double_precision(matrix_param.weight());
  bool bias_double_precision = is_double_precision(matrix_param.bias());

  ACCUMULATE_T *outputs = new ACCUMULATE_T[X * Y * K];
  for (int i = 0; i < X * Y; i++) {
    for (int k = 0; k < K; k++) {
      if (matrix_param.has_bias()) {
        outputs[i * K + k] = bias[k];
      } else {
        outputs[i * K + k] = ACCUMULATE_T(0.0);
      }
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
                        int input_addr = (STRIDE * y + fy) * STRIDE * X * C +
                                         (STRIDE * x + fx) * C + c;
                        int weight_addr = (fy + (FY - 1) / 2) * FX * C * K +
                                          (fx + (FX - 1) / 2) * C * K + c * K +
                                          k;
                        if (STRIDE * x + fx >= 0 &&
                            STRIDE * x + fx < STRIDE * X &&
                            STRIDE * y + fy >= 0 &&
                            STRIDE * y + fy < STRIDE * Y) {
                          INTERMEDIATE_T input = inputs[input_addr];
                          INTERMEDIATE_T weight = weights[weight_addr];
#ifdef CHECK_PE
                          int pe_num = ic0 * OC_DIMENSION + oc0;
                          if (tiling.replication) {
                            pe_num =
                                (fx / FX_UNROLL) * 3 + ic0 + (fx % FX_UNROLL);
                          }
                          pe_checker.addReference(pe_num, input, weight,
                                                  outputs[output_addr]);
#endif
                          fused_multiply_add(input, weight,
                                             outputs[output_addr]);
                        } else {
#ifdef CHECK_PE
                          int pe_num = ic0 * OC_DIMENSION + oc0;
                          if (tiling.replication) {
                            pe_num =
                                (fx / FX_UNROLL) * 3 + ic0 + (fx % FX_UNROLL);
                          }
                          INTERMEDIATE_T input;
                          input.setZero();
                          INTERMEDIATE_T weight = weights[weight_addr];
                          pe_checker.addReference(pe_num, input, weight,
                                                  outputs[output_addr]);
#endif
                        }
                      }
                      if (tiling.replication) {
                        if (IC_DIMENSION == 16) {
                          if (counters[1][tiling.fx_index] == 3 ||
                              counters[1][tiling.fx_index] == 6) {
                            outputs[output_addr] = static_cast<INTERMEDIATE_T>(
                                outputs[output_addr]);
                          }
                        } else if (IC_DIMENSION == 32) {
                          if (counters[1][tiling.fx_index] == 6) {
                            outputs[output_addr] = static_cast<INTERMEDIATE_T>(
                                outputs[output_addr]);
                          }
                        }
                      } else {
                        outputs[output_addr] =
                            static_cast<INTERMEDIATE_T>(outputs[output_addr]);
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

  delete[] inputs;
  delete[] weights;
  if (matrix_param.has_bias()) {
    delete[] bias;
  }

  return outputs;
}

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T,
          typename VECTOR_T>
inline VECTOR_T *matrix_vector_multiply(std::any input_tensor,
                                        std::any weight_tensor,
                                        std::any bias_tensor,
                                        const codegen::MatrixParam &param) {
  const auto weight = param.weight();
  int K = weight.shape(0);
  int C = weight.shape(1);

  VECTOR_T *inputs = std::any_cast<VECTOR_T *>(input_tensor);
  VECTOR_T *weights = std::any_cast<VECTOR_T *>(weight_tensor);
  VECTOR_T *bias = std::any_cast<VECTOR_T *>(bias_tensor);

  bool input_double_precision = is_double_precision(param.input());
  bool weight_double_precision = is_double_precision(param.weight());
  bool bias_double_precision = is_double_precision(param.bias());

  VECTOR_T *outputs = new VECTOR_T[K];
  for (int i = 0; i < K; i++) {
    outputs[i] = 0.0;
  }

  for (int k = 0; k < K; k++) {
    VECTOR_T product[C];
    for (int c = 0; c < C; c++) {
      VECTOR_T input = inputs[c];
      VECTOR_T weight = weights[k * C + c];
      product[c] = static_cast<VECTOR_T>(input * weight);
    }

    // perform a tree addition
    for (int i = 0; i < C; i += OC_DIMENSION) {
      VECTOR_T buffer[OC_DIMENSION];
      for (int j = 0; j < OC_DIMENSION; j++) {
        buffer[j] = product[i + j];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int j = 0; j < depth; j += 2) {
          buffer[j / 2] = static_cast<VECTOR_T>(buffer[j] + buffer[j + 1]);
        }
        depth = depth / 2;
      }
      outputs[k] = static_cast<VECTOR_T>(outputs[k] + buffer[0]);
    }

    if (param.has_bias()) {
      outputs[k] += bias[k];
    }
  }

  delete[] inputs;
  delete[] weights;
  delete[] bias;

  return outputs;
}
