#pragma once

#include "test/common/Tiling.h"
#include "test/common/operations/Common.h"

#ifdef CHECK_PE
#include "test/checker/PEChecker.h"
#endif

template <typename T1, typename T2>
inline void fused_multiply_add(T1 a, T1 b, T2 &c) {
  typename T1::decoded v1 = a;
  typename T1::decoded v2 = b;
  c = v1.fma(v2, c);
}

template <typename Input, typename Psum, typename Buffer, typename Scale>
inline Buffer *gemm(std::any input_ptr, std::any input_scale_ptr,
                    std::any weight_ptr, std::any weight_scale_ptr,
                    std::any bias_ptr, const Tiling &tiling,
                    const int block_size) {
  LOG("Performing GEMM");

  Input *inputs = std::any_cast<Input *>(input_ptr);
  Scale *input_scales = std::any_cast<Scale *>(input_scale_ptr);

  Input *weights = std::any_cast<Input *>(weight_ptr);
  Scale *weight_scales = std::any_cast<Scale *>(weight_scale_ptr);

  Buffer *biases = std::any_cast<Buffer *>(bias_ptr);

  int X = tiling.loops[0][tiling.x_loop_index[0]] *
          tiling.loops[1][tiling.x_loop_index[1]];
  int Y = tiling.loops[0][tiling.y_loop_index[0]] *
          tiling.loops[1][tiling.y_loop_index[1]];
  int C = tiling.loops[0][tiling.reduction_loop_index[0]] *
          tiling.loops[1][tiling.reduction_loop_index[1]] * IC_DIMENSION;
  int K = tiling.loops[0][tiling.weight_loop_index[0]] *
          tiling.loops[1][tiling.weight_loop_index[1]] * OC_DIMENSION;
  int FX = tiling.loops[1][tiling.fx_index];
  int FY = tiling.loops[1][tiling.fy_index];
  int STRIDE = tiling.stride;

  int X0 = tiling.loops[1][tiling.x_loop_index[1]];
  int Y0 = tiling.loops[1][tiling.y_loop_index[1]];
  int K0 = tiling.loops[1][tiling.weight_loop_index[1]];
  int C1 = tiling.loops[1][tiling.reduction_loop_index[1]];
  int IC_UNROLL = IC_DIMENSION;
  int FX_UNROLL = 1;

  if (tiling.replication) {
    FX = 7;
    C = 3;
    IC_UNROLL = 3;

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

  Buffer *outputs = new Buffer[X * Y * K];

#if SUPPORT_MX
  // only used for replication on MX-based designs
  Psum *accumulations = new Psum[X * Y * K];
  for (int i = 0; i < X * Y * K; i++) {
    accumulations[i] = Psum(0.0);
  }
#endif

  // Store biases in the accumulation buffer
  for (int i = 0; i < X * Y; i++) {
    for (int k = 0; k < K; k++) {
      if (biases != nullptr) {
        outputs[i * K + k] = biases[k];
      } else {
        outputs[i * K + k] = Buffer(0.0);
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
        for (counters[0][3] = 0; counters[0][3] < tiling.loops[0][3];
             counters[0][3]++) {
          int x1 = counters[0][tiling.x_loop_index[0]];
          int y1 = counters[0][tiling.y_loop_index[0]];
          int k1 = counters[0][tiling.weight_loop_index[0]];
          int c2 = counters[0][tiling.reduction_loop_index[0]];
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
                    for (counters[1][5] = 0;
                         counters[1][5] < tiling.loops[1][5];
                         counters[1][5]++) {
                      int x0 = counters[1][tiling.x_loop_index[1]];
                      int y0 = counters[1][tiling.y_loop_index[1]];
                      int c1 = counters[1][tiling.reduction_loop_index[1]];
                      int k0 = counters[1][tiling.weight_loop_index[1]];
                      int fx = counters[1][tiling.fx_index] - (FX - 1) / 2;
                      int fy = counters[1][tiling.fy_index] - (FY - 1) / 2;

                      int x = x1 * X0 + x0;
                      int y = y1 * Y0 + y0;

                      for (int oc0 = 0; oc0 < OC_DIMENSION; oc0++) {
                        int k = (k1 * K0 + k0) * OC_DIMENSION + oc0;
                        int output_addr = y * X * K + x * K + k;

#if SUPPORT_MX
                        Psum psum = Psum(0.0);
#else
                        Psum psum = outputs[output_addr];
#endif

                        for (int ic0 = 0; ic0 < IC_UNROLL; ic0++) {
                          int c = c2 * C1 * IC_UNROLL + c1 * IC_UNROLL + ic0;
                          int input_addr = (STRIDE * y + fy) * STRIDE * X * C +
                                           (STRIDE * x + fx) * C + c;
                          int weight_addr = (fy + (FY - 1) / 2) * FX * C * K +
                                            (fx + (FX - 1) / 2) * C * K +
                                            c * K + k;
                          if (STRIDE * x + fx >= 0 &&
                              STRIDE * x + fx < STRIDE * X &&
                              STRIDE * y + fy >= 0 &&
                              STRIDE * y + fy < STRIDE * Y) {
                            Input input = inputs[input_addr];
                            Input weight = weights[weight_addr];
#ifdef CHECK_PE
                            int pe_num = ic0 * OC_DIMENSION + oc0;
                            if (tiling.replication) {
                              pe_num =
                                  ic0 * OC_DIMENSION +
                                  (counters[1][tiling.fx_index] % FX_UNROLL) *
                                      IC_UNROLL * OC_DIMENSION +
                                  oc0;
                            }
                            pe_checker.add_reference(pe_num, input, weight,
                                                     psum);
#endif
                            fused_multiply_add(input, weight, psum);
                          } else {
#ifdef CHECK_PE
                            int pe_num = ic0 * OC_DIMENSION + oc0;
                            if (tiling.replication) {
                              pe_num =
                                  ic0 * OC_DIMENSION +
                                  (counters[1][tiling.fx_index] % FX_UNROLL) *
                                      IC_UNROLL * OC_DIMENSION +
                                  oc0;
                            }
                            Input input;
                            input.set_zero();
                            Input weight = weights[weight_addr];
                            pe_checker.add_reference(pe_num, input, weight,
                                                     psum);
#endif
                          }
                        }

#if SUPPORT_MX
                        if (input_scales && weight_scales) {
                          // only perform scaling if within bounds
                          if (STRIDE * x + fx >= 0 &&
                              STRIDE * x + fx < STRIDE * X &&
                              STRIDE * y + fy >= 0 &&
                              STRIDE * y + fy < STRIDE * Y) {
                            int c = c2 * C1 + c1;
                            int num_blocks = C / block_size;
                            int input_scale_addr =
                                (y * STRIDE + fy) * X * STRIDE * num_blocks +
                                (x * STRIDE + fx) * num_blocks + c;
                            assert(input_scale_addr >= 0);
                            int weight_scale_addr =
                                (fy + (FY - 1) / 2) * FX * num_blocks * K +
                                (fx + (FX - 1) / 2) * num_blocks * K + c * K +
                                k;
                            assert(weight_scale_addr >= 0);

                            Scale input_scale = input_scales[input_scale_addr];
                            Scale weight_scale =
                                weight_scales[weight_scale_addr];
                            Buffer scale = static_cast<Buffer>(input_scale) *
                                           static_cast<Buffer>(weight_scale);
                            outputs[output_addr] +=
                                static_cast<Buffer>(psum) * scale;
                          }

                        } else {
                          if (tiling.replication) {
                            accumulations[output_addr] += psum;
                            if (IC_DIMENSION == 4) {
                              Buffer scaled_psum = static_cast<Buffer>(
                                  accumulations[output_addr]);
                              outputs[output_addr] += scaled_psum;
                              accumulations[output_addr] = Psum(0.0);
                            } else if (IC_DIMENSION == 8) {
                              if (counters[1][tiling.fx_index] == 1 ||
                                  counters[1][tiling.fx_index] == 3 ||
                                  counters[1][tiling.fx_index] == 5 ||
                                  counters[1][tiling.fx_index] == 6) {
                                Buffer scaled_psum = static_cast<Buffer>(
                                    accumulations[output_addr]);
                                outputs[output_addr] += scaled_psum;
                                accumulations[output_addr] = Psum(0.0);
                              }
                            } else if (IC_DIMENSION == 16) {
                              if (counters[1][tiling.fx_index] == 3 ||
                                  counters[1][tiling.fx_index] == 6) {
                                Buffer scaled_psum = static_cast<Buffer>(
                                    accumulations[output_addr]);
                                outputs[output_addr] += scaled_psum;
                                accumulations[output_addr] = Psum(0.0);
                              }
                            } else if (IC_DIMENSION == 32) {
                              if (counters[1][tiling.fx_index] == 6) {
                                Buffer scaled_psum = static_cast<Buffer>(
                                    accumulations[output_addr]);
                                outputs[output_addr] += scaled_psum;
                                accumulations[output_addr] = Psum(0.0);
                              }
                            }
                          } else {
                            // use a scale factor of 1 to directly convert the
                            // int value into a float
                            Buffer scaled_psum = static_cast<Buffer>(psum);
                            outputs[output_addr] += scaled_psum;
                          }
                        }
#else
                        outputs[output_addr] = static_cast<Psum>(psum);

                        if (tiling.replication) {
                          if (IC_DIMENSION == 16) {
                            if (counters[1][tiling.fx_index] == 3 ||
                                counters[1][tiling.fx_index] == 6) {
                              outputs[output_addr] = static_cast<Buffer>(psum);
                            }
                          } else if (IC_DIMENSION == 32) {
                            if (counters[1][tiling.fx_index] == 6) {
                              outputs[output_addr] = static_cast<Buffer>(psum);
                            }
                          }
                        } else {
                          outputs[output_addr] = static_cast<Buffer>(psum);
                        }
#endif
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

  LOG("GEMM done");

  delete[] inputs;
  delete[] weights;

  if (input_scales != nullptr) {
    delete[] input_scales;
  }

  if (weight_scales != nullptr) {
    delete[] weight_scales;
  }

  if (biases != nullptr) {
    delete[] biases;
  }

  return outputs;
}

template <typename Input, typename Psum, typename Buffer, typename Scale>
inline Buffer *gemm(std::any input_ptr, std::any input_scale_ptr,
                    std::any weight_ptr, std::any weight_scale_ptr,
                    std::any bias_ptr, const Operation &operation) {
  Tiling tiling = get_tiling(operation);
  std::cerr << "GEMM tiling: " << tiling << std::endl;

  if (tiling.replication) {
    tiling.loops[1][tiling.fx_index] = tiling.loops[1][tiling.fy_index];
    tiling.loops[1][tiling.reduction_loop_index[1]] = 1;
  }

  const auto op_list = get_op_list(operation.param);
  const auto matrix_op = op_list.front();

  bool is_mx = matrix_op.target().find("mx") != std::string::npos;
  int block_size = is_mx ? matrix_op.kwargs().at("block_size").int_value() : 0;

  return gemm<Input, Psum, Buffer, Scale>(input_ptr, input_scale_ptr,
                                          weight_ptr, weight_scale_ptr,
                                          bias_ptr, tiling, block_size);
}

template <typename Input, typename Psum, typename Vector>
inline Vector *matrix_vector_multiply(std::any input_ptr, std::any weight_ptr,
                                      std::any bias_ptr,
                                      const std::vector<int> &weight_shape) {
  const int K = weight_shape[0];
  const int C = weight_shape[1];

  Vector *inputs = std::any_cast<Vector *>(input_ptr);
  Vector *weights = std::any_cast<Vector *>(weight_ptr);
  Vector *biases = std::any_cast<Vector *>(bias_ptr);

  Vector *outputs = new Vector[K];
  for (int i = 0; i < K; i++) {
    outputs[i] = 0.0;
  }

  for (int k = 0; k < K; k++) {
    Vector product[C];
    for (int c = 0; c < C; c++) {
      Vector input = inputs[c];
      Vector weight = weights[k * C + c];
      product[c] = static_cast<Vector>(input * weight);
    }

    // perform a tree addition
    for (int i = 0; i < C; i += OC_DIMENSION) {
      Vector buffer[OC_DIMENSION];
      for (int j = 0; j < OC_DIMENSION; j++) {
        buffer[j] = product[i + j];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int j = 0; j < depth; j += 2) {
          buffer[j / 2] = static_cast<Vector>(buffer[j] + buffer[j + 1]);
        }
        depth = depth / 2;
      }
      outputs[k] = static_cast<Vector>(outputs[k] + buffer[0]);
    }

    if (biases != nullptr) {
      outputs[k] += biases[k];
    }
  }

  delete[] inputs;
  delete[] weights;

  if (biases != nullptr) {
    delete[] biases;
  }

  return outputs;
}
