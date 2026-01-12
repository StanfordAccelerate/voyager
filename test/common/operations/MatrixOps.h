#pragma once

#include "test/common/Tiling.h"
#include "test/common/operations/Common.h"

#ifdef CHECK_PE
#include "test/checker/PEChecker.h"
#endif

template <typename T1, typename T2>
inline void fused_multiply_add(T1 a, T1 b, T2& c) {
  typename T1::decoded v1 = a;
  typename T1::decoded v2 = b;
  c = v1.fma(v2, c);
}

template <typename Input, typename Weight, typename Psum, typename Buffer,
          typename Scale>
inline Buffer* gemm(std::any input_ptr, std::any input_scale_ptr,
                    std::any weight_ptr, std::any weight_scale_ptr,
                    std::any bias_ptr, Tiling tiling, const int block_size) {
  spdlog::debug("Performing GEMM\n");

  Input* inputs = std::any_cast<Input*>(input_ptr);
  Scale* input_scales = std::any_cast<Scale*>(input_scale_ptr);

  Weight* weights = std::any_cast<Weight*>(weight_ptr);
  Scale* weight_scales = std::any_cast<Scale*>(weight_scale_ptr);

  Buffer* biases = std::any_cast<Buffer*>(bias_ptr);

  int X = tiling.loops[0][tiling.x_loop_idx[0]] *
          tiling.loops[1][tiling.x_loop_idx[1]];
  int Y = tiling.loops[0][tiling.y_loop_idx[0]] *
          tiling.loops[1][tiling.y_loop_idx[1]];
  int C = tiling.loops[0][tiling.reduction_loop_idx[0]] *
          tiling.loops[1][tiling.reduction_loop_idx[1]] * IC_DIMENSION;
  int K = tiling.loops[0][tiling.weight_loop_idx[0]] *
          tiling.loops[1][tiling.weight_loop_idx[1]] * OC_DIMENSION;
  int FX = tiling.loops[1][tiling.fx_loop_idx];
  int FY = tiling.loops[0][tiling.fy_loop_idx[0]] *
           tiling.loops[1][tiling.fy_loop_idx[1]];
  int STRIDE = tiling.stride;
  int IY = tiling.input_y;
  int IX = tiling.input_x;

  int X0 = tiling.loops[1][tiling.x_loop_idx[1]];
  int Y0 = tiling.loops[1][tiling.y_loop_idx[1]];
  int K0 = tiling.loops[1][tiling.weight_loop_idx[1]];
  int FY0 = tiling.loops[1][tiling.fy_loop_idx[1]];
  int C1 = tiling.loops[1][tiling.reduction_loop_idx[1]];
  int IC_UNROLL = IC_DIMENSION;
  int FX_UNROLL = 1;

  if (tiling.generic_replication) {
    C = tiling.num_channels;
    FX_UNROLL = tiling.fx_unrolling;
    FX = FX * FX_UNROLL;
    tiling.loops[1][tiling.fx_loop_idx] = FX;
    IY = Y * STRIDE;
    IX = X * STRIDE;
  }

  if (C < IC_DIMENSION) {
    IC_UNROLL = C;
  }

  if (tiling.resnet_replication) {
    FX = 7;
    C = 3;
    IC_UNROLL = 3;

    if (IC_DIMENSION == 4) {
      FX_UNROLL = 1;
    } else if (IC_DIMENSION == 8) {
      FX_UNROLL = 2;
    } else if (IC_DIMENSION == 16) {
      FX_UNROLL = 4;
    } else if (IC_DIMENSION == 32 || IC_DIMENSION == 64) {
      FX_UNROLL = 7;
    }
    tiling.loops[1][tiling.fx_loop_idx] = FX;
  }

  spdlog::debug("Performing GEMM: {}x{}x{} * {}x{}x{}x{} -> {}x{}x{}\n", Y, X,
                C, FY, FX, C, K, Y, X, K);

  // assert that none of tiling.loops are 0
  for (int j = 0; j < 3; j++) {
    assert(tiling.loops[0][j] != 0);
  }
  for (int j = 0; j < 6; j++) {
    assert(tiling.loops[1][j] != 0);
  }

  Buffer* outputs = new Buffer[X * Y * K];

  // only used for replication
  Psum* accumulations = new Psum[X * Y * K];
  for (int i = 0; i < X * Y * K; i++) {
    accumulations[i] = Psum(0.0);
  }

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
          for (counters[0][4] = 0; counters[0][4] < tiling.loops[0][4];
               counters[0][4]++) {
            int x1 = counters[0][tiling.x_loop_idx[0]];
            int y1 = counters[0][tiling.y_loop_idx[0]];
            int k1 = counters[0][tiling.weight_loop_idx[0]];
            int c2 = counters[0][tiling.reduction_loop_idx[0]];
            int fy1 = counters[0][tiling.fy_loop_idx[0]];
            for (counters[1][0] = 0; counters[1][0] < tiling.loops[1][0];
                 counters[1][0]++) {
              for (counters[1][1] = 0; counters[1][1] < tiling.loops[1][1];
                   counters[1][1]++) {
                for (counters[1][2] = 0; counters[1][2] < tiling.loops[1][2];
                     counters[1][2]++) {
                  for (counters[1][3] = 0; counters[1][3] < tiling.loops[1][3];
                       counters[1][3]++) {
                    for (counters[1][4] = 0;
                         counters[1][4] < tiling.loops[1][4];
                         counters[1][4]++) {
                      for (counters[1][5] = 0;
                           counters[1][5] < tiling.loops[1][5];
                           counters[1][5]++) {
                        int x0 = counters[1][tiling.x_loop_idx[1]];
                        int y0 = counters[1][tiling.y_loop_idx[1]];
                        int c1 = counters[1][tiling.reduction_loop_idx[1]];
                        int k0 = counters[1][tiling.weight_loop_idx[1]];
                        int fx = counters[1][tiling.fx_loop_idx];
                        int fy0 = counters[1][tiling.fy_loop_idx[1]];
                        int fy = fy1 * FY0 + fy0;

                        int x = x1 * X0 + x0;
                        int y = y1 * Y0 + y0;
                        int ix = x * STRIDE + fx - tiling.padding;
                        int iy = y * STRIDE + fy - tiling.padding;

                        for (int oc0 = 0; oc0 < OC_DIMENSION; oc0++) {
                          int k = (k1 * K0 + k0) * OC_DIMENSION + oc0;
                          int output_addr = y * X * K + x * K + k;

                          // accumulation is only updated during replication
                          Psum psum = accumulations[output_addr];

                          for (int ic0 = 0; ic0 < IC_UNROLL; ic0++) {
                            int c = c2 * C1 * IC_UNROLL + c1 * IC_UNROLL + ic0;
                            int input_addr = iy * IX * C + ix * C + c;
                            int weight_addr =
                                fy * FX * C * K + fx * C * K + c * K + k;
                            if (ix >= 0 && ix < IX && iy >= 0 && iy < IY) {
                              Input input = inputs[input_addr];
                              Input weight = weights[weight_addr];
#ifdef CHECK_PE
                              int pe_num = ic0 * OC_DIMENSION + oc0;
                              if (tiling.resnet_replication ||
                                  tiling.generic_replication) {
                                pe_num = ic0 * OC_DIMENSION +
                                         (counters[1][tiling.fx_loop_idx] %
                                          FX_UNROLL) *
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
                              if (tiling.resnet_replication ||
                                  tiling.generic_replication) {
                                pe_num = ic0 * OC_DIMENSION +
                                         (counters[1][tiling.fx_loop_idx] %
                                          FX_UNROLL) *
                                             IC_UNROLL * OC_DIMENSION +
                                         oc0;
                              }
                              Weight weight = weights[weight_addr];
                              pe_checker.add_reference(pe_num, Input::zero(),
                                                       weight, psum);
#endif
                            }
                          }

#if SUPPORT_MX
                          if (input_scales && weight_scales) {
                            // only perform scaling if within bounds
                            if (ix >= 0 && ix < IX && iy >= 0 && iy < IY) {
                              int c = c2 * C1 + c1;
                              int num_blocks = C / block_size;
                              int input_scale_addr =
                                  iy * IX * num_blocks + ix * num_blocks + c;
                              assert(input_scale_addr >= 0);

                              int weight_scale_addr = fy * FX * num_blocks * K +
                                                      fx * num_blocks * K +
                                                      c * K + k;
                              assert(weight_scale_addr >= 0);

                              Scale input_scale =
                                  input_scales[input_scale_addr];
                              Scale weight_scale =
                                  weight_scales[weight_scale_addr];
                              Buffer scale = static_cast<Buffer>(input_scale) *
                                             static_cast<Buffer>(weight_scale);
                              outputs[output_addr] +=
                                  static_cast<Buffer>(psum) * scale;
                            }
                          } else
#endif
                              if (tiling.resnet_replication) {
                            accumulations[output_addr] = psum;
                            if (IC_DIMENSION == 4) {
                              outputs[output_addr] += static_cast<Buffer>(
                                  accumulations[output_addr]);
                              accumulations[output_addr] = Psum(0.0);
                            } else if (IC_DIMENSION == 8) {
                              if (counters[1][tiling.fx_loop_idx] == 1 ||
                                  counters[1][tiling.fx_loop_idx] == 3 ||
                                  counters[1][tiling.fx_loop_idx] == 5 ||
                                  counters[1][tiling.fx_loop_idx] == 6) {
                                outputs[output_addr] += static_cast<Buffer>(
                                    accumulations[output_addr]);
                                accumulations[output_addr] = Psum(0.0);
                              }
                            } else if (IC_DIMENSION == 16) {
                              if (counters[1][tiling.fx_loop_idx] == 3 ||
                                  counters[1][tiling.fx_loop_idx] == 6) {
                                outputs[output_addr] += static_cast<Buffer>(
                                    accumulations[output_addr]);
                                accumulations[output_addr] = Psum(0.0);
                              }
                            } else if (IC_DIMENSION == 32 ||
                                       IC_DIMENSION == 64) {
                              if (counters[1][tiling.fx_loop_idx] == 6) {
                                outputs[output_addr] += static_cast<Buffer>(
                                    accumulations[output_addr]);
                                accumulations[output_addr] = Psum(0.0);
                              }
                            }
                          } else if (tiling.generic_replication) {
                            accumulations[output_addr] = psum;
                            if (tiling.fx_unrolling == 1 ||
                                ((counters[1][tiling.fx_loop_idx] > 0) &&
                                 (counters[1][tiling.fx_loop_idx] %
                                      tiling.fx_unrolling ==
                                  tiling.fx_unrolling - 1))) {
                              outputs[output_addr] += static_cast<Buffer>(
                                  accumulations[output_addr]);
                              accumulations[output_addr] = Psum(0.0);
                            }
                          } else {
                            outputs[output_addr] += static_cast<Buffer>(psum);
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
    }
  }

  spdlog::debug("GEMM done\n");

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

template <typename Input, typename Weight, typename Psum, typename Buffer,
          typename Scale>
inline Buffer* gemm(std::any input_ptr, std::any input_scale_ptr,
                    std::any weight_ptr, std::any weight_scale_ptr,
                    std::any bias_ptr, const Operation& operation) {
  Tiling tiling = get_tiling(operation);

  std::ostringstream oss;
  oss << "GEMM Tiling: " << tiling << std::endl;
  spdlog::debug(oss.str());

  const auto op_list = get_op_list(operation.param);
  const auto matrix_op = op_list.front();

  bool is_mx = matrix_op.target().find("mx") != std::string::npos;
  int block_size = is_mx ? matrix_op.kwargs().at("block_size").int_value() : 0;

  return gemm<Input, Weight, Psum, Buffer, Scale>(input_ptr, input_scale_ptr,
                                                  weight_ptr, weight_scale_ptr,
                                                  bias_ptr, tiling, block_size);
}

template <typename Input, typename Weight, typename Psum, typename Output,
          typename Scale, int N>
inline Output* gemv_quantized(std::any input_ptr, std::any input_scale_ptr,
                              std::any weight_ptr, std::any weight_scale_ptr,
                              std::any bias_ptr, const Operation& operation) {
  spdlog::debug("Performing matrix-vector multiply\n");

  const auto op_list = get_op_list(operation.param);
  const auto matrix_op = op_list.front();

  const auto input = matrix_op.kwargs().at("input").tensor();
  const auto output = get_op_outputs(operation.param).back();

  int block_size = 1;
  if (matrix_op.kwargs().contains("block_size")) {
    block_size = matrix_op.kwargs().at("block_size").int_value();
  }

  int C = get_size(input);
  int K = get_size(output);
  int num_tiles = (C + N - 1) / N;
  int num_blocks = N / block_size;

  Input* inputs = std::any_cast<Input*>(input_ptr);
  Weight* weights = std::any_cast<Weight*>(weight_ptr);
  Output* biases = std::any_cast<Output*>(bias_ptr);
  Scale* input_scales = std::any_cast<Scale*>(input_scale_ptr);
  Scale* weight_scales = std::any_cast<Scale*>(weight_scale_ptr);

  Output* outputs = new Output[K];
  for (int i = 0; i < K; i++) {
    outputs[i] = biases != nullptr ? biases[i] : Output::zero();
  }

  for (int k = 0; k < K; k++) {
    for (int c = 0; c < num_tiles; c++) {
      Psum product[N];
      for (int i = 0; i < N; i++) {
        int index = c * N + i;
        if (index < C) {
          product[i] = (Psum)inputs[index] * (Psum)weights[k * C + index];
        } else {
          product[i] = Psum::zero();
        }
      }

      Output output_block[num_blocks];

      for (int i = 0; i < num_blocks; i++) {
        Psum buffer[block_size];
        for (int j = 0; j < block_size; j++) {
          buffer[j] = product[i * block_size + j];
        }

        Psum psum = tree_reduce(buffer, block_size);

        if (input_scales && weight_scales) {
          Scale input_scale = input_scales[c * num_blocks + i];
          Scale weight_scale =
              weight_scales[k * C / block_size + c * num_blocks + i];
          output_block[i] = static_cast<Output>(input_scale) *
                            static_cast<Output>(weight_scale) *
                            static_cast<Output>(psum);
        } else {
          output_block[i] = static_cast<Output>(product[0]);
        }
      }

      // tree reduce over output blocks
      outputs[k] += tree_reduce(output_block, num_blocks);
    }
  }

  delete[] inputs;
  delete[] weights;

  if (biases != nullptr) {
    delete[] biases;
  }

  if (input_scales != nullptr) {
    delete[] input_scales;
  }

  if (weight_scales != nullptr) {
    delete[] weight_scales;
  }

  return outputs;
}

template <typename T>
inline T* gemv_bfloat16(std::any input_ptr, std::any weight_ptr,
                        std::any bias_ptr,
                        const std::vector<int>& weight_shape) {
  const int K = weight_shape[0];
  const int C = weight_shape[1];

  T* inputs = std::any_cast<T*>(input_ptr);
  T* weights = std::any_cast<T*>(weight_ptr);
  T* biases = std::any_cast<T*>(bias_ptr);

  T* outputs = new T[K];
  for (int i = 0; i < K; i++) {
    outputs[i] = 0.0;
  }

  for (int k = 0; k < K; k++) {
    T sums[2] = {0, 0};
    int index = 0;

    for (int i = 0; i < C; i += VECTOR_UNIT_WIDTH) {
      T buffer[VECTOR_UNIT_WIDTH];
      for (int j = 0; j < VECTOR_UNIT_WIDTH; j++) {
        buffer[j] = inputs[i + j] * weights[k * C + i + j];
      }
      sums[index++ % 2] += tree_reduce(buffer, VECTOR_UNIT_WIDTH);
    }

    outputs[k] = tree_reduce(sums, 2);

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

// arb kernel size model based on 3*3 kernel
template <typename Input, typename Psum, typename Buffer, typename Scale>
// Input: quantized input/weight type
// Psum: accumulation type
// Buffer: output type
inline Buffer* DwC(std::any input_ptr, std::any input_scale_ptr,
                   std::any weight_ptr, std::any weight_scale_ptr,
                   std::any bias_ptr, const Operation& operation) {
  std::vector<int> BUFFER_DIM = {2, 9, 9};  // IC, X, Y

  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto first_op = op_list.front();
  bool is_mx = first_op.target().find("mx") != std::string::npos;
  int block_size = is_mx ? first_op.kwargs().at("block_size").int_value() : 0;

  const auto& stride_vals = first_op.kwargs().at("stride").int_list().values();
  int stride_y = stride_vals[0];
  int stride_x = stride_vals[1];

  int ibatch = first_op.kwargs().at("input").tensor().shape(0);
  int ic = first_op.kwargs().at("input").tensor().shape(3);
  int iy = first_op.kwargs().at("input").tensor().shape(1);  // input height
  int ix = first_op.kwargs().at("input").tensor().shape(2);  // input weight

  int k_h = first_op.kwargs().at("weight").tensor().shape(2);
  int k_w = first_op.kwargs().at("weight").tensor().shape(3);
  int kernel_size = 3;

  int x_pad = first_op.kwargs().at("padding").int_list().values()[1];
  int y_pad = first_op.kwargs().at("padding").int_list().values()[0];

  int obatch = 1;
  int oc = ic;
  int ox = floor((ix + 2 * x_pad - kernel_size) / stride_x) + 1;
  int oy = floor((iy + 2 * y_pad - kernel_size) / stride_y) + 1;

  Buffer* outputs = new Buffer[obatch * oc * ox * oy];
  Input* inputs = std::any_cast<Input*>(input_ptr);
  Scale* input_scales = std::any_cast<Scale*>(input_scale_ptr);

  Input* weights = std::any_cast<Input*>(weight_ptr);
  Scale* weight_scales = std::any_cast<Scale*>(weight_scale_ptr);

  Buffer* biases = std::any_cast<Buffer*>(bias_ptr);

  spdlog::debug(
      "Input shape: [{} {} {} {}], Weight shape: [{} {} {} {}], Output shape: "
      "[{} {} {} {}]",
      ibatch, ic, ix, iy, oc, 1, k_w, k_h, obatch, oc, ox, oy);

  int IC1 = (ic + BUFFER_DIM[0] - 1) / BUFFER_DIM[0];
  int IC0 = BUFFER_DIM[0];
  int OX1 = (ox + BUFFER_DIM[1] - 1) / BUFFER_DIM[1];
  int OX0 = BUFFER_DIM[1];
  int OY1 = (oy + BUFFER_DIM[2] - 1) / BUFFER_DIM[2];
  int OY0 = BUFFER_DIM[2];

#if SUPPORT_MX
  Buffer psums[7];
  Buffer products[kernel_size * kernel_size];
#else
  Psum psums[7];
  Psum products[kernel_size * kernel_size];
#endif

  int counter[2][3] = {0};
  for (counter[0][0] = 0; counter[0][0] < IC1; counter[0][0]++) {
    for (counter[0][1] = 0; counter[0][1] < OX1; counter[0][1]++) {
      for (counter[0][2] = 0; counter[0][2] < OY1; counter[0][2]++) {
        int ic1 = counter[0][0];
        int ox1 = counter[0][1];
        int oy1 = counter[0][2];

        for (counter[1][0] = 0; counter[1][0] < IC0; counter[1][0]++) {
          for (counter[1][1] = 0; counter[1][1] < OX0; counter[1][1]++) {
            for (counter[1][2] = 0; counter[1][2] < OY0; counter[1][2]++) {
              int ic0 = counter[1][0];
              int ox0 = counter[1][1];
              int oy0 = counter[1][2];

              int ic_g = ic1 * IC0 + ic0;
              int x = ox1 * OX0 + ox0;
              int y = oy1 * OY0 + oy0;

              if (x >= ox || y >= oy || ic_g >= ic) continue;

              // For pad=0 (no padding), start from actual stride position
              int x_start = x * stride_x;
              int y_start = y * stride_y;

              int output_addr = (y * ox + x) * oc + ic_g;

              // Apply 3x3 kernel directly on input (pad=0 behavior)
              for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                  int input_x = x_start + kx - x_pad;
                  int input_y = y_start + ky - y_pad;

                  // Check bounds to prevent core dump
                  if (input_x >= 0 && input_x < ix && input_y >= 0 &&
                      input_y < iy) {
                    int input_addr = (input_y * ix + input_x) * ic + ic_g;
                    Input input = inputs[input_addr];
                    int weight_addr = ic_g * kernel_size * kernel_size +
                                      ky * kernel_size + kx;
                    Input weight = weights[weight_addr];  // Use padded_weights

#if SUPPORT_MX
                    if (block_size == 0) {
                      products[ky * kernel_size + kx] =
                          static_cast<Buffer>(input) *
                          static_cast<Buffer>(weight);
                    } else {
                      int num_blocks = (ic + block_size - 1) / block_size;
                      int input_scale_addr =
                          (input_y * ix + input_x) * num_blocks +
                          ic_g / block_size;
                      int weight_scale_addr = weight_addr;
                      Scale input_scale = input_scales[input_scale_addr];
                      Scale weight_scale = weight_scales[weight_scale_addr];
                      Buffer scale = static_cast<Buffer>(input_scale) *
                                     static_cast<Buffer>(weight_scale);
                      products[ky * kernel_size + kx] =
                          static_cast<Buffer>(static_cast<Psum>(input) *
                                              static_cast<Psum>(weight)) *
                          scale;
                    }
#else
                    products[ky * kernel_size + kx] =
                        static_cast<Psum>(input) * static_cast<Psum>(weight);
#endif
                  } else {
#if SUPPORT_MX
                    products[ky * kernel_size + kx] = Buffer(0.0);
#else
                    products[ky * kernel_size + kx] = Psum(0.0);
#endif
                  }
                }
              }
              psums[0] = products[0] + products[1];
              psums[1] = products[2] + products[3];
              psums[2] = products[4] + products[5];
              psums[3] = products[6] + products[7];
              psums[4] = psums[0] + psums[1];
              psums[5] = psums[2] + psums[3];
              psums[6] = psums[4] + psums[5] + products[8];
              outputs[output_addr] =
                  biases[ic_g] + static_cast<Buffer>(psums[6]);
            }
          }
        }
      }
    }
  }

  delete[] inputs;
  if (input_scales != nullptr) {
    delete[] input_scales;
  }
  delete[] weights;
  if (weight_scales != nullptr) {
    delete[] weight_scales;
  }
  delete[] biases;

  spdlog::debug("DwC done\n");
  return outputs;
}