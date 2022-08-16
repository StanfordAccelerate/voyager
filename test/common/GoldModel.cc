#include "test/common/GoldModel.h"

#include <algorithm>
#include <cstring>
#include <vector>

#ifndef NO_UNIVERSAL
inline UniversalPositAccum gold_fma(UniversalPosit a, UniversalPosit b,
                                    UniversalPositAccum c) {
  UniversalPositAccum product;
  sw::universal::value<15> internal = sw::universal::fma<8, 1>(a, b, 0);
  sw::universal::convert<16, 1, 15>(internal, product);
  return product + c;
}
#endif

inline ACCUM_DATATYPE gold_fma(ACCUM_DATATYPE a, ACCUM_DATATYPE b,
                               ACCUM_DATATYPE c) {
  INPUT_DATATYPE::DecomposedPosit f1 = a;
  INPUT_DATATYPE::DecomposedPosit f2 = b;
  ACCUM_DATATYPE::DecomposedPosit f3 = c;
  return decomposed_fma<8, 1, 16, 1>(f1, f2, f3);
}

inline float gold_fma(float a, float b, float c) { return a * b + c; }

#ifndef NO_UNIVERSAL
inline void gold_relu(UniversalPositAccum &a) { a = a < 0 ? 0 : a; }
#endif
inline void gold_relu(ACCUM_DATATYPE &a) { a.relu(); }
inline void gold_relu(float &a) { a = a < 0 ? 0 : a; }

#ifndef NO_UNIVERSAL
inline void gold_exp(UniversalPositAccum &a) { a = sw::universal::exp(a); }
#endif
inline void gold_exp(ACCUM_DATATYPE &a) { a = posit_exp(a); }
inline void gold_exp(float &a) { a = exp(a); }

#ifndef NO_UNIVERSAL
inline void gold_reciprocal(UniversalPositAccum &a) { a = a.reciprocate(); }
#endif
inline void gold_reciprocal(ACCUM_DATATYPE &a) { a.reciprocal(); }
inline void gold_reciprocal(float &a) { a = 1.0f / a; }

#ifndef NO_UNIVERSAL
inline void gold_inv_sqrt(UniversalPositAccum &a) {
  a = sw::universal::rsqrt(a);
}
#endif
inline void gold_inv_sqrt(ACCUM_DATATYPE &a) {
  ACCUM_DATATYPE::DecomposedPosit decomposed = a;
  a = decomposed.inv_sqrt();
}
inline void gold_inv_sqrt(float &a) { a = 1.0f / std::sqrt(a); }

#ifndef NO_UNIVERSAL
inline UniversalPositAccum readInput(UniversalPosit *matrix, int index,
                                     bool accType) {
  if (!accType) {
    return static_cast<UniversalPositAccum>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].encoding();
  int encoding2 = matrix[2 * index + 1].encoding();
  UniversalPositAccum p16;
  p16.setbits((encoding1 << 8) + encoding2);
  return p16;
}

inline UniversalPositAccum readInput2(UniversalPosit *matrix, int index,
                                      bool accType, int expBias = 0) {
  UniversalPositAccum p16;
  if (!accType) {
    p16 = matrix[index];
  } else {
    int encoding1 = matrix[2 * index].encoding();
    int encoding2 = matrix[2 * index + 1].encoding();
    p16.setbits((encoding1 << 8) + encoding2);
  }
  sw::universal::value<12> val = p16.to_value();
  val.setExponent(val.scale() + expBias);
  sw::universal::convert<16, 1>(val, p16);
  return p16;
}
#endif

inline ACCUM_DATATYPE readInput(INPUT_DATATYPE *matrix, int index,
                                bool accType) {
  if (!accType) {
    return static_cast<ACCUM_DATATYPE>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].bits;
  int encoding2 = matrix[2 * index + 1].bits;
  ACCUM_DATATYPE p16;
  p16.setbits((encoding1 << 8) + encoding2);
  return p16;
}

inline ACCUM_DATATYPE readInput2(INPUT_DATATYPE *matrix, int index,
                                 bool accType, int expBias = 0) {
  ACCUM_DATATYPE p16;
  if (!accType) {
    p16 = matrix[index];
  } else {
    int encoding1 = matrix[2 * index].bits;
    int encoding2 = matrix[2 * index + 1].bits;
    p16.setbits((encoding1 << 8) + encoding2);
  }
  ACCUM_DATATYPE::DecomposedPosit val = p16;
  val.scale += expBias;
  return val;
}

inline float readInput(float *matrix, int index, bool accType) {
  return accType ? matrix[2 * index] : matrix[index];
}

inline float readInput2(float *matrix, int index, bool accType, int expBias) {
  return accType ? matrix[2 * index] : matrix[index];
}

#ifndef NO_UNIVERSAL
inline void saveOutput(UniversalPosit *matrix, int index,
                       UniversalPositAccum value, bool accType) {
  if (!accType) {
    matrix[index] = static_cast<UniversalPosit>(value);
  } else {
    int bits = value.encoding();
    matrix[2 * index].setbits((bits >> 8) & 0xFF);
    matrix[2 * index + 1].setbits(bits & 0xFF);
  }
}
#endif

inline void saveOutput(INPUT_DATATYPE *matrix, int index, ACCUM_DATATYPE value,
                       bool accType) {
  if (!accType) {
    matrix[index] = static_cast<INPUT_DATATYPE>(value);
  } else {
    int bits = value.bits;
    matrix[2 * index].setbits((bits >> 8) & 0xFF);
    matrix[2 * index + 1].setbits(bits & 0xFF);
  }
}

inline void saveOutput(float *matrix, int index, float value, bool accType) {
  if (!accType) {
    matrix[index] = value;
  } else {
    matrix[2 * index] = value;
    matrix[2 * index + 1] = 0;
  }
}

template <typename T, typename ACC_T>
void run_gold_op(const SimplifiedParams params, T *matrixA, T *matrixB,
                 T *matrixC, T *biasMatrix, T *residualMatrix,
                 T *weightGradMatrix, T *biasGradMatrix) {
  std::cerr << "Running gold model " << std::endl;

  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;

  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  ACC_T learningRate = params.learningRate;

  if (params.SOFTMAX) {
    for (int x = 0; x < X; x++) {
      ACC_T outputMatrix[Y];
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          outputMatrix[y] = readInput(matrixA, x * Y + y, params.ACC_T_INPUT);
        } else {
          outputMatrix[y] = 0;
        }
      }

      ACC_T max = 0;
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          max = outputMatrix[y] > max ? outputMatrix[y] : max;
        }
      }

      ACC_T sum = 0;
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          ACC_T exp = outputMatrix[y] - max;
          gold_exp(exp);
          outputMatrix[y] = exp;
          sum += exp;
        }
      }

      ACC_T divisor = sum;
      gold_reciprocal(divisor);
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          outputMatrix[y] *= divisor;
        }
        saveOutput(matrixC, x * Y + y, outputMatrix[y], params.ACC_T_OUTPUT);
      }
    }
  } else if (params.FC) {
    // fully connected layer (matrix-vector)
    for (int k = 0; k < K; k++) {
      ACC_T acc = 0;
      for (int c = 0; c < C; c++) {
        ACC_T a = readInput(matrixA, c, params.ACC_T_INPUT);
        ACC_T b = readInput(matrixB, k * C + c, params.ACC_T_WEIGHT);

        if (params.WEIGHT_SPLITTING) {
          ACC_T grad = weightGradMatrix[k * C + c];
          b += learningRate * b;
        }

        acc += a * b;
      }

      if (params.BIAS) {
        acc += readInput(biasMatrix, k, true);
        if (params.WEIGHT_SPLITTING) {
          acc += learningRate * readInput(biasGradMatrix, k, true);
        }
      }

      saveOutput(matrixC, k, acc, params.ACC_T_OUTPUT);
    }
  } else if (params.NO_NORM) {
    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        ACC_T a = readInput(matrixA, x * K + k, params.ACC_T_INPUT);
        ACC_T b = readInput(matrixB, k, params.ACC_T_WEIGHT);
        ACC_T acc = a * b;

        if (params.WEIGHT_SPLITTING) {
          ACC_T grad = weightGradMatrix[k];
          acc += learningRate * grad * a;
        }

        if (params.BIAS) {
          acc += readInput(biasMatrix, k, true);
          if (params.WEIGHT_SPLITTING) {
            acc += learningRate * readInput(biasGradMatrix, k, true);
          }
        }

        saveOutput(matrixC, x * K + k, acc, params.ACC_T_OUTPUT);
      }
    }
  } else if (params.SOFTMAX_GRAD) {
    ACC_T outputMatrix[X * Y];
    ACC_T attentionProbs[X * Y];
    ACC_T sums[X];
    memset(sums, 0, sizeof(sums));

    for (int i = 0; i < X * Y; i++) {
      outputMatrix[i] = readInput(matrixA, i, params.ACC_T_INPUT);
      attentionProbs[i] = readInput(residualMatrix, i, params.ACC_T_OUTPUT);
    }

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        int index = x * Y + y;
        outputMatrix[index] *= attentionProbs[index];
        sums[x] += outputMatrix[index];
      }
    }

    ACC_T scale = 1.0f / std::sqrt(32);
    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        outputMatrix[x * Y + y] -= sums[x] * attentionProbs[x * Y + y];
        outputMatrix[x * Y + y] *= scale;
        saveOutput(matrixC, x * Y + y, outputMatrix[x * Y + y],
                   params.ACC_T_OUTPUT);
      }
    }
  } else if (params.FC_GRAD) {
    ACC_T outputMatrix[X * K];
    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        ACC_T a = readInput(matrixA, x, params.ACC_T_INPUT);
        ACC_T b = readInput(matrixB, k, params.ACC_T_WEIGHT);
        outputMatrix[x * K + k] = a * b;
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < X * K; i++) {
        acc += outputMatrix[i] * outputMatrix[i];
      }

      gold_inv_sqrt(acc);
      acc = std::min(static_cast<float>(acc), 1.0f);
      for (int i = 0; i < X * K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    for (int i = 0; i < X * K; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else if (params.NO_NORM_GRAD) {
    ACC_T outputMatrix[K];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int i = 0; i < X; i++) {
      for (int j = 0; j < K; j++) {
        ACC_T a = readInput(matrixA, i * K + j, params.ACC_T_INPUT);
        ACC_T b = readInput(matrixB, i * K + j, params.ACC_T_WEIGHT);
        outputMatrix[j] += a * b;
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < K; i++) {
        acc += outputMatrix[i] * outputMatrix[i];
      }

      gold_inv_sqrt(acc);
      acc = std::min(static_cast<float>(acc), 1.0f);
      for (int i = 0; i < K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    for (int i = 0; i < K; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else if (params.BIAS_GRAD) {
    ACC_T accumMatrixB[C * K];
    for (int i = 0; i < C * K; i++) {
      accumMatrixB[i] = readInput(matrixB, i, params.ACC_T_WEIGHT);
    }

    if (params.CONCAT_WEIGHT) {
      ACC_T copyMatrixB[C * K];
      memcpy(copyMatrixB, accumMatrixB, sizeof(copyMatrixB));
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < C; j++) {
          for (int k = 0; k < K / 4; k++) {
            accumMatrixB[i * K / 4 + j * K + k] =
                copyMatrixB[i * C * K / 4 + j * K / 4 + k];
          }
        }
      }
    }

    ACC_T outputMatrix[K];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int i = 0; i < C; i++) {
      for (int j = 0; j < K; j++) {
        outputMatrix[j] += accumMatrixB[i * K + j];
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < K; i++) {
        acc += outputMatrix[i] * outputMatrix[i];
      }

      gold_inv_sqrt(acc);
      acc = std::min(static_cast<float>(acc), 1.0f);
      for (int i = 0; i < K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    for (int i = 0; i < K; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else if (params.CROSS_ENTROPY_GRAD) {
    ACC_T labels[X];
    ACC_T logits[X];
    ACC_T gradients[X];

    for (int i = 0; i < X; i++) {
      labels[i] = readInput(matrixA, i, params.ACC_T_INPUT);
      logits[i] = readInput(matrixB, i, params.ACC_T_WEIGHT);
    }

    ACC_T max = 0;
    for (int i = 0; i < X; i++) {
      max = logits[i] > max ? logits[i] : max;
    }

    ACC_T sum = 0;
    for (int i = 0; i < X; i++) {
      ACC_T exp = logits[i] - max;
      gold_exp(exp);
      gradients[i] = exp;
      sum += exp;
    }

    gold_reciprocal(sum);
    for (int i = 0; i < X; i++) {
      gradients[i] = gradients[i] * sum - labels[i];
      saveOutput(matrixC, i, gradients[i], params.ACC_T_OUTPUT);
    }
  } else if (params.MSE_GRAD) {
    ACC_T divisor = 2 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<ACC_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else if (params.BCE_WITH_LOGITS_GRAD) {
    ACC_T divisor = 1 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<ACC_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else if (params.GRAD_CLIPPING_UNIT_TEST) {
    ACC_T outputMatrix[X * C];
    for (int i = 0; i < X * C; i++) {
      outputMatrix[i] = readInput(matrixA, i, params.ACC_T_INPUT);
    }

    ACC_T acc = 0;
    for (int i = 0; i < X * C; i++) {
      acc += outputMatrix[i] * outputMatrix[i];
    }

    gold_reciprocal(acc);
    acc = std::min(static_cast<float>(acc), 1.0f);
    for (int i = 0; i < X * C; i++) {
      outputMatrix[i] *= acc;
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else {
    ACC_T accumMatrixA[(STRIDE * X) * (STRIDE * Y) * C];
    ACC_T accumMatrixB[FX * FY * C * K];
    ACC_T accumResidualMatrix[X * Y * K];
    ACC_T outputMatrix[X * Y * K];

    for (int i = 0; i < X * C; i++) {
      accumMatrixA[i] = readInput(matrixA, i, params.ACC_T_INPUT);
    }

    for (int i = 0; i < C * K; i++) {
      accumMatrixB[i] = readInput(matrixB, i, params.ACC_T_WEIGHT);
    }

    if (params.RESIDUAL || params.RELU_GRAD) {
      for (int i = 0; i < X * K; i++) {
        accumResidualMatrix[i] =
            readInput(residualMatrix, i, params.ACC_T_OUTPUT);
      }
    }

    if (params.CONCAT_INPUT) {
      ACC_T copyMatrixA[X * C];
      memcpy(copyMatrixA, accumMatrixA, sizeof(copyMatrixA));
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < X; j++) {
          for (int k = 0; k < C / 4; k++) {
            accumMatrixA[i * C / 4 + j * C + k] =
                copyMatrixA[i * X * C / 4 + j * C / 4 + k];
          }
        }
      }
    }

    if (params.INPUT_TRANSPOSE) {
      ACC_T copyMatrixA[X * C];
      memcpy(copyMatrixA, accumMatrixA, sizeof(copyMatrixA));
      for (int x = 0; x < X; x++) {
        for (int c = 0; c < C; c++) {
          accumMatrixA[x * C + c] = copyMatrixA[c * X + x];
        }
      }
    }

    if (params.CONCAT_WEIGHT) {
      ACC_T copyMatrixB[C * K];
      memcpy(copyMatrixB, accumMatrixB, sizeof(copyMatrixB));
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < C; j++) {
          for (int k = 0; k < K / 4; k++) {
            accumMatrixB[i * K / 4 + j * K + k] =
                copyMatrixB[i * C * K / 4 + j * K / 4 + k];
          }
        }
      }
    }

    if (params.WEIGHT_TRANSPOSE) {
      ACC_T copyMatrixB[C * K];
      memcpy(copyMatrixB, accumMatrixB, sizeof(copyMatrixB));
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          accumMatrixB[c * K + k] = copyMatrixB[k * C + c];
        }
      }
    }

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        for (int k = 0; k < K; k++) {
          ACC_T acc = 0;
          for (int c = 0; c < C; c++) {
            for (int fy = -(FY - 1) / 2; fy <= (FY - 1) / 2; fy++) {
              for (int fx = -(FX - 1) / 2; fx <= (FX - 1) / 2; fx++) {
                if (STRIDE * x + fx >= 0 && STRIDE * x + fx < STRIDE * X &&
                    STRIDE * y + fy >= 0 &&
                    STRIDE * y + fy < STRIDE * Y) {  // check if in bounds
                  ACC_T a = accumMatrixA[(STRIDE * y + fy) * STRIDE * X * C +
                                         (STRIDE * x + fx) * C + c];
                  ACC_T b =
                      accumMatrixB[(fy + (FY - 1) / 2) * FX * C * K +
                                   (fx + (FX - 1) / 2) * C * K + c * K + k];

                  if (params.WEIGHT_SPLITTING) {
                    ACC_T grad =
                        weightGradMatrix[(fy + (FY - 1) / 2) * FX * C * K +
                                         (fx + (FX - 1) / 2) * C * K + c * K +
                                         k];
                    b += learningRate * grad;
                  }

                  acc = gold_fma(a, b, acc);
                }
              }
            }
          }

          if (params.BIAS) {
            acc += readInput(biasMatrix, k, true);
            if (params.WEIGHT_SPLITTING) {
              acc += learningRate * readInput(biasGradMatrix, k, true);
            }
          }

          if (params.RELU) {
            gold_relu(acc);
          }

          if (params.RELU_GRAD && accumResidualMatrix[x * K + k] <= 0) {
            acc = 0;
          }

          if (params.ATTENTION_SCALING) {
            ACC_T scale = 1.0f / sqrt(32);
            acc *= scale;
          }

          outputMatrix[y * X * K + x * K + k] = acc;
        }
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < X * K; i++) {
        acc += outputMatrix[i] * outputMatrix[i];
      }

      gold_inv_sqrt(acc);
      acc = std::min(static_cast<float>(acc), 1.0f);
      for (int i = 0; i < X * K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    if (params.RESIDUAL) {
      for (int i = 0; i < X * K; i++) {
        outputMatrix[i] += accumResidualMatrix[i];
      }
    }

    if (params.SPLIT_OUTPUT) {
      ACC_T copyMatrixC[X * K];
      memcpy(copyMatrixC, outputMatrix, sizeof(copyMatrixC));
      for (int i = 0; i < X; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < K / 4; k++) {
            outputMatrix[j * X * K / 4 + i * K / 4 + k] =
                copyMatrixC[i * K + j * K / 4 + k];
          }
        }
      }
    }

    for (int i = 0; i < X * Y * K; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }

    if (params.MAXPOOL) {
      // create copy
      T tmpMatrixC[X * Y * K];
      memcpy(tmpMatrixC, matrixC, sizeof(T) * X * Y * K);

      for (int y = 0; y < Y / 2; y++) {
        for (int x = 0; x < X / 2; x++) {
          for (int k = 0; k < K; k++) {
            std::vector<T> v;

            for (int x_window = 0; x_window < 2; x_window++) {
              for (int y_window = 0; y_window < 2; y_window++) {
                int x_maxpool = x * 2 + x_window;
                int y_maxpool = y * 2 + y_window;
                v.push_back(tmpMatrixC[y_maxpool * X * K + x_maxpool * K + k]);
              }
            }

            matrixC[y * (X / 2) * K + x * K + k] =
                *std::max_element(v.begin(), v.end());
          }
        }
      }
    }

    if (params.AVGPOOL) {
      // create copy
      T tmpMatrixC[X * Y * K];
      memcpy(tmpMatrixC, matrixC, sizeof(T) * X * Y * K);

      for (int k = 0; k < K; k++) {
        ACC_T acc = 0;
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            acc += tmpMatrixC[y * X * K + x * K + k];
          }
        }
        matrixC[k] = acc;  /// (Y * X);  // Average
      }
    }
  }
}

void run_custom_posit_gold_model(
    const SimplifiedParams params, INPUT_DATATYPE *matrixA,
    INPUT_DATATYPE *matrixB, INPUT_DATATYPE *matrixC,
    INPUT_DATATYPE *biasMatrix, INPUT_DATATYPE *residualMatrix,
    INPUT_DATATYPE *weightGradMatrix, INPUT_DATATYPE *biasGradMatrix) {
  run_gold_op<INPUT_DATATYPE, ACCUM_DATATYPE>(params, matrixA, matrixB, matrixC,
                                              biasMatrix, residualMatrix,
                                              weightGradMatrix, biasGradMatrix);
}

void run_universal_posit_gold_model(
    const SimplifiedParams params, UniversalPosit *matrixA,
    UniversalPosit *matrixB, UniversalPosit *matrixC,
    UniversalPosit *biasMatrix, UniversalPosit *residualMatrix,
    UniversalPosit *weightGradMatrix, UniversalPosit *biasGradMatrix) {
  run_gold_op<UniversalPosit, UniversalPositAccum>(
      params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix,
      weightGradMatrix, biasGradMatrix);
}

void run_fp_gold_model(const SimplifiedParams params, float *matrixA,
                       float *matrixB, float *matrixC, float *biasMatrix,
                       float *residualMatrix, float *weightGradMatrix,
                       float *biasGradMatrix) {
  run_gold_op<float, float>(params, matrixA, matrixB, matrixC, biasMatrix,
                            residualMatrix, weightGradMatrix, biasGradMatrix);
}
