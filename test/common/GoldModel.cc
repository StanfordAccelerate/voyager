#include "test/common/GoldModel.h"

#include <algorithm>
#include <cstring>
#include <vector>
#ifndef NO_UNIVERSAL
inline UniversalPositAccum gold_fma(UniversalPosit a, UniversalPosit b,
                                    UniversalPositAccum c) {
  UniversalPositAccum tmp;
  sw::universal::value<15> internal = sw::universal::fma<8, 1>(a, b, 0);
  sw::universal::convert<16, 1, 15>(internal, tmp);
  tmp += c;
  return tmp;
}
#endif

inline ACCUM_DATATYPE gold_fma(INPUT_DATATYPE a, INPUT_DATATYPE b,
                               ACCUM_DATATYPE c) {
  return fma(a, b, c);
}

inline float gold_fma(float a, float b, float c) { return a * b + c; }

#ifndef NO_UNIVERSAL
inline void gold_relu(UniversalPositAccum &a) {
  if (a < 0) {
    a = 0;
  }
}
#endif

inline void gold_relu(ACCUM_DATATYPE &a) { a.relu(); }

inline void gold_relu(float &a) {
  if (a < 0.0f) {
    a = 0.0f;
  }
}

inline void gold_exp(ACCUM_DATATYPE &a) { a = posit_exp(a); }
inline void gold_exp(float &a) { a = exp(a); }

#ifndef NO_UNIVERSAL
inline void gold_exp(UniversalPositAccum &a) {
  // TODO
}
#endif

inline void gold_reciprocal(ACCUM_DATATYPE &a) { a.reciprocal(); }
inline void gold_reciprocal(float &a) { a = 1.0 / a; }

#ifndef NO_UNIVERSAL
inline void gold_reciprocal(UniversalPositAccum &a) { a = 1 / a; }
#endif

template <typename T, typename ACC_T>
void run_gold_op(const SimplifiedParams params, T *matrixA, T *matrixB,
                 T *matrixC, ACC_T *biasMatrix, T *residualMatrix,
                 T *weightGradMatrix, ACC_T *biasGradMatrix, bool inputScaling,
                 bool weightScaling) {
#ifndef PIPE_INPUT
  std::cout << "Running gold model " << std::endl;
#endif

  if (params.SOFTMAX) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int Y = params.loops[0][params.inputYLoopIndex[0]] *
            params.loops[1][params.inputYLoopIndex[1]];

    const int rows = inputScaling ? X + 1 : X;
    ACC_T outputMatrix[rows * Y];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          outputMatrix[x * Y + y] = matrixA[x * Y + y];
          if (inputScaling) {
            outputMatrix[x * Y + y] *= static_cast<ACC_T>(matrixA[X * Y + y]);
          }
        } else {
          outputMatrix[x * Y + y] = 0;
        }
      }
    }

    for (int x = 0; x < X; x++) {
      ACC_T max = 0;
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          max = outputMatrix[x * Y + y] > max ? outputMatrix[x * Y + y] : max;
        }
      }

      ACC_T sum = 0;
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          ACC_T adjustedVal = outputMatrix[x * Y + y] - max;
          gold_exp(adjustedVal);
          outputMatrix[x * Y + y] = adjustedVal;

          // outputMatrix[x * Y + y] =
          //     exp(static_cast<float>(outputMatrix[x * Y + y] - max));
          sum += outputMatrix[x * Y + y];
        }
      }

      ACC_T divisor = sum;
      gold_reciprocal(divisor);
      for (int y = 0; y < Y; y++) {
        if (!params.ATTENTION_MASK || static_cast<float>(matrixB[y])) {
          // ACC_T divisor = sum.reciprocal();
          outputMatrix[x * Y + y] *= divisor;
          // outputMatrix[x * Y + y] /= sum;
          if (inputScaling) {
            outputMatrix[X * Y + y] += outputMatrix[x * Y + y];
          }
        }
      }
    }

    for (int y = 0; y < Y; y++) {
      if (inputScaling) {
        float sum = static_cast<float>(outputMatrix[X * Y + y]);
        ACC_T scalingFactor = sum ? pow(2, round(log2(sum / X))) : 1;
        matrixC[X * Y + y] = scalingFactor;
        for (int x = 0; x < X; x++) {
          matrixC[x * Y + y] = outputMatrix[x * Y + y] / scalingFactor;
        }
      } else {
        for (int x = 0; x < X; x++) {
          matrixC[x * Y + y] = outputMatrix[x * Y + y];
        }
      }
    }
  } else if (params.FC) {
    // fully connected layer (matrix-vector)
    int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
    ACC_T learningRate = params.learningRate;

    for (int k = 0; k < K; k++) {
      ACC_T acc = 0;
      for (int c = 0; c < C; c++) {
        ACC_T a = matrixA[c];
        ACC_T b = matrixB[k * C + c];
        acc = gold_fma(a, b, acc);

        if (params.WEIGHT_SPLITTING) {
          b = weightGradMatrix[k * C + c];
          acc += learningRate * a * b;
        }

        if (inputScaling) {
          acc *= static_cast<ACC_T>(matrixA[C + c]);
        }
      }

      if (weightScaling) {
        acc *= static_cast<ACC_T>(matrixB[C * K]);
      }

      if (params.BIAS) {
        acc += biasMatrix[k];
        if (params.WEIGHT_SPLITTING) {
          acc += learningRate * biasGradMatrix[k];
        }
      }

      matrixC[k] = acc;
    }
  } else if (params.NO_NORM) {
    // elementwise multiplication and addition between a matrix and a vector
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
    ACC_T learningRate = params.learningRate;

    ACC_T outputMatrix[X * K + K];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        ACC_T a = matrixA[x * K + k];
        ACC_T b = matrixB[k];
        ACC_T acc = a * b;

        if (params.WEIGHT_SPLITTING) {
          b = weightGradMatrix[k];
          acc += learningRate * a * b;
        }

        if (params.BIAS) {
          acc += biasMatrix[k];
          if (params.WEIGHT_SPLITTING) {
            acc += learningRate * biasGradMatrix[k];
          }
        }

        if (inputScaling) {
          acc *= static_cast<ACC_T>(matrixA[X * K + k]);
        }

        if (weightScaling) {
          acc *= static_cast<ACC_T>(matrixB[K]);
        }

        outputMatrix[x * K + k] = acc;
        if (inputScaling) {
          outputMatrix[X * K + k] += abs(static_cast<float>(acc));
        }
      }
    }

    for (int k = 0; k < K; k++) {
      if (inputScaling) {
        float sum = static_cast<float>(outputMatrix[X * K + k]);
        ACC_T scalingFactor = sum ? pow(2, round(log2(sum / X))) : 1;
        matrixC[X * K + k] = scalingFactor;
        for (int x = 0; x < X; x++) {
          matrixC[x * K + k] = outputMatrix[x * K + k] / scalingFactor;
        }
      } else {
        for (int x = 0; x < X; x++) {
          matrixC[x * K + k] = outputMatrix[x * K + k];
        }
      }
    }
  } else if (params.OUTER_PRODUCT) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

    ACC_T outputMatrix[X * K + K];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        outputMatrix[x * K + k] = matrixA[x] * matrixB[k];
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < X * K; i++) {
        acc = gold_fma(outputMatrix[i], outputMatrix[i], acc);
      }

      acc = std::min(1.0f / std::sqrt(static_cast<float>(acc)), 1.0f);
      for (int i = 0; i < X * K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    for (int i = 0; i < X * K; i++) {
      matrixC[i] = outputMatrix[i];
    }
  } else if (params.SOFTMAX_GRAD) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int Y = params.loops[0][params.inputYLoopIndex[0]] *
            params.loops[1][params.inputYLoopIndex[1]];

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        ACC_T acc = 0;
        for (int k = 0; k < Y; k++) {
          ACC_T prob_kj;
          prob_kj = -residualMatrix[x * Y + k] * residualMatrix[x * Y + y];
          if (k == y) {
            prob_kj += residualMatrix[x * Y + y];
          }
          prob_kj *= matrixA[x * Y + k];
          acc += prob_kj;
        }
        matrixC[x * Y + y] = static_cast<float>(acc) / sqrt(32);
      }
    }
  } else if (params.NO_NORM_GRAD) {
    // elementwise multiplication and addition of matrices
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

    ACC_T outputMatrix[K];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int i = 0; i < X; i++) {
      for (int j = 0; j < K; j++) {
        ACC_T a = matrixA[i * K + j];
        ACC_T b = matrixB[i * K + j];

        if (inputScaling) {
          a *= static_cast<ACC_T>(matrixA[X * K + j]);
        }

        if (weightScaling) {
          b *= static_cast<ACC_T>(matrixB[X * K + j]);
        }

        outputMatrix[j] = gold_fma(a, b, outputMatrix[j]);
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < K; i++) {
        acc = gold_fma(outputMatrix[i], outputMatrix[i], acc);
      }

      acc = std::min(1.0f / std::sqrt(static_cast<float>(acc)), 1.0f);
      for (int i = 0; i < K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    for (int i = 0; i < K; i++) {
      matrixC[i] = outputMatrix[i];
    }
    // Cross Entropy Loss
  } else if (params.BIAS_GRAD) {
    int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

    T inputMatrixB[C * K];
    memcpy(inputMatrixB, matrixB, sizeof(inputMatrixB));

    if (params.CONCAT_WEIGHT) {
      for (int i = 0; i < C; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < K / 4; k++) {
            inputMatrixB[i * K + j * K / 4 + k] =
                matrixB[(i + j * C) * K / 4 + k];
          }
        }
      }
    }

    ACC_T outputMatrix[K];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int i = 0; i < K; i++) {
      for (int j = 0; j < C; j++) {
        ACC_T b = inputMatrixB[j * K + i];

        if (inputScaling) {
          b *= static_cast<ACC_T>(matrixB[C * K + i]);
        }

        outputMatrix[i] += b;
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < K; i++) {
        acc = gold_fma(outputMatrix[i], outputMatrix[i], acc);
      }

      acc = std::min(1.0f / std::sqrt(static_cast<float>(acc)), 1.0f);
      for (int i = 0; i < K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    for (int i = 0; i < K; i++) {
      matrixC[i] = outputMatrix[i];
    }
  } else if (params.CROSS_ENTROPY_GRAD) {
    // matrix A: one-hot encoded targets
    // matrix B: output logits
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];

    ACC_T outputMatrix[X];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    for (int i = 0; i < X; i++) {
      outputMatrix[i] = matrixB[i];
      if (inputScaling) {
        outputMatrix[i] *= static_cast<ACC_T>(matrixB[X + i]);
      }
    }

    ACC_T max = 0;
    for (int i = 0; i < X; i++) {
      max = outputMatrix[i] > max ? outputMatrix[i] : max;
    }

    ACC_T sum = 0;
    for (int i = 0; i < X; i++) {
      outputMatrix[i] = exp(static_cast<float>(outputMatrix[i] - max));
      sum += outputMatrix[i];
    }

    for (int i = 0; i < X; i++) {
      matrixC[i] = outputMatrix[i] / sum - static_cast<ACC_T>(matrixA[i]);
    }
  } else if (params.MSE_GRAD) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];

    ACC_T divisor = 2 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<ACC_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else if (params.BCE_WITH_LOGITS_GRAD) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];

    ACC_T divisor = 1 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<ACC_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else {  // normal operation
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
    ACC_T learningRate = params.learningRate;

    if (params.REPLICATION) {
      FX = 7;
      C = 3;
    }

    const int rows = inputScaling ? X + 1 : X;

    T inputMatrixA[(STRIDE * X) * (STRIDE * Y) * C];
    T inputMatrixB[FX * FY * C * K];
    T gradientMatrixB[FX * FY * C * K];
    ACC_T outputMatrix[rows * Y * K];

    memcpy(inputMatrixA, matrixA, sizeof(inputMatrixA));
    memcpy(inputMatrixB, matrixB, sizeof(inputMatrixB));
    memset(outputMatrix, 0, sizeof(outputMatrix));

    if (params.CONCAT_INPUT) {
      T tmpMatrixA[rows * C];
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < 32; k++) {
            tmpMatrixA[i * 128 + j * 32 + k] =
                inputMatrixA[(j * rows + i) * 32 + k];
          }
        }
      }
      memcpy(inputMatrixA, tmpMatrixA, sizeof(tmpMatrixA));
    }

    if (params.INPUT_TRANSPOSE) {
      T tmpMatrixA[X * C];
      for (int x = 0; x < X; x++) {
        for (int c = 0; c < C; c++) {
          tmpMatrixA[x * C + c] = inputMatrixA[c * X + x];
        }
      }
      memcpy(inputMatrixA, tmpMatrixA, sizeof(tmpMatrixA));
    }

    if (params.CONCAT_WEIGHT) {
      T tmpMatrixB[C * K];
      for (int i = 0; i < C; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < 32; k++) {
            tmpMatrixB[i * 128 + j * 32 + k] =
                inputMatrixB[(j * C + i) * 32 + k];
          }
        }
      }
      memcpy(inputMatrixB, tmpMatrixB, sizeof(tmpMatrixB));
    }

    if (params.WEIGHT_TRANSPOSE) {
      T tmpMatrixB[C * K];
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          tmpMatrixB[c * K + k] = inputMatrixB[k * C + c];
        }
      }
      memcpy(inputMatrixB, tmpMatrixB, sizeof(tmpMatrixB));
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

                  T a = inputMatrixA[(STRIDE * y + fy) * STRIDE * X * C +
                                     (STRIDE * x + fx) * C + c];
                  T b = inputMatrixB[(fy + (FY - 1) / 2) * FX * C * K +
                                     (fx + (FX - 1) / 2) * C * K + c * K + k];

                  acc = gold_fma(a, b, acc);

                  if (params.WEIGHT_SPLITTING) {
                    b = weightGradMatrix[(fy + (FY - 1) / 2) * FX * C * K +
                                         (fx + (FX - 1) / 2) * C * K + c * K +
                                         k];
                    ACC_T grad = a * b;
                    acc += learningRate * grad;
                  }

                  if (inputScaling) {
                    acc *= static_cast<ACC_T>(params.INPUT_TRANSPOSE
                                                  ? matrixA[X * K + x]
                                                  : matrixA[X * K + k]);
                    if (!params.WEIGHT) {
                      acc *= static_cast<ACC_T>(params.WEIGHT_TRANSPOSE
                                                    ? matrixB[C * K + c]
                                                    : matrixB[C * K + k]);
                    }
                  }
                }
              }
            }
          }

          if (weightScaling) {
            acc *= static_cast<ACC_T>(matrixB[C * K]);
          }

          if (params.BIAS) {
            acc += biasMatrix[k];

            if (params.WEIGHT_SPLITTING) {
              acc += learningRate * biasGradMatrix[k];
            }
          }

          if (params.RELU) {
#ifdef POSIT
            gold_relu(acc);
#else
            acc = std::max(0, (int)acc);
#endif
          }

          if (params.RELU_GRAD) {
            ACC_T residual = residualMatrix[y * X * K + x * K + k];
            if (residual <= 0) acc = 0;
          }

          if (params.ATTENTION_SCALING) {
            ACC_T scale = static_cast<ACC_T>(static_cast<T>(1.0 / sqrt(32)));
            acc *= scale;
          }

          outputMatrix[y * X * K + x * K + k] = acc;
          if (inputScaling) {
            outputMatrix[Y * X * K + X * K + k] += abs(static_cast<float>(acc));
          }
        }
      }
    }

    if (params.GRAD_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < X * K; i++) {
        acc = gold_fma(outputMatrix[i], outputMatrix[i], acc);
      }

      // TODO: implement posit reciprocal square root
      acc = std::min(1.0f / std::sqrt(static_cast<float>(acc)), 1.0f);
      for (int i = 0; i < X * K; i++) {
        outputMatrix[i] *= acc;
      }
    }

    if (params.RESIDUAL) {
      for (int i = 0; i < X; i++) {
        for (int j = 0; j < K; j++) {
          ACC_T residual = residualMatrix[i * K + j];
          if (inputScaling) {
            residual *= static_cast<ACC_T>(residualMatrix[X * K + j]);
          }
          outputMatrix[i * K + j] += residual;
        }
      }
    }

    for (int k = 0; k < K; k++) {
      if (inputScaling) {
        float sum = static_cast<float>(outputMatrix[X * K + k]);
        ACC_T scalingFactor =
            (sum && !params.RELU) ? pow(2, round(log2(sum / X))) : 1;
        matrixC[X * K + k] = scalingFactor;
        for (int x = 0; x < X; x++) {
          matrixC[x * K + k] = outputMatrix[x * K + k] / scalingFactor;
        }
      } else {
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            matrixC[y * X * K + x * K + k] =
                outputMatrix[y * X * K + x * K + k];
          }
        }
      }
    }

    if (params.SPLIT_OUTPUT) {
      T tmpMatrixC[rows * K];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < rows; j++) {
          for (int k = 0; k < 32; k++) {
            tmpMatrixC[(i * rows + j) * 32 + k] = matrixC[j * 128 + i * 32 + k];
          }
        }
      }
      memcpy(matrixC, tmpMatrixC, sizeof(tmpMatrixC));
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
    INPUT_DATATYPE *residualWeightMatrix, INPUT_DATATYPE *residualBiasMatrix,
    bool inputScaling, bool weightScaling) {
  run_gold_op<INPUT_DATATYPE, ACCUM_DATATYPE>(
      params, matrixA, matrixB, matrixC,
      reinterpret_cast<ACCUM_DATATYPE *>(biasMatrix), residualMatrix,
      residualWeightMatrix,
      reinterpret_cast<ACCUM_DATATYPE *>(residualBiasMatrix), inputScaling,
      weightScaling);
}

void run_universal_posit_gold_model(
    const SimplifiedParams params, UniversalPosit *matrixA,
    UniversalPosit *matrixB, UniversalPosit *matrixC,
    UniversalPosit *biasMatrix, UniversalPosit *residualMatrix,
    UniversalPosit *residualWeightMatrix, UniversalPosit *residualBiasMatrix,
    bool inputScaling, bool weightScaling) {
  run_gold_op<UniversalPosit, UniversalPositAccum>(
      params, matrixA, matrixB, matrixC,
      reinterpret_cast<UniversalPositAccum *>(biasMatrix), residualMatrix,
      residualWeightMatrix,
      reinterpret_cast<UniversalPositAccum *>(residualBiasMatrix), inputScaling,
      weightScaling);
}

void run_fp_gold_model(const SimplifiedParams params, float *matrixA,
                       float *matrixB, float *matrixC, float *biasMatrix,
                       float *residualMatrix, float *residualWeightMatrix,
                       float *residualBiasMatrix, bool inputScaling,
                       bool weightScaling) {
  run_gold_op<float, float>(params, matrixA, matrixB, matrixC, biasMatrix,
                            residualMatrix, residualWeightMatrix,
                            residualBiasMatrix, inputScaling, weightScaling);
}
