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

inline void gold_exp(INPUT_DATATYPE &a) { posit_exp(a); }
inline void gold_exp(float &a) { a = exp(a); }

#ifndef NO_UNIVERSAL
inline void gold_exp(UniversalPosit &a) {
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
                 bool inputScaling, bool weightScaling) {
#ifndef PIPE_INPUT
  std::cout << "Running gold model " << std::endl;
#endif

  if (params.SOFTMAX) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int Y = params.loops[0][params.inputYLoopIndex[0]] *
            params.loops[1][params.inputYLoopIndex[1]];

    const int numRows = inputScaling ? X + 1 : X;
    ACC_T accumMatrix[numRows * Y];
    memset(accumMatrix, 0, sizeof(accumMatrix));

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        accumMatrix[x * Y + y] = matrixA[x * Y + y];
        if (inputScaling) {
          accumMatrix[x * Y + y] *= static_cast<ACC_T>(matrixA[X * Y + y]);
        }
      }
    }

    for (int x = 0; x < X; x++) {
      ACC_T max = 0;
      ACC_T sum = 0;

      for (int y = 0; y < Y; y++) {
        if (accumMatrix[x * Y + y] > max) {
          max = accumMatrix[x * Y + y];
        }
      }

      for (int y = 0; y < Y; y++) {
        accumMatrix[x * Y + y] =
            exp(static_cast<float>(accumMatrix[x * Y + y] - max));
        sum += accumMatrix[x * Y + y];
      }

      for (int y = 0; y < Y; y++) {
        accumMatrix[x * Y + y] /= sum;
        if (inputScaling) {
          accumMatrix[X * Y + y] += accumMatrix[x * Y + y];
        }
      }
    }

    for (int y = 0; y < Y; y++) {
      if (inputScaling) {
        float sum = static_cast<float>(accumMatrix[X * Y + y]);
        ACC_T scalingFactor = sum ? pow(2, round(log2(sum / X))) : 1;
        matrixC[X * Y + y] = scalingFactor;
        for (int x = 0; x < X; x++) {
          matrixC[x * Y + y] = accumMatrix[x * Y + y] / scalingFactor;
        }
      } else {
        for (int x = 0; x < X; x++) {
          matrixC[x * Y + y] = accumMatrix[x * Y + y];
        }
      }
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
  } else if (params.FC) {
    // fully connected layer (matrix-vector)
    int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

    for (int k = 0; k < K; k++) {
      ACC_T acc = 0;
      for (int c = 0; c < C; c++) {
        ACC_T a = matrixA[c];
        ACC_T b = matrixB[k * C + c];
        acc = gold_fma(a, b, acc);

        if (inputScaling) {
          acc *= static_cast<ACC_T>(matrixA[C + c]);
        }
      }

      if (weightScaling) {
        acc *= static_cast<ACC_T>(matrixB[C * K]);
      }

      if (params.BIAS) {
        acc += biasMatrix[k];
      }

      matrixC[k] = acc;
    }
  } else if (params.NO_NORM) {
    // elementwise multiplication and addition between a matrix and a vector
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

    ACC_T accumMatrix[X * K + K];
    memset(accumMatrix, 0, sizeof(accumMatrix));

    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        ACC_T a = matrixA[x * K + k];
        ACC_T b = matrixB[k];
        ACC_T acc;

        if (params.BIAS) {
          ACC_T bias = biasMatrix[k];
          acc = gold_fma(a, b, bias);
        } else {
          acc = a * b;
        }

        if (inputScaling) {
          acc *= static_cast<ACC_T>(matrixA[X * K + k]);
        }

        if (weightScaling) {
          acc *= static_cast<ACC_T>(matrixB[K]);
        }

        accumMatrix[x * K + k] = acc;
        accumMatrix[X * K + k] += abs(static_cast<float>(acc));
      }
    }

    for (int k = 0; k < K; k++) {
      if (inputScaling) {
        float sum = static_cast<float>(accumMatrix[X * K + k]);
        ACC_T scalingFactor = sum ? pow(2, round(log2(sum / X))) : 1;
        matrixC[X * K + k] = scalingFactor;
        for (int x = 0; x < X; x++) {
          matrixC[x * K + k] = accumMatrix[x * K + k] / scalingFactor;
        }
      } else {
        for (int x = 0; x < X; x++) {
          matrixC[x * K + k] = accumMatrix[x * K + k];
        }
      }
    }
  } else if (params.NO_NORM_GRAD) {
    // elementwise multiplication and addition of matrices
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

    ACC_T accumMatrix[K];
    memset(accumMatrix, 0, sizeof(accumMatrix));

    for (int i = 0; i < X; i++) {
      for (int j = 0; j < K; j++) {
        ACC_T a = matrixA[i * K + j];
        ACC_T b = matrixB[i * K + j];
        ACC_T acc = a * b;

        if (inputScaling) {
          acc *= static_cast<ACC_T>(matrixA[X * K + j]);
        }

        if (weightScaling) {
          acc *= static_cast<ACC_T>(matrixB[X * K + j]);
        }

        accumMatrix[j] += acc;
      }
    }

    for (int i = 0; i < K; i++) {
      matrixC[i] = accumMatrix[i];
    }
    // Cross Entropy Loss
  } else if (params.CROSS_ENTROPY_LOSS_GRAD) {
    // matrix A: logits
    // matrix B: one-hot encoded targets
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];

    ACC_T accumMatrix[X];
    memset(accumMatrix, 0, sizeof(accumMatrix));

    for (int i = 0; i < X; i++) {
      accumMatrix[i] = matrixA[i];
      if (inputScaling) {
        accumMatrix[i] *= static_cast<ACC_T>(matrixA[X + i]);
      }
    }

    ACC_T max = 0;
    ACC_T sum = 0;

    for (int i = 0; i < X; i++) {
      if (accumMatrix[i] > max) {
        max = accumMatrix[i];
      }
    }

    for (int i = 0; i < X; i++) {
      accumMatrix[i] = exp(static_cast<float>(accumMatrix[i] - max));
      sum += accumMatrix[i];
    }

    for (int i = 0; i < X; i++) {
      accumMatrix[i] /= sum;
      accumMatrix[i] -= matrixB[i];
      matrixC[i] = accumMatrix[i];
    }
  } else if (params.MSE_LOSS_GRAD) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];

    ACC_T divisor = 2 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<ACC_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else if (params.BCE_LOGITS_LOSS_GRAD) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];

    ACC_T divisor = 1 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<ACC_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else if (params.BIAS_GRAD) {
    int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

    ACC_T accumMatrix[K];
    memset(accumMatrix, 0, sizeof(accumMatrix));

    T permuteMatrixB[C * K];
    memcpy(permuteMatrixB, matrixB, sizeof(permuteMatrixB));

    // TODO: Add input/weight tranpose/permute to all operations
    if (params.WEIGHT_PERMUTE) {
      for (int i = 0; i < C; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < K / 4; k++) {
            permuteMatrixB[i * K + j * K / 4 + k] =
                matrixB[(i + j * C) * K / 4 + k];
          }
        }
      }
    }

    for (int i = 0; i < K; i++) {
      for (int j = 0; j < C; j++) {
        ACC_T acc = permuteMatrixB[j * K + i];

        if (inputScaling) {
          acc *= static_cast<ACC_T>(matrixB[C * K + i]);
        }

        accumMatrix[i] += acc;
      }
    }

    for (int i = 0; i < K; i++) {
      matrixC[i] = accumMatrix[i];
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

    if (params.REPLICATION) {
      FX = 7;
      C = 3;
    }

    const int numRows = inputScaling ? X + 1 : X;
    ACC_T accumMatrix[numRows * K];
    memset(accumMatrix, 0, sizeof(accumMatrix));

    T permuteMatrixA[(STRIDE * X) * (STRIDE * Y) * C];
    T permuteMatrixB[FX * FY * C * K];

    memcpy(permuteMatrixA, matrixA, sizeof(permuteMatrixA));
    memcpy(permuteMatrixB, matrixB, sizeof(permuteMatrixB));

    if (params.CONCAT_HEAD) {
      T tmpMatrixA[numRows * C];
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < 32; k++) {
            tmpMatrixA[i * 128 + j * 32 + k] =
                permuteMatrixA[(j * numRows + i) * 32 + k];
          }
        }
      }
      memcpy(permuteMatrixA, tmpMatrixA, sizeof(tmpMatrixA));
    }

    if (params.INPUT_TRANSPOSE) {
      T tmpMatrixA[X * C];
      for (int x = 0; x < X; x++) {
        for (int c = 0; c < C; c++) {
          tmpMatrixA[x * C + c] = permuteMatrixA[c * X + x];
        }
      }
      memcpy(permuteMatrixA, tmpMatrixA, sizeof(tmpMatrixA));
    }

    if (params.WEIGHT_PERMUTE) {
      T tmpMatrixB[C * K];
      for (int i = 0; i < C; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < 32; k++) {
            tmpMatrixB[i * 128 + j * 32 + k] =
                permuteMatrixB[(j * C + i) * 32 + k];
          }
        }
      }
      memcpy(permuteMatrixB, tmpMatrixB, sizeof(tmpMatrixB));
    }

    if (params.TRANSPOSE) {
      T tmpMatrixB[C * K];
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          tmpMatrixB[c * K + k] = permuteMatrixB[k * C + c];
        }
      }
      memcpy(permuteMatrixB, tmpMatrixB, sizeof(tmpMatrixB));
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

                  T a = permuteMatrixA[(STRIDE * y + fy) * STRIDE * X * C +
                                       (STRIDE * x + fx) * C + c];
                  T b = permuteMatrixB[(fy + (FY - 1) / 2) * FX * C * K +
                                       (fx + (FX - 1) / 2) * C * K + c * K + k];
                  acc = gold_fma(a, b, acc);

                  if (inputScaling) {
                    acc *= static_cast<ACC_T>(params.INPUT_TRANSPOSE
                                                  ? matrixA[X * K + x]
                                                  : matrixA[X * K + k]);
                    if (!params.WEIGHT) {
                      acc *= static_cast<ACC_T>(params.TRANSPOSE
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
          }

          if (params.RESIDUAL) {
            ACC_T residual = residualMatrix[y * X * K + x * K + k];
            if (inputScaling) {
              residual *= static_cast<ACC_T>(residualMatrix[X * K + k]);
            }
            acc += residual;
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

          accumMatrix[y * X * K + x * K + k] = acc;
          if (inputScaling) {
            accumMatrix[Y * X * K + X * K + k] += abs(static_cast<float>(acc));
          }
        }
      }
    }

    if (params.GRADIENT_CLIPPING) {
      ACC_T acc = 0;
      for (int i = 0; i < X; i++) {
        for (int j = 0; j < K; j++) {
          acc = gold_fma(accumMatrix[i * K + j], accumMatrix[i * K + j], acc);
        }
      }

      // TODO: implement posit square root
      acc = std::sqrt(static_cast<float>(acc));

      if (acc > 1) {
        for (int i = 0; i < X; i++) {
          for (int j = 0; j < K; j++) {
            accumMatrix[i * K + j] =
                static_cast<float>(accumMatrix[i * K + j]) /
                static_cast<float>(acc);
          }
        }
      }
    }

    for (int k = 0; k < K; k++) {
      if (inputScaling) {
        float sum = static_cast<float>(accumMatrix[X * K + k]);
        ACC_T scalingFactor =
            (sum && !params.RELU) ? pow(2, round(log2(sum / X))) : 1;
        matrixC[X * K + k] = scalingFactor;
        for (int x = 0; x < X; x++) {
          matrixC[x * K + k] = accumMatrix[x * K + k] / scalingFactor;
        }
      } else {
        for (int x = 0; x < X; x++) {
          matrixC[x * K + k] = accumMatrix[x * K + k];
        }
      }
    }

    if (params.SPLIT_HEAD) {
      T tmpMatrixC[numRows * K];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < numRows; j++) {
          for (int k = 0; k < 32; k++) {
            tmpMatrixC[(i * numRows + j) * 32 + k] =
                matrixC[j * 128 + i * 32 + k];
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

void run_custom_posit_gold_model(const SimplifiedParams params,
                                 INPUT_DATATYPE *matrixA,
                                 INPUT_DATATYPE *matrixB,
                                 INPUT_DATATYPE *matrixC,
                                 INPUT_DATATYPE *biasMatrix,
                                 INPUT_DATATYPE *residualMatrix,
                                 bool inputScaling, bool weightScaling) {
  run_gold_op<INPUT_DATATYPE, ACCUM_DATATYPE>(
      params, matrixA, matrixB, matrixC,
      reinterpret_cast<ACCUM_DATATYPE *>(biasMatrix), residualMatrix,
      inputScaling, weightScaling);
}

void run_universal_posit_gold_model(const SimplifiedParams params,
                                    UniversalPosit *matrixA,
                                    UniversalPosit *matrixB,
                                    UniversalPosit *matrixC,
                                    UniversalPosit *biasMatrix,
                                    UniversalPosit *residualMatrix,
                                    bool inputScaling, bool weightScaling) {
  run_gold_op<UniversalPosit, UniversalPositAccum>(
      params, matrixA, matrixB, matrixC,
      reinterpret_cast<UniversalPositAccum *>(biasMatrix), residualMatrix,
      inputScaling, weightScaling);
}

void run_fp_gold_model(const SimplifiedParams params, float *matrixA,
                       float *matrixB, float *matrixC, float *biasMatrix,
                       float *residualMatrix, bool inputScaling,
                       bool weightScaling) {
  run_gold_op<float, float>(params, matrixA, matrixB, matrixC, biasMatrix,
                            residualMatrix, inputScaling, weightScaling);
}
