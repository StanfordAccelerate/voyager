#include "test/common/GoldModel.h"

#include <algorithm>
#include <cstring>

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
                 T *matrixC, T *biasMatrix, T *residualMatrix) {
  std::cout << "Running gold model " << std::endl;

  if (params.SOFTMAX) {
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int Y = params.loops[0][params.inputYLoopIndex[0]] *
            params.loops[1][params.inputYLoopIndex[1]];
    ACC_T scaledMatrixA[X * Y];
    ACC_T columnSums[Y];
    memset(columnSums, 0, sizeof(columnSums));
    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        scaledMatrixA[x * Y + y] = matrixA[x * Y + y];
#ifdef INPUT_SCALING
        scaledMatrixA[x * Y + y] *= static_cast<ACC_T>(matrixA[X * Y + y]);
#endif
      }
    }

    // 2D softmax
    for (int x = 0; x < X; x++) {
      ACC_T max = 0;
      for (int y = 0; y < Y; y++) {
        int index = x * Y + y;
        if (scaledMatrixA[index] > max) {
          max = scaledMatrixA[index];
        }
      }

      ACC_T sum = 0;
      for (int y = 0; y < Y; y++) {
        int index = x * Y + y;
        scaledMatrixA[index] =
            exp(static_cast<float>(scaledMatrixA[index] - max));
        sum += scaledMatrixA[index];
      }

      for (int y = 0; y < Y; y++) {
        int index = x * Y + y;
#ifdef INPUT_SCALING
        scaledMatrixA[index] /= sum;
        columnSums[y] += scaledMatrixA[index];
#else
        matrixC[index] = scaledMatrixA[index] / sum;
#endif
      }
    }
#ifdef INPUT_SCALING
    for (int y = 0; y < Y; y++) {
      float sum = static_cast<float>(columnSums[y]);
      ACC_T scalingFactor = sum ? pow(2, round(log2(sum / X))) : 1;
      matrixC[X * Y + y] = scalingFactor;
      for (int x = 0; x < X; x++) {
        matrixC[x * Y + y] = scaledMatrixA[x * Y + y] / scalingFactor;
      }
    }
#endif
  } else if (params.FC) {
    // fully connected layer (matrix-vector)
    int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
    // for (int c = 0; c < C; c++) {
    //   ACC_T a = matrixA[c];
    //   std::cout << a << " ";
    // }
    std::cout << std::endl << std::endl;
    for (int k = 0; k < K; k++) {
      ACC_T acc = 0;
      // for (int c = 0; c < C; c++) {
      //   ACC_T b = matrixB[k * C + c];
      //   std::cout << b << " ";
      // }
      // std::cout << std::endl;

      for (int c = 0; c < C; c++) {
        ACC_T a = matrixA[c];
        ACC_T b = matrixB[k * C + c];
#ifdef INPUT_SCALING
        a *= static_cast<ACC_T>(matrixA[C + c]);
#endif
        acc = gold_fma(a, b, acc);
      }
#ifdef WEIGHT_SCALING
      acc *= static_cast<ACC_T>(matrixB[C * K]);
#endif

      if (params.BIAS) {
        ACC_T bias = biasMatrix[k];
#ifdef WEIGHT_SCALING
        bias *= static_cast<ACC_T>(biasMatrix[K]);
#endif
        acc += bias;
      }

      matrixC[k] = acc;
    }
  } else if (params.NO_NORM) {
    // elementwise multiplication and addition of matrices
    int X = params.loops[0][params.inputXLoopIndex[0]] *
            params.loops[1][params.inputXLoopIndex[1]];
    int K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

#ifdef INPUT_SCALING
    ACC_T unscaledMatrix[X * K];
    ACC_T columnSums[K];
    memset(columnSums, 0, sizeof(columnSums));
#endif

    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        ACC_T a = matrixA[x * K + k];
        ACC_T b = matrixB[k];
        ACC_T bias = biasMatrix[k];
#ifdef WEIGHT_SCALING
        b *= static_cast<ACC_T>(matrixB[K]);
        bias *= static_cast<ACC_T>(biasMatrix[K]);
#endif
#ifdef INPUT_SCALING
        a *= static_cast<ACC_T>(matrixA[X * K + k]);
        unscaledMatrix[x * K + k] = gold_fma(a, b, bias);
        columnSums[k] += abs(static_cast<float>(unscaledMatrix[x * K + k]));
#else
        matrixC[x * K + k] = gold_fma(a, b, bias);
#endif
      }
    }
#ifdef INPUT_SCALING
    for (int k = 0; k < K; k++) {
      float sum = static_cast<float>(columnSums[k]);
      ACC_T scalingFactor = sum ? pow(2, round(log2(sum / X))) : 1;
      matrixC[X * K + k] = scalingFactor;
      for (int x = 0; x < X; x++) {
        matrixC[x * K + k] = unscaledMatrix[x * K + k] / scalingFactor;
      }
    }
#endif
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

#ifdef INPUT_SCALING
    ACC_T unscaledMatrix[X * Y * K];
    ACC_T columnSums[K];
    memset(columnSums, 0, sizeof(columnSums));
#endif

    if (params.CONCAT_HEAD) {
      T tmpMatrixA[X * C + C];
      int addr = 0;
      for (int i = 0; i <= 128; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < 32; k++) {
#ifdef INPUT_SCALING
            tmpMatrixA[addr++] = matrixA[(j * 129 + i) * 32 + k];
#else
            tmpMatrixA[addr++] = matrixA[(j * 128 + i) * 32 + k];
#endif
          }
        }
      }
      memcpy(matrixA, tmpMatrixA, (X * C + C) * sizeof(T));
    }

    if (params.INPUT_TRANSPOSE) {
      T tmpMatrixA[X * C];
      for (int x = 0; x < X; x++) {
        for (int c = 0; c < C; c++) {
          tmpMatrixA[x * C + c] = matrixA[c * X + x];
        }
      }
      memcpy(matrixA, tmpMatrixA, sizeof(T) * X * C);
    }

    if (params.TRANSPOSE) {
      T tmpMatrixB[C * K];
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          tmpMatrixB[c * K + k] = matrixB[k * C + c];
        }
      }
      memcpy(matrixB, tmpMatrixB, sizeof(T) * C * K);
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

                  T a = matrixA[(STRIDE * y + fy) * STRIDE * X * C +
                                (STRIDE * x + fx) * C + c];
                  T b = matrixB[(fy + (FY - 1) / 2) * FX * C * K +
                                (fx + (FX - 1) / 2) * C * K + c * K + k];
#ifdef INPUT_SCALING
                  a *= static_cast<ACC_T>(matrixA[X * C + c]);
                  if (!params.WEIGHT) {
                    b *= params.TRANSPOSE
                             ? static_cast<ACC_T>(matrixB[C * K + c])
                             : static_cast<ACC_T>(matrixB[C * K + k]);
                  }
#endif
                  acc = gold_fma(a, b, acc);
                }
              }
            }
          }
#ifdef WEIGHT_SCALING
          if (params.WEIGHT) {
            acc *= static_cast<ACC_T>(matrixB[C * K]);
          }
#endif
          if (params.BIAS) {
            ACC_T bias = biasMatrix[k];
#ifdef WEIGHT_SCALING
            bias *= static_cast<ACC_T>(biasMatrix[K]);
#endif
            acc += bias;
          }

          if (params.RESIDUAL) {
            ACC_T residual = residualMatrix[y * X * K + x * K + k];
#ifdef INPUT_SCALING
            residual *= static_cast<ACC_T>(residualMatrix[X * K + k]);
#endif
            acc += residual;
          }

          if (params.RELU) {
#ifdef POSIT
            gold_relu(acc);
#else
            acc = std::max(0, (int)acc);
#endif
          }

          if (params.ATTENTION_SCALING) {
            acc = static_cast<float>(acc) / sqrt(32);
          }
#ifdef INPUT_SCALING
          columnSums[k] += abs(static_cast<float>(acc));
          unscaledMatrix[y * X * K + x * K + k] = acc;
#else
          matrixC[y * X * K + x * K + k] = acc;
#endif
        }
      }
    }
#ifdef INPUT_SCALING
    for (int k = 0; k < K; k++) {
      float sum = static_cast<float>(columnSums[k]);
      ACC_T scalingFactor = sum ? pow(2, round(log2(sum / X))) : 1;
      matrixC[X * K + k] = scalingFactor;
      for (int x = 0; x < X; x++) {
        matrixC[x * K + k] = unscaledMatrix[x * K + k] / scalingFactor;
      }
    }
#endif

    if (params.SPLIT_HEAD) {
      T tmpMatrixC[X * K + K];
      int addr = 0;
      for (int i = 0; i < 4; i++) {
#ifdef INPUT_SCALING
        for (int j = 0; j <= 128; j++) {
#else
        for (int j = 0; j < 128; j++) {
#endif
          for (int k = 0; k < 32; k++) {
            tmpMatrixC[addr++] = matrixC[j * 128 + i * 32 + k];
          }
        }
      }
      memcpy(matrixC, tmpMatrixC, (X * K + K) * sizeof(T));
    }

    if (params.MAXPOOL) {
      // create copy
      T *tmpMatrixC = new T[X * Y * K];
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

      delete[] tmpMatrixC;
    }

    if (params.AVGPOOL) {
      // create copy
      T *tmpMatrixC = new T[X * Y * K];
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

      delete[] tmpMatrixC;
    }
  }
}

void run_custom_posit_gold_model(const SimplifiedParams params,
                                 INPUT_DATATYPE *matrixA,
                                 INPUT_DATATYPE *matrixB,
                                 INPUT_DATATYPE *matrixC,
                                 INPUT_DATATYPE *biasMatrix,
                                 INPUT_DATATYPE *residualMatrix) {
  run_gold_op<INPUT_DATATYPE, ACCUM_DATATYPE>(params, matrixA, matrixB, matrixC,
                                              biasMatrix, residualMatrix);
}

void run_universal_posit_gold_model(const SimplifiedParams params,
                                    UniversalPosit *matrixA,
                                    UniversalPosit *matrixB,
                                    UniversalPosit *matrixC,
                                    UniversalPosit *biasMatrix,
                                    UniversalPosit *residualMatrix) {
  run_gold_op<UniversalPosit, UniversalPositAccum>(
      params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix);
}

void run_fp_gold_model(const SimplifiedParams params, float *matrixA,
                       float *matrixB, float *matrixC, float *biasMatrix,
                       float *residualMatrix) {
  run_gold_op<float, float>(params, matrixA, matrixB, matrixC, biasMatrix,
                            residualMatrix);
}
