#pragma once

#include "src/ArchitectureParams.h"
#include "src/PositTypes.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"

void run_custom_posit_gold_model(const SimplifiedParams params,
                                 INPUT_DATATYPE *matrixA,
                                 INPUT_DATATYPE *matrixB,
                                 INPUT_DATATYPE *matrixC,
                                 INPUT_DATATYPE *biasMatrix,
                                 INPUT_DATATYPE *residualMatrix);

void run_universal_posit_gold_model(const SimplifiedParams params,
                                    UniversalPosit *matrixA,
                                    UniversalPosit *matrixB,
                                    UniversalPosit *matrixC,
                                    UniversalPosit *biasMatrix,
                                    UniversalPosit *residualMatrix);

void run_fp_gold_model(const SimplifiedParams params, float *matrixA,
                       float *matrixB, float *matrixC, float *biasMatrix,
                       float *residualMatrix);

// Performs a*b + c
inline UniversalPositAccum gold_fma(UniversalPosit a, UniversalPosit b,
                                    UniversalPositAccum c) {
  UniversalPositAccum tmp;
  sw::universal::value<15> internal = sw::universal::fma<8, 1>(a, b, 0);
  sw::universal::convert<16, 1, 15>(internal, tmp);
  tmp += c;
  return tmp;
}

inline ACCUM_DATATYPE gold_fma(INPUT_DATATYPE a, INPUT_DATATYPE b,
                               ACCUM_DATATYPE c) {
  return fma(a, b, c);
}

inline float gold_fma(float a, float b, float c) { return a * b + c; }

inline void gold_relu(UniversalPositAccum &a) {
  if (a < 0) {
    a = 0;
  }
}

inline void gold_relu(ACCUM_DATATYPE &a) { a.relu(); }

inline void gold_relu(float &a) {
  if (a < 0.0f) {
    a = 0.0f;
  }
}

inline void gold_exp(INPUT_DATATYPE &a) { a.exp(); }
inline void gold_exp(float &a) { a = exp(a); }
inline void gold_exp(UniversalPosit &a) {
  // TODO
}

inline void gold_reciprocal(ACCUM_DATATYPE &a) { a.reciprocal(); }
inline void gold_reciprocal(float &a) { a = 1.0 / a; }
inline void gold_reciprocal(UniversalPositAccum &a) { a = 1 / a; }

template <typename T, typename ACC_T>
void run_gold_op(const SimplifiedParams params, T *matrixA, T *matrixB,
                 T *matrixC, T *biasMatrix, T *residualMatrix) {
  std::cout << "Running gold model " << std::endl;

  if (params.SOFTMAX) {
    // 2D softmax
    for (int i = 0; i < params.loops[1][params.inputYLoopIndex[1]]; i++) {
      // find max value
      ACC_T max = 0;
      for (int j = 0; j < params.loops[1][params.inputXLoopIndex[1]]; j++) {
        int index = i * params.loops[1][params.inputXLoopIndex[1]] + j;

        if (static_cast<ACC_T>(matrixA[index]) > max) {
          max = static_cast<ACC_T>(matrixA[index]);
        }
      }

      // subtract max, exp, and sum
      ACC_T sum = 0;
      for (int j = 0; j < params.loops[1][params.inputXLoopIndex[1]]; j++) {
        int index = i * params.loops[1][params.inputXLoopIndex[1]] + j;
        matrixC[index] = static_cast<ACC_T>(matrixA[index]) - max;
        gold_exp(matrixC[index]);
        sum += matrixC[index];
      }

      gold_reciprocal(sum);

      // div
      for (int j = 0; j < params.loops[1][params.inputXLoopIndex[1]]; j++) {
        int index = i * params.loops[1][params.inputXLoopIndex[1]] + j;
        matrixC[index] = static_cast<ACC_T>(matrixC[index]) * sum;
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
        std::cout << a << " ";
      }
      std::cout << " * ";
      for (int c = 0; c < C; c++) {
        ACC_T b = matrixB[c * K + k];
        std::cout << b << " ";
      }
      std::cout << std::endl;

      for (int c = 0; c < C; c++) {
        ACC_T a = matrixA[c];
        ACC_T b = matrixB[k * C + c];

        acc = gold_fma(a, b, acc);
      }

      if (params.BIAS) {
        acc += biasMatrix[k];
      }

      matrixC[k] = acc;
    }
  } else if (params.NO_NORM) {
    // elementwise multiplication and addition of matrices
    int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
    int Y = params.loops[0][params.inputYLoopIndex[0]] *
            params.loops[1][params.inputYLoopIndex[1]];
    for (int y = 0; y < Y; y++) {
      for (int c = 0; c < C; c++) {
        ACC_T a = matrixA[y * C + c];
        ACC_T b = matrixB[c];
        ACC_T bias = biasMatrix[c];
        ACC_T acc = gold_fma(a, b, bias);

        matrixC[y * C + c] = acc;
      }
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

    if (params.TRANSPOSE) {
      std::cout << "Before Transpose:" << std::endl;
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          std::cout << matrixB[c * K + k] << " ";
        }
        std::cout << std::endl;
      }
      // create copy
      T *tmpMatrixB = new T[C * K];
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          tmpMatrixB[c * K + k] = matrixB[k * C + c];
        }
      }
      memcpy(matrixB, tmpMatrixB, sizeof(T) * C * K);

      std::cout << "After Transpose:" << std::endl;
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          std::cout << matrixB[c * K + k] << " ";
        }
        std::cout << std::endl;
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

                  T a = matrixA[(STRIDE * y + fy) * STRIDE * X * C +
                                (STRIDE * x + fx) * C + c];
                  T b = matrixB[(fy + (FY - 1) / 2) * FX * C * K +
                                (fx + (FX - 1) / 2) * C * K + c * K + k];

                  acc = gold_fma(a, b, acc);
                }
              }
            }
          }
          if (params.BIAS) {
            acc += biasMatrix[k];
          }

          if (params.RESIDUAL) {
            acc += residualMatrix[y * X * K + x * K + k];
          }

          if (params.RELU) {
#ifdef POSIT
            gold_relu(acc);
#else
            acc = std::max(0, (int)acc);
#endif
          }

          matrixC[y * X * K + x * K + k] = acc;
        }
      }
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
