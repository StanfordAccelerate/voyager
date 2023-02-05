#include "test/common/GoldModel.h"

#include <algorithm>
#include <cstring>
#include <vector>

#ifndef NO_UNIVERSAL
inline void gold_fma(UniversalPosit a, UniversalPosit b,
                     UniversalPositAccum &c) {
  UniversalPositAccum product;
  sw::universal::value<15> internal = sw::universal::fma<8, 1>(a, b, 0);
  sw::universal::convert<16, 1, 15>(internal, product);
  c += product;
}
#endif

inline void gold_fma(INPUT_DATATYPE a, INPUT_DATATYPE b,
                     ACCUM_DATATYPE::DecomposedPosit &c) {
  INPUT_DATATYPE::DecomposedPosit v1 = a;
  INPUT_DATATYPE::DecomposedPosit v2 = b;
  c = decomposed_fma<8, 1, 16, 1>(v1, v2, c);
}

inline void gold_fma(float a, float b, float &c) { c += a * b; }

#ifndef NO_UNIVERSAL
inline void gold_relu(UniversalPositAccum &a) { a = a < 0 ? 0 : a; }
#endif
inline void gold_relu(ACCUM_DATATYPE::DecomposedPosit &a) { a.relu(); }
inline void gold_relu(float &a) { a = a < 0 ? 0 : a; }

#ifndef NO_UNIVERSAL
inline void gold_exp(UniversalPositAccum &a) { a = sw::universal::exp(a); }
#endif
inline void gold_exp(ACCUM_DATATYPE::DecomposedPosit &a) {
  a = static_cast<ACCUM_DATATYPE::DecomposedPosit>(
      posit_exp(static_cast<ACCUM_DATATYPE>(a)));
}
inline void gold_exp(float &a) { a = exp(a); }

#ifndef NO_UNIVERSAL
inline void gold_reciprocal(UniversalPositAccum &a) { a = a.reciprocate(); }
#endif
inline void gold_reciprocal(ACCUM_DATATYPE::DecomposedPosit &a) {
  ACCUM_DATATYPE tmp = static_cast<ACCUM_DATATYPE>(a);
  tmp.reciprocal();
  a = static_cast<ACCUM_DATATYPE::DecomposedPosit>(tmp);
}
inline void gold_reciprocal(float &a) { a = 1.0f / a; }

#ifndef NO_UNIVERSAL
inline void gold_inv_sqrt(UniversalPositAccum &a) {
  a = sw::universal::rsqrt(a);
}
#endif
inline void gold_inv_sqrt(ACCUM_DATATYPE::DecomposedPosit &a) {
  a = a.inv_sqrt();
}
inline void gold_inv_sqrt(float &a) { a = 1.0f / std::sqrt(a); }

#ifndef NO_UNIVERSAL
inline UniversalPositAccum readInput(UniversalPosit *matrix, int index,
                                     bool doublePrecision) {
  if (!doublePrecision) {
    return static_cast<UniversalPositAccum>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].encoding();
  int encoding2 = matrix[2 * index + 1].encoding();
  UniversalPositAccum p16;
  p16.setbits((encoding2 << 8) + encoding1);
  return p16;
}
#endif

inline ACCUM_DATATYPE readInput(INPUT_DATATYPE *matrix, int index,
                                bool doublePrecision) {
  if (!doublePrecision) {
    return static_cast<ACCUM_DATATYPE>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].bits;
  int encoding2 = matrix[2 * index + 1].bits;
  ACCUM_DATATYPE p16;
  p16.setbits((encoding2 << 8) + encoding1);
  return p16;
}

inline float readInput(float *matrix, int index, bool doublePrecision) {
  return doublePrecision ? matrix[2 * index] : matrix[index];
}

#ifndef NO_UNIVERSAL
inline void saveOutput(UniversalPosit *matrix, int index,
                       UniversalPositAccum value, bool doublePrecision) {
  if (!doublePrecision) {
    matrix[index] = static_cast<UniversalPosit>(value);
  } else {
    int bits = value.encoding();
    matrix[2 * index].setbits(bits & 0xFF);
    matrix[2 * index + 1].setbits((bits >> 8) & 0xFF);
  }
}
#endif

inline void saveOutput(INPUT_DATATYPE *matrix, int index,
                       ACCUM_DATATYPE::DecomposedPosit value,
                       bool doublePrecision) {
  if (!doublePrecision) {
    matrix[index] = static_cast<INPUT_DATATYPE>(value);
  } else {
    ACCUM_DATATYPE p16 = value;
    int bits = p16.bits;
    matrix[2 * index].setbits(bits & 0xFF);
    matrix[2 * index + 1].setbits((bits >> 8) & 0xFF);
  }
}

inline void saveOutput(float *matrix, int index, float value,
                       bool doublePrecision) {
  if (!doublePrecision) {
    matrix[index] = value;
  } else {
    matrix[2 * index] = value;
    matrix[2 * index + 1] = 0;
  }
}

#ifndef NO_UNIVERSAL
inline void adjustExp(UniversalPositAccum &value, int expBias) {
  value *= pow(2, expBias);
}
#endif
inline void adjustExp(ACCUM_DATATYPE::DecomposedPosit &value, int expBias) {
  value.scale += expBias;
}
inline void adjustExp(float &value, int expBias) { value *= pow(2, expBias); }

template <typename ACC_T>
void grad_clip_norm(ACC_T *matrix, int size) {
  // tree add
  ACC_T norm = 0;
  for (int reductionCount = 0; reductionCount < size;
       reductionCount += DIMENSION) {
    // perform a tree addition
    ACC_T accum[DIMENSION];
    for (int i = 0; i < DIMENSION; i++) {
      accum[i] = static_cast<ACC_T>(matrix[i] * matrix[i]);
    }

    int depth = DIMENSION;
    while (depth > 1) {
      for (int i = 0; i < depth; i += 2) {
        accum[i / 2] = static_cast<ACC_T>(accum[i] + accum[i + 1]);
      }
      depth = depth / 2;
    }
    norm = static_cast<ACC_T>(norm + accum[0]);
  }

  gold_inv_sqrt(norm);

  if (static_cast<float>(norm) < 1) {
    for (int i = 0; i < size; i++) {
      matrix[i] *= norm;
    }
  }
}

template <typename T, typename ACC_T, typename INT_T>
void run_gold_op(SimplifiedParams params, T *matrixA, T *matrixB, T *matrixC,
                 T *biasMatrix, T *residualMatrix, T *weightResidualMatrix) {
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

  ACC_T learningRate = static_cast<T>(params.learningRate);

  if (params.SOFTMAX) {
    for (int x = 0; x < X; x++) {
      ACC_T outputMatrix[Y];
      for (int y = 0; y < Y; y++) {
        outputMatrix[y] = readInput(matrixA, x * Y + y, params.ACC_T_INPUT);
      }

      ACC_T max = 0;
      for (int y = 0; y < Y; y++) {
        max = outputMatrix[y] > max ? outputMatrix[y] : max;
      }

      ACC_T sum = 0;
      for (int y = 0; y < Y; y++) {
        ACC_T exp = static_cast<ACC_T>(outputMatrix[y] - max);
        gold_exp(exp);
        outputMatrix[y] = exp;
      }

      for (int reductionCount = 0; reductionCount < Y;
           reductionCount += DIMENSION) {
        // perform a tree addition
        ACC_T accum[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
          accum[i] = outputMatrix[reductionCount + i];
        }

        int depth = DIMENSION;
        while (depth > 1) {
          for (int i = 0; i < depth; i += 2) {
            accum[i / 2] = static_cast<ACC_T>(accum[i] + accum[i + 1]);
          }
          depth = depth / 2;
        }
        sum = static_cast<ACC_T>(sum + accum[0]);
      }

      ACC_T divisor = sum;
      gold_reciprocal(divisor);
      for (int y = 0; y < Y; y++) {
        outputMatrix[y] *= divisor;
        saveOutput(matrixC, x * Y + y, outputMatrix[y], params.ACC_T_OUTPUT);
      }
    }
  } else if (params.FC) {
    // fully connected layer (matrix-vector)
    for (int k = 0; k < K; k++) {
      ACC_T acc = 0;

      ACC_T flattened_mult[C];
      for (int c = 0; c < C; c++) {
        ACC_T a = readInput(matrixA, c, params.ACC_T_INPUT);
        ACC_T b = readInput(matrixB, k * C + c, params.ACC_T_WEIGHT);

        if (params.WEIGHT_SPLITTING) {
          ACC_T grad = readInput(weightResidualMatrix, k * C + c, false);
          b -= static_cast<ACC_T>(learningRate * grad);
        }

        flattened_mult[c] =
            static_cast<ACC_T>(static_cast<ACC_T>(a) * static_cast<ACC_T>(b));
      }

      for (int reductionCount = 0; reductionCount < C;
           reductionCount += DIMENSION) {
        // perform a tree addition
        ACC_T accum[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
          accum[i] = flattened_mult[reductionCount + i];
        }

        int depth = DIMENSION;
        while (depth > 1) {
          for (int i = 0; i < depth; i += 2) {
            accum[i / 2] = static_cast<ACC_T>(accum[i] + accum[i + 1]);
          }
          depth = depth / 2;
        }
        acc = static_cast<ACC_T>(acc + accum[0]);
      }

      if (params.BIAS) {
        acc += readInput(biasMatrix, k, true);
      }

      saveOutput(matrixC, k, acc, params.ACC_T_OUTPUT);
    }
  } else if (params.NO_NORM) {
    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        ACC_T a = readInput(matrixA, x * K + k, params.ACC_T_INPUT);
        ACC_T b = readInput(matrixB, k, true);
        ACC_T acc = static_cast<ACC_T>(a * b);

        if (params.BIAS) {
          ACC_T bias = readInput(biasMatrix, k, true);
          acc += bias;
        }

        saveOutput(matrixC, x * K + k, acc, params.ACC_T_OUTPUT);
      }
    }
  } else if (params.SOFTMAX_GRAD) {
    ACC_T outputMatrix[X * Y];
    ACC_T attentionProbs[X * Y];
    ACC_T sums[X];
    for (int i = 0; i < X; i++) {
      sums[i] = 0;
    }

    for (int i = 0; i < X * Y; i++) {
      outputMatrix[i] = readInput(matrixA, i, params.ACC_T_INPUT);
      attentionProbs[i] = readInput(residualMatrix, i, params.ACC_T_OUTPUT);
    }

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        outputMatrix[x * Y + y] *= attentionProbs[x * Y + y];
      }

      // perform sum with a tree addition
      for (int reductionCount = 0; reductionCount < Y;
           reductionCount += DIMENSION) {
        // perform a tree addition
        ACC_T accum[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
          accum[i] = outputMatrix[x * Y + reductionCount + i];
        }

        int depth = DIMENSION;
        while (depth > 1) {
          for (int i = 0; i < depth; i += 2) {
            accum[i / 2] = static_cast<ACC_T>(accum[i] + accum[i + 1]);
          }
          depth = depth / 2;
        }
        sums[x] = static_cast<ACC_T>(sums[x] + accum[0]);
      }
    }

    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        outputMatrix[x * Y + y] -=
            static_cast<ACC_T>(sums[x] * attentionProbs[x * Y + y]);
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
        outputMatrix[x * K + k] = static_cast<ACC_T>(a * b);
      }
    }

    for (int i = 0; i < X * K; i++) {
      adjustExp(outputMatrix[i], params.outputExpBias);
    }

    if (params.RESIDUAL) {
      for (int i = 0; i < X * K; i++) {
        ACC_T grad = readInput(residualMatrix, i, params.ACC_T_RESIDUAL);
        outputMatrix[i] += grad;
      }
    }

    if (params.GRAD_CLIPPING) {
      grad_clip_norm(outputMatrix, X * K);
    }

    for (int i = 0; i < X * K; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else if (params.NO_NORM_GRAD) {
    ACC_T outputMatrix[K];
    for (int i = 0; i < K; i++) {
      outputMatrix[i] = 0;
    }

    for (int i = 0; i < X; i++) {
      for (int j = 0; j < K; j++) {
        ACC_T a = readInput(matrixA, i * K + j, params.ACC_T_INPUT);
        ACC_T b = readInput(matrixB, i * K + j, params.ACC_T_WEIGHT);
        outputMatrix[j] += static_cast<ACC_T>(a * b);
      }
    }

    for (int i = 0; i < K; i++) {
      adjustExp(outputMatrix[i], params.outputExpBias);
    }

    if (params.RESIDUAL) {
      for (int i = 0; i < K; i++) {
        ACC_T grad = readInput(residualMatrix, i, params.ACC_T_RESIDUAL);
        outputMatrix[i] += grad;
      }
    }

    if (params.GRAD_CLIPPING) {
      grad_clip_norm(outputMatrix, K);
    }

    for (int i = 0; i < K; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else if (params.BIAS_GRAD) {
    INT_T inputMatrixB[C * K];
    for (int i = 0; i < C * K; i++) {
      inputMatrixB[i] = readInput(matrixB, i, params.ACC_T_WEIGHT);
    }

    if (params.CONCAT_WEIGHT) {
      INT_T copyMatrixB[C * K];
      memcpy(copyMatrixB, inputMatrixB, sizeof(copyMatrixB));
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < C; j++) {
          for (int k = 0; k < K / 4; k++) {
            inputMatrixB[i * K / 4 + j * K + k] =
                copyMatrixB[i * C * K / 4 + j * K / 4 + k];
          }
        }
      }
    }

    ACC_T outputMatrix[K];
    for (int i = 0; i < K; i++) {
      outputMatrix[i] = 0;
    }

    for (int i = 0; i < C; i++) {
      for (int j = 0; j < K; j++) {
        outputMatrix[j] += inputMatrixB[i * K + j];
      }
    }

    for (int i = 0; i < X * K; i++) {
      adjustExp(outputMatrix[i], params.outputExpBias);
    }

    if (params.RESIDUAL) {
      for (int i = 0; i < X * K; i++) {
        ACC_T grad = readInput(residualMatrix, i, params.ACC_T_RESIDUAL);
        outputMatrix[i] += grad;
      }
    }

    if (params.GRAD_CLIPPING) {
      grad_clip_norm(outputMatrix, K);
    }

    for (int i = 0; i < K; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else if (params.CROSS_ENTROPY_GRAD) {
    ACC_T labels[X];
    ACC_T logits[X];
    ACC_T outputs[X];

    for (int i = 0; i < X; i++) {
      logits[i] = readInput(matrixA, i, params.ACC_T_INPUT);
      labels[i] = readInput(matrixB, i, params.ACC_T_WEIGHT);
    }

    ACC_T max = 0;
    for (int i = 0; i < X; i++) {
      max = logits[i] > max ? logits[i] : max;
    }

    ACC_T sum = 0;
    for (int i = 0; i < X; i++) {
      ACC_T exp = static_cast<ACC_T>(logits[i] - max);
      gold_exp(exp);
      outputs[i] = exp;
      sum += exp;
    }

    ACC_T divisor = sum;
    gold_reciprocal(divisor);
    for (int i = 0; i < X; i++) {
      outputs[i] *= divisor;
      outputs[i] -= labels[i];
      if (params.outputExpBias) {
        adjustExp(outputs[i], params.outputExpBias);
      }
      saveOutput(matrixC, i, outputs[i], params.ACC_T_OUTPUT);
    }
  } else if (params.MSE_GRAD) {
    INT_T divisor = 2 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<INT_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else if (params.BCE_WITH_LOGITS_GRAD) {
    INT_T divisor = 1 / X;
    for (int i = 0; i < X; i++) {
      matrixC[i] = static_cast<INT_T>(matrixA[i] - matrixB[i]) * divisor;
    }
  } else if (params.GRAD_CLIPPING_UNIT_TEST) {
    ACC_T outputMatrix[X * C];
    for (int i = 0; i < X * C; i++) {
      outputMatrix[i] = readInput(matrixA, i, params.ACC_T_INPUT);
    }

    for (int i = 0; i < X * K; i++) {
      adjustExp(outputMatrix[i], params.outputExpBias);
    }

    if (params.RESIDUAL) {
      for (int i = 0; i < X * K; i++) {
        ACC_T grad = readInput(residualMatrix, i, params.ACC_T_RESIDUAL);
        outputMatrix[i] += grad;
      }
    }

    grad_clip_norm(outputMatrix, X * C);

    for (int i = 0; i < X * C; i++) {
      saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
    }
  } else if (params.WEIGHT_UPDATE) {
    ACC_T *weights = new ACC_T[X * C];
    ACC_T *gradients = new ACC_T[X * C];
    for (int i = 0; i < X * C; i++) {
      gradients[i] = readInput(matrixA, i, params.ACC_T_INPUT);
      weights[i] = readInput(matrixB, i, params.ACC_T_WEIGHT);

      weights[i] -= static_cast<ACC_T>(learningRate * gradients[i]);

      // save 8-bit quantized weight to RRAM
      saveOutput(matrixB, i, weights[i], params.ACC_T_OUTPUT);

      if (!params.ACC_T_OUTPUT && params.ERROR_FEEDBACK) {
        INT_T lr = learningRate;
        gradients[i] =
            static_cast<ACC_T>(static_cast<ACC_T>(matrixA[i]) - weights[i]);
        gradients[i] *= static_cast<ACC_T>(1 / lr);
        // float feedback = ((float)matrixA[i] - val) / learningRate;
        // Save 8-bit weight residual to SRAM
        saveOutput(matrixA, i, gradients[i], params.ACC_T_OUTPUT);
      }
    }
  } else {
    // Large arrays need to go on the heap
    INT_T *inputMatrixA = new INT_T[(STRIDE * X) * (STRIDE * Y) * C];
    INT_T *inputMatrixB = new INT_T[FX * FY * C * K];
    ACC_T *inputResidualMatrix = new ACC_T[X * Y * K];
    ACC_T *outputMatrix = new ACC_T[X * Y * K];

    for (int i = 0; i < X * Y * K; i++) {
      outputMatrix[i] = 0;
    }

    for (int i = 0; i < (STRIDE * X) * (STRIDE * Y) * C; i++) {
      inputMatrixA[i] = readInput(matrixA, i, params.ACC_T_INPUT);
    }

    for (int i = 0; i < FX * FY * C * K; i++) {
      inputMatrixB[i] = readInput(matrixB, i, params.ACC_T_WEIGHT);

      if (params.WEIGHT_SPLITTING) {
        ACC_T grad = readInput(weightResidualMatrix, i, false);
        inputMatrixB[i] -= learningRate * grad;
        inputMatrixB[i] = static_cast<T>(inputMatrixB[i]);
      }
    }

    if (params.RESIDUAL || params.RELU_GRAD) {
      for (int i = 0; i < X * Y * K; i++) {
        inputResidualMatrix[i] =
            readInput(residualMatrix, i, params.ACC_T_RESIDUAL);
      }
    }

    if (params.CONCAT_INPUT) {
      INT_T copyMatrixA[X * C];
      memcpy(copyMatrixA, inputMatrixA, sizeof(copyMatrixA));
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < X; j++) {
          for (int k = 0; k < C / 4; k++) {
            inputMatrixA[i * C / 4 + j * C + k] =
                copyMatrixA[i * X * C / 4 + j * C / 4 + k];
          }
        }
      }
    }

    if (params.INPUT_TRANSPOSE) {
      INT_T copyMatrixA[X * C];
      memcpy(copyMatrixA, inputMatrixA, sizeof(copyMatrixA));
      for (int x = 0; x < X; x++) {
        for (int c = 0; c < C; c++) {
          inputMatrixA[x * C + c] = copyMatrixA[c * X + x];
        }
      }
    }

    if (params.CONCAT_WEIGHT) {
      INT_T copyMatrixB[C * K];
      memcpy(copyMatrixB, inputMatrixB, sizeof(copyMatrixB));
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < C; j++) {
          for (int k = 0; k < K / 4; k++) {
            inputMatrixB[i * K / 4 + j * K + k] =
                copyMatrixB[i * C * K / 4 + j * K / 4 + k];
          }
        }
      }
    }

    if (params.WEIGHT_TRANSPOSE) {
      INT_T copyMatrixB[C * K];
      memcpy(copyMatrixB, inputMatrixB, sizeof(copyMatrixB));
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          inputMatrixB[c * K + k] = copyMatrixB[k * C + c];
        }
      }
    }

    int loop_counters[2][6] = {0};

    int X0 = params.loops[1][params.inputXLoopIndex[1]];
    // int X1 = params.loops[0][params.inputXLoopIndex[0]];
    int Y0 = params.loops[1][params.inputYLoopIndex[1]];
    // int Y1 = params.loops[0][params.inputYLoopIndex[0]];
    // int C0 = params.loops[1][params.reductionLoopIndex[1]];
    // int C1 = params.loops[0][params.reductionLoopIndex[0]];
    int K0 = params.loops[1][params.weightLoopIndex[1]];
    // int K1 = params.loops[0][params.weightLoopIndex[0]];
    int IC_unroll = DIMENSION;

    if (params.REPLICATION) {
      params.loops[1][params.fxIndex] = 7;
      IC_unroll = 3;
    }

    for (loop_counters[0][0] = 0; loop_counters[0][0] < params.loops[0][0];
         loop_counters[0][0]++) {
      for (loop_counters[0][1] = 0; loop_counters[0][1] < params.loops[0][1];
           loop_counters[0][1]++) {
        for (loop_counters[0][2] = 0; loop_counters[0][2] < params.loops[0][2];
             loop_counters[0][2]++) {
          int x1 = loop_counters[0][params.inputXLoopIndex[0]];
          int y1 = loop_counters[0][params.inputYLoopIndex[0]];
          int k1 = loop_counters[0][params.weightLoopIndex[0]];

          for (loop_counters[1][0] = 0;
               loop_counters[1][0] < params.loops[1][0];
               loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;
                 loop_counters[1][1] < params.loops[1][1];
                 loop_counters[1][1]++) {
              for (loop_counters[1][2] = 0;
                   loop_counters[1][2] < params.loops[1][2];
                   loop_counters[1][2]++) {
                for (loop_counters[1][3] = 0;
                     loop_counters[1][3] < params.loops[1][3];
                     loop_counters[1][3]++) {
                  for (loop_counters[1][4] = 0;
                       loop_counters[1][4] < params.loops[1][4];
                       loop_counters[1][4]++) {
                    for (loop_counters[1][5] = 0;
                         loop_counters[1][5] < params.loops[1][5];
                         loop_counters[1][5]++) {
                      int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                      int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                      int c0 = loop_counters[1][params.reductionLoopIndex[1]];
                      int k0 = loop_counters[1][params.weightLoopIndex[1]];
                      int fx = loop_counters[1][params.fxIndex] - (FX - 1) / 2;
                      int fy = loop_counters[1][params.fyIndex] - (FY - 1) / 2;

                      int x = x1 * X0 + x0;
                      int y = y1 * Y0 + y0;

                      for (int oc0 = 0; oc0 < DIMENSION; oc0++) {
                        int k = (k1 * K0 + k0) * DIMENSION + oc0;
                        int outputAddress = y * X * K + x * K + k;

                        for (int ic0 = 0; ic0 < IC_unroll; ic0++) {
                          int c = c0 * IC_unroll + ic0;
                          int inputAddress =
                              (STRIDE * y + fy) * STRIDE * X * C +
                              (STRIDE * x + fx) * C + c;
                          int weightAddress = (fy + (FY - 1) / 2) * FX * C * K +
                                              (fx + (FX - 1) / 2) * C * K +
                                              c * K + k;
                          if (STRIDE * x + fx >= 0 &&
                              STRIDE * x + fx < STRIDE * X &&
                              STRIDE * y + fy >= 0 &&
                              STRIDE * y + fy < STRIDE * Y) {
                            gold_fma(inputMatrixA[inputAddress],
                                     inputMatrixB[weightAddress],
                                     outputMatrix[outputAddress]);
                          }
                        }
                        if (params.REPLICATION) {
                          if (loop_counters[1][params.fxIndex] == 3 ||
                              loop_counters[1][params.fxIndex] == 6) {
                            outputMatrix[outputAddress] =
                                static_cast<INT_T>(outputMatrix[outputAddress]);
                          }
                        } else {
                          outputMatrix[outputAddress] =
                              static_cast<INT_T>(outputMatrix[outputAddress]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }

          int X0 = params.loops[1][params.inputXLoopIndex[1]];
          int Y0 = params.loops[1][params.inputYLoopIndex[1]];

          for (int y0 = 0; y0 < Y0; y0++) {
            for (int x0 = 0; x0 < X0; x0++) {
              for (int k0 = 0; k0 < K0; k0++) {
                for (int oc0 = 0; oc0 < DIMENSION; oc0++) {
                  int x = x1 * X0 + x0;
                  int y = y1 * Y0 + y0;
                  int k = (k1 * K0 + k0) * DIMENSION + oc0;
                  int outputAddress = y * X * K + x * K + k;

                  if (params.ATTENTION_SCALING) {
                    T scale = 1.0f / sqrt(32);
                    outputMatrix[outputAddress] *= static_cast<ACC_T>(scale);
                  }

                  adjustExp(outputMatrix[outputAddress], params.outputExpBias);

                  if (params.RESIDUAL) {
                    outputMatrix[outputAddress] +=
                        inputResidualMatrix[outputAddress];
                  }

                  if (params.BIAS) {
                    ACC_T bias = readInput(biasMatrix, k, true);
                    outputMatrix[outputAddress] += bias;
                  }

                  if (params.RELU) {
                    gold_relu(outputMatrix[outputAddress]);
                  }

                  if (params.RELU_GRAD &&
                      inputResidualMatrix[outputAddress] == 0) {
                    outputMatrix[outputAddress] = 0;
                  }
                }
              }
            }
          }
        }
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

    if (params.GRAD_CLIPPING) {
      grad_clip_norm(outputMatrix, Y * X * K);
    }

    if (params.MAXPOOL) {
      ACC_T *tmpMatrixC = new ACC_T[X * Y * K];
      memcpy(tmpMatrixC, outputMatrix, sizeof(ACC_T) * X * Y * K);

      for (int y = 0; y < Y / 2; y++) {
        for (int x = 0; x < X / 2; x++) {
          for (int k = 0; k < K; k++) {
            std::vector<ACC_T> v;

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
    } else if (params.AVGPOOL) {
      ACC_T *tmpMatrixC = new ACC_T[X * Y * K];
      memcpy(tmpMatrixC, outputMatrix, sizeof(ACC_T) * X * Y * K);

      for (int k = 0; k < K; k++) {
        ACC_T acc = 0;
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            acc += tmpMatrixC[y * X * K + x * K + k];
          }
        }
        float scale = 1.0 / (X * Y);
        T divisor = static_cast<T>(scale);
        matrixC[k] = acc * static_cast<ACC_T>(divisor);
      }
      delete[] tmpMatrixC;
    } else {
      for (int i = 0; i < X * Y * K; i++) {
        saveOutput(matrixC, i, outputMatrix[i], params.ACC_T_OUTPUT);
      }
    }

    delete[] inputMatrixA;
    delete[] inputMatrixB;
    delete[] inputResidualMatrix;
    delete[] outputMatrix;
  }
}

void run_custom_posit_gold_model(const SimplifiedParams params,
                                 INPUT_DATATYPE *matrixA,
                                 INPUT_DATATYPE *matrixB,
                                 INPUT_DATATYPE *matrixC,
                                 INPUT_DATATYPE *biasMatrix,
                                 INPUT_DATATYPE *residualMatrix,
                                 INPUT_DATATYPE *weightResidualMatrix) {
  run_gold_op<INPUT_DATATYPE, ACCUM_DATATYPE::DecomposedPosit, ACCUM_DATATYPE>(
      params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix,
      weightResidualMatrix);
}

void run_universal_posit_gold_model(const SimplifiedParams params,
                                    UniversalPosit *matrixA,
                                    UniversalPosit *matrixB,
                                    UniversalPosit *matrixC,
                                    UniversalPosit *biasMatrix,
                                    UniversalPosit *residualMatrix,
                                    UniversalPosit *weightResidualMatrix) {
  run_gold_op<UniversalPosit, UniversalPositAccum, UniversalPositAccum>(
      params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix,
      weightResidualMatrix);
}

void run_fp_gold_model(const SimplifiedParams params, float *matrixA,
                       float *matrixB, float *matrixC, float *biasMatrix,
                       float *residualMatrix, float *weightResidualMatrix) {
  run_gold_op<float, float, float>(params, matrixA, matrixB, matrixC,
                                   biasMatrix, residualMatrix,
                                   weightResidualMatrix);
}
