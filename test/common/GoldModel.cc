#include "test/common/GoldModel.h"

void run_gold_op(const Params params, INPUT_DATATYPE *matrixA,
                 INPUT_DATATYPE *matrixB, OUTPUT_DATATYPE *matrixC) {
  std::cout << "Running gold model " << std::endl;

  if (params.VEC_OP) {
    if (params.VEC_REDUCE) {
      for (int m = 0; m < params.M0 * params.M1; m++) {
        OUTPUT_DATATYPE acc = 0;
        for (int p = 0; p < params.P1 * params.P2 * DIMENSION; p++) {
          int index = m * (params.P1 * params.P2 * DIMENSION) + p;
          OUTPUT_DATATYPE tmp = matrixA[index];
          if (params.VEC_SUB) {
            tmp -= matrixB[m];
          }

          if (params.VEC_SQUARE) {
            tmp *= tmp;
          }

          acc += tmp;
        }

        matrixC[m] = acc / params.SCALE;
      }
    } else {
      for (int m = 0; m < params.M0 * params.M1; m++) {
        for (int p = 0; p < params.P1 * params.P2 * DIMENSION; p++) {
          int index = m * (params.P1 * params.P2 * DIMENSION) + p;
          OUTPUT_DATATYPE tmp = matrixA[index];

          if (params.VEC_SUB) {
            tmp -= matrixB[m];
          }
          if (!params.CONST_SCALE) {
            tmp /= matrixC[m];
          }

          matrixA[index] = tmp;
        }
      }
    }
  } else {
    for (int p = 0; p < params.P1 * params.P2 * DIMENSION; p++) {
      for (int m = 0; m < params.M0 * params.M1; m++) {
        OUTPUT_DATATYPE acc = 0;
        for (int n = 0; n < DIMENSION * params.N1; n++) {
          int matrixAIndex = m * params.N1 * DIMENSION + n;
          int matrixBIndex = n * (params.P1 * params.P2 * DIMENSION) + p;

          if (params.TRANSPOSE) {
            matrixBIndex = p * (DIMENSION * params.N1) + n;
          }

          acc = matrixA[matrixAIndex] * matrixB[matrixBIndex] + acc;
        }

        matrixC[m * (params.P1 * params.P2 * DIMENSION) + p] =
            acc * params.SCALE;
      }
    }
  }
}