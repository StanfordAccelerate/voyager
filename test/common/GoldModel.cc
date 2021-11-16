#include "test/common/GoldModel.h"

void run_gold_op(const Params params, INPUT_DATATYPE *matrixA,
                 INPUT_DATATYPE *matrixB, OUTPUT_DATATYPE *matrixC) {
  std::cout << "Running gold model " << std::endl;

  int inputs = params.loops[0][params.inputLoopIndex[0]] *
               params.loops[1][params.inputLoopIndex[1]];
  int weights = params.loops[0][params.weightLoopIndex[0]] *
                params.loops[1][params.weightLoopIndex[1]];
  int reduction = params.loops[1][params.reductionLoopIndex[1]];

  if (params.VEC_OP) {
    if (params.VEC_REDUCE) {
      for (int m = 0; m < inputs; m++) {
        OUTPUT_DATATYPE acc = 0;
        for (int p = 0; p < weights * DIMENSION; p++) {
          int index = m * (weights * DIMENSION) + p;
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
      for (int m = 0; m < inputs; m++) {
        for (int p = 0; p < weights * DIMENSION; p++) {
          int index = m * (weights * DIMENSION) + p;
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
    for (int p = 0; p < weights * DIMENSION; p++) {
      for (int m = 0; m < inputs; m++) {
        OUTPUT_DATATYPE acc = 0;
        for (int n = 0; n < DIMENSION * reduction; n++) {
          int matrixAIndex = m * reduction * DIMENSION + n;
          int matrixBIndex = n * (weights * DIMENSION) + p;

          if (params.TRANSPOSE) {
            matrixBIndex = p * (DIMENSION * reduction) + n;
          }

          acc = matrixA[matrixAIndex] * matrixB[matrixBIndex] + acc;
        }

        matrixC[m * (weights * DIMENSION) + p] = acc * params.SCALE;
      }
    }
  }
}