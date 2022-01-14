#include "test/common/GoldModel.h"

void run_gold_op(const Params params, INPUT_DATATYPE *matrixA,
                 INPUT_DATATYPE *matrixB, OUTPUT_DATATYPE *matrixC,
                 INPUT_DATATYPE *biasMatrix, OUTPUT_DATATYPE *residualMatrix) {
  std::cout << "Running gold model " << std::endl;

  if (params.VEC_OP) {
    // FIXME!
    // if (params.VEC_REDUCE) {
    //   for (int m = 0; m < inputs; m++) {
    //     OUTPUT_DATATYPE acc = 0;
    //     for (int p = 0; p < weights * DIMENSION; p++) {
    //       int index = m * (weights * DIMENSION) + p;
    //       OUTPUT_DATATYPE tmp = matrixA[index];
    //       if (params.VEC_SUB) {
    //         tmp -= matrixB[m];
    //       }

    //       if (params.VEC_SQUARE) {
    //         tmp *= tmp;
    //       }

    //       acc += tmp;
    //     }

    //     matrixC[m] = acc / params.SCALE;
    //   }
    // } else {
    //   for (int m = 0; m < inputs; m++) {
    //     for (int p = 0; p < weights * DIMENSION; p++) {
    //       int index = m * (weights * DIMENSION) + p;
    //       OUTPUT_DATATYPE tmp = matrixA[index];

    //       if (params.VEC_SUB) {
    //         tmp -= matrixB[m];
    //       }
    //       if (!params.CONST_SCALE) {
    //         tmp /= matrixC[m];
    //       }

    //       matrixA[index] = tmp;
    //     }
    //   }
    // }
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
      INPUT_DATATYPE *tmpMatrixB = new INPUT_DATATYPE[C * K];
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          tmpMatrixB[c * K + k] = matrixB[k * C + c];
        }
      }
      memcpy(matrixB, tmpMatrixB, sizeof(INPUT_DATATYPE) * C * K);

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
          OUTPUT_DATATYPE acc = 0;
          for (int c = 0; c < C; c++) {
            for (int fy = -(FY - 1) / 2; fy <= (FY - 1) / 2; fy++) {
              for (int fx = -(FX - 1) / 2; fx <= (FX - 1) / 2; fx++) {
                if (STRIDE * x + fx >= 0 && STRIDE * x + fx < STRIDE * X &&
                    STRIDE * y + fy >= 0 &&
                    STRIDE * y + fy < STRIDE * Y) {  // check if in bounds
                  acc += matrixA[(STRIDE * y + fy) * STRIDE * X * C +
                                 (STRIDE * x + fx) * C + c] *
                         matrixB[(fy + (FY - 1) / 2) * FX * C * K +
                                 (fx + (FX - 1) / 2) * C * K + c * K + k];
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
          matrixC[y * X * K + x * K + k] = acc;
        }
      }
    }

    if (params.MAXPOOL) {
      std::cout << "Before MP" << std::endl;
      for (int y = 0; y < Y; y++) {
        for (int x = 0; x < X; x++) {
          std::cout << matrixC[y * X * K + x * K] << " ";
        }
        std::cout << std::endl;
      }

      // create copy
      OUTPUT_DATATYPE *tmpMatrixC = new OUTPUT_DATATYPE[X * Y * K];
      memcpy(tmpMatrixC, matrixC, sizeof(OUTPUT_DATATYPE) * X * Y * K);

      for (int y = 0; y < Y / 2; y++) {
        for (int x = 0; x < X / 2; x++) {
          for (int k = 0; k < K; k++) {
            std::vector<OUTPUT_DATATYPE> v;

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

      std::cout << "after MP" << std::endl;
      for (int y = 0; y < Y / 2; y++) {
        for (int x = 0; x < X / 2; x++) {
          std::cout << matrixC[y * (X / 2) * K + x * K] << " ";
        }
        std::cout << std::endl;
      }
    }

    if (params.AVGPOOL) {
      // create copy
      OUTPUT_DATATYPE *tmpMatrixC = new OUTPUT_DATATYPE[X * Y * K];
      memcpy(tmpMatrixC, matrixC, sizeof(OUTPUT_DATATYPE) * X * Y * K);

      for (int k = 0; k < K; k++) {
        int acc = 0;
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            acc += tmpMatrixC[y * X * K + x * K + k];
          }
        }
        matrixC[k] = acc;
      }

      delete[] tmpMatrixC;
    }
  }
}