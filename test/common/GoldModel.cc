#include "test/common/GoldModel.h"

#include <algorithm>
#include <cstring>

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
