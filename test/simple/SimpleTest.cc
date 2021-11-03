#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"

void basic() {
  std::cout << "Basic Test" << std::endl;
  std::cout << "----------" << std::endl;

  INPUT_DATATYPE *mainMemory = new INPUT_DATATYPE[4 * 1024 * 1024];

  const Params params = {
      8,                       // M0
      1,                       // P1
      1,                       // N1
      1,                       // M1
      1,                       // P2
      0,                       // INPUT_OFFSET
      1024 * 1024,             // WEIGHT_OFFSET
      2 * 1024 * 1024,         // OUTPUT_OFFSET
      false,                   // SOFTMAX
      1,                       // SCALE
      false,                   // TRANSPOSE
      0,                       // VECTOR_OFFSET
      false,                   // VEC_OP
      false,                   // VEC_SUB
      false,                   // VEC_SQUARE
      false,                   // VEC_REDUCE
      true,                    // CONST_SCALE
      0,                       // VEC_SCALE_OFFSET
      0,                       // VEC_SUB_OFFSET
      false,                   // RELU
      {{1, 1, 1}, {1, 1, 8}},  // LOOPS
      {1, 2},                  // INPUT
      {2, 0},                  // REDUCTION
      {0, 1}                   // WEIGHT
  };

  // Create matrix A
  INPUT_DATATYPE *matrixA =
      new INPUT_DATATYPE[params.M0 * params.M1 * params.N1 * DIMENSION];
  for (int i = 0; i < params.M0 * params.M1; i++) {
    for (int j = 0; j < params.N1 * DIMENSION; j++) {
      int val = i * 10 + j;

      mainMemory[params.INPUT_OFFSET + i * (params.N1 * DIMENSION) + j] = val;
      matrixA[i * (params.N1 * DIMENSION) + j] = val;
    }
  }

  INPUT_DATATYPE *matrixB =
      new INPUT_DATATYPE[params.N1 * DIMENSION * params.P1 * params.P2 *
                         DIMENSION];
  for (int i = 0; i < params.N1 * DIMENSION; i++) {
    for (int j = 0; j < params.P1 * params.P2 * DIMENSION; j++) {
      int val = i;

      mainMemory[params.WEIGHT_OFFSET +
                 i * (params.P1 * params.P2 * DIMENSION) + j] = val;
      matrixB[i * (params.P1 * params.P2 * DIMENSION) + j] = val;
    }
  }

  OUTPUT_DATATYPE *matrixC =
      new OUTPUT_DATATYPE[params.M0 * params.M1 * params.P1 * params.P2 *
                          DIMENSION];

  run_op(params, mainMemory);
  run_gold_op(params, matrixA, matrixB, matrixC);
  compare_arrays(&mainMemory[params.OUTPUT_OFFSET], matrixC,
                 params.M0 * params.M1 * params.P1 * params.P2 * DIMENSION);

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] mainMemory;
}

int sc_main(int argc, char *argv[]) { basic(); }