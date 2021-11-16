#include <locale>
#include <string>

#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/simple/params.h"

void run_test(Params params) {
  INPUT_DATATYPE *mainMemory = new INPUT_DATATYPE[4 * 1024 * 1024];

  int inputRows = params.loops[0][params.inputLoopIndex[0]] *
                  params.loops[1][params.inputLoopIndex[1]];
  int inputCols = params.loops[1][params.reductionLoopIndex[1]];
  int weightCols = params.loops[0][params.weightLoopIndex[0]] *
                   params.loops[1][params.weightLoopIndex[1]];

  // Create matrix A
  INPUT_DATATYPE *matrixA =
      new INPUT_DATATYPE[inputRows * inputCols * DIMENSION];
  for (int i = 0; i < inputRows; i++) {
    for (int j = 0; j < inputCols * DIMENSION; j++) {
      // int val = i * 10 + j;
      int val = rand() % 128;

      mainMemory[params.INPUT_OFFSET + i * (inputCols * DIMENSION) + j] = val;
      matrixA[i * (inputCols * DIMENSION) + j] = val;
    }
  }

  INPUT_DATATYPE *matrixB =
      new INPUT_DATATYPE[inputCols * DIMENSION * weightCols * DIMENSION];
  for (int i = 0; i < inputCols * DIMENSION; i++) {
    for (int j = 0; j < weightCols * DIMENSION; j++) {
      // int val = i;
      int val = rand() % 128;

      mainMemory[params.WEIGHT_OFFSET + i * (weightCols * DIMENSION) + j] = val;
      matrixB[i * (weightCols * DIMENSION) + j] = val;
    }
  }

  OUTPUT_DATATYPE *matrixC =
      new OUTPUT_DATATYPE[inputRows * weightCols * DIMENSION];

  run_op(params, mainMemory);
  run_gold_op(params, matrixA, matrixB, matrixC);
  compare_arrays(&mainMemory[params.OUTPUT_OFFSET], matrixC,
                 inputRows * weightCols * DIMENSION);

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] mainMemory;
}

int sc_main(int argc, char *argv[]) {
  Params params = simple;

  const char *testName = std::getenv("TEST");
  if (testName) {
    std::string test(testName);
    std::cout << "Running test: " << test << std::endl;

    if (test == "simple") {
      params = simple;
    } else if (test == "inputBottleneck") {
      params = inputBottleneck;
    } else if (test == "qkvProjection") {
      params = qkvProjection;
    } else if (test == "qkAttention") {
      params = qkAttention;
    } else if (test == "vAttention") {
      params = vAttention;
    } else if (test == "wProjection") {
      params = wProjection;
    } else if (test == "ffn1") {
      params = ffn1;
    } else if (test == "ffn2") {
      params = ffn2;
    } else if (test == "outputBottleneck") {
      params = outputBottleneck;
    } else {
      std::cout << "Warning! Test not found!" << std::endl;
    }
  } else {
    std::cout << "Warning! No test specified! Please set the environment "
                 "variable TEST"
              << std::endl;
  }

  run_test(params);
  return 0;
}
