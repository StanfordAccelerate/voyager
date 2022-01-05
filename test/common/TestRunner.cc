#include <locale>
#include <string>

#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/resnet/params.h"
#include "test/simple/params.h"

void run_test(Params params) {
  INPUT_DATATYPE *mainMemory = new INPUT_DATATYPE[4 * 1024 * 1024];

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

  std::cout << "Performing the following operation:" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;

  // Create matrix A
  INPUT_DATATYPE *matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];

  if (params.REPLICATION) {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x_o = 0; x_o < (STRIDE * X) / 4; x_o++) {
        for (int x_i = 0; x_i < 4; x_i++) {  // 4 packed together
          for (int c = 0; c < C; c++) {
            int x = x_o * 4 + x_i;
            int val = rand() % 128;
            // int val = x;

            int address = y * ((STRIDE * X) / 4) * 16 + x_o * 16 + x_i * 3 + c;
            mainMemory[params.INPUT_OFFSET + address] = val;

            address = y * (STRIDE * X) * C + x * C + c;
            matrixA[address] = val;
          }
        }
      }
    }
  } else {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x = 0; x < STRIDE * X; x++) {
        for (int c = 0; c < C; c++) {
          // int val = i * 10 + j;
          // int val = rand() % 128;
          int val = x;

          int address = y * (STRIDE * X) * C + x * C + c;

          mainMemory[params.INPUT_OFFSET + address] = val;
          matrixA[address] = val;
        }
      }
    }
  }

  for (int i = 0; i < 512; i++) {
    if (i % 16 == 0) {
      std::cout << std::endl;
    }
    std::cout << mainMemory[params.INPUT_OFFSET + i] << " ";
  }

  INPUT_DATATYPE *matrixB = new INPUT_DATATYPE[FX * FY * C * K];
  for (int fy = 0; fy < FY; fy++) {
    for (int fx = 0; fx < FX; fx++) {
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          // int val = i;
          int val = rand() % 128;

          int address = fy * FX * C * K + fx * C * K + c * K + k;
          mainMemory[params.WEIGHT_OFFSET + address] = val;
          matrixB[address] = val;
        }
      }
    }
  }

  INPUT_DATATYPE *biasMatrix = new INPUT_DATATYPE[K];

  if (params.BIAS) {
    for (int k = 0; k < K; k++) {
      int val = rand() % 128;
      mainMemory[params.BIAS_OFFSET + k] = val;
      biasMatrix[k] = val;
    }
  }

  INPUT_DATATYPE *residualMatrix = new INPUT_DATATYPE[X * Y * K];
  if (params.RESIDUAL) {
    for (int y = 0; y < Y; y++) {
      for (int x = 0; x < X; x++) {
        for (int k = 0; k < K; k++) {
          int val = rand() % 128;

          int address = y * X * K + x * K + k;
          mainMemory[params.RESIDUAL_OFFSET + address] = val;
          residualMatrix[address] = val;
        }
      }
    }
  }

  OUTPUT_DATATYPE *matrixC = new OUTPUT_DATATYPE[X * Y * K];

  if (params.MAXPOOL) {
    X = X / 2;
    Y = Y / 2;
  }

  if (params.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  run_op(params, mainMemory);
  run_gold_op(params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix);
  compare_arrays(&mainMemory[params.OUTPUT_OFFSET], matrixC, X * Y * K);

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] mainMemory;
}

int sc_main(int argc, char *argv[]) {
  Params params;

  const char *groupName = std::getenv("GROUP");
  const char *testName = std::getenv("TEST");
  if (testName && groupName) {
    std::string group(groupName);
    std::string test(testName);

    std::cout << "Running: " << group << ", " << test << std::endl;

    std::map<std::string, Params> *mapPtr;

    if (group == "simple") {
      mapPtr = &simple;
    } else if (group == "mobilebert") {
      mapPtr = &mobilebert;
    } else if (group == "resnet") {
      mapPtr = &resnet;
    } else {
      std::cout << "Warning! Group " << group << " not found!" << std::endl;
    }

    auto search = mapPtr->find(test);
    if (search != mapPtr->end()) {
      params = search->second;
    } else {
      std::cout << "Warning! Test " << test << " not found!" << std::endl;
    }

  } else {
    std::cout << "Warning! No group/test specified! Please set the environment "
                 "variables GROUP and TEST"
              << std::endl;
  }

  run_test(params);
  return 0;
}
