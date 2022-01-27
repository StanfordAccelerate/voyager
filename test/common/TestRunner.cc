#include <sys/wait.h>
#include <unistd.h>

#include <locale>
#include <stdexcept>
#include <string>

#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/resnet/params.h"
#include "test/simple/params.h"

#define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
#define RRAM_MEMORY_SIZE (12 * 1024 * 1024)

// NOTE: Binary data files are always supplied in [Y][X][C][K] ordering

void validateMapping(Params params) {
  int x0 = params.loops[1][params.inputXLoopIndex[1]];
  int y0 = params.loops[1][params.inputYLoopIndex[1]];
  int c0 = params.loops[1][params.reductionLoopIndex[1]];
  int k0 = params.loops[1][params.weightLoopIndex[1]];
  int fx = params.loops[1][params.fxIndex];
  int fy = params.loops[1][params.fyIndex];
  int stride = params.STRIDE;

  // Input buffer
  int input_buffer_tile_size = (x0 * stride + fx - 1) * (y0 * stride + fy - 1);
  if (params.REPLICATION) {
    // don't check temporarily
    input_buffer_tile_size = 1;
  }
  if (input_buffer_tile_size > INPUT_BUFFER_SIZE) {
    std::cout << "[ERROR] Input buffer tile size violation." << std::endl;
    std::terminate();
  }

  // Weight buffer
  if (fx * fy * k0 > WEIGHT_BUFFER_SIZE) {
    std::cout << "[ERROR] Weight buffer tile size violation." << std::endl;
    std::terminate();
  }

  if (x0 * y0 * k0 > ACCUMULATION_BUFFER_SIZE) {
    std::cout << "[ERROR] Accumulation buffer tile size violation."
              << std::endl;
    std::terminate();
  }
}

int run_test(const Params params, const std::string& dataDir,
             const Files& files, const MemoryMap& memoryMap, bool useDataFile) {
  validateMapping(params);

  INPUT_DATATYPE* sramMemory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* rramMemory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];

  if (sramMemory == nullptr || rramMemory == nullptr)
    throw std::runtime_error("Failed to allocate accelerator memory");

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

  INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];
  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[FX * FY * C * K];
  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[K];
  INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[X * Y * K];
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[X * Y * K];
  OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[X * Y * K];

  load_memory(params, dataDir, files, memoryMap, useDataFile, sramMemory,
              rramMemory, matrixA, matrixB, biasMatrix, residualMatrix, matrixC,
              dataFileOutput);

  if (params.MAXPOOL) {
    X = X / 2;
    Y = Y / 2;
  }

  if (params.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  run_op(params, sramMemory, rramMemory, memoryMap);
  run_gold_op(params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix);

  std::cout << "Accelerator vs. Gold Model" << std::endl;
  std::cout << "(reveals bugs in accelerator or memory placement)" << std::endl;

  int errors =
      compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC, X * Y * K);

  std::cout << "Gold Model vs. Pytorch" << std::endl;
  std::cout << "(reveals bugs in mapping operations to accelerator)"
            << std::endl;
  errors += compare_arrays(matrixC, dataFileOutput, X * Y * K);

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] sramMemory;
  delete[] rramMemory;

  if (errors == 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }

  return errors;
}

int sc_main(int argc, char* argv[]) {
  Params params;

  const char* groupName = std::getenv("GROUP");
  const char* testName = std::getenv("TEST");

  if (!(testName && groupName)) {
    std::cout << "Warning! No group/test specified! Please set the environment "
                 "variables GROUP and TEST"
              << std::endl;
    return -1;
  }

  std::string group(groupName);
  std::string test(testName);

  std::cout << "Running: " << group << ": " << test << std::endl;

  std::map<std::string, Params>* mapPtr;

  if (group == "simple") {
    mapPtr = &simple;
  } else if (group == "mobilebert") {
    mapPtr = &mobilebert;
  } else if (group == "resnet") {
    mapPtr = &resnetParams;
  } else {
    throw std::runtime_error("Group: " + group + " not found");
  }

  auto search = mapPtr->find(test);
  if (search != mapPtr->end()) {
    params = search->second;
  } else {
    throw std::runtime_error("Test: " + test + " not found");
  }

  bool useDataFiles = true;
  std::string dataDir;
  Files files;
  MemoryMap memoryMap;
  if (group == "resnet") {  // currently only resnet has data files
    useDataFiles = true;

    dataDir = resnetDataDir;

    auto fileSearch = resnetFiles.find(test);
    if (fileSearch != resnetFiles.end()) {
      files = fileSearch->second;
    } else {
      throw std::runtime_error("Files for " + test + " not found");
    }

    auto memoryMapSearch = resnetMemoryMap.find(test);
    if (fileSearch != resnetFiles.end()) {
      files = fileSearch->second;
    } else {
      throw std::runtime_error("Memory map for " + test + " not found");
    }
  }

  run_test(params, dataDir, files, memoryMap, useDataFiles);
}
