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
#include "test/mobilebert/params.h"

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

int run_complete(const std::string& dataDir,
             const Files& files)
             {
  bool useDataFile = true;
  MemoryMap& memoryMap = resnetMemoryMap["conv1"];
  Params params = resnetParams["conv1"];

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

  std::cout << "Using the following memory map:" << std::endl;
  std::cout << "Inputs: " << (memoryMap.inputs == 0 ? "SRAM" : "RRAM")
            << std::endl;
  std::cout << "Weights: " << (memoryMap.weights == 0 ? "SRAM" : "RRAM")
            << std::endl;
  std::cout << "Bias: " << (memoryMap.bias == 0 ? "SRAM" : "RRAM") << std::endl;
  std::cout << "Residual: " << (memoryMap.residual == 0 ? "SRAM" : "RRAM")
            << std::endl;
  std::cout << "Outputs: " << (memoryMap.outputs == 0 ? "SRAM" : "RRAM")
            << std::endl;

  INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[SRAM_MEMORY_SIZE];
  OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[SRAM_MEMORY_SIZE];

  load_memory(resnetParams["conv1"], dataDir, resnetFiles["conv1"], memoryMap, useDataFile, sramMemory,
              rramMemory, matrixA, matrixB, biasMatrix, residualMatrix, matrixC,
              dataFileOutput);

  for (const std::string& param_name : resnet_order)
  {
    std::cout << param_name << std::endl;
    load_wb(resnetParams[param_name], dataDir, resnetFiles[param_name], memoryMap, useDataFile, sramMemory,
              rramMemory, matrixA, matrixB, biasMatrix, residualMatrix, matrixC,
              dataFileOutput);
  }

  if (params.MAXPOOL) {
    X = X / 2;
    Y = Y / 2;
  }

  if (params.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  std::vector<Params> ops;
  for (const std::string& param_name : resnet_order)
  {
    ops.push_back(resnetParams[param_name]);
  }
  run_op(ops, sramMemory, rramMemory, memoryMap);
  // run_gold_op(params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix);

  std::ofstream wf("pybuild/output", std::ios::out | std::ios::binary);
  if (!wf.good())
    throw std::runtime_error("File write failed");

  wf.write((char*)sramMemory + resnetParams["fc"].OUTPUT_OFFSET, 1000*sizeof(char));
  wf.close();

  // delete[] matrixA;
  // delete[] matrixB;
  // delete[] matrixC;
  // delete[] sramMemory;
  // delete[] rramMemory;
  // delete[] dataFileOutput;

  return 0;

}


int run_test(const Params params, const std::string& dataDir,
             const Files& files, const MemoryMap& memoryMap, bool useDataFile,
             std::string& fileOutputPrefix) {
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

  run_op({params}, sramMemory, rramMemory, memoryMap);
  run_gold_op(params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix);

  std::ofstream wf("pybuild/output", std::ios::out | std::ios::binary);
  if (!wf.good())
    throw std::runtime_error("File write failed");

  wf.write((char*)sramMemory + resnetParams["fc"].OUTPUT_OFFSET, 1000*sizeof(char));
  wf.close();

  std::cout << "Accelerator vs. Gold Model" << std::endl;
  std::cout << "(reveals bugs in accelerator or memory placement)" << std::endl;
  std::string diffFile = fileOutputPrefix + "accel_vs_gold.txt";
  int errors = compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC,
                              X * Y * K, diffFile);

  if (useDataFile) {
    std::cout << "Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals bugs in mapping operations to accelerator)"
              << std::endl;
    diffFile = fileOutputPrefix + "gold_vs_pytorch.txt";
    errors += compare_arrays(matrixC, dataFileOutput, X * Y * K, diffFile);
  }

  // delete[] matrixA;
  // delete[] matrixB;
  // delete[] matrixC;
  // delete[] sramMemory;
  // delete[] rramMemory;
  // delete[] dataFileOutput;

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

  std::string fullName = "test_outputs/" + group + "." + test + ".";

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
    if (memoryMapSearch != resnetMemoryMap.end()) {
      memoryMap = memoryMapSearch->second;
    } else {
      throw std::runtime_error("Memory map for " + test + " not found");
    }
  } else if (group == "mobilebert") {
    useDataFiles = true;

    dataDir = mobilebertDataDir;

    auto fileSearch = mobilebertFiles.find(test);
    if (fileSearch != mobilebertFiles.end()) {
      files = fileSearch->second;
    } else {
      throw std::runtime_error("Files for " + test + " not found");
    }

    auto memoryMapSearch = mobilebertMemoryMap.find(test);
    if (memoryMapSearch != mobilebertMemoryMap.end()) {
      memoryMap = memoryMapSearch->second;
    } else {
      throw std::runtime_error("Memory map for " + test + " not found");
    }
  }

  // run_complete(dataDir, files);
  run_test(params, dataDir, files, memoryMap, useDataFiles, fullName);
}
