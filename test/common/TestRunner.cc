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

void validateMapping(SimplifiedParams params) {
  int x0 = params.loops[1][params.inputXLoopIndex[1]];
  int y0 = params.loops[1][params.inputYLoopIndex[1]];
  int c0 = params.loops[1][params.reductionLoopIndex[1]];
  int k0 = params.loops[1][params.weightLoopIndex[1]];
  int fx = params.loops[1][params.fxIndex];
  int fy = params.loops[1][params.fyIndex];
  int stride = params.STRIDE;

  if (params.FC || params.SOFTMAX ||
      params.NO_NORM) {  // don't check for vector ops
    return;
  }

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

// int run_complete(const std::string& dataDir,
//              const Files& files){

//   bool useDataFile = true;

//   INPUT_DATATYPE* sramMemory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
//   INPUT_DATATYPE* rramMemory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];

//   if (sramMemory == nullptr || rramMemory == nullptr)
//     throw std::runtime_error("Failed to allocate accelerator memory");


//   INPUT_DATATYPE* trash = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
//   load_memory(resnetParams["conv1"], dataDir, resnetFiles["conv1"], resnetMemoryMap["conv1"], useDataFile, sramMemory,
//               rramMemory, trash, trash, trash, trash, trash,
//               trash);

//   // run_op({params}, sramMemory, rramMemory, memoryMap);
//   for(const std::string& param_name : resnet_order)
//   {
//   validateMapping(resnetParams[param_name]);
//   int X = resnetParams[param_name].loops[0][resnetParams[param_name].inputXLoopIndex[0]] *
//           resnetParams[param_name].loops[1][resnetParams[param_name].inputXLoopIndex[1]];
//   int Y = resnetParams[param_name].loops[0][resnetParams[param_name].inputYLoopIndex[0]] *
//           resnetParams[param_name].loops[1][resnetParams[param_name].inputYLoopIndex[1]];
//   int C = resnetParams[param_name].loops[1][resnetParams[param_name].reductionLoopIndex[1]] * DIMENSION;
//   int K = resnetParams[param_name].loops[0][resnetParams[param_name].weightLoopIndex[0]] *
//           resnetParams[param_name].loops[1][resnetParams[param_name].weightLoopIndex[1]] * DIMENSION;
//   int FX = resnetParams[param_name].loops[1][resnetParams[param_name].fxIndex];
//   int FY = resnetParams[param_name].loops[1][resnetParams[param_name].fyIndex];
//   int STRIDE = resnetParams[param_name].STRIDE;

//   if (resnetParams[param_name].REPLICATION) {
//     FX = 7;
//     C = 3;
//   }

//   if (resnetParams[param_name].MAXPOOL) {
//     X = X / 2;
//     Y = Y / 2;
//   }

//   if (resnetParams[param_name].AVGPOOL) {
//     X = 1;
//     Y = 1;
//   }

//   // INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];
//   // INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[FX * FY * C * K];
//   // INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[K];
//   // INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[X * Y * K];
//   // OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[X * Y * K];

//   std::cout << "Performing "+ param_name+ ":" << std::endl;
//   std::cout << "(" << X << "x" << Y << "x" << C << ")"
//             << " * "
//             << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
//             << std::endl;

//   // load_inputs(resnetParams["conv1"], "")
//   load_wb(resnetParams[param_name], dataDir, resnetFiles[param_name], resnetMemoryMap[param_name], useDataFile, sramMemory,
//               rramMemory, trash, trash, trash, trash, trash,
//               trash);
//   run_gold_op(resnetParams[param_name], sramMemory + resnetParams[param_name].INPUT_OFFSET, rramMemory + resnetParams[param_name].WEIGHT_OFFSET, sramMemory + resnetParams[param_name].OUTPUT_OFFSET, rramMemory + resnetParams[param_name].BIAS_OFFSET, sramMemory + resnetParams[param_name].RESIDUAL_OFFSET);

// // if (param_name == "layer4_1_conv1") {
//   if (param_name == "softmax"){
//     continue;
//   // if (false){
//   OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[X * Y * K];
//   load_datafile_outputs(resnetParams[param_name], "data/resnet/" + param_name + "_comp",
//                            dataFileOutput);
//   std::cout << "Gold vs. Pytorch" << std::endl;
//   std::cout << "(reveals bugs in accelerator or memory placement)" << std::endl;
//   std::string diffFile = "test_outputs/resnet."+ param_name + "gold_vs_pytorch.txt";
//   int errors = compare_arrays(&sramMemory[resnetParams[param_name].OUTPUT_OFFSET], dataFileOutput,
//                               X * Y * K, diffFile);

//   delete[] dataFileOutput;
//   // return errors;
//   // }
//   }

//   std::ofstream wf("pybuild/output", std::ios::out | std::ios::binary);
//   if (!wf.good())
//     throw std::runtime_error("File write failed");

//   for (int i = 0; i< 1000; i++)
//   {
//     wf.write((char*)(sramMemory + resnetParams["fc"].OUTPUT_OFFSET + i * 4), sizeof(char));
//   }
//   wf.close();

//   delete[] sramMemory;
//   delete[] rramMemory;

//   return 0;
// }

int run_test(const SimplifiedParams params, const std::string& dataDir,
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

  // run_op({params}, sramMemory, rramMemory, memoryMap);
  run_gold_op(params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix);

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
  SimplifiedParams params;

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

  std::map<std::string, SimplifiedParams>* mapPtr;

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
