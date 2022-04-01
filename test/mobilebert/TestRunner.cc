#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
// #include "test/common/Harness.h"
#include "test/common/UniversalPosit.h"
#include "test/common/Utils.h"

#define TRAINING
#ifndef TRAINING
#include "test/mobilebert/params.h"
#else
#include "test/mobilebert/training.h"
#endif

#define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
// #define RRAM_MEMORY_SIZE (12 * 1024 * 1024)  // RRAM size for TinyBERT
#define RRAM_MEMORY_SIZE (20 * 1024 * 1024)  // RRAM size for MobileBERT

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

int run_test(const SimplifiedParams params, const std::string& dataDir,
             const Files& files, bool useDataFile,
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
  if (params.SOFTMAX) {
    K = 1;
  }
  if (params.FC) {
    K = params.loops[0][params.weightLoopIndex[0]] *
        params.loops[1][params.weightLoopIndex[1]];
  }

  int size = X * Y * K;
  int inputChannelWidth = C;
  int outputChannelWidth = K;

  std::cout << "Performing the following operation:" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;

  INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];
  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[FX * FY * C * K];
  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[K];
  INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[X * Y * K];
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[size];
  OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[size];

  UniversalPosit* universalMatrixA =
      new UniversalPosit[(STRIDE * X) * (STRIDE * Y) * C];
  UniversalPosit* universalMatrixB = new UniversalPosit[FX * FY * C * K];
  UniversalPosit* universalBiasMatrix = new UniversalPosit[K];
  UniversalPosit* universalResidualMatrix = new UniversalPosit[X * Y * K];
  UniversalPosit* universalMatrixC = new UniversalPosit[size];
  UniversalPosit* universalDataFileOutput = new UniversalPosit[size];

  float* floatMatrixA = new float[(STRIDE * X) * (STRIDE * Y) * C];
  float* floatMatrixB = new float[FX * FY * C * K];
  float* floatBiasMatrix = new float[K];
  float* floatResidualMatrix = new float[X * Y * K];
  float* floatMatrixC = new float[size];
  float* floatDataFileOutput = new float[size];

  load_inputs(params, dataDir + files.inputs_file, useDataFile, sramMemory,
              matrixA, universalMatrixA, floatMatrixA);
  if (!files.weights_file.empty()) {
    load_weights(params, dataDir + files.weights_file, useDataFile,
                 params.WEIGHT ? sramMemory : rramMemory, matrixB,
                 universalMatrixB, floatMatrixB);
  }
  if (params.BIAS) {
    load_bias(params, dataDir + files.bias_file, useDataFile, rramMemory,
              biasMatrix, universalBiasMatrix, floatBiasMatrix);
  }
  if (params.RESIDUAL) {
    load_residual(params, dataDir + files.residual_file, useDataFile,
                  sramMemory, residualMatrix, universalResidualMatrix,
                  floatResidualMatrix);
  }
  if (useDataFile) {
    load_datafile_outputs(params, dataDir + files.outputs_file, dataFileOutput,
                          universalDataFileOutput, floatDataFileOutput);
  }

  // run_op({params}, sramMemory, rramMemory, memoryMap);
  run_custom_posit_gold_model(params, matrixA, matrixB, matrixC, biasMatrix,
                              residualMatrix);
  run_universal_posit_gold_model(params, universalMatrixA, universalMatrixB,
                                 universalMatrixC, universalBiasMatrix,
                                 universalResidualMatrix);
  run_fp_gold_model(params, floatMatrixA, floatMatrixB, floatMatrixC,
                    floatBiasMatrix, floatResidualMatrix);

  std::string diffFile;
  int errors = 0;

  // std::cout << "Accelerator vs. HLS Posit Gold Model" << std::endl;
  // std::cout << "(reveals bugs in accelerator or memory placement)" <<
  // std::endl; diffFile = fileOutputPrefix + "accel_vs_hlsgold.txt";
  // compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC, size, diffFile);

  std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
  std::cout << "(reveals bugs in mapping operations to accelerator)"
            << std::endl;
  diffFile = fileOutputPrefix + "hlsgold_vs_pytorch.txt";
  compare_arrays(matrixC, dataFileOutput, size, diffFile);

  std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
  std::cout << "(reveals issues in representing float as Posit)" << std::endl;
  diffFile = fileOutputPrefix + "universalgold_vs_pytorch.txt";
  compare_arrays(universalMatrixC, universalDataFileOutput, size, diffFile);

  std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
            << std::endl;
  std::cout << "(reveals bugs in implementation of custom HLS Posit operators)"
            << std::endl;
  diffFile = fileOutputPrefix + "hlsgold_vs_universalgold.txt";
  errors += compare_arrays(matrixC, universalMatrixC, size, diffFile);

  std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
  std::cout << "(reveals issues in data loading or mapping)" << std::endl;
  diffFile = fileOutputPrefix + "fpgold_vs_pytorch.txt";
  errors += compare_arrays(floatMatrixC, floatDataFileOutput, size, diffFile);

  delete[] sramMemory;
  delete[] rramMemory;

  delete[] floatMatrixA;
  delete[] floatMatrixB;
  delete[] floatMatrixC;
  delete[] floatBiasMatrix;
  delete[] floatResidualMatrix;
  delete[] floatDataFileOutput;

  delete[] universalMatrixA;
  delete[] universalMatrixB;
  delete[] universalBiasMatrix;
  delete[] universalResidualMatrix;
  delete[] universalMatrixC;
  delete[] universalDataFileOutput;

  delete[] matrixA;
  delete[] matrixB;
  delete[] biasMatrix;
  delete[] residualMatrix;
  delete[] matrixC;
  delete[] dataFileOutput;

  if (errors == 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }

  return errors;
}

int run_mobilebert() {
  SimplifiedParams params;
  Offsets offsets;
  Files files;
  std::string operation;
  std::string dataDirectory;
  std::string fileOutputPrefix = "test_outputs/";
  bool useDataFile = true;

  // Memory allocation
  INPUT_DATATYPE* acc_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* acc_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  INPUT_DATATYPE* hls_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* hls_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  UniversalPosit* uni_sram_memory = new UniversalPosit[SRAM_MEMORY_SIZE];
  UniversalPosit* uni_rram_memory = new UniversalPosit[RRAM_MEMORY_SIZE];
  float* float_sram_memory = new float[SRAM_MEMORY_SIZE];
  float* float_rram_memory = new float[RRAM_MEMORY_SIZE];

  if (!acc_sram_memory || !acc_rram_memory || !hls_sram_memory ||
      !hls_rram_memory || !uni_sram_memory || !uni_rram_memory ||
      !float_sram_memory || !float_rram_memory) {
    throw std::runtime_error("Failed to allocate simulation memory");
  }

  // Load first layer input
  std::string firstTest = mobilebertOrder[0];
  operation = mobilebertOperations[firstTest];
  params = mobilebert[operation];
  files = mobilebertFiles[firstTest];
  offsets = mobilebertOffsets[firstTest];
  dataDirectory = mobilebertDataDir + "mobilebert_encoder_layer_0_";
  load_inputs(params, dataDirectory + files.inputs_file, useDataFile,
              acc_sram_memory, hls_sram_memory + offsets.INPUT_OFFSET,
              uni_sram_memory + offsets.INPUT_OFFSET,
              float_sram_memory + offsets.INPUT_OFFSET);

  // Set initial scaling factor to 1
  for (int i = 0; i < 512; i++) {
    hls_sram_memory[offsets.INPUT_OFFSET + 128 * 512 + i] = 1;
    uni_sram_memory[offsets.INPUT_OFFSET + 128 * 512 + i] = 1;
    float_sram_memory[offsets.INPUT_OFFSET + 128 * 512 + i] = 1;
  }

  // Execute 24 encoder layers
  for (int i = 0; i < 24; i++) {
    dataDirectory = mobilebertDataDir + "mobilebert_encoder_layer_" +
                    std::to_string(i) + "_";
    for (const std::string& test : mobilebertOrder) {
      if (test.empty()) continue;

      if (test == "classifier") {
        if (i != 23) {
          continue;
        } else {
          dataDirectory = mobilebertDataDir;
          // We take the first token of the entire matrix and copy the scaling
          // factor
          memcpy(hls_sram_memory + 512, hls_sram_memory + 128 * 512,
                 512 * sizeof(INPUT_DATATYPE));
          memcpy(uni_sram_memory + 512, uni_sram_memory + 128 * 512,
                 512 * sizeof(UniversalPosit));
          memcpy(float_sram_memory + 512, float_sram_memory + 128 * 512,
                 512 * sizeof(float));
        }
      }

      auto operationSearch = mobilebertOperations.find(test);
      if (operationSearch != mobilebertOperations.end()) {
        operation = operationSearch->second;
      } else {
        throw std::runtime_error("Operation for " + test + " not found");
      }

#ifndef PIPE_INPUT
      std::cout << "Test input: " << dataDirectory << test << std::endl;
      std::cout << "Operation: " << operation << std::endl;
#endif

      auto paramSearch = mobilebert.find(operation);
      if (paramSearch != mobilebert.end()) {
        params = paramSearch->second;
      } else {
        throw std::runtime_error("Parameters for " + test + " not found");
      }

      auto offsetsSearch = mobilebertOffsets.find(test);
      if (offsetsSearch != mobilebertOffsets.end()) {
        offsets = offsetsSearch->second;
      } else {
        throw std::runtime_error("Offsets for " + test + " not found");
      }

      auto filesSearch = mobilebertFiles.find(test);
      if (filesSearch != mobilebertFiles.end()) {
        files = filesSearch->second;
      } else {
        throw std::runtime_error("Files for " + test + " not found");
      }

      validateMapping(params);
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

      if (params.SOFTMAX) {
        K = 1;
      }

      if (params.FC) {
        K = params.loops[0][params.weightLoopIndex[0]] *
            params.loops[1][params.weightLoopIndex[1]];
      }

      int size = X * Y * K;

#ifndef PIPE_INPUT
      std::cout << "Performing the following operation:" << std::endl;
      std::cout << "(" << X << "x" << Y << "x" << C << ")"
                << " * "
                << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
                << std::endl;

      INPUT_DATATYPE hlsDataFileOutput[size];
      UniversalPosit uniDataFileOutput[size];
      float dataFileOutput[size];

      if (!hlsDataFileOutput || !uniDataFileOutput || !dataFileOutput) {
        throw std::runtime_error(
            "Failed to allocate simulation memory in sequence");
      }

      load_datafile_outputs(params, dataDirectory + files.outputs_file,
                            hlsDataFileOutput, uniDataFileOutput,
                            dataFileOutput);
#endif

      if (params.WEIGHT) {
        load_weights(params, dataDirectory + files.weights_file, useDataFile,
                     acc_rram_memory, hls_rram_memory + offsets.WEIGHT_OFFSET,
                     uni_rram_memory + offsets.WEIGHT_OFFSET,
                     float_rram_memory + offsets.WEIGHT_OFFSET);
      }

      if (params.BIAS) {
        load_bias(params, dataDirectory + files.bias_file, useDataFile,
                  acc_rram_memory, hls_rram_memory + offsets.BIAS_OFFSET,
                  uni_rram_memory + offsets.BIAS_OFFSET,
                  float_rram_memory + offsets.BIAS_OFFSET);
      }

      run_custom_posit_gold_model(
          params, hls_sram_memory + offsets.INPUT_OFFSET,
          (params.WEIGHT ? hls_rram_memory : hls_sram_memory) +
              offsets.WEIGHT_OFFSET,
          hls_sram_memory + offsets.OUTPUT_OFFSET,
          hls_rram_memory + offsets.BIAS_OFFSET,
          hls_sram_memory + offsets.RESIDUAL_OFFSET);

#ifndef PIPE_INPUT
      run_universal_posit_gold_model(
          params, uni_sram_memory + offsets.INPUT_OFFSET,
          (params.WEIGHT ? uni_rram_memory : uni_sram_memory) +
              offsets.WEIGHT_OFFSET,
          uni_sram_memory + offsets.OUTPUT_OFFSET,
          uni_rram_memory + offsets.BIAS_OFFSET,
          uni_sram_memory + offsets.RESIDUAL_OFFSET);
#endif

      run_fp_gold_model(
          params, float_sram_memory + offsets.INPUT_OFFSET,
          (params.WEIGHT ? float_rram_memory : float_sram_memory) +
              offsets.WEIGHT_OFFSET,
          float_sram_memory + offsets.OUTPUT_OFFSET,
          float_rram_memory + offsets.BIAS_OFFSET,
          float_sram_memory + offsets.RESIDUAL_OFFSET);

#ifndef PIPE_INPUT
      std::string diffFile;
      int errors;

      INPUT_DATATYPE hlsScaledMatrix[size];
      UniversalPosit uniScaledMatrix[size];
      float floatScaledMatrix[size];

      memcpy(hlsScaledMatrix, hls_sram_memory + offsets.OUTPUT_OFFSET,
             sizeof(hlsScaledMatrix));
      memcpy(uniScaledMatrix, uni_sram_memory + offsets.OUTPUT_OFFSET,
             sizeof(uniScaledMatrix));
      memcpy(floatScaledMatrix, float_sram_memory + offsets.OUTPUT_OFFSET,
             sizeof(floatScaledMatrix));

#ifdef INPUT_SCALING
      if (test != "classifier") {
        int rowWidth = params.SOFTMAX ? Y : K;
        for (int i = 0; i < X; i++) {
          for (int j = 0; j < rowWidth; j++) {
            hlsScaledMatrix[i * rowWidth + j] *=
                hls_sram_memory[offsets.OUTPUT_OFFSET + size + j];
            uniScaledMatrix[i * rowWidth + j] *=
                uni_sram_memory[offsets.OUTPUT_OFFSET + size + j];
            floatScaledMatrix[i * rowWidth + j] *=
                float_sram_memory[offsets.OUTPUT_OFFSET + size + j];
          }
        }
      }
#endif

      std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
      std::cout << "(reveals bugs in mapping operations to accelerator)"
                << std::endl;
      diffFile = fileOutputPrefix + "hlsgold_vs_pytorch.txt";
      errors =
          compare_arrays(hlsScaledMatrix, hlsDataFileOutput, size, diffFile);

      std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
      std::cout << "(reveals issues in representing float as Posit)"
                << std::endl;
      diffFile = fileOutputPrefix + "universalgold_vs_pytorch.txt";
      errors =
          compare_arrays(uniScaledMatrix, uniDataFileOutput, size, diffFile);

      std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
                << std::endl;
      std::cout
          << "(reveals bugs in implementation of custom HLS Posit operators)"
          << std::endl;
      diffFile = fileOutputPrefix + "hlsgold_vs_universalgold.txt";
      errors = compare_arrays(hlsScaledMatrix, uniScaledMatrix, size, diffFile);

      std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
      std::cout << "(reveals issues in data loading or mapping)" << std::endl;
      diffFile = fileOutputPrefix + "fpgold_vs_pytorch.txt";
      errors =
          compare_arrays(floatScaledMatrix, dataFileOutput, size, diffFile);
      if (errors) {
        std::cout << "Test failed!" << std::endl;
        return errors;
      }

      std::cout << "Test passed!" << std::endl;
#endif

      if (test == "attention_self_query" or test == "attention_self_key" or
          test == "attention_self_value") {
        INPUT_DATATYPE tmpHlsMatrix[size + 128];
        UniversalPosit tmpUniMatrix[size + 128];
        float tmpFloatMatrix[size + 128];

        int addr = offsets.OUTPUT_OFFSET;
        memcpy(tmpHlsMatrix, hls_sram_memory + addr, sizeof(tmpHlsMatrix));
        memcpy(tmpUniMatrix, uni_sram_memory + addr, sizeof(tmpUniMatrix));
        memcpy(tmpFloatMatrix, float_sram_memory + addr,
               sizeof(tmpFloatMatrix));

        for (int i = 0; i < 4; i++) {
          for (int j = 0; j <= 128; j++) {
            for (int k = 0; k < 32; k++) {
              hls_sram_memory[addr] = tmpHlsMatrix[j * 128 + i * 32 + k];
              uni_sram_memory[addr] = tmpUniMatrix[j * 128 + i * 32 + k];
              float_sram_memory[addr++] = tmpFloatMatrix[j * 128 + i * 32 + k];
            }
          }
        }
      }

      if (test == "attention_self_context_layer_3") {
        INPUT_DATATYPE tmpHlsMatrix[128 * 128 + 128];
        UniversalPosit tmpUniMatrix[128 * 128 + 128];
        float tmpFloatMatrix[128 * 128 + 128];

        int addr = 6 * (128 * 128 + 128);
        memcpy(tmpHlsMatrix, hls_sram_memory + addr, sizeof(tmpHlsMatrix));
        memcpy(tmpUniMatrix, uni_sram_memory + addr, sizeof(tmpUniMatrix));
        memcpy(tmpFloatMatrix, float_sram_memory + addr,
               sizeof(tmpFloatMatrix));

        for (int i = 0; i <= 128; i++) {
          for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 32; k++) {
              hls_sram_memory[addr] = tmpHlsMatrix[(j * 129 + i) * 32 + k];
              uni_sram_memory[addr] = tmpUniMatrix[(j * 129 + i) * 32 + k];
              float_sram_memory[addr++] =
                  tmpFloatMatrix[(j * 129 + i) * 32 + k];
            }
          }
        }
      }
    }
  }

  float logit0 = (float)hls_sram_memory[offsets.OUTPUT_OFFSET];
  float logit1 = (float)hls_sram_memory[offsets.OUTPUT_OFFSET + 1];
  int hls_index = logit0 >= logit1 ? 0 : 1;
  std::cout << logit0 << " " << logit1 << " " << hls_index << " ";

  logit0 = float_sram_memory[offsets.OUTPUT_OFFSET];
  logit1 = float_sram_memory[offsets.OUTPUT_OFFSET + 1];
  int float_index = logit0 >= logit1 ? 0 : 1;
  std::cout << logit0 << " " << logit1 << " " << float_index;

  delete acc_sram_memory;
  delete acc_rram_memory;
  delete hls_sram_memory;
  delete hls_rram_memory;
  delete uni_sram_memory;
  delete uni_rram_memory;
  delete float_sram_memory;
  delete float_rram_memory;

  return 0;
}

extern "C" int sc_main(int argc, char* argv[]) {
  const char* testName = std::getenv("TEST");

  if (!testName) {
    return run_mobilebert();
  }

  std::string test(testName);
  std::string dataDir = test == "classifier"
                            ? mobilebertDataDir
                            : mobilebertDataDir + "mobilebert_encoder_layer_0_";
  std::string operation;
  SimplifiedParams params;
  Files files;
  bool useDataFiles = true;

  auto operationSearch = mobilebertOperations.find(test);
  if (operationSearch != mobilebertOperations.end()) {
    operation = operationSearch->second;
  } else {
    throw std::runtime_error("Operation for " + test + " not found");
  }

  auto search = mobilebert.find(operation);
  if (search != mobilebert.end()) {
    params = search->second;
  } else {
    throw std::runtime_error("Test: " + test + " not found");
  }

  auto fileSearch = mobilebertFiles.find(test);
  if (fileSearch != mobilebertFiles.end()) {
    files = fileSearch->second;
  } else {
    throw std::runtime_error("Files for " + test + " not found");
  }

  std::string fileOutputPrefix = "test_outputs/";
  return run_test(params, dataDir, files, useDataFiles, fileOutputPrefix);
}