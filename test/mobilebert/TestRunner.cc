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
#include "test/mobilebert/params.h"
#include "test/mobilebert/training.h"

#define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
// #define RRAM_MEMORY_SIZE (12 * 1024 * 1024)  // RRAM size for TinyBERT
#define RRAM_MEMORY_SIZE (20 * 1024 * 1024)  // RRAM size for MobileBERT

// #define TRAINING

std::string activationDataDir = "data/mobilebert/activations/";
std::string weightDataDir = "data/mobilebert/weights/";
std::string weightScaledDataDir = "data/mobilebert/weights_scaled/";
std::string errorDataDir = "data/mobilebert/errors/";
std::string gradientDataDir = "data/mobilebert/gradients/";

void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
            MemoryMap memoryMap);

std::vector<std::string> parse_csv(std::string& csv) {
  // Remove spaces
  std::remove(csv.begin(), csv.end(), ' ');

  std::stringstream ss(csv);
  std::string token;
  std::vector<std::string> result;

  // Tokenize
  while (std::getline(ss, token, ',')) {
    result.push_back(token);
  }

  return result;
}

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

int run_test(const SimplifiedParams params, const Files files,
             const MemoryMap memoryMap, const std::string dataDir,
             const std::string layerName, const std::string outfilePrefix) {
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

  std::cout << "Performing the following operation:" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;

  INPUT_DATATYPE* matrixA =
      new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C + C];
  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[FX * FY * C * K];
  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[K];
  INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[X * Y * K];
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[X * Y * K + K];
  OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[X * Y * K];

  UniversalPosit* universalMatrixA =
      new UniversalPosit[(STRIDE * X) * (STRIDE * Y) * C + C];
  UniversalPosit* universalMatrixB = new UniversalPosit[FX * FY * C * K];
  UniversalPosit* universalBiasMatrix = new UniversalPosit[K];
  UniversalPosit* universalResidualMatrix = new UniversalPosit[X * Y * K];
  UniversalPosit* universalMatrixC = new UniversalPosit[X * Y * K + K];
  UniversalPosit* universalDataFileOutput = new UniversalPosit[X * Y * K];

  float* floatMatrixA = new float[(STRIDE * X) * (STRIDE * Y) * C + C];
  float* floatMatrixB = new float[FX * FY * C * K];
  float* floatBiasMatrix = new float[K];
  float* floatResidualMatrix = new float[X * Y * K];
  float* floatMatrixC = new float[X * Y * K + K];
  float* floatDataFileOutput = new float[X * Y * K];

  if (files.inputs_file == "attention_self_attention_probs_3") {
  }
  std::string datafile = dataDir + layerName + files.inputs_file;
  load_inputs(params, datafile, true, sramMemory, matrixA, universalMatrixA,
              floatMatrixA);

  if (!files.weights_file.empty()) {
    datafile = (params.WEIGHT ? weightDataDir : dataDir) + layerName +
               files.weights_file;
    load_weights(params, datafile, true,
                 params.WEIGHT ? rramMemory : sramMemory, matrixB,
                 universalMatrixB, floatMatrixB);
  }

  if (params.BIAS) {
    datafile = weightDataDir + layerName + files.bias_file;
    load_bias(params, datafile, true, rramMemory, biasMatrix,
              universalBiasMatrix, floatBiasMatrix);
  }

  if (params.RESIDUAL) {
    datafile = dataDir + layerName + files.residual_file;
    load_residual(params, datafile, true, sramMemory, residualMatrix,
                  universalResidualMatrix, floatResidualMatrix);
  }

  datafile = dataDir + layerName + files.outputs_file;
  load_datafile_outputs(params, datafile, dataFileOutput,
                        universalDataFileOutput, floatDataFileOutput);

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
  // std::endl; diffFile = outfilePrefix + "accel_vs_hlsgold.txt";
  // compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC, X * Y * K,
  //                diffFile);

  std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
  std::cout << "(reveals bugs in mapping operations to accelerator)"
            << std::endl;
  diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
  compare_arrays(matrixC, dataFileOutput, X * Y * K, diffFile);

  std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
  std::cout << "(reveals issues in representing float as Posit)" << std::endl;
  diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
  compare_arrays(universalMatrixC, universalDataFileOutput, X * Y * K,
                 diffFile);

  std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
            << std::endl;
  std::cout << "(reveals bugs in implementation of custom HLS Posit operators)"
            << std::endl;
  diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
  errors += compare_arrays(matrixC, universalMatrixC, X * Y * K, diffFile);

  std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
  std::cout << "(reveals issues in data loading or mapping)" << std::endl;
  diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
  errors +=
      compare_arrays(floatMatrixC, floatDataFileOutput, X * Y * K, diffFile);

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

int run_network(
    const std::map<std::string, SimplifiedParams> mobilebertParams,
    const std::map<std::string, MemoryOffsets> memOffsets,
    const std::map<std::string, Files> testFiles,
    const std::vector<std::pair<std::string, std::string>> operations,
    const std::vector<std::string> comparisons, const std::string dataDir,
    std::string outfilePrefix, bool isBackward) {
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
  auto operation = operations[0];
  SimplifiedParams params = mobilebertParams.at(operation.second);
  MemoryOffsets offsets = memOffsets.at(operation.first);
  Files files = testFiles.at(operation.first);
  std::string datafile = isBackward
                             ? "mobilebert_classifier"
                             : "mobilebert_encoder_layer_0_hidden_states";
  bool useDataFile = true;
  load_inputs(params, dataDir + datafile, useDataFile, acc_sram_memory,
              hls_sram_memory + offsets.INPUT_OFFSET,
              uni_sram_memory + offsets.INPUT_OFFSET,
              float_sram_memory + offsets.INPUT_OFFSET);

#ifdef INPUT_SCALING
  // Set initial scaling factor to 1
  for (int i = 0; i < 512; i++) {
    hls_sram_memory[offsets.INPUT_OFFSET + 128 * 512 + i] = 1;
    uni_sram_memory[offsets.INPUT_OFFSET + 128 * 512 + i] = 1;
    float_sram_memory[offsets.INPUT_OFFSET + 128 * 512 + i] = 1;
  }
#endif

  for (int count = 0; count < 24; count++) {
    int layer = isBackward ? 23 - count : count;
    for (const auto& it : operations) {
      // if (it.first == "attention_self_context_layer_3") return 0;

      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";

      if (it.first == "classifier") {
        if (layer != 23) continue;
        layerName = "";
#ifdef INPUT_SCALING
        // We take the first token of the entire matrix and copy the scaling
        // factor
        memcpy(hls_sram_memory + 512, hls_sram_memory + 128 * 512,
               512 * sizeof(INPUT_DATATYPE));
        memcpy(uni_sram_memory + 512, uni_sram_memory + 128 * 512,
               512 * sizeof(UniversalPosit));
        memcpy(float_sram_memory + 512, float_sram_memory + 128 * 512,
               512 * sizeof(float));
#endif
      }

#ifndef PIPE_INPUT
      std::cout << "Test input: " << layerName << it.first << std::endl;
      std::cout << "Operation: " << it.second << std::endl;
#endif

      std::string test = it.first;
      params = mobilebertParams.at(it.second);
      offsets = memOffsets.at(test);
      files = testFiles.at(test);

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

      int outputSize = X * Y * K;

#ifndef PIPE_INPUT
      std::cout << "Performing the following operation:" << std::endl;
      std::cout << "(" << X << "x" << Y << "x" << C << ")"
                << " * "
                << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
                << std::endl;

      INPUT_DATATYPE hlsDataFileOutput[outputSize];
      UniversalPosit uniDataFileOutput[outputSize];
      float dataFileOutput[outputSize];

      if (!hlsDataFileOutput || !uniDataFileOutput || !dataFileOutput) {
        throw std::runtime_error(
            "Failed to allocate simulation memory in sequence");
      }

      datafile = dataDir + layerName + files.outputs_file;
      load_datafile_outputs(params, datafile, hlsDataFileOutput,
                            uniDataFileOutput, dataFileOutput);
#endif

      if (params.WEIGHT) {
        datafile = weightDataDir + layerName + files.weights_file;
        load_weights(params, datafile, useDataFile, acc_rram_memory,
                     hls_rram_memory + offsets.WEIGHT_OFFSET,
                     uni_rram_memory + offsets.WEIGHT_OFFSET,
                     float_rram_memory + offsets.WEIGHT_OFFSET);
      }

      if (params.BIAS) {
        datafile = weightDataDir + layerName + files.bias_file;
        load_bias(params, datafile, useDataFile, acc_rram_memory,
                  hls_rram_memory + offsets.BIAS_OFFSET,
                  uni_rram_memory + offsets.BIAS_OFFSET,
                  float_rram_memory + offsets.BIAS_OFFSET);
      }

      bool runAccelerator = std::find(comparisons.begin(), comparisons.end(),
                                      "accelerator") != comparisons.end();
      bool runPosit = std::find(comparisons.begin(), comparisons.end(),
                                "posit") != comparisons.end();
      bool runUniversal = std::find(comparisons.begin(), comparisons.end(),
                                    "universal") != comparisons.end();
      bool runFloat32 = std::find(comparisons.begin(), comparisons.end(),
                                  "fp32") != comparisons.end();

      if (runAccelerator) {
        MemoryMap memoryMap = {SRAM, (params.WEIGHT ? RRAM : SRAM), RRAM, SRAM,
                               SRAM};
        params.INPUT_OFFSET = offsets.INPUT_OFFSET;
        params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET;
        params.OUTPUT_OFFSET = offsets.OUTPUT_OFFSET;
        params.BIAS_OFFSET = offsets.BIAS_OFFSET;
        params.RESIDUAL_OFFSET = offsets.RESIDUAL_OFFSET;
        run_op({params}, acc_sram_memory, acc_rram_memory, memoryMap);
      }

      if (runPosit) {
        run_custom_posit_gold_model(
            params, hls_sram_memory + offsets.INPUT_OFFSET,
            (params.WEIGHT ? hls_rram_memory : hls_sram_memory) +
                offsets.WEIGHT_OFFSET,
            hls_sram_memory + offsets.OUTPUT_OFFSET,
            hls_rram_memory + offsets.BIAS_OFFSET,
            hls_sram_memory + offsets.RESIDUAL_OFFSET);
      }

      if (runUniversal) {
        run_universal_posit_gold_model(
            params, uni_sram_memory + offsets.INPUT_OFFSET,
            (params.WEIGHT ? uni_rram_memory : uni_sram_memory) +
                offsets.WEIGHT_OFFSET,
            uni_sram_memory + offsets.OUTPUT_OFFSET,
            uni_rram_memory + offsets.BIAS_OFFSET,
            uni_sram_memory + offsets.RESIDUAL_OFFSET);
      }

      if (runFloat32) {
        run_fp_gold_model(
            params, float_sram_memory + offsets.INPUT_OFFSET,
            (params.WEIGHT ? float_rram_memory : float_sram_memory) +
                offsets.WEIGHT_OFFSET,
            float_sram_memory + offsets.OUTPUT_OFFSET,
            float_rram_memory + offsets.BIAS_OFFSET,
            float_sram_memory + offsets.RESIDUAL_OFFSET);
      }

#ifndef PIPE_INPUT
      INPUT_DATATYPE hlsScaledMatrix[outputSize];
      UniversalPosit uniScaledMatrix[outputSize];
      float floatScaledMatrix[outputSize];

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
                hls_sram_memory[offsets.OUTPUT_OFFSET + outputSize + j];
            uniScaledMatrix[i * rowWidth + j] *=
                uni_sram_memory[offsets.OUTPUT_OFFSET + outputSize + j];
            floatScaledMatrix[i * rowWidth + j] *=
                float_sram_memory[offsets.OUTPUT_OFFSET + outputSize + j];
          }
        }
      }
#endif

      std::string diffFile;
      int errors;
      if (runAccelerator and runPosit) {
        std::cout << "Accelerator vs. HLS Posit Gold Model" << std::endl;
        std::cout << "(reveals bugs in accelerator or memory placement)"
                  << std::endl;
        diffFile = outfilePrefix + "accel_vs_hlsgold.txt";
        errors = compare_arrays(&acc_sram_memory[params.OUTPUT_OFFSET],
                                hlsScaledMatrix, outputSize, diffFile);
      }

      if (runPosit and runFloat32) {
        std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
        std::cout << "(reveals bugs in mapping operations to accelerator)"
                  << std::endl;
        diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
        errors = compare_arrays(hlsScaledMatrix, hlsDataFileOutput, outputSize,
                                diffFile);
      }

      if (runUniversal and runFloat32) {
        std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
        std::cout << "(reveals issues in representing float as Posit)"
                  << std::endl;
        diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
        errors = compare_arrays(uniScaledMatrix, uniDataFileOutput, outputSize,
                                diffFile);
      }

      if (runPosit && runUniversal) {
        std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
                  << std::endl;
        std::cout
            << "(reveals bugs in implementation of custom HLS Posit operators)"
            << std::endl;
        diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
        errors = compare_arrays(hlsScaledMatrix, uniScaledMatrix, outputSize,
                                diffFile);
      }

      if (runFloat32) {
        std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
        std::cout << "(reveals issues in data loading or mapping)" << std::endl;
        diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
        errors = compare_arrays(floatScaledMatrix, dataFileOutput, outputSize,
                                diffFile);
      }

      if (errors) {
        std::cout << "Test failed!" << std::endl;
        return errors;
      }

      std::cout << "Test passed!" << std::endl;
#endif

      // if (test == "attention_self_context_layer_3") {
      //   INPUT_DATATYPE tmpHlsMatrix[HIDDEN_SIZE];
      //   UniversalPosit tmpUniMatrix[HIDDEN_SIZE];
      //   float tmpFloatMatrix[HIDDEN_SIZE];

      //   int addr = 6 * HIDDEN_SIZE;
      //   memcpy(tmpHlsMatrix, hls_sram_memory + addr, sizeof(tmpHlsMatrix));
      //   memcpy(tmpUniMatrix, uni_sram_memory + addr, sizeof(tmpUniMatrix));
      //   memcpy(tmpFloatMatrix, float_sram_memory + addr,
      //          sizeof(tmpFloatMatrix));

      //   for (int i = 0; i <= 128; i++) {
      //     for (int j = 0; j < 4; j++) {
      //       for (int k = 0; k < 32; k++) {
      //         hls_sram_memory[addr] = tmpHlsMatrix[(j * 129 + i) * 32 + k];
      //         uni_sram_memory[addr] = tmpUniMatrix[(j * 129 + i) * 32 + k];
      //         float_sram_memory[addr++] =
      //             tmpFloatMatrix[(j * 129 + i) * 32 + k];
      //       }
      //     }
      //   }
      // }

      if (isBackward) {
        // Testbench code handling ReLU activation
        if (test == "output_dense" || test == "ffn_2_output_dense" ||
            test == "ffn_1_output_dense" || test == "ffn_0_output_dense") {
          std::cerr << "Testbench code handling ReLU activation" << std::endl;
          INPUT_DATATYPE tmpHlsMatrix[INTERMEDIATE_SIZE];
          UniversalPosit tmpUniMatrix[INTERMEDIATE_SIZE];
          float tmpFloatMatrix[INTERMEDIATE_SIZE];

          params = mobilebertParams.at("ffn1");
          std::size_t pos = test.find("output_dense");
          datafile = activationDataDir + layerName + test.substr(0, pos) +
                     "intermediate_intermediate_act_fn";
          load_datafile_outputs(params, datafile, tmpHlsMatrix, tmpUniMatrix,
                                tmpFloatMatrix);

          for (int i = 0; i < INTERMEDIATE_SIZE; i++) {
            if (tmpFloatMatrix[i] <= 0) {
              hls_sram_memory[offsets.OUTPUT_OFFSET + i] = 0;
              uni_sram_memory[offsets.OUTPUT_OFFSET + i] = 0;
              float_sram_memory[offsets.OUTPUT_OFFSET + i] = 0;
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
  const char* compNames = std::getenv("SIMS");

  if (!compNames) {
    std::cout << "You must set the environment variable SIMS" << std::endl;
  }

  std::string test(testName);
  std::string comps(compNames);

  std::vector<std::string> compList = parse_csv(comps);
  std::string outfilePrefix = "test_outputs/";

  if (test == "inference") {
    return run_network(inferenceParams, inferenceMemOffsets, inferenceTestFiles,
                       inferenceOperations, compList, activationDataDir,
                       outfilePrefix, false);
  }

  if (test == "training") {
    return run_network(trainingParams, trainingMemOffsets, trainingTestFiles,
                       trainingOperations, compList, errorDataDir,
                       outfilePrefix, true);
  }

#ifndef TRAINING
  for (auto op : inferenceOperations) {
#else
  for (auto op : trainingOperations) {
#endif
    if (op.first == test) {
#ifndef TRAINING
      SimplifiedParams params = inferenceParams.at(op.second);
      Files files = inferenceTestFiles.at(op.first);
      MemoryOffsets memOffsets = inferenceMemOffsets.at(op.first);
      std::string dataDir = activationDataDir;
#else
      SimplifiedParams params = trainingParams.at(op.second);
      Files files = trainingTestFiles.at(op.first);
      MemoryOffsets memOffsets = trainingMemOffsets.at(op.first);
      std::string dataDir = errorDataDir;
#endif
      params.INPUT_OFFSET = memOffsets.INPUT_OFFSET;
      params.WEIGHT_OFFSET = memOffsets.WEIGHT_OFFSET;
      params.OUTPUT_OFFSET = memOffsets.OUTPUT_OFFSET;
      params.BIAS_OFFSET = memOffsets.BIAS_OFFSET;
      params.RESIDUAL_OFFSET = memOffsets.RESIDUAL_OFFSET;

      MemoryMap memoryMap = {SRAM, params.WEIGHT ? RRAM : SRAM, SRAM, SRAM,
                             SRAM};

      std::string layerName =
          test == "classifier" ? "" : "mobilebert_encoder_layer_0_";

      std::cerr << "test name: " << op.first << std::endl;
      std::cerr << "operation name: " << op.second << std::endl;

      return run_test(params, files, memoryMap, dataDir, layerName,
                      outfilePrefix);
    }
  }

  throw;
}