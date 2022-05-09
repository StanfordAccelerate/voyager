#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
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

std::string activationDataDir =
    "/sim/jeffreyy/accelerator/data/mobilebert/activations/";
std::string weightDataDir =
    "/sim/jeffreyy/accelerator/data/mobilebert/weights/";
std::string weightScaledDataDir =
    "/sim/jeffreyy/accelerator/data/mobilebert/weights_scaled/";
std::string errorDataDir = "/sim/jeffreyy/accelerator/data/mobilebert/errors/";
std::string gradientDataDir =
    "/sim/jeffreyy/accelerator/data/mobilebert/gradients/";

void validateMapping(SimplifiedParams params);
void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
            MemoryMap memoryMap);

int runMbUnitTest(const SimplifiedParams params, const Files files,
                  const MemoryMap memoryMap, const std::string dataDir,
                  const std::string layerName, const std::string outfilePrefix,
                  const std::vector<std::string> models) {
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

  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    K = 1;
    C = 1;
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
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[X * Y * K + K];
  OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[X * Y * K];

  UniversalPosit* universalMatrixA =
      new UniversalPosit[(STRIDE * X) * (STRIDE * Y) * C];
  UniversalPosit* universalMatrixB = new UniversalPosit[FX * FY * C * K];
  UniversalPosit* universalBiasMatrix = new UniversalPosit[K];
  UniversalPosit* universalResidualMatrix = new UniversalPosit[X * Y * K];
  UniversalPosit* universalMatrixC = new UniversalPosit[X * Y * K + K];
  UniversalPosit* universalDataFileOutput = new UniversalPosit[X * Y * K];

  float* floatMatrixA = new float[(STRIDE * X) * (STRIDE * Y) * C];
  float* floatMatrixB = new float[FX * FY * C * K];
  float* floatBiasMatrix = new float[K];
  float* floatResidualMatrix = new float[X * Y * K];
  float* floatMatrixC = new float[X * Y * K + K];
  float* floatDataFileOutput = new float[X * Y * K];

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

  bool runAccelerator =
      std::find(models.begin(), models.end(), "accelerator") != models.end();
  bool runHlsposit =
      std::find(models.begin(), models.end(), "hlsposit") != models.end();
  bool runUniversal =
      std::find(models.begin(), models.end(), "universal") != models.end();
  bool runFloat32 =
      std::find(models.begin(), models.end(), "fp32") != models.end();

  if (runAccelerator) {
    run_op({params}, sramMemory, rramMemory, memoryMap);
  }

  if (runHlsposit) {
    run_custom_posit_gold_model(params, matrixA, matrixB, matrixC, biasMatrix,
                                residualMatrix);
  }

  if (runUniversal) {
    run_universal_posit_gold_model(params, universalMatrixA, universalMatrixB,
                                   universalMatrixC, universalBiasMatrix,
                                   universalResidualMatrix);
  }

  if (runFloat32) {
    run_fp_gold_model(params, floatMatrixA, floatMatrixB, floatMatrixC,
                      floatBiasMatrix, floatResidualMatrix);
  }

  std::string diffFile;
  int errors = 0;

  if (runUniversal) {
    std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in representing float as Posit)" << std::endl;
    diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
    compare_arrays(universalMatrixC, universalDataFileOutput, X * Y * K,
                   diffFile);
  }

  if (runHlsposit) {
    std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals bugs in mapping operations to accelerator)"
              << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
    compare_arrays(matrixC, dataFileOutput, X * Y * K, diffFile);
  }

  if (runAccelerator) {
    std::cout << "Accelerator vs. PyTorch" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_pytorch.txt";
    errors += compare_arrays(&sramMemory[params.OUTPUT_OFFSET], dataFileOutput,
                             X * Y * K, diffFile);
  }

  if (runAccelerator && runHlsposit) {
    std::cout << "Accelerator vs. HLS Posit Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_hlsgold.txt";
    errors += compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC,
                             X * Y * K, diffFile);
  }

  if (runHlsposit && runUniversal) {
    std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
              << std::endl;
    std::cout
        << "(reveals bugs in implementation of custom HLS Posit operators)"
        << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
    errors += compare_arrays(matrixC, universalMatrixC, X * Y * K, diffFile);
  }

  if (runFloat32) {
    std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
    errors +=
        compare_arrays(floatMatrixC, floatDataFileOutput, X * Y * K, diffFile);
  }

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

int runMobilebert(
    const std::map<std::string, SimplifiedParams> mobilebertParams,
    const std::map<std::string, MemoryOffsets> memOffsets,
    const std::map<std::string, Files> testFiles,
    const std::vector<std::pair<std::string, std::string>> operations,
    const std::vector<std::string> models, const std::string dataDir,
    std::string outfilePrefix, bool backprop) {
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

  // Load the input of the first layer
  auto operation = operations[0];
  SimplifiedParams params = mobilebertParams.at(operation.second);
  MemoryOffsets offsets = memOffsets.at(operation.first);
  Files files = testFiles.at(operation.first);
  std::string datafile = backprop ? "mobilebert_classifier"
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

  // Load weights and biases
  for (int layer = 0; layer < 24; layer++) {
    for (const auto& it : operations) {
      if (it.first == "classifier" && layer != 23) continue;
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      params = mobilebertParams.at(it.second);
      offsets = memOffsets.at(it.first);
      files = testFiles.at(it.first);

      params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET + layer * ENCODER_SIZE;
      params.BIAS_OFFSET = offsets.BIAS_OFFSET + layer * ENCODER_SIZE;

      if (it.first == "classifier") {
        layerName = "";
        params.WEIGHT_OFFSET += ENCODER_SIZE;
        params.BIAS_OFFSET += ENCODER_SIZE;
      }

      if (params.WEIGHT) {
        datafile = weightDataDir + layerName + files.weights_file;
        load_weights(params, datafile, useDataFile, acc_rram_memory,
                     hls_rram_memory + params.WEIGHT_OFFSET,
                     uni_rram_memory + params.WEIGHT_OFFSET,
                     float_rram_memory + params.WEIGHT_OFFSET);
      }

      if (params.BIAS) {
        datafile = weightDataDir + layerName + files.bias_file;
        load_bias(params, datafile, useDataFile, acc_rram_memory,
                  hls_rram_memory + params.BIAS_OFFSET,
                  uni_rram_memory + params.BIAS_OFFSET,
                  float_rram_memory + params.BIAS_OFFSET);
      }
    }
  }

  std::vector<SimplifiedParams> opParams;

  for (int count = 0; count < 24; count++) {
    int layer = backprop ? 23 - count : count;
    for (const auto& it : operations) {
      std::string test = it.first;
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      params = mobilebertParams.at(it.second);
      offsets = memOffsets.at(test);
      files = testFiles.at(test);

      params.INPUT_OFFSET = offsets.INPUT_OFFSET;
      params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET;
      params.OUTPUT_OFFSET = offsets.OUTPUT_OFFSET;
      params.BIAS_OFFSET = offsets.BIAS_OFFSET + layer * ENCODER_SIZE;
      params.RESIDUAL_OFFSET = offsets.RESIDUAL_OFFSET;

      opParams.push_back(params);

      if (params.WEIGHT) {
        params.WEIGHT_OFFSET += layer * ENCODER_SIZE;
      }

      if (test == "classifier") {
        if (layer != 23) continue;
        layerName = "";
        params.WEIGHT_OFFSET += ENCODER_SIZE;
        params.BIAS_OFFSET += ENCODER_SIZE;
#ifdef INPUT_SCALING
        // We take the first token of the entire matrix and copy the scaling
        // factor
        std::memcpy(hls_sram_memory + 512, hls_sram_memory + 128 * 512,
                    512 * sizeof(INPUT_DATATYPE));
        std::memcpy(uni_sram_memory + 512, uni_sram_memory + 128 * 512,
                    512 * sizeof(UniversalPosit));
        std::memcpy(float_sram_memory + 512, float_sram_memory + 128 * 512,
                    512 * sizeof(float));
#endif
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

      if (params.SOFTMAX || params.SOFTMAX_GRAD) {
        K = 1;
      }

      int outputSize = X * Y * K;

#ifndef PIPE_INPUT
      std::cout << "Performing " + layerName + test + ":" << std::endl;
      std::cout << "(" << X << "x" << Y << "x" << C << ")"
                << " * "
                << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
                << std::endl;
#endif

      if (backprop) {
        // ReLU activation
        if (files.inputs_file.find("intermediate_dense") != std::string::npos) {
          INPUT_DATATYPE tmpHlsMatrix[INTERMEDIATE_SIZE];
          UniversalPosit tmpUniMatrix[INTERMEDIATE_SIZE];
          float tmpFloatMatrix[INTERMEDIATE_SIZE];

          params = mobilebertParams.at("ffn1");
          std::size_t pos = files.inputs_file.find("intermediate_dense");
          datafile = activationDataDir + layerName +
                     files.inputs_file.substr(0, pos) + "intermediate_dense";
          load_datafile_outputs(params, datafile, tmpHlsMatrix, tmpUniMatrix,
                                tmpFloatMatrix);

          for (int i = 0; i < INTERMEDIATE_SIZE; i++) {
            if (tmpFloatMatrix[i] <= 0) {
              hls_sram_memory[params.OUTPUT_OFFSET + i] = 0;
              uni_sram_memory[params.OUTPUT_OFFSET + i] = 0;
              float_sram_memory[params.OUTPUT_OFFSET + i] = 0;
            }
          }
        }

        if (test.find("attention_self_value_layer") != std::string::npos) {
          datafile = activationDataDir + layerName + files.inputs_file;
          load_inputs(params, datafile, useDataFile, acc_sram_memory,
                      hls_sram_memory + params.INPUT_OFFSET,
                      uni_sram_memory + params.INPUT_OFFSET,
                      float_sram_memory + params.INPUT_OFFSET);
        }

        if (test.find("attention_self_attention_probs") != std::string::npos ||
            test.find("attention_self_query_layer") != std::string::npos ||
            test.find("attention_self_key_layer") != std::string::npos) {
          datafile = activationDataDir + layerName + files.weights_file;
          load_weights(params, datafile, useDataFile, acc_sram_memory,
                       hls_sram_memory + params.WEIGHT_OFFSET,
                       uni_sram_memory + params.WEIGHT_OFFSET,
                       float_sram_memory + params.WEIGHT_OFFSET);
        }

        if (test.find("attention_self_attention_scores") != std::string::npos) {
          datafile = activationDataDir + layerName + files.residual_file;
          load_residual(params, datafile, useDataFile, acc_sram_memory,
                        hls_sram_memory + params.RESIDUAL_OFFSET,
                        uni_sram_memory + params.RESIDUAL_OFFSET,
                        float_sram_memory + params.RESIDUAL_OFFSET);
        }
      }

      bool runHlsposit =
          std::find(models.begin(), models.end(), "hlsposit") != models.end();
      bool runUniversal =
          std::find(models.begin(), models.end(), "universal") != models.end();
      bool runFloat32 =
          std::find(models.begin(), models.end(), "fp32") != models.end();

      if (runHlsposit) {
        run_custom_posit_gold_model(
            params, hls_sram_memory + params.INPUT_OFFSET,
            (params.WEIGHT ? hls_rram_memory : hls_sram_memory) +
                params.WEIGHT_OFFSET,
            hls_sram_memory + params.OUTPUT_OFFSET,
            hls_rram_memory + params.BIAS_OFFSET,
            hls_sram_memory + params.RESIDUAL_OFFSET);
      }

      if (runUniversal) {
        run_universal_posit_gold_model(
            params, uni_sram_memory + params.INPUT_OFFSET,
            (params.WEIGHT ? uni_rram_memory : uni_sram_memory) +
                params.WEIGHT_OFFSET,
            uni_sram_memory + params.OUTPUT_OFFSET,
            uni_rram_memory + params.BIAS_OFFSET,
            uni_sram_memory + params.RESIDUAL_OFFSET);
      }

      if (runFloat32) {
        run_fp_gold_model(
            params, float_sram_memory + params.INPUT_OFFSET,
            (params.WEIGHT ? float_rram_memory : float_sram_memory) +
                params.WEIGHT_OFFSET,
            float_sram_memory + params.OUTPUT_OFFSET,
            float_rram_memory + params.BIAS_OFFSET,
            float_sram_memory + params.RESIDUAL_OFFSET);
      }

#ifndef PIPE_INPUT
      // Multiply the tensor by scaling factor for comparison
      INPUT_DATATYPE hlsOutputMatrix[outputSize];
      UniversalPosit uniOutputMatrix[outputSize];
      float floatOutputMatrix[outputSize];

      std::memcpy(hlsOutputMatrix, hls_sram_memory + params.OUTPUT_OFFSET,
                  sizeof(hlsOutputMatrix));
      std::memcpy(uniOutputMatrix, uni_sram_memory + params.OUTPUT_OFFSET,
                  sizeof(uniOutputMatrix));
      std::memcpy(floatOutputMatrix, float_sram_memory + params.OUTPUT_OFFSET,
                  sizeof(floatOutputMatrix));

#ifdef INPUT_SCALING
      if (test != "classifier") {
        int rowWidth = params.SOFTMAX ? Y : K;
        for (int i = 0; i < X; i++) {
          for (int j = 0; j < rowWidth; j++) {
            hlsOutputMatrix[i * rowWidth + j] *=
                hls_sram_memory[params.OUTPUT_OFFSET + outputSize + j];
            uniOutputMatrix[i * rowWidth + j] *=
                uni_sram_memory[params.OUTPUT_OFFSET + outputSize + j];
            floatOutputMatrix[i * rowWidth + j] *=
                float_sram_memory[params.OUTPUT_OFFSET + outputSize + j];
          }
        }
      }
#endif

      if (params.SPLIT_HEAD) continue;

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

      std::string diffFile;
      int errors;
      if (runHlsposit) {
        std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
        std::cout << "(reveals bugs in mapping operations to accelerator)"
                  << std::endl;
        diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
        errors = compare_arrays(hlsOutputMatrix, hlsDataFileOutput, outputSize,
                                diffFile);
      }

      if (runUniversal) {
        std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
        std::cout << "(reveals issues in representing float as Posit)"
                  << std::endl;
        diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
        errors = compare_arrays(uniOutputMatrix, uniDataFileOutput, outputSize,
                                diffFile);
      }

      if (runHlsposit && runUniversal) {
        std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
                  << std::endl;
        std::cout
            << "(reveals bugs in implementation of custom HLS Posit operators)"
            << std::endl;
        diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
        errors = compare_arrays(hlsOutputMatrix, uniOutputMatrix, outputSize,
                                diffFile);
      }

      if (runFloat32) {
        std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
        std::cout << "(reveals issues in data loading or mapping)" << std::endl;
        diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
        errors = compare_arrays(floatOutputMatrix, dataFileOutput, outputSize,
                                diffFile);
      }

      if (errors) {
        std::cout << "Test failed!" << std::endl;
        return errors;
      }

      std::cout << "Test passed!" << std::endl;

      // if (test == "ffn_0_intermediate_dense") return 1;
#endif
    }
  }

  // if (std::find(models.begin(), models.end(), "accelerator") != models.end())
  // {
  //   MemoryMap memoryMap = {SRAM, RRAM, RRAM, SRAM, SRAM};
  //   params.INPUT_OFFSET = params.INPUT_OFFSET;
  //   params.WEIGHT_OFFSET = params.WEIGHT_OFFSET;
  //   params.OUTPUT_OFFSET = params.OUTPUT_OFFSET;
  //   params.BIAS_OFFSET = params.BIAS_OFFSET;
  //   params.RESIDUAL_OFFSET = params.RESIDUAL_OFFSET;
  //   run_op(opParams, acc_sram_memory, acc_rram_memory, memoryMap);
  // }

  float logit0 = (float)hls_sram_memory[params.OUTPUT_OFFSET];
  float logit1 = (float)hls_sram_memory[params.OUTPUT_OFFSET + 1];
  int hls_index = logit0 >= logit1 ? 0 : 1;
  std::cout << logit0 << " " << logit1 << " " << hls_index << " ";

  logit0 = float_sram_memory[params.OUTPUT_OFFSET];
  logit1 = float_sram_memory[params.OUTPUT_OFFSET + 1];
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

int runMbTest(std::string task, std::string test,
              std::vector<std::string> compList) {
  std::string outfilePrefix = "test_outputs/";

  std::vector<std::pair<std::string, std::string>> operations;
  std::map<std::string, SimplifiedParams> mobilebertParams;
  std::map<std::string, Files> mobilebertFiles;
  std::map<std::string, MemoryOffsets> mobilebertMemOffsets;
  std::string dataDir;

  if (task == "inference") {
    operations = inferenceOperations;
    mobilebertParams = inferenceParams;
    mobilebertFiles = inferenceTestFiles;
    mobilebertMemOffsets = inferenceMemOffsets;
    dataDir = activationDataDir;
  } else if (task == "training") {
    operations = trainingOperations;
    mobilebertParams = trainingParams;
    mobilebertFiles = trainingTestFiles;
    mobilebertMemOffsets = trainingMemOffsets;
    dataDir = errorDataDir;
  } else {
    throw std::invalid_argument("ERROR: Task unfound!");
  }

  if (test == "all") {
    return runMobilebert(mobilebertParams, mobilebertMemOffsets,
                         mobilebertFiles, operations, compList, dataDir,
                         outfilePrefix, task == "training");
  }

  for (auto op : operations) {
    if (op.first == test) {
      SimplifiedParams params = mobilebertParams.at(op.second);
      Files files = mobilebertFiles.at(op.first);
      MemoryOffsets memOffsets = mobilebertMemOffsets.at(op.first);
      std::string layerName =
          test == "classifier" ? "" : "mobilebert_encoder_layer_23_";

      params.INPUT_OFFSET = memOffsets.INPUT_OFFSET;
      params.WEIGHT_OFFSET = memOffsets.WEIGHT_OFFSET;
      params.OUTPUT_OFFSET = memOffsets.OUTPUT_OFFSET;
      params.BIAS_OFFSET = memOffsets.BIAS_OFFSET;
      params.RESIDUAL_OFFSET = memOffsets.RESIDUAL_OFFSET;

      MemoryMap memoryMap = {SRAM, (params.WEIGHT ? RRAM : SRAM), RRAM, SRAM,
                             SRAM};

      std::cout << "test name: " << op.first << std::endl;
      std::cout << "operation name: " << op.second << std::endl;

      return runMbUnitTest(params, files, memoryMap, dataDir, layerName,
                           outfilePrefix, compList);
    }
  }

  return 1;
}