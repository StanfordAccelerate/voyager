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
#include "test/mobilebert/backprop.h"
#include "test/mobilebert/gradient.h"
#include "test/mobilebert/inference.h"

#ifndef SRAM_MEMORY_SIZE
#define SRAM_MEMORY_SIZE (30 * 1024 * 1024)
#endif

#ifndef RRAM_MEMORY_SIZE
// #define RRAM_MEMORY_SIZE (12 * 1024 * 1024)  // RRAM size for TinyBERT
#define RRAM_MEMORY_SIZE (22 * 1024 * 1024)  // RRAM size for MobileBERT
#endif

void validateMapping(SimplifiedParams params);
void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
            MemoryMap memoryMap);

int runOperation(const SimplifiedParams params, const Files files,
                 const MemoryMap memoryMap, const std::string inputDataDir,
                 const std::string weightDataDir,
                 const std::string outputDataDir,
                 const std::string residualDataDir,
                 const std::string gradDataDir, const std::string layerName,
                 const std::string outfilePrefix,
                 const std::vector<std::string> groups) {
  validateMapping(params);

  INPUT_DATATYPE* sramMemory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* rramMemory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];

  if (sramMemory == nullptr || rramMemory == nullptr) {
    throw std::runtime_error("Failed to allocate accelerator memory");
  }

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

  if (params.SOFTMAX && params.ATTENTION_MASK) {
    C = X;
    K = 1;
  } else if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    K = 1;
    C = 1;
  }

  std::cout << "Performing the following operation:" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;

  int inputSize = (STRIDE * X) * (STRIDE * Y) * C;
  int weightSize = FX * FY * C * K;
  int biasSize = 2 * K;
  int outputSize = X * Y * K;

  if (params.NO_NORM_GRAD) {
    outputSize = K;
  } else if (params.CROSS_ENTROPY_GRAD) {
    outputSize = X;
  }

  if (params.ACC_T_INPUT) {
    inputSize = 2 * inputSize;
  }
  if (params.ACC_T_WEIGHT) {
    weightSize = 2 * weightSize;
  }

  INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[inputSize];
  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[weightSize];
  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[biasSize];
  INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[2 * outputSize];
  INPUT_DATATYPE* weightGradMatrix = new INPUT_DATATYPE[weightSize];
  INPUT_DATATYPE* biasGradMatrix = new INPUT_DATATYPE[biasSize];
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[2 * outputSize];
  OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[2 * outputSize];

  UniversalPosit* universalMatrixA = new UniversalPosit[inputSize];
  UniversalPosit* universalMatrixB = new UniversalPosit[weightSize];
  UniversalPosit* universalBiasMatrix = new UniversalPosit[biasSize];
  UniversalPosit* universalResidualMatrix = new UniversalPosit[2 * outputSize];
  UniversalPosit* universalWeightGradMatrix = new UniversalPosit[weightSize];
  UniversalPosit* universalBiasGradMatrix = new UniversalPosit[biasSize];
  UniversalPosit* universalMatrixC = new UniversalPosit[2 * outputSize];
  UniversalPosit* universalDataFileOutput = new UniversalPosit[2 * outputSize];

  float* floatMatrixA = new float[inputSize];
  float* floatMatrixB = new float[weightSize];
  float* floatBiasMatrix = new float[biasSize];
  float* floatResidualMatrix = new float[2 * outputSize];
  float* floatWeightGradMatrix = new float[weightSize];
  float* floatBiasGradMatrix = new float[biasSize];
  float* floatMatrixC = new float[2 * outputSize];
  float* floatDataFileOutput = new float[2 * outputSize];

  std::string datafile;
  if (!files.inputs_file.empty()) {
    datafile = inputDataDir + layerName + files.inputs_file;
    load_inputs(params, datafile, true, sramMemory, matrixA, universalMatrixA,
                floatMatrixA);
  }

  if (!files.weights_file.empty()) {
    datafile = params.ATTENTION_MASK
                   ? inputDataDir + files.weights_file
                   : weightDataDir + layerName + files.weights_file;
    load_weights(params, datafile, true,
                 params.WEIGHT ? rramMemory : sramMemory, matrixB,
                 universalMatrixB, floatMatrixB);
  }

  if (params.BIAS) {
    datafile = weightDataDir + layerName + files.bias_file;
    load_bias(params, datafile, true, rramMemory, biasMatrix,
              universalBiasMatrix, floatBiasMatrix);
  }

  if (params.RESIDUAL || params.RELU_GRAD || params.SOFTMAX_GRAD) {
    datafile = residualDataDir + layerName + files.residual_file;
    load_residual(params, datafile, true, sramMemory, residualMatrix,
                  universalResidualMatrix, floatResidualMatrix);
  }

  if (params.WEIGHT_SPLITTING && params.WEIGHT) {
    datafile = gradDataDir + layerName + files.weights_file;
    load_weights(params, datafile, true, sramMemory, weightGradMatrix,
                 universalWeightGradMatrix, floatWeightGradMatrix);
  }

  if (params.WEIGHT_SPLITTING && params.BIAS) {
    datafile = gradDataDir + layerName + files.bias_file;
    load_bias(params, datafile, true, sramMemory, biasGradMatrix,
              universalBiasGradMatrix, floatBiasGradMatrix);
  }

  datafile = outputDataDir + layerName + files.outputs_file;
  load_datafile_outputs(params, datafile, dataFileOutput,
                        universalDataFileOutput, floatDataFileOutput);

  bool accelerator =
      std::find(groups.begin(), groups.end(), "accelerator") != groups.end();
  bool customposit =
      std::find(groups.begin(), groups.end(), "customposit") != groups.end();
  bool universal =
      std::find(groups.begin(), groups.end(), "universal") != groups.end();
  bool fp32 = std::find(groups.begin(), groups.end(), "fp32") != groups.end();

  if (accelerator) {
    run_op({params}, sramMemory, rramMemory, memoryMap);
  }

  if (customposit) {
    run_custom_posit_gold_model(params, matrixA, matrixB, matrixC, biasMatrix,
                                residualMatrix, weightGradMatrix,
                                biasGradMatrix);
  }

  if (universal) {
    run_universal_posit_gold_model(
        params, universalMatrixA, universalMatrixB, universalMatrixC,
        universalBiasMatrix, universalResidualMatrix, universalWeightGradMatrix,
        universalBiasGradMatrix);
  }

  if (fp32) {
    run_fp_gold_model(params, floatMatrixA, floatMatrixB, floatMatrixC,
                      floatBiasMatrix, floatResidualMatrix,
                      floatWeightGradMatrix, floatBiasGradMatrix);
  }

  std::string diffFile;
  int errors = 0;
  if (universal) {
    std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in representing float as Posit)" << std::endl;
    diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
    compare_arrays(universalMatrixC, floatDataFileOutput, outputSize, diffFile,
                   params.ACC_T_OUTPUT);
  }

  if (customposit) {
    std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals bugs in mapping operations to accelerator)"
              << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
    compare_arrays(matrixC, floatDataFileOutput, outputSize, diffFile,
                   params.ACC_T_OUTPUT);
    // compare_arrays(matrixC, dataFileOutput, outputSize, diffFile,
    //                params.ACC_T_OUTPUT);
  }

  if (accelerator) {
    std::cout << "Accelerator vs. PyTorch" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_pytorch.txt";
    errors +=
        compare_arrays(&sramMemory[params.OUTPUT_OFFSET], floatDataFileOutput,
                       outputSize, diffFile, params.ACC_T_OUTPUT);
  }

  if (accelerator && customposit) {
    std::cout << "Accelerator vs. HLS Posit Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_hlsgold.txt";
    errors += compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC,
                             outputSize, diffFile, params.ACC_T_OUTPUT);
  }

  if (customposit && universal) {
    std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
              << std::endl;
    std::cout
        << "(reveals bugs in implementation of custom HLS Posit operators)"
        << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
    errors += compare_arrays(matrixC, universalMatrixC, outputSize, diffFile,
                             params.ACC_T_OUTPUT);
  }

  if (fp32) {
    std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
    errors += compare_arrays(floatMatrixC, floatDataFileOutput, outputSize,
                             diffFile, params.ACC_T_OUTPUT);
  }

  delete[] sramMemory;
  delete[] rramMemory;

  delete[] floatMatrixA;
  delete[] floatMatrixB;
  delete[] floatMatrixC;
  delete[] floatBiasMatrix;
  delete[] floatResidualMatrix;
  delete[] floatDataFileOutput;
  delete[] floatWeightGradMatrix;
  delete[] floatBiasGradMatrix;

  delete[] universalMatrixA;
  delete[] universalMatrixB;
  delete[] universalBiasMatrix;
  delete[] universalResidualMatrix;
  delete[] universalMatrixC;
  delete[] universalDataFileOutput;
  delete[] universalWeightGradMatrix;
  delete[] universalBiasGradMatrix;

  delete[] matrixA;
  delete[] matrixB;
  delete[] biasMatrix;
  delete[] residualMatrix;
  delete[] matrixC;
  delete[] dataFileOutput;
  delete[] weightGradMatrix;
  delete[] biasGradMatrix;

  if (errors == 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }

  return errors;
}

int runMobileBertUnitTest(std::string task, std::string test,
                          std::vector<std::string> compList,
                          std::string datapath) {
  std::string activationDataDir = datapath + "activations/";
  std::string weightDataDir = datapath + "weights/";
  std::string errorDataDir = datapath + "errors/";
  std::string gradientDataDir = datapath + "gradients/";
  std::string outfilePrefix = "test_outputs/" + test + ".";

  std::map<std::string, std::string> paramsMapping;
  std::map<std::string, SimplifiedParams> mobileBertParams;
  std::map<std::string, MemoryOffsets> mobileBertMemOffsets;
  std::map<std::string, Files> mobileBertTestFiles;

  if (task == "forward") {
    paramsMapping = inferenceParamsMapping;
    mobileBertParams = inferenceParams;
    mobileBertMemOffsets = inferenceMemOffsets;
    mobileBertTestFiles = inferenceTestFiles;
  } else if (task == "backward") {
    paramsMapping = backpropParamsMapping;
    mobileBertParams = backpropParams;
    mobileBertMemOffsets = backpropMemOffsets;
    mobileBertTestFiles = backpropTestFiles;
  } else if (task == "gradient") {
    paramsMapping = gradientParamsMapping;
    mobileBertParams = gradientParams;
    mobileBertMemOffsets = gradientMemOffsets;
    mobileBertTestFiles = gradientTestFiles;
  } else {
    throw std::invalid_argument("ERROR: Task unfound!");
  }

  if (paramsMapping.find(test) == paramsMapping.end()) {
    throw std::invalid_argument("ERROR: Operation unfound!");
  }

  std::string paramsName = paramsMapping.at(test);
  SimplifiedParams params = mobileBertParams.at(paramsName);
  Files files = mobileBertTestFiles.at(test);
  MemoryOffsets offsets = mobileBertMemOffsets.at(test);
  std::string layerName = "mobilebert_encoder_layer_23_";

  if (test.find("classifier") != std::string::npos ||
      (task == "backward" && test == "output_bottleneck_LayerNorm")) {
    layerName = "";
  }

  params.INPUT_OFFSET = offsets.INPUT_OFFSET + STACK_SIZE;
  params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET;
  params.OUTPUT_OFFSET = offsets.OUTPUT_OFFSET + STACK_SIZE;
  params.BIAS_OFFSET = offsets.BIAS_OFFSET;
  params.RESIDUAL_OFFSET = offsets.RESIDUAL_OFFSET + STACK_SIZE;
  params.WEIGHT_SPLITTING = false;

  if (!params.WEIGHT) {
    params.WEIGHT_OFFSET += STACK_SIZE;
  }

  MemoryMap memoryMap = {SRAM, (params.WEIGHT ? RRAM : SRAM), RRAM, SRAM, SRAM};

  std::cout << "test name: " << test << std::endl;
  std::cout << "operation name: " << paramsName << std::endl;

  if (task == "forward") {
    return runOperation(params, files, memoryMap, activationDataDir,
                        params.WEIGHT ? weightDataDir : activationDataDir,
                        activationDataDir, activationDataDir, gradientDataDir,
                        layerName, outfilePrefix, compList);
  } else if (task == "backward") {
    if (test.find("attention_self_value_layer") != std::string::npos) {
      return runOperation(params, files, memoryMap, activationDataDir,
                          errorDataDir, errorDataDir, errorDataDir,
                          gradientDataDir, layerName, outfilePrefix, compList);
    } else if (test.find("attention_self_query_layer") != std::string::npos ||
               test.find("attention_self_key_layer") != std::string::npos ||
               test.find("attention_self_attention_probs") !=
                   std::string::npos) {
      return runOperation(params, files, memoryMap, errorDataDir,
                          activationDataDir, errorDataDir, errorDataDir,
                          gradientDataDir, layerName, outfilePrefix, compList);
    } else if (params.SOFTMAX_GRAD || params.RELU_GRAD) {
      return runOperation(params, files, memoryMap, errorDataDir, weightDataDir,
                          errorDataDir, activationDataDir, gradientDataDir,
                          layerName, outfilePrefix, compList);
    } else if (params.CROSS_ENTROPY_GRAD) {
      return runOperation(params, files, memoryMap, activationDataDir,
                          activationDataDir, errorDataDir, activationDataDir,
                          gradientDataDir, layerName, outfilePrefix, compList);
    }
    return runOperation(params, files, memoryMap, errorDataDir, weightDataDir,
                        errorDataDir, errorDataDir, gradientDataDir, layerName,
                        outfilePrefix, compList);
  } else if (task == "gradient") {
    if (test.find("classifier_weight") != std::string::npos) {
      return runOperation(params, files, memoryMap, errorDataDir,
                          activationDataDir, gradientDataDir, gradientDataDir,
                          gradientDataDir, layerName, outfilePrefix, compList);
    }
    return runOperation(params, files, memoryMap, activationDataDir,
                        errorDataDir, gradientDataDir, gradientDataDir,
                        gradientDataDir, layerName, outfilePrefix, compList);
  }

  return 1;
}
