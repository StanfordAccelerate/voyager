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
                 const std::string residualDataDir, const std::string layerName,
                 const std::string outfilePrefix,
                 const std::vector<std::string> groups, bool inputScaling,
                 bool weightScaling) {
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

  int outputSize = X * Y * K;
  if (params.NO_NORM_GRAD) {
    outputSize = C;
  } else if (params.CROSS_ENTROPY_LOSS_GRAD) {
    outputSize = X;
  }

  std::cout << "Performing the following operation:" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;

  INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];
  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[FX * FY * C * K];
  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[K];
  INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[outputSize];
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[outputSize];
  OUTPUT_DATATYPE* dataFileOutput = new OUTPUT_DATATYPE[outputSize];

  UniversalPosit* universalMatrixA =
      new UniversalPosit[(STRIDE * X) * (STRIDE * Y) * C];
  UniversalPosit* universalMatrixB = new UniversalPosit[FX * FY * C * K];
  UniversalPosit* universalBiasMatrix = new UniversalPosit[K];
  UniversalPosit* universalResidualMatrix = new UniversalPosit[outputSize];
  UniversalPosit* universalMatrixC = new UniversalPosit[outputSize];
  UniversalPosit* universalDataFileOutput = new UniversalPosit[outputSize];

  float* floatMatrixA = new float[(STRIDE * X) * (STRIDE * Y) * C];
  float* floatMatrixB = new float[FX * FY * C * K];
  float* floatBiasMatrix = new float[K];
  float* floatResidualMatrix = new float[outputSize];
  float* floatMatrixC = new float[outputSize];
  float* floatDataFileOutput = new float[outputSize];

  std::string datafile = inputDataDir + layerName + files.inputs_file;
  if (!files.inputs_file.empty()) {
    load_inputs(params, datafile, true, sramMemory, matrixA, universalMatrixA,
                floatMatrixA);
  }

  if (!files.weights_file.empty()) {
    datafile = weightDataDir + layerName + files.weights_file;
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
                                residualMatrix, inputScaling, weightScaling);
  }

  if (universal) {
    run_universal_posit_gold_model(params, universalMatrixA, universalMatrixB,
                                   universalMatrixC, universalBiasMatrix,
                                   universalResidualMatrix, inputScaling,
                                   weightScaling);
  }

  if (fp32) {
    run_fp_gold_model(params, floatMatrixA, floatMatrixB, floatMatrixC,
                      floatBiasMatrix, floatResidualMatrix, inputScaling,
                      weightScaling);
  }

  std::string diffFile;
  int errors = 0;

  if (universal) {
    std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in representing float as Posit)" << std::endl;
    diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
    compare_arrays(universalMatrixC, universalDataFileOutput, outputSize,
                   diffFile);
  }

  if (customposit) {
    std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals bugs in mapping operations to accelerator)"
              << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
    compare_arrays(matrixC, dataFileOutput, outputSize, diffFile);
  }

  if (accelerator) {
    std::cout << "Accelerator vs. PyTorch" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_pytorch.txt";
    errors += compare_arrays(&sramMemory[params.OUTPUT_OFFSET], dataFileOutput,
                             outputSize, diffFile);
  }

  if (accelerator && customposit) {
    std::cout << "Accelerator vs. HLS Posit Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_hlsgold.txt";
    errors += compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC,
                             outputSize, diffFile);
  }

  if (customposit && universal) {
    std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
              << std::endl;
    std::cout
        << "(reveals bugs in implementation of custom HLS Posit operators)"
        << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
    errors += compare_arrays(matrixC, universalMatrixC, outputSize, diffFile);
  }

  if (fp32) {
    std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
    errors +=
        compare_arrays(floatMatrixC, floatDataFileOutput, outputSize, diffFile);
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

int runMobileBertUnitTest(std::string task, std::string test,
                          std::vector<std::string> compList,
                          std::string datapath) {
  // std::string datapath = "/sim2/shared/MINOTAUR/nn_data/mobilebert/";
  std::string activationDataDir = datapath + "activations/";
  std::string weightDataDir = datapath + "weights/";
  std::string errorDataDir = datapath + "errors/";
  std::string gradientDataDir = datapath + "gradients/";
  std::string outfilePrefix = "test_outputs/";

  std::map<std::string, std::string> paramsMapping;
  std::map<std::string, SimplifiedParams> mobileBertParams;
  std::map<std::string, MemoryOffsets> mobileBertMemOffsets;
  std::map<std::string, Files> mobileBertTestFiles;

  if (task == "inference") {
    paramsMapping = inferenceParamsMapping;
    mobileBertParams = inferenceParams;
    mobileBertMemOffsets = inferenceMemOffsets;
    mobileBertTestFiles = inferenceTestFiles;
  } else if (task == "backprop") {
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
  std::string layerName = "mobilebert_encoder_layer_0_";

  if (test.find("classifier") != std::string::npos ||
      (task == "backprop" && test == "output_bottleneck_LayerNorm")) {
    layerName = "";
  }

  params.INPUT_OFFSET = offsets.INPUT_OFFSET + STACK_SIZE;
  params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET;
  params.OUTPUT_OFFSET = offsets.OUTPUT_OFFSET + STACK_SIZE;
  params.BIAS_OFFSET = offsets.BIAS_OFFSET;
  params.RESIDUAL_OFFSET = offsets.RESIDUAL_OFFSET + STACK_SIZE;

  if (!params.WEIGHT) {
    params.WEIGHT_OFFSET += STACK_SIZE;
  }

  MemoryMap memoryMap = {SRAM, (params.WEIGHT ? RRAM : SRAM), RRAM, SRAM, SRAM};

  std::cout << "test name: " << test << std::endl;
  std::cout << "operation name: " << paramsName << std::endl;

  if (task == "inference") {
    return runOperation(params, files, memoryMap, activationDataDir,
                        params.WEIGHT ? weightDataDir : activationDataDir,
                        activationDataDir, activationDataDir, layerName,
                        outfilePrefix, compList, false, false);
  } else if (task == "backprop") {
    if (test.find("attention_self_value_layer") != std::string::npos) {
      return runOperation(params, files, memoryMap, activationDataDir,
                          errorDataDir, errorDataDir, errorDataDir, layerName,
                          outfilePrefix, compList, false, false);
    } else if (test.find("attention_self_query_layer") != std::string::npos ||
               test.find("attention_self_key_layer") != std::string::npos ||
               test.find("attention_self_attention_probs") !=
                   std::string::npos) {
      return runOperation(params, files, memoryMap, errorDataDir,
                          activationDataDir, errorDataDir, errorDataDir,
                          layerName, outfilePrefix, compList, false, false);
    } else if (params.SOFTMAX_GRAD || params.RELU_GRAD) {
      return runOperation(params, files, memoryMap, errorDataDir, weightDataDir,
                          errorDataDir, activationDataDir, layerName,
                          outfilePrefix, compList, false, false);
    } else if (params.CROSS_ENTROPY_LOSS_GRAD) {
      return runOperation(params, files, memoryMap, activationDataDir,
                          activationDataDir, errorDataDir, activationDataDir,
                          layerName, outfilePrefix, compList, false, false);
    }
    return runOperation(params, files, memoryMap, errorDataDir, weightDataDir,
                        errorDataDir, errorDataDir, layerName, outfilePrefix,
                        compList, false, false);
  } else if (task == "gradient") {
    if (test.find("classifier_weight") != std::string::npos) {
      return runOperation(params, files, memoryMap, errorDataDir,
                          activationDataDir, gradientDataDir, gradientDataDir,
                          layerName, outfilePrefix, compList, false, false);
    }
    return runOperation(params, files, memoryMap, activationDataDir,
                        errorDataDir, gradientDataDir, gradientDataDir,
                        layerName, outfilePrefix, compList, false, false);
  }

  return 1;
}
