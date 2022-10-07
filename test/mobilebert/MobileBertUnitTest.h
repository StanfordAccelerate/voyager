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
#include "test/mobilebert/mobilebert_tiny/inference.h"

#ifndef SRAM_MEMORY_SIZE
#define SRAM_MEMORY_SIZE (30 * 1024 * 1024)
#endif

#ifndef RRAM_MEMORY_SIZE
// #define RRAM_MEMORY_SIZE (12 * 1024 * 1024)  // RRAM size for TinyBERT
#define RRAM_MEMORY_SIZE (22 * 1024 * 1024)  // RRAM size for MobileBERT
#endif

void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE* acc_sram_memory, INPUT_DATATYPE* acc_rram_memory,
            MemoryMap memoryMap);

int runOperation(const SimplifiedParams params, const Files files,
                 const MemoryMap memoryMap, const std::string outfilePrefix,
                 const std::vector<std::string> groups) {
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

  if (params.SOFTMAX && params.ATTENTION_MASK) {
    C = X;
    K = 1;
  } else if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    K = 1;
    C = 1;
  }

  int outputSize = X * Y * K;
  if (params.NO_NORM_GRAD) {
    outputSize = K;
  } else if (params.CROSS_ENTROPY_GRAD) {
    outputSize = X;
  }

  std::cout << "Performing the following operation:" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;

  INPUT_DATATYPE* acc_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* acc_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  INPUT_DATATYPE* hls_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* hls_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  UniversalPosit* uni_sram_memory = new UniversalPosit[SRAM_MEMORY_SIZE];
  UniversalPosit* uni_rram_memory = new UniversalPosit[RRAM_MEMORY_SIZE];
  float* float_sram_memory = new float[SRAM_MEMORY_SIZE];
  float* float_rram_memory = new float[RRAM_MEMORY_SIZE];

  if (!files.inputs_file.empty()) {
    std::cerr << "Load input file: " << files.inputs_file << std::endl;
    load_inputs(params, files.inputs_file, true, acc_sram_memory,
                hls_sram_memory + params.INPUT_OFFSET,
                uni_sram_memory + params.INPUT_OFFSET,
                float_sram_memory + params.INPUT_OFFSET);
  }

  if (params.WEIGHT) {
    std::cerr << "Load weight file: " << files.weights_file << std::endl;
    load_weights(params, files.weights_file, true, acc_rram_memory,
                 hls_rram_memory + params.WEIGHT_OFFSET,
                 uni_rram_memory + params.WEIGHT_OFFSET,
                 float_rram_memory + params.WEIGHT_OFFSET);
  } else if (!files.weights_file.empty()) {
    std::cerr << "Load weight file: " << files.weights_file << std::endl;
    load_weights(params, files.weights_file, true, acc_sram_memory,
                 hls_sram_memory + params.WEIGHT_OFFSET,
                 uni_sram_memory + params.WEIGHT_OFFSET,
                 float_sram_memory + params.WEIGHT_OFFSET);
  }

  if (params.BIAS) {
    std::cerr << "Load bias file: " << files.bias_file << std::endl;
    load_bias(params, files.bias_file, true, acc_rram_memory,
              hls_rram_memory + params.BIAS_OFFSET,
              uni_rram_memory + params.BIAS_OFFSET,
              float_rram_memory + params.BIAS_OFFSET);
  }

  if (params.RESIDUAL || params.RELU_GRAD || params.SOFTMAX_GRAD) {
    std::cerr << "Load residual file: " << files.residual_file << std::endl;
    load_residual(params, files.residual_file, true, acc_sram_memory,
                  hls_sram_memory + params.RESIDUAL_OFFSET,
                  uni_sram_memory + params.RESIDUAL_OFFSET,
                  float_sram_memory + params.RESIDUAL_OFFSET);
  }

  if (params.WEIGHT_SPLITTING && params.WEIGHT) {
    load_weights(params, files.weight_grad_file, true, acc_sram_memory,
                 hls_sram_memory + params.WEIGHT_GRADIENT_OFFSET,
                 uni_sram_memory + params.WEIGHT_GRADIENT_OFFSET,
                 float_sram_memory + params.WEIGHT_GRADIENT_OFFSET);
  }

  if (params.WEIGHT_SPLITTING && params.BIAS) {
    load_bias(params, files.bias_grad_file, true, acc_sram_memory,
              hls_sram_memory + params.BIAS_GRADIENT_OFFSET,
              uni_sram_memory + params.BIAS_GRADIENT_OFFSET,
              float_sram_memory + params.BIAS_GRADIENT_OFFSET);
  }

  INPUT_DATATYPE hlsDataFileOutput[2 * outputSize];
  UniversalPosit uniDataFileOutput[2 * outputSize];
  float dataFileOutput[outputSize];

  std::cerr << "Load output file: " << files.outputs_file << std::endl;
  load_datafile_outputs(params, files.outputs_file, hlsDataFileOutput,
                        uniDataFileOutput, dataFileOutput);

  bool accelerator =
      std::find(groups.begin(), groups.end(), "accelerator") != groups.end();
  bool customposit =
      std::find(groups.begin(), groups.end(), "customposit") != groups.end();
  bool universal =
      std::find(groups.begin(), groups.end(), "universal") != groups.end();
  bool fp32 = std::find(groups.begin(), groups.end(), "fp32") != groups.end();

  if (accelerator) {
    run_op({params}, acc_sram_memory, acc_rram_memory, memoryMap);
  }

  if (customposit) {
    run_custom_posit_gold_model(
        params, hls_sram_memory + params.INPUT_OFFSET,
        (params.WEIGHT ? hls_rram_memory : hls_sram_memory) +
            params.WEIGHT_OFFSET,
        hls_sram_memory + params.OUTPUT_OFFSET,
        hls_rram_memory + params.BIAS_OFFSET,
        hls_sram_memory + params.RESIDUAL_OFFSET,
        hls_sram_memory + params.WEIGHT_GRADIENT_OFFSET,
        hls_sram_memory + params.BIAS_GRADIENT_OFFSET);
  }

  if (universal) {
    run_universal_posit_gold_model(
        params, uni_sram_memory + params.INPUT_OFFSET,
        (params.WEIGHT ? uni_rram_memory : uni_sram_memory) +
            params.WEIGHT_OFFSET,
        uni_sram_memory + params.OUTPUT_OFFSET,
        uni_rram_memory + params.BIAS_OFFSET,
        uni_sram_memory + params.RESIDUAL_OFFSET,
        uni_sram_memory + params.WEIGHT_GRADIENT_OFFSET,
        uni_sram_memory + params.BIAS_GRADIENT_OFFSET);
  }

  if (fp32) {
    run_fp_gold_model(params, float_sram_memory + params.INPUT_OFFSET,
                      (params.WEIGHT ? float_rram_memory : float_sram_memory) +
                          params.WEIGHT_OFFSET,
                      float_sram_memory + params.OUTPUT_OFFSET,
                      float_rram_memory + params.BIAS_OFFSET,
                      float_sram_memory + params.RESIDUAL_OFFSET,
                      float_sram_memory + params.WEIGHT_GRADIENT_OFFSET,
                      float_sram_memory + params.BIAS_GRADIENT_OFFSET);
  }

  std::string diffFile;
  int errors = 0;
  if (universal) {
    std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in representing float as Posit)" << std::endl;
    diffFile = outfilePrefix + "universal_vs_pytorch.txt";
    errors += compare_arrays(uni_sram_memory + params.OUTPUT_OFFSET,
                             uniDataFileOutput, outputSize, diffFile,
                             params.ACC_T_OUTPUT);
  }

  if (customposit) {
    std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals bugs in mapping operations to accelerator)"
              << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
    errors += compare_arrays(hls_sram_memory + params.OUTPUT_OFFSET,
                             hlsDataFileOutput, outputSize, diffFile,
                             params.ACC_T_OUTPUT);
  }

  if (accelerator) {
    std::cout << "Accelerator vs. PyTorch" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_pytorch.txt";
    errors += compare_arrays(acc_sram_memory + params.OUTPUT_OFFSET,
                             hlsDataFileOutput, outputSize, diffFile,
                             params.ACC_T_OUTPUT);
  }

  if (accelerator && customposit) {
    std::cout << "Accelerator vs. HLS Posit Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    diffFile = outfilePrefix + "accel_vs_hlsgold.txt";
    errors += compare_arrays(acc_sram_memory + params.OUTPUT_OFFSET,
                             hls_sram_memory + params.OUTPUT_OFFSET, outputSize,
                             diffFile, params.ACC_T_OUTPUT);
  }

  if (customposit && universal) {
    std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
              << std::endl;
    std::cout
        << "(reveals bugs in implementation of custom HLS Posit operators)"
        << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_universal.txt";
    errors += compare_arrays(hls_sram_memory + params.OUTPUT_OFFSET,
                             uni_sram_memory + params.OUTPUT_OFFSET, outputSize,
                             diffFile, params.ACC_T_OUTPUT);
  }

  if (fp32) {
    std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
    errors +=
        compare_arrays(float_sram_memory + params.OUTPUT_OFFSET, dataFileOutput,
                       outputSize, diffFile, params.ACC_T_OUTPUT);
  }

  delete acc_sram_memory;
  delete acc_rram_memory;
  delete hls_sram_memory;
  delete hls_rram_memory;
  delete uni_sram_memory;
  delete uni_rram_memory;
  delete float_sram_memory;
  delete float_rram_memory;

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
  SimplifiedParams params;
  Files files;
  MemoryOffsets offsets;

  std::string paramName;
  std::string layerName = "mobilebert_encoder_layer_0_";
  std::string outfilePrefix = "test_outputs/" + test + "_";

  std::string inputDataDir;
  std::string weightDataDir;
  std::string outputDataDir;
  std::string residualDataDir;
  std::string gradientDataDir;

  if (task == "forward") {
    paramName = inferenceParamsMapping.at(test);
    params = inferenceParams.at(paramName);
    files = inferenceTestFiles.at(test);
    offsets = inferenceMemOffsets.at(test);

    inputDataDir = datapath + "activations/";
    weightDataDir = datapath + "weights/";
    outputDataDir = datapath + "activations/";
    residualDataDir = datapath + "activations/";
    gradientDataDir = datapath + "gradients/";

    if (!params.WEIGHT || params.ATTENTION_MASK) {
      weightDataDir = datapath + "activations/";
    }

    params.WEIGHT_SPLITTING = false;
  } else if (task == "backward") {
    paramName = backpropParamsMapping.at(test);
    params = backpropParams.at(paramName);
    files = backpropTestFiles.at(test);
    offsets = backpropMemOffsets.at(test);

    inputDataDir = datapath + "errors/";
    weightDataDir = datapath + "weights/";
    outputDataDir = datapath + "errors/";
    residualDataDir = datapath + "errors/";

    if (test.find("attention_self_value_layer") != std::string::npos) {
      inputDataDir = datapath + "activations/";
      weightDataDir = datapath + "errors/";
    } else if (params.CROSS_ENTROPY_GRAD) {
      inputDataDir = datapath + "activations/";
      weightDataDir = datapath + "activations/";
    } else if (!params.WEIGHT) {
      weightDataDir = datapath + "activations/";
    } else if (params.SOFTMAX_GRAD || params.RELU_GRAD) {
      residualDataDir = datapath + "activations/";
    }
  } else if (task == "gradient") {
    paramName = gradientParamsMapping.at(test);
    params = gradientParams.at(paramName);
    files = gradientTestFiles.at(test);
    offsets = gradientMemOffsets.at(test);

    inputDataDir = datapath + "activations/";
    weightDataDir = datapath + "errors/";
    outputDataDir = datapath + "gradients/";
    residualDataDir = datapath + "gradients/";

    if (test.find("classifier") != std::string::npos) {
      inputDataDir = datapath + "errors/";
      weightDataDir = datapath + "activations/";
    }

    // FIXME: accelerator doesn't support these functionalities
    params.ACC_T_OUTPUT = false;
    params.GRAD_CLIPPING = false;
  }

  if (test.find("classifier") != std::string::npos ||
      (task == "backward" && test == "output_bottleneck_LayerNorm")) {
    layerName = "";
  }

  if (!files.inputs_file.empty()) {
    files.inputs_file = inputDataDir + layerName + files.inputs_file;
  }

  if (params.ATTENTION_MASK) {
    files.weights_file = weightDataDir + files.weights_file;
  } else if (!files.weights_file.empty()) {
    files.weights_file = weightDataDir + layerName + files.weights_file;
  }

  files.outputs_file = outputDataDir + layerName + files.outputs_file;
  files.bias_file = weightDataDir + layerName + files.bias_file;
  files.residual_file = residualDataDir + layerName + files.residual_file;

  // Fake memory offsets used for unit tests
  params.INPUT_OFFSET = STACK_SIZE;
  params.WEIGHT_OFFSET = STACK_SIZE + INTERMEDIATE_SIZE;
  params.OUTPUT_OFFSET = STACK_SIZE + 2 * INTERMEDIATE_SIZE;
  params.BIAS_OFFSET = STACK_SIZE + 3 * INTERMEDIATE_SIZE;
  params.RESIDUAL_OFFSET = STACK_SIZE + 4 * INTERMEDIATE_SIZE;

  MemoryMap memoryMap = {SRAM, (params.WEIGHT ? RRAM : SRAM), RRAM, SRAM, SRAM};

  std::cout << "Test name: " << test << std::endl;
  std::cout << "Operation name: " << paramName << std::endl;

  return runOperation(params, files, memoryMap, outfilePrefix, compList);
}
