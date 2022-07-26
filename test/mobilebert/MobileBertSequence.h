#include <cstring>

#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
// #include "test/common/Harness.h"
#include "test/common/UniversalPosit.h"
#include "test/common/Utils.h"
#include "test/mobilebert/backprop.h"
#include "test/mobilebert/gradient.h"
#include "test/mobilebert/inference.h"

#ifndef SRAM_MEMORY_SIZE
// #define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
#define SRAM_MEMORY_SIZE (30 * 1024 * 1024)
#endif

#ifndef RRAM_MEMORY_SIZE
// #define RRAM_MEMORY_SIZE (12 * 1024 * 1024)  // RRAM size for TinyBERT
#define RRAM_MEMORY_SIZE (22 * 1024 * 1024)  // RRAM size for MobileBERT
#endif

// Data memory
INPUT_DATATYPE* acc_sram_memory;
INPUT_DATATYPE* acc_rram_memory;
INPUT_DATATYPE* hls_sram_memory;
INPUT_DATATYPE* hls_rram_memory;
UniversalPosit* uni_sram_memory;
UniversalPosit* uni_rram_memory;
float* float_sram_memory;
float* float_rram_memory;

// Training parameters
const int num_epochs = 3;
float learningRate = 2e-5;

void validateMapping(SimplifiedParams params);

SimplifiedParams paramsLookup(std::string operation, std::string task) {
  std::map<std::string, std::string> paramsMapping;
  std::map<std::string, SimplifiedParams> mobileBertParams;
  std::map<std::string, MemoryOffsets> mobileBertMemOffsets;

  if (task == "inference") {
    paramsMapping = inferenceParamsMapping;
    mobileBertParams = inferenceParams;
    mobileBertMemOffsets = inferenceMemOffsets;
  } else if (task == "backprop") {
    paramsMapping = backpropParamsMapping;
    mobileBertParams = backpropParams;
    mobileBertMemOffsets = backpropMemOffsets;
  } else if (task == "gradient") {
    paramsMapping = gradientParamsMapping;
    mobileBertParams = gradientParams;
    mobileBertMemOffsets = gradientMemOffsets;
  } else {
    throw std::invalid_argument("ERROR: Task unfound!");
  }

  if (paramsMapping.find(operation) == paramsMapping.end()) {
    throw std::invalid_argument("ERROR: Operation unfound!");
  }

  std::string paramsName = paramsMapping.at(operation);
  SimplifiedParams params = mobileBertParams.at(paramsName);
  MemoryOffsets offsets = mobileBertMemOffsets.at(operation);

  params.INPUT_OFFSET = offsets.INPUT_OFFSET;
  params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET;
  params.OUTPUT_OFFSET = offsets.OUTPUT_OFFSET;
  params.BIAS_OFFSET = offsets.BIAS_OFFSET;
  params.RESIDUAL_OFFSET = offsets.RESIDUAL_OFFSET;

  return params;
}

void formatOperation(const SimplifiedParams params) {
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
}

int allocateMemory() {
  acc_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  acc_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  hls_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  hls_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  uni_sram_memory = new UniversalPosit[SRAM_MEMORY_SIZE];
  uni_rram_memory = new UniversalPosit[RRAM_MEMORY_SIZE];
  float_sram_memory = new float[SRAM_MEMORY_SIZE];
  float_rram_memory = new float[RRAM_MEMORY_SIZE];

  return (!acc_sram_memory || !acc_rram_memory || !hls_sram_memory ||
          !hls_rram_memory || !uni_sram_memory || !uni_rram_memory ||
          !float_sram_memory || !float_rram_memory);
}

void deleteMemory() {
  delete acc_sram_memory;
  delete acc_rram_memory;
  delete hls_sram_memory;
  delete hls_rram_memory;
  delete uni_sram_memory;
  delete uni_rram_memory;
  delete float_sram_memory;
  delete float_rram_memory;
}

void loadWeights(std::string weightDataDir) {
  for (int layer = 0; layer < 24; layer++) {
    for (const auto& operation : inferenceOrder) {
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      SimplifiedParams params = paramsLookup(operation, "inference");
      Files files = inferenceTestFiles.at(operation);
      std::string datafile;

      params.WEIGHT_OFFSET += layer * ENCODER_WEIGHT_OFFSET;
      params.BIAS_OFFSET += layer * ENCODER_WEIGHT_OFFSET;

      if (operation == "classifier") {
        if (layer != 23) continue;
        layerName = "";
      }

      if (params.WEIGHT) {
        datafile = weightDataDir + layerName + files.weights_file;
        load_weights(params, datafile, true, acc_rram_memory,
                     hls_rram_memory + params.WEIGHT_OFFSET,
                     uni_rram_memory + params.WEIGHT_OFFSET,
                     float_rram_memory + params.WEIGHT_OFFSET);
      }

      if (params.BIAS) {
        datafile = weightDataDir + layerName + files.bias_file;
        load_bias(params, datafile, true, acc_rram_memory,
                  hls_rram_memory + params.BIAS_OFFSET,
                  uni_rram_memory + params.BIAS_OFFSET,
                  float_rram_memory + params.BIAS_OFFSET);
      }
    }
  }
}

void loadActivation(std::string dataDir) {
  for (int layer = 0; layer < 24; layer++) {
    for (const auto& operation : inferenceOrder) {
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      SimplifiedParams params = paramsLookup(operation, "inference");
      Files files = inferenceTestFiles.at(operation);
      std::string datafile;

      params.OUTPUT_OFFSET += layer * ENCODER_ACTIVATION_OFFSET;

      if (operation == "classifier") {
        if (layer != 23) continue;
        layerName = "";
      }

      datafile = dataDir + layerName + files.outputs_file;
      load_datafile_outputs(params, datafile,
                            hls_sram_memory + params.OUTPUT_OFFSET,
                            uni_sram_memory + params.OUTPUT_OFFSET,
                            float_sram_memory + params.OUTPUT_OFFSET);
    }
  }
}

int runOperation(const SimplifiedParams params,
                 const std::string outputDataFile,
                 const std::string outfilePrefix, const std::string testName,
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

  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    K = 1;
    C = 1;
  }

  int outputSize = params.NO_NORM_GRAD ? C : X * Y * K;

#ifndef PIPE_INPUT
  std::cout << "Performing " + testName + ":" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;
#endif

  bool customposit =
      std::find(groups.begin(), groups.end(), "customposit") != groups.end();
  bool universal =
      std::find(groups.begin(), groups.end(), "universal") != groups.end();
  bool fp32 = std::find(groups.begin(), groups.end(), "fp32") != groups.end();

  if (customposit) {
    run_custom_posit_gold_model(
        params, hls_sram_memory + params.INPUT_OFFSET,
        (params.WEIGHT ? hls_rram_memory : hls_sram_memory) +
            params.WEIGHT_OFFSET,
        hls_sram_memory + params.OUTPUT_OFFSET,
        hls_rram_memory + params.BIAS_OFFSET,
        hls_sram_memory + params.RESIDUAL_OFFSET, false, false);
  }

  if (universal) {
    run_universal_posit_gold_model(
        params, uni_sram_memory + params.INPUT_OFFSET,
        (params.WEIGHT ? uni_rram_memory : uni_sram_memory) +
            params.WEIGHT_OFFSET,
        uni_sram_memory + params.OUTPUT_OFFSET,
        uni_rram_memory + params.BIAS_OFFSET,
        uni_sram_memory + params.RESIDUAL_OFFSET, false, false);
  }

  if (fp32) {
    run_fp_gold_model(params, float_sram_memory + params.INPUT_OFFSET,
                      (params.WEIGHT ? float_rram_memory : float_sram_memory) +
                          params.WEIGHT_OFFSET,
                      float_sram_memory + params.OUTPUT_OFFSET,
                      float_rram_memory + params.BIAS_OFFSET,
                      float_sram_memory + params.RESIDUAL_OFFSET, false, false);
  }

#ifdef PIPE_INPUT
  return 0;
#else

  if (params.SPLIT_OUTPUT) return 0;

  INPUT_DATATYPE hlsDataFileOutput[outputSize];
  UniversalPosit uniDataFileOutput[outputSize];
  float dataFileOutput[outputSize];

  load_datafile_outputs(params, outputDataFile, hlsDataFileOutput,
                        uniDataFileOutput, dataFileOutput);

  std::string diffFile;
  int errors;
  if (customposit) {
    std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals bugs in mapping operations to accelerator)"
              << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
    errors = compare_arrays(hls_sram_memory + params.OUTPUT_OFFSET,
                            hlsDataFileOutput, outputSize, diffFile);
  }

  if (universal) {
    std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in representing float as Posit)" << std::endl;
    diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
    errors = compare_arrays(uni_sram_memory + params.OUTPUT_OFFSET,
                            uniDataFileOutput, outputSize, diffFile);
  }

  if (customposit && universal) {
    std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
              << std::endl;
    std::cout
        << "(reveals bugs in implementation of custom HLS Posit operators)"
        << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
    errors = compare_arrays(hls_sram_memory + params.OUTPUT_OFFSET,
                            uni_sram_memory + params.OUTPUT_OFFSET, outputSize,
                            diffFile);
  }

  if (fp32) {
    std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
    errors = compare_arrays(float_sram_memory + params.OUTPUT_OFFSET,
                            dataFileOutput, outputSize, diffFile);
  }

  return errors;

#endif
}

int runInference(std::string datapath, std::vector<std::string> groups) {
  std::string activationDataDir = datapath + "activations/";
  std::string outfilePrefix = "test_outputs/";

  int errors = 0;

  // Load input of the first layer
  std::string operation = inferenceOrder[0];
  SimplifiedParams params = paramsLookup(operation, "inference");
  Files files = inferenceTestFiles.at(operation);
  std::string datafile =
      activationDataDir + "mobilebert_encoder_layer_0_hidden_states";
  bool useDataFile = true;

  load_inputs(params, datafile, useDataFile, acc_sram_memory,
              hls_sram_memory + params.INPUT_OFFSET,
              uni_sram_memory + params.INPUT_OFFSET,
              float_sram_memory + params.INPUT_OFFSET);

  // Perform forward pass
  for (int layer = 0; layer < 24; layer++) {
    for (const auto& test : inferenceOrder) {
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      params = paramsLookup(test, "inference");
      files = inferenceTestFiles.at(test);

      params.INPUT_OFFSET += layer * ENCODER_ACTIVATION_OFFSET;
      params.WEIGHT_OFFSET += params.WEIGHT ? layer * ENCODER_WEIGHT_OFFSET
                                            : layer * ENCODER_ACTIVATION_OFFSET;
      params.OUTPUT_OFFSET += layer * ENCODER_ACTIVATION_OFFSET;
      params.BIAS_OFFSET += layer * ENCODER_WEIGHT_OFFSET;
      params.RESIDUAL_OFFSET += layer * ENCODER_ACTIVATION_OFFSET;

      if (test == "classifier") {
        if (layer != 23) continue;
        layerName = "";
      }

      outfilePrefix = "test_outputs/" + test + "_activation_";
      datafile = activationDataDir + layerName + files.outputs_file;
      errors += runOperation(params, datafile, outfilePrefix, test, groups);

      // if (errors) return 1;
    }
  }

  for (int i = 0; i < 2; i++) {
    std::cout << (float)hls_sram_memory[params.OUTPUT_OFFSET + i] << "\t"
              << float_sram_memory[params.OUTPUT_OFFSET + i] << std::endl;
  }

  return errors;
}

int runBackprop(std::string datapath, std::vector<std::string> groups) {
  std::string errorDataDir = datapath + "errors/";
  std::string gradientDataDir = datapath + "gradients/";
  std::string outfilePrefix = "test_outputs/";

  int errors = 0;

  std::string operation = backpropOrder[0];
  SimplifiedParams params = paramsLookup(operation, "backprop");
  Files files = backpropTestFiles.at(operation);
  std::string datafile = errorDataDir + "mobilebert_logits";
  bool useDataFile = true;

  params.INPUT_OFFSET += ACTIVATION_OFFSET;

  load_inputs(params, datafile, useDataFile, acc_sram_memory,
              hls_sram_memory + params.INPUT_OFFSET,
              uni_sram_memory + params.INPUT_OFFSET,
              float_sram_memory + params.INPUT_OFFSET);

  // TODO: Compute cross entropy gradient

  // Perform back propagation
  for (int layer = 23; layer >= 0; layer--) {
    for (const auto& test : backpropOrder) {
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      params = paramsLookup(test, "backprop");
      files = backpropTestFiles.at(test);
      MemoryOffsets offsets = backpropMemOffsets.at(test);

      params.INPUT_OFFSET += ACTIVATION_OFFSET;
      params.WEIGHT_OFFSET += layer * ENCODER_WEIGHT_OFFSET;
      params.OUTPUT_OFFSET += ACTIVATION_OFFSET;
      params.RESIDUAL_OFFSET += ACTIVATION_OFFSET;

      if (!params.WEIGHT) {
        params.WEIGHT_OFFSET =
            offsets.WEIGHT_OFFSET + layer * ENCODER_ACTIVATION_OFFSET;
      }

      if (params.RELU_GRAD || params.SOFTMAX_GRAD) {
        params.RESIDUAL_OFFSET =
            offsets.RESIDUAL_OFFSET + layer * ENCODER_ACTIVATION_OFFSET;
      }

      if (test.find("attention_self_value_layer") != std::string::npos) {
        params.INPUT_OFFSET =
            offsets.INPUT_OFFSET + layer * ENCODER_ACTIVATION_OFFSET;
        params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET + ACTIVATION_OFFSET;
      }

      if (test == "output_bottleneck_LayerNorm") {
        if (layer != 23) continue;
        layerName = "";
      }

      outfilePrefix = "test_outputs/" + test + "_error_";
      datafile = errorDataDir + layerName + files.outputs_file;
      std::string testName = "layer " + std::to_string(layer) + " " + test;
      errors += runOperation(params, datafile, outfilePrefix, testName, groups);

      // if (errors) return 1;

      // Gradient compute
      if (gradientParamsMapping.find(test) != gradientParamsMapping.end()) {
        params = paramsLookup(test, "gradient");
        files = gradientTestFiles.at(test);
        layerName = "mobilebert_encoder_layer_" + std::to_string(layer) + "_";

        if (test == "hidden_states_2") {
          if (layer == 0) continue;
          layerName =
              "mobilebert_encoder_layer_" + std::to_string(layer - 1) + "_";
        }

        params.INPUT_OFFSET += offsets.OUTPUT_OFFSET + ACTIVATION_OFFSET;
        params.WEIGHT_OFFSET += layer * ENCODER_ACTIVATION_OFFSET;
        params.OUTPUT_OFFSET += offsets.OUTPUT_OFFSET + ACTIVATION_OFFSET;

        outfilePrefix = "test_outputs/" + test + "_gradient_";
        datafile = gradientDataDir + layerName + files.outputs_file;
        errors += runOperation(params, datafile, outfilePrefix,
                               test + "_gradient", groups);

        // if (errors) return 1;

        // TODO: Gradient norm clipping

        // TODO: Weight update
      }
    }
  }

  return errors;
}
