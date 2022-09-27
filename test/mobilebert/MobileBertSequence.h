#include <cstring>

#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
// #include "test/common/Harness.h"
#include "test/common/UniversalPosit.h"
#include "test/common/Utils.h"
#include "test/mobilebert/backprop.h"
#include "test/mobilebert/gradient.h"
#include "test/mobilebert/inference.h"
#include "test/mobilebert/utils.h"

#define VERBOSE
// #define ACCELERATOR
// #define DUMP_PARAMS
// #define ACC_T_ERROR 1

// TODO(fpedd): Redefinition of SRAM/RRAM, see TestRunner.cc
#ifdef ACCELERATOR
#define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
#define RRAM_MEMORY_SIZE (12 * 1024 * 1024)
#else
#define SRAM_MEMORY_SIZE (50 * 1024 * 1024)
#define RRAM_MEMORY_SIZE (22 * 1024 * 1024)
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
const int numTrainEpochs = 3;
const int gradientAccumulationSteps = 32;
float learningRate = 2e-3;

void run_op(std::vector<SimplifiedParams> params_list,
            INPUT_DATATYPE* acc_sram_memory, INPUT_DATATYPE* acc_rram_memory,
            MemoryMap memoryMap);

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
    throw std::invalid_argument("ERROR: Task not found!");
  }

  if (paramsMapping.find(operation) == paramsMapping.end()) {
    throw std::invalid_argument("ERROR: Operation not found!");
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

int allocateMemory() {
  acc_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  acc_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  hls_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  hls_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  uni_sram_memory = new UniversalPosit[SRAM_MEMORY_SIZE];
  uni_rram_memory = new UniversalPosit[RRAM_MEMORY_SIZE];
  float_sram_memory = new float[SRAM_MEMORY_SIZE];
  float_rram_memory = new float[RRAM_MEMORY_SIZE];

  // TODO(fpedd): new never returns nullptr, use try/catch instead
  if (!acc_sram_memory || !acc_rram_memory || !hls_sram_memory ||
      !hls_rram_memory || !uni_sram_memory || !uni_rram_memory ||
      !float_sram_memory || !float_rram_memory) {
    return 1;
  }

  memset(acc_sram_memory, 0, SRAM_MEMORY_SIZE * sizeof(INPUT_DATATYPE));
  memset(acc_rram_memory, 0, RRAM_MEMORY_SIZE * sizeof(INPUT_DATATYPE));
  memset(hls_sram_memory, 0, SRAM_MEMORY_SIZE * sizeof(INPUT_DATATYPE));
  memset(hls_rram_memory, 0, RRAM_MEMORY_SIZE * sizeof(INPUT_DATATYPE));
  memset(uni_sram_memory, 0, SRAM_MEMORY_SIZE * sizeof(UniversalPosit));
  memset(uni_rram_memory, 0, RRAM_MEMORY_SIZE * sizeof(UniversalPosit));
  memset(float_sram_memory, 0, SRAM_MEMORY_SIZE * sizeof(float));
  memset(float_rram_memory, 0, RRAM_MEMORY_SIZE * sizeof(float));

  return 0;
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
      SimplifiedParams params = paramsLookup(operation, "inference");
      Files files = inferenceTestFiles.at(operation);
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      std::string datafile;

      params.WEIGHT_OFFSET += layer * ENCODER_WEIGHT_SIZE;
      params.BIAS_OFFSET += layer * ENCODER_WEIGHT_SIZE;

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
#ifdef ACCELERATOR
    break;
#endif
  }
}

void loadActivation(std::string dataDir) {
  std::string operation = inferenceOrder[0];
  SimplifiedParams params = paramsLookup(operation, "inference");
  Files files = inferenceTestFiles.at(operation);
  std::string layerName = "mobilebert_encoder_layer_0_";
  std::string datafile = dataDir + layerName + files.inputs_file;
  bool useDataFile = true;

  params.INPUT_OFFSET += ACTIVATION_OFFSET;
  load_inputs(params, datafile, useDataFile, acc_sram_memory,
              hls_sram_memory + params.INPUT_OFFSET,
              uni_sram_memory + params.INPUT_OFFSET,
              float_sram_memory + params.INPUT_OFFSET);

  for (int layer = 0; layer < 24; layer++) {
    for (const auto& operation : inferenceOrder) {
      layerName = "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      params = paramsLookup(operation, "inference");
      files = inferenceTestFiles.at(operation);

      params.OUTPUT_OFFSET +=
          ACTIVATION_OFFSET + layer * ENCODER_ACTIVATION_SIZE;

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

void verifyGradients(std::string dataDir, std::string outfilePrefix) {
  std::string datafile;
  std::string diffFile;
  int errors;
  for (int layer = 0; layer < 24; layer++) {
    for (const auto& operation : inferenceOrder) {
      SimplifiedParams params = paramsLookup(operation, "inference");
      Files files = inferenceTestFiles.at(operation);
      std::string layerName =
          "mobilebert_encoder_layer_" + std::to_string(layer) + "_";

      if (operation == "classifier") {
        if (layer != 23) continue;
        layerName = "";
      }

      params.WEIGHT_OFFSET += GRADIENT_OFFSET + layer * ENCODER_WEIGHT_SIZE;
      params.BIAS_OFFSET += GRADIENT_OFFSET + layer * ENCODER_WEIGHT_SIZE;

      int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
      int K = params.loops[0][params.weightLoopIndex[0]] *
              params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

      if (params.WEIGHT) {
        int weightSize = params.NO_NORM ? K : C * K;
        float floatMatrixB[weightSize];
        INPUT_DATATYPE hlsMatrixB[weightSize];
        UniversalPosit uniMatrixB[weightSize];

        datafile = dataDir + layerName + files.weights_file;
        std::cout << "Checking " << datafile << std::endl;
        load_weights(params, datafile, true, acc_sram_memory, hlsMatrixB,
                     uniMatrixB, floatMatrixB);
        errors = compare_arrays(float_sram_memory + params.WEIGHT_OFFSET,
                                floatMatrixB, weightSize,
                                outfilePrefix + files.weights_file, false);

        if (errors) {
          std::cerr << "ERROR: " << errors << " mismatches found" << std::endl;
        }
      }

      if (params.BIAS) {
        int biasSize = 2 * K;
        float floatBiasMatrix[biasSize];
        INPUT_DATATYPE hlsBiasMatrix[biasSize];
        UniversalPosit uniBiasMatrix[biasSize];

        datafile = dataDir + layerName + files.bias_file;
        std::cout << "Checking " << datafile << std::endl;
        load_bias(params, datafile, true, acc_sram_memory, hlsBiasMatrix,
                  uniBiasMatrix, floatBiasMatrix);
        errors = compare_arrays(float_sram_memory + params.BIAS_OFFSET,
                                floatBiasMatrix, K,
                                outfilePrefix + files.bias_file, true);

        if (errors) {
          std::cerr << "ERROR: " << errors << " mismatches found" << std::endl;
        }
      }
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

  int outputSize = X * Y * K;
  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    outputSize = X * Y;
  } else if (params.NO_NORM_GRAD) {
    outputSize = K;
  } else if (params.CROSS_ENTROPY_GRAD) {
    outputSize = X;
  }

#ifdef VERBOSE
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

  int errors = 0;

#ifdef VERBOSE
  INPUT_DATATYPE hlsDataFileOutput[2 * outputSize];
  UniversalPosit uniDataFileOutput[2 * outputSize];
  float dataFileOutput[outputSize];

  load_datafile_outputs(params, outputDataFile, hlsDataFileOutput,
                        uniDataFileOutput, dataFileOutput);

  std::string diffFile;
  if (customposit) {
    std::cout << "HLS Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals bugs in mapping operations to accelerator)"
              << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_pytorch.txt";
    compare_arrays(hls_sram_memory + params.OUTPUT_OFFSET, hlsDataFileOutput,
                   outputSize, diffFile, params.ACC_T_OUTPUT);
  }

  if (universal) {
    std::cout << "Universal Posit Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in representing float as Posit)" << std::endl;
    diffFile = outfilePrefix + "universalgold_vs_pytorch.txt";
    compare_arrays(uni_sram_memory + params.OUTPUT_OFFSET, uniDataFileOutput,
                   outputSize, diffFile, params.ACC_T_OUTPUT);
  }

  if (customposit && universal) {
    std::cout << "HLS Posit Gold Model vs. Universal Posit Gold Model"
              << std::endl;
    std::cout
        << "(reveals bugs in implementation of custom HLS Posit operators)"
        << std::endl;
    diffFile = outfilePrefix + "hlsgold_vs_universalgold.txt";
    compare_arrays(hls_sram_memory + params.OUTPUT_OFFSET,
                   uni_sram_memory + params.OUTPUT_OFFSET, outputSize, diffFile,
                   params.ACC_T_OUTPUT);
  }

  if (fp32) {
    std::cout << "FP32 Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    diffFile = outfilePrefix + "fpgold_vs_pytorch.txt";
    errors =
        compare_arrays(float_sram_memory + params.OUTPUT_OFFSET, dataFileOutput,
                       outputSize, diffFile, params.ACC_T_OUTPUT);
  }

  if (errors) {
    std::cerr << "ERROR: " << errors << " mismatches found" << std::endl;
  }
#endif

  return errors;
}

void checkAcceleratorOutputs(const SimplifiedParams params,
                             const std::string outputDataFile,
                             const std::string outfilePrefix,
                             const std::vector<std::string> groups) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;

  int outputSize = X * Y * K;
  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    outputSize = X * Y;
  } else if (params.NO_NORM_GRAD) {
    outputSize = K;
  } else if (params.CROSS_ENTROPY_GRAD) {
    outputSize = X;
  }

  INPUT_DATATYPE hlsDataFileOutput[2 * outputSize];
  UniversalPosit uniDataFileOutput[2 * outputSize];
  float dataFileOutput[outputSize];

  load_datafile_outputs(params, outputDataFile, hlsDataFileOutput,
                        uniDataFileOutput, dataFileOutput);

  std::cout << "Accelerator vs. PyTorch" << std::endl;
  std::cout << "(reveals bugs in accelerator or memory placement)" << std::endl;
  std::string diffFile = outfilePrefix + "accel_vs_pytorch.txt";
  compare_arrays(acc_sram_memory + params.OUTPUT_OFFSET, hlsDataFileOutput,
                 outputSize, diffFile, params.ACC_T_OUTPUT);

  std::cout << "Accelerator vs. HLS Posit Gold Model" << std::endl;
  std::cout << "(reveals bugs in accelerator or memory placement)" << std::endl;
  diffFile = outfilePrefix + "accel_vs_hlsgold.txt";
  int errors = compare_arrays(acc_sram_memory + params.OUTPUT_OFFSET,
                              hls_sram_memory + params.OUTPUT_OFFSET,
                              outputSize, diffFile, params.ACC_T_OUTPUT);

  if (errors) {
    std::cerr << "ERROR: " << errors << " mismatches found" << std::endl;
  }
}

void runForward(std::string datapath, std::vector<std::string> groups) {
  std::string inputDataDir = datapath + "activations/";
  std::string outfilePrefix;

#ifdef DUMP_PARAMS
  std::ofstream myfile;
  myfile.open("test/mobilebert/mobilebert_inference_params.h");
#endif

  // Load embedding inputs
  std::string operation = inferenceOrder[0];
  SimplifiedParams params = paramsLookup(operation, "inference");
  Files files = inferenceTestFiles.at(operation);
  std::string layerName = "mobilebert_encoder_layer_0_";
  std::string datafile = inputDataDir + layerName + files.inputs_file;
  bool useDataFile = true;

  params.INPUT_OFFSET += ACTIVATION_OFFSET;
  load_inputs(params, datafile, useDataFile, acc_sram_memory,
              hls_sram_memory + params.INPUT_OFFSET,
              uni_sram_memory + params.INPUT_OFFSET,
              float_sram_memory + params.INPUT_OFFSET);

  // Load attention mask
  params = paramsLookup("attention_self_attention_probs_0", "inference");
  datafile = inputDataDir + "mobilebert_attention_mask";
  params.WEIGHT_OFFSET += STACK_SIZE;
  load_weights(params, datafile, useDataFile, acc_sram_memory,
               hls_sram_memory + params.WEIGHT_OFFSET,
               uni_sram_memory + params.WEIGHT_OFFSET,
               float_sram_memory + params.WEIGHT_OFFSET);

  std::vector<SimplifiedParams> params_list;
  std::vector<std::string> files_list;

  for (int layer = 0; layer < 24; layer++) {
    for (const auto& op : inferenceOrder) {
      layerName = "mobilebert_encoder_layer_" + std::to_string(layer) + "_";
      params = paramsLookup(op, "inference");
      files = inferenceTestFiles.at(op);

      const int inputOffset =
          ACTIVATION_OFFSET + layer * ENCODER_ACTIVATION_SIZE;
      const int weightOffset = layer * ENCODER_WEIGHT_SIZE;

      params.INPUT_OFFSET += inputOffset;
      params.WEIGHT_OFFSET += params.WEIGHT ? weightOffset : inputOffset;
      params.OUTPUT_OFFSET += inputOffset;
      params.BIAS_OFFSET += weightOffset;
      params.RESIDUAL_OFFSET += inputOffset;

      params.WEIGHT_GRADIENT_OFFSET = GRADIENT_OFFSET + params.WEIGHT_OFFSET;
      params.BIAS_GRADIENT_OFFSET = GRADIENT_OFFSET + params.BIAS_OFFSET;
      params.WEIGHT_SPLITTING = false;

      if (params.ATTENTION_MASK) {
        params.WEIGHT_OFFSET = STACK_SIZE;
      }

      if (op == "classifier") {
        if (layer != 23) continue;
        layerName = "";
      }

      outfilePrefix = "test_outputs/" + op + "_activation_";
      datafile = inputDataDir + layerName + files.outputs_file;
      runOperation(params, datafile, outfilePrefix, layerName + op, groups);

      params_list.push_back(params);
      files_list.push_back(datafile);

#ifdef DUMP_PARAMS
      if (layer == 0 || op == "classifier") {
        params.INPUT_OFFSET = ACTIVATION_OFFSET;
        params.OUTPUT_OFFSET = ACTIVATION_OFFSET + INTERMEDIATE_SIZE;
        params.RESIDUAL_OFFSET = ACTIVATION_OFFSET + 2 * INTERMEDIATE_SIZE;
        params.WEIGHT_OFFSET = ACTIVATION_OFFSET + 3 * INTERMEDIATE_SIZE;
        params.BIAS_OFFSET = ACTIVATION_OFFSET + 4 * INTERMEDIATE_SIZE;
        myfile << formatOperation2(params, op) << std::endl;
      }
#endif
    }
#ifdef ACCELERATOR
    break;  // FIXME: layer by layer
#endif
  }

  if (std::find(groups.begin(), groups.end(), "accelerator") != groups.end()) {
    const MemoryMap memMap = {SRAM, RRAM, RRAM, SRAM,
                              SRAM};  // don't need this anymore
    run_op(params_list, acc_sram_memory, acc_rram_memory, memMap);

    for (int i = 0; i < params_list.size(); i++) {
      params = params_list[i];
      datafile = files_list[i];
      std::string op = inferenceOrder[i];
      outfilePrefix = "test_outputs/" + op + "_activation_";
      checkAcceleratorOutputs(params, datafile, outfilePrefix, groups);
    }
  }

#ifdef DUMP_PARAMS
  myfile.close();
#endif

  for (int i = 0; i < 2; i++) {
    std::cout << hls_sram_memory[params.OUTPUT_OFFSET + i] << "\t"
              << float_sram_memory[params.OUTPUT_OFFSET + i] << std::endl;
  }
}

void runBackward(std::string datapath, std::vector<std::string> groups) {
#ifdef DUMP_PARAMS
  std::ofstream errFile, gradFile;
  errFile.open("test/mobilebert/mobilebert_backprop_params.h");
  gradFile.open("test/mobilebert/mobilebert_gradient_params.h");
#endif
  std::string inputDataDir = datapath + "activations/";
  std::string errorDataDir = datapath + "errors/";
  std::string gradDataDir = datapath + "gradients/";
  std::string outfilePrefix;

  std::string operation = "classifier";
  SimplifiedParams params = paramsLookup(operation, "backprop");
  Files files = backpropTestFiles.at(operation);
  MemoryOffsets offsets;
  std::string layerName;
  std::string datafile = inputDataDir + files.weights_file;
  bool useDataFile = true;

  params.WEIGHT_OFFSET += ERROR_OFFSET;
  load_weights(params, datafile, useDataFile, acc_sram_memory,
               hls_sram_memory + params.WEIGHT_OFFSET,
               uni_sram_memory + params.WEIGHT_OFFSET,
               float_sram_memory + params.WEIGHT_OFFSET);

  for (int layer = 23; layer >= 0; layer--) {
    for (const auto& backOp : backpropOrder) {
      params = paramsLookup(backOp, "backprop");
      files = backpropTestFiles.at(backOp);
      offsets = backpropMemOffsets.at(backOp);
      layerName = "mobilebert_encoder_layer_" + std::to_string(layer) + "_";

#ifdef ACC_T_ERROR
      params.INPUT_OFFSET = 2 * offsets.INPUT_OFFSET;
      params.OUTPUT_OFFSET = 2 * offsets.OUTPUT_OFFSET;
      params.RESIDUAL_OFFSET = 2 * offsets.RESIDUAL_OFFSET;
#endif

      const int inputOffset =
          ACTIVATION_OFFSET + layer * ENCODER_ACTIVATION_SIZE;
      const int weightOffset = layer * ENCODER_WEIGHT_SIZE;
      const int gradOffset = GRADIENT_OFFSET + layer * ENCODER_WEIGHT_SIZE;

      params.INPUT_OFFSET += ERROR_OFFSET;
      params.WEIGHT_OFFSET += weightOffset;
      params.OUTPUT_OFFSET += ERROR_OFFSET;
      params.RESIDUAL_OFFSET += ERROR_OFFSET;

      if (!params.WEIGHT) {
        params.WEIGHT_OFFSET = inputOffset + offsets.WEIGHT_OFFSET;
      }

      if (params.RELU_GRAD || params.SOFTMAX_GRAD) {
        params.RESIDUAL_OFFSET = inputOffset + offsets.RESIDUAL_OFFSET;
      }

      if (backOp.find("attention_self_value_layer") != std::string::npos ||
          params.CROSS_ENTROPY_GRAD) {
        params.INPUT_OFFSET = inputOffset + offsets.INPUT_OFFSET;
        params.WEIGHT_OFFSET = ERROR_OFFSET + offsets.WEIGHT_OFFSET;
      }

      if (backOp == "classifier" || backOp == "output_bottleneck_LayerNorm") {
        if (layer != 23) continue;
        layerName = "";
      }

      outfilePrefix = "test_outputs/" + backOp + "_error_";
      datafile = errorDataDir + layerName + files.outputs_file;
      runOperation(params, datafile, outfilePrefix, layerName + backOp, groups);

#ifdef DUMP_PARAMS
      if (layer == 23) {
        SimplifiedParams dummyParams = params;
        dummyParams.INPUT_OFFSET = STACK_SIZE;
        dummyParams.WEIGHT_OFFSET = STACK_SIZE + INTERMEDIATE_SIZE;
        dummyParams.OUTPUT_OFFSET = STACK_SIZE + 2 * INTERMEDIATE_SIZE;
        dummyParams.BIAS_OFFSET = 0;
        dummyParams.RESIDUAL_OFFSET = STACK_SIZE + 3 * INTERMEDIATE_SIZE;
        errFile << formatOperation2(dummyParams, backOp + "_backward")
                << std::endl;
      }
#endif

      operation = backOp;
      if (backOp == "bottlenecked_hidden_states" && layer > 0) {
        operation = "output_bottleneck_LayerNorm";
      } else if (backOp == "attention_self_query_layer_0") {
        operation = "attention_self_query";
      } else if (backOp == "attention_self_key_layer_0") {
        operation = "attention_self_key";
      } else if (backOp == "attention_self_value_layer_0") {
        operation = "attention_self_value";
      } else if (backOp == "bottleneck_attention_LayerNorm_k") {
        operation = "bottleneck_attention_LayerNorm";
      }

      std::vector<std::string> gradOperations = {operation + "_weight",
                                                 operation + "_bias"};

      if (backOp == "attention_output_dense") {
        gradOperations.push_back("bottleneck_input_LayerNorm_weight");
        gradOperations.push_back("bottleneck_input_LayerNorm_bias");
      }

      layerName = "mobilebert_encoder_layer_" + std::to_string(layer) + "_";

      for (const auto& gradOp : gradOperations) {
        if (gradientParamsMapping.find(gradOp) != gradientParamsMapping.end()) {
          params = paramsLookup(gradOp, "gradient");
          files = gradientTestFiles.at(gradOp);
          MemoryOffsets gradOffsets = gradientMemOffsets.at(gradOp);

          params.INPUT_OFFSET += inputOffset;
          params.WEIGHT_OFFSET = ERROR_OFFSET + offsets.OUTPUT_OFFSET;
          params.OUTPUT_OFFSET += gradOffset;
          params.RESIDUAL_OFFSET = params.OUTPUT_OFFSET;
          params.RESIDUAL = true;

          if (backOp == "classifier") {
            params.INPUT_OFFSET = ERROR_OFFSET + offsets.OUTPUT_OFFSET;
            params.WEIGHT_OFFSET = inputOffset + gradOffsets.WEIGHT_OFFSET;
            layerName = "";
          }

          if (backOp == "bottlenecked_hidden_states") {
            params.INPUT_OFFSET -= ENCODER_ACTIVATION_SIZE;
            params.OUTPUT_OFFSET -= ENCODER_WEIGHT_SIZE;
            layerName =
                "mobilebert_encoder_layer_" + std::to_string(layer - 1) + "_";
          }

          outfilePrefix = "test_outputs/" + gradOp + "_";
          datafile = gradDataDir + layerName + files.outputs_file;
          runOperation(params, datafile, outfilePrefix, layerName + gradOp,
                       groups);

#ifdef DUMP_PARAMS
          if (layer == 23) {
            SimplifiedParams dummyParams = params;
            dummyParams.INPUT_OFFSET = STACK_SIZE;
            dummyParams.WEIGHT_OFFSET = STACK_SIZE + INTERMEDIATE_SIZE;
            dummyParams.OUTPUT_OFFSET = STACK_SIZE + 2 * INTERMEDIATE_SIZE;
            dummyParams.BIAS_OFFSET = 0;
            dummyParams.RESIDUAL_OFFSET = 0;
            dummyParams.RESIDUAL = false;
            gradFile << formatOperation2(dummyParams, gradOp + "_grad")
                     << std::endl;
          }
#endif
        }
      }
    }
  }

#ifdef DUMP_PARAMS
  errFile.close();
  gradFile.close();
#endif
}
