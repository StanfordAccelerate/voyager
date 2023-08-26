#include "test/mobilebert/MobileBERT.h"

#include <algorithm>

MobileBERT::MobileBERT(const std::string modelName, const std::string task,
                       const std::string dataDir)
    : Network(modelName, dataDir), task(task) {
  if (codegen) {
#if __has_include("test/mobilebert/paramsCodeGen.h")
#include "test/mobilebert/paramsCodeGen.h"
#else
    const std::map<std::string, SimplifiedParams> paramsCodeGen;
    const std::map<std::string, Files> filesCodeGen;
    const std::vector<std::string> orderCodeGen;
#endif

    if (paramsCodeGen.empty() || filesCodeGen.empty() || orderCodeGen.empty()) {
      throw std::runtime_error(
          "Codegen files for MobileBERT not found. Did you run the codegen "
          "script?");
    }
    this->order = orderCodeGen;
    this->params = paramsCodeGen;
    this->files = filesCodeGen;

    // Prepend dataDir to all files
    for (auto& it : files) {
      it.second.inputs_file.insert(0, this->dataDir);
      it.second.weights_file.insert(0, this->dataDir);
      it.second.bias_file.insert(0, this->dataDir);
      it.second.outputs_file.insert(0, this->dataDir);
      it.second.residual_file.insert(0, this->dataDir);
    }
  } else {
    setTask(task);
  }
}

void MobileBERT::setTask(std::string task) {
#include "test/mobilebert/mobilebert_tiny2/backprop.h"
#include "test/mobilebert/mobilebert_tiny2/gradient.h"
#include "test/mobilebert/mobilebert_tiny2/inference.h"
#include "test/mobilebert/mobilebert_tiny2/weight.h"

  this->task = task;
  if (task == "inference" || task == "forward_with_weight_splitting") {
    order = inferenceOrder;
    params = inferenceParams;
    memOffsets = inferenceMemOffsets;
    files = inferenceTestFiles;
  } else if (task == "backward" || task == "backward_with_weight_splitting") {
    order = backpropOrder;
    params = backpropParams;
    memOffsets = backpropMemOffsets;
    files = backpropTestFiles;
  } else if (task == "gradient") {
    params = gradientParams;
    memOffsets = gradientMemOffsets;
    files = gradientTestFiles;

    order.clear();
    for (auto it = gradientParams.begin(); it != gradientParams.end(); it++) {
      order.push_back(it->first);
    }
  } else if (task == "weight_update" || task == "sram_weight_update") {
    params = weightParams;
    memOffsets = weightMemOffsets;

    order.clear();
    files.clear();
    for (auto it = weightParams.begin(); it != weightParams.end(); it++) {
      order.push_back(it->first);
      files.insert({it->first, Files{it->first, it->first, "", it->first}});
    }
  } else {
    std::cerr << "ERROR: unrecognized task: " << task << std::endl;
    std::abort();
  }
}

std::vector<Workload> MobileBERT::getWorkloads(
    const std::vector<std::string>& layers, bool useOffsets,
    int encoderIndex = 0, bool useCustomDataDir = false) const {
  std::vector<Workload> workloads;
  workloads.reserve(10000);

  for (const std::string& layer : layers) {
    // Setup workload
    Workload workload;
    workload.name = layer;
    workload.params = params.at(layer);
    workload.files = files.at(layer);

    // Handle hardcoded optimizations
    if (opt == O0) {
      // force all banks to be on
      for (int i = 0; i < NUM_SRAM_BANKS; i++) {
        workload.params.sram_banks[i] = ON;
      }
      for (int i = 0; i < NUM_RRAM_BANKS; i++) {
        workload.params.rram_banks[i] = ON;
      }
    }
    if (opt == O0 || opt == O1) {
      // force full bandwidth mode
      workload.params.bandwidth_mode = QUAD;
    }

    // If codegen, use generated models
    if (codegen) {
      // inputs, weights, bias, residual, outputs
      workload.memoryMap = {SRAM, workload.params.WEIGHT ? RRAM : SRAM,
                            workload.params.ATTENTION_MASK ? SRAM : RRAM, SRAM,
                            SRAM};

      // Otherwise, use handwritten models
    } else {
      std::string encoderPrefix =
          "mobilebert_encoder_layer_" + std::to_string(encoderIndex) + "_";
      // Prepend encoder prefix to workload name in order to avoid name
      // collisions across encoders (codegen doesnt need that since it uses
      // unique names directly from the ONNX model)
      workload.name = encoderPrefix + workload.name;

      // adjust files path
      std::string inputDataDir;
      std::string weightDataDir;
      std::string outputDataDir;
      std::string residualDataDir;
      std::string gradientDataDir;

      int inputOffset = 0;
      int weightOffset = 0;
      int outputOffset = 0;
      int residualOffset = 0;

      const int activationSize =
          ENCODER_ACTIVATION_SIZE + INTERMEDIATE_SIZE + 32;
      const int weightSize = ENCODER_WEIGHT_SIZE + 16 * 512 * 2;

      if (task == "inference") {
        if (!useCustomDataDir) {
          inputDataDir = "step_51_activations/";
          weightDataDir = "step_51_weights/";
          outputDataDir = "step_51_activations/";
          residualDataDir = "step_51_activations/";

          if (!workload.params.WEIGHT) {
            weightDataDir = "step_51_activations/";
          }
        }

        workload.memoryMap = {SRAM, workload.params.WEIGHT ? RRAM : SRAM, RRAM,
                              SRAM, SRAM};
        // if operation involves attention mask, use SRAM as bias
        if (workload.files.bias_file.find("mobilebert_attention_mask") !=
            std::string::npos) {
          workload.memoryMap.bias = SRAM;
          workload.params.ATTENTION_MASK = true;
        }
      } else if (task == "backward") {
        inputDataDir = "step_51_activation_gradients/";
        weightDataDir = "step_51_weights/";
        outputDataDir = "step_51_activation_gradients/";
        residualDataDir = "step_51_activation_gradients/";

        inputOffset = activationSize + weightSize;
        outputOffset = activationSize + weightSize;
        residualOffset = activationSize + weightSize;

        if (!workload.params.WEIGHT) {
          weightDataDir = "step_51_activations/";
        }
        if (workload.params.SOFTMAX_GRAD || workload.params.RELU_GRAD) {
          residualDataDir = "step_51_activations/";
          residualOffset = 0;
        }
        if (layer.find("attention_self_value_layer") != std::string::npos) {
          inputDataDir = "step_51_activations/";
          weightDataDir = "step_51_activation_gradients/";
          inputOffset = 0;
          weightOffset = activationSize + weightSize;
        }
        if (workload.params.CROSS_ENTROPY_GRAD) {
          inputDataDir = "step_51_activations/";
          weightDataDir = "step_51_activations/";
          inputOffset = 0;
          weightOffset = 0;
        }

        workload.memoryMap = {SRAM, workload.params.WEIGHT ? RRAM : SRAM, RRAM,
                              SRAM, SRAM};
      } else if (task == "gradient") {
        workload.params.outputExpBias = -13;
        workload.params.RESIDUAL = true;
        workload.params.GRAD_CLIPPING = true;
        workload.params.ACC_T_OUTPUT = true;
        workload.files.residual_file = workload.files.outputs_file;

        inputDataDir = "step_51_activations/";
        weightDataDir = "step_51_activation_gradients/";
        outputDataDir = "step_51_weight_gradients/";
        residualDataDir = "step_50_weight_gradients/";

        weightOffset = activationSize + weightSize;
        outputOffset = activationSize;
        residualOffset = activationSize;

        if (layer.find("classifier") != std::string::npos) {
          inputDataDir = "step_51_activation_gradients/";
          weightDataDir = "step_51_activations/";

          inputOffset = activationSize + weightSize;
          weightOffset = 0;
        }

        workload.memoryMap = {SRAM, SRAM, RRAM, SRAM, SRAM};
      } else if (task == "weight_update") {
        workload.params.learningRate = 0.02995417748587139;

        inputDataDir = "step_51_weight_gradients/";
        weightDataDir = "step_51_weights/";
        outputDataDir = "step_52_weights/";

        inputOffset = activationSize;

        workload.memoryMap = {SRAM, RRAM, RRAM, SRAM, RRAM};
      } else if (task == "sram_weight_update") {
        workload.params.learningRate = 0.02995417748587139;

        inputDataDir = "step_51_weight_gradients/";
        weightDataDir = "step_51_weights/";
        outputDataDir = "step_52_weights/";

        workload.memoryMap = {SRAM, SRAM, SRAM, SRAM, SRAM};
      } else if (task == "forward_with_weight_splitting") {
        workload.params.WEIGHT_SPLITTING =
            workload.params.WEIGHT && !workload.params.NO_NORM;
        workload.params.learningRate = 0.02995417748587139;
        workload.files.weight_grad_file = workload.files.weights_file;

        inputDataDir = "step_52_activations/";
        weightDataDir = "step_52_ws_weights/";
        outputDataDir = "step_52_activations/";
        residualDataDir = "step_52_activations/";
        gradientDataDir = "step_51_weight_gradients/";

        if (!workload.params.WEIGHT) {
          weightDataDir = "step_52_activations/";
        }

        workload.memoryMap = {SRAM, workload.params.WEIGHT ? RRAM : SRAM, RRAM,
                              SRAM, SRAM};
      } else if (task == "backward_with_weight_splitting") {
        workload.params.WEIGHT_SPLITTING = workload.params.WEIGHT;
        workload.params.learningRate = 0.02995417748587139;
        workload.files.weight_grad_file = workload.files.weights_file;

        if (workload.params.CROSS_ENTROPY_GRAD) {
          workload.params.outputExpBias = 8;
        }

        inputDataDir = "step_52_activation_gradients/";
        weightDataDir = "step_52_ws_weights/";
        outputDataDir = "step_52_activation_gradients/";
        residualDataDir = "step_52_activation_gradients/";
        gradientDataDir = "step_51_weight_gradients/";

        inputOffset = activationSize + weightSize;
        outputOffset = activationSize + weightSize;
        residualOffset = activationSize + weightSize;

        if (!workload.params.WEIGHT) {
          weightDataDir = "step_52_activations/";
        }
        if (workload.params.SOFTMAX_GRAD || workload.params.RELU_GRAD) {
          residualDataDir = "step_52_activations/";
          residualOffset = 0;
        }
        if (layer.find("attention_self_value_layer") != std::string::npos) {
          inputDataDir = "step_52_activations/";
          weightDataDir = "step_52_activation_gradients/";
          inputOffset = 0;
          weightOffset = activationSize + weightSize;
        }
        if (workload.params.CROSS_ENTROPY_GRAD) {
          inputDataDir = "step_52_activations/";
          weightDataDir = "step_52_activations/";
          inputOffset = 0;
          weightOffset = 0;
        }

        workload.memoryMap = {SRAM, workload.params.WEIGHT ? RRAM : SRAM, RRAM,
                              SRAM, SRAM};
      }

      if (layer.find("classifier") != std::string::npos ||
          (layer == "output_bottleneck_LayerNorm" &&
           task.find("backward") != std::string::npos)) {
        encoderPrefix = "";
      }

      if (!workload.files.inputs_file.empty()) {
        workload.files.inputs_file.insert(
            0, dataDir + inputDataDir + encoderPrefix);
      }

      if (!workload.files.weights_file.empty()) {
        workload.files.weights_file.insert(
            0, dataDir + weightDataDir + encoderPrefix);
      }

      if (workload.files.bias_file == "mobilebert_attention_mask") {
        workload.files.bias_file.insert(0, dataDir + inputDataDir);
      } else {
        workload.files.bias_file.insert(
            0, dataDir + weightDataDir + encoderPrefix);
      }

      workload.files.outputs_file.insert(
          0, dataDir + outputDataDir + encoderPrefix);
      workload.files.residual_file.insert(
          0, dataDir + residualDataDir + encoderPrefix);
      workload.files.weight_grad_file.insert(
          0, dataDir + gradientDataDir + encoderPrefix);

      const size_t rramOffsets =
          WEIGHT_OFFSET + encoderIndex * ENCODER_WEIGHT_SIZE;

      MemoryOffsets offsets = memOffsets.at(layer);
      workload.params.INPUT_OFFSET = offsets.INPUT_OFFSET;
      workload.params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET;
      workload.params.OUTPUT_OFFSET = offsets.OUTPUT_OFFSET;
      workload.params.BIAS_OFFSET = offsets.BIAS_OFFSET;
      workload.params.RESIDUAL_OFFSET = offsets.RESIDUAL_OFFSET;

      if (useOffsets) {
        workload.params.INPUT_OFFSET += ACTIVATION_OFFSET + inputOffset;

        if (workload.memoryMap.weights == RRAM) {
          workload.params.WEIGHT_OFFSET += rramOffsets;
        } else {
          workload.params.WEIGHT_OFFSET += ACTIVATION_OFFSET;
        }

        workload.params.OUTPUT_OFFSET += ACTIVATION_OFFSET + outputOffset;

        if (workload.memoryMap.residual == SRAM) {
          workload.params.RESIDUAL_OFFSET += ACTIVATION_OFFSET + residualOffset;
        }
        workload.params.WEIGHT_RESIDUAL_OFFSET +=
            ACTIVATION_OFFSET + activationSize;

        if (workload.params.WEIGHT_UPDATE) {
          workload.params.OUTPUT_OFFSET = workload.params.WEIGHT_OFFSET;
        }

        if (workload.params.BIAS) {
          if (workload.files.bias_file.find("mobilebert_attention_mask") ==
              std::string::npos) {
            workload.params.BIAS_OFFSET += rramOffsets;
          } else {  // attention_mask is being used as bias
            workload.params.BIAS_OFFSET += ATTENTION_MASK_OFFSET;
          }
        }
      }
    }

    workloads.push_back(workload);
  }
  return workloads;
}

/*
 * Returns a vector of workloads to run.
 * Layers specifies either a single layer or range of layers.
 */
std::vector<Workload> MobileBERT::getWorkloadsInRange(
    const std::vector<std::string>& layers) {
  std::vector<std::string> layersInRange;

  // End to end tests
  if (layers.front() == "all") {
    std::vector<Workload> workloads;
    std::vector<Workload> forward = getForwardWorkloads();
    workloads.insert(workloads.end(), forward.begin(), forward.end());
    std::vector<Workload> backward = getBackwardWorkloads();
    workloads.insert(workloads.end(), backward.begin(), backward.end());
    return workloads;
  }

  // Single layer
  if (layers.size() == 1) {
    layersInRange.push_back(layers.front());
    return getWorkloads(layersInRange, true);
  }

  // Multi-layer test
  if (task == "gradient") {
    std::cerr << "Task gradient does not have an order defined. Please define "
                 "one before attempting to run a sequence."
              << std::endl;
    std::abort();
  }

  auto firstLayer = std::find(order.begin(), order.end(), layers.at(0));
  auto lastLayer = std::find(order.begin(), order.end(), layers.at(1));
  layersInRange = std::vector<std::string>(firstLayer, lastLayer + 1);

  if (layersInRange.empty()) {
    throw std::runtime_error("Layer list is empty.");
  }

  return getWorkloads(layersInRange, true);
}

std::vector<Workload> MobileBERT::getAllWorkloads() {
  if (task == "backward") {
    std::vector<std::string> tests;
    std::vector<std::string> skipTests{
        "query_to_bottleneck_attention_LayerNorm",
        "key_to_bottleneck_attention_LayerNorm",
        "shared_attention_input_to_hidden_states",
        "value_to_hidden_states",
        "bottlenecked_hidden_states",
    };
    for (auto it = order.begin(); it != order.end(); it++) {
      if (std::find(skipTests.begin(), skipTests.end(), *it) ==
          skipTests.end()) {
        tests.push_back(*it);
      }
    }
    return getWorkloads(tests, true);
  }
  return getWorkloads(order, true);
}

std::vector<Workload> MobileBERT::getForwardWorkloads() {
  setTask("inference");
  std::vector<Workload> inferenceWorkloads;

  int inputOffset;
  int weightOffset;

  auto encoderOrder = std::vector<std::string>(order.begin(), order.end() - 1);

  for (int layer = 0; layer < 21; layer++) {
    std::vector<Workload> workloads = getWorkloads(encoderOrder, false, layer);

    inputOffset = ACTIVATION_OFFSET + layer * ENCODER_ACTIVATION_SIZE;
    weightOffset = WEIGHT_OFFSET + layer * ENCODER_WEIGHT_SIZE;

    for (auto workload : workloads) {
      workload.params.INPUT_OFFSET += inputOffset;
      workload.params.WEIGHT_OFFSET +=
          workload.params.WEIGHT ? weightOffset : inputOffset;
      workload.params.OUTPUT_OFFSET += inputOffset;
      workload.params.BIAS_OFFSET += weightOffset;
      workload.params.RESIDUAL_OFFSET += inputOffset;

      if (workload.files.bias_file.find("mobilebert_attention_mask") !=
          std::string::npos) {
        workload.params.BIAS_OFFSET = 0;
      }

      inferenceWorkloads.push_back(workload);
    }
  }

  Workload classifier = getWorkloads({"classifier"}, false, 20).front();
  classifier.params.INPUT_OFFSET += inputOffset;
  classifier.params.WEIGHT_OFFSET += weightOffset;
  classifier.params.OUTPUT_OFFSET += inputOffset;
  classifier.params.BIAS_OFFSET += weightOffset;
  inferenceWorkloads.push_back(classifier);

  // inferenceWorkloads = std::vector<Workload>(inferenceWorkloads.begin(),
  //                                            inferenceWorkloads.begin() +
  //                                            23);

  return inferenceWorkloads;
}

std::vector<Workload> MobileBERT::getFullForwardPass() {
  setTask("inference");
  std::vector<Workload> inferenceWorkloads;

  int inputOffset;
  int weightOffset;

  auto encoderOrder = std::vector<std::string>(order.begin(), order.end() - 1);

  for (int encoderIndex = 0; encoderIndex < 21; encoderIndex++) {
    std::vector<Workload> encoderLayerWorkloads =
        getWorkloads(encoderOrder, true, encoderIndex, true);

    inferenceWorkloads.insert(inferenceWorkloads.end(),
                              encoderLayerWorkloads.begin(),
                              encoderLayerWorkloads.end());
  }

  // add final classifier
  Workload classifier = getWorkloads({"classifier"}, true, 20, true).front();
  inferenceWorkloads.push_back(classifier);

  return inferenceWorkloads;
}

std::vector<Workload> MobileBERT::getFullBackwardPass() {
  setTask("backward");
  std::vector<Workload> backwardWorkloads;

  int inputOffset = ACTIVATION_OFFSET;
  int weightOffset = WEIGHT_OFFSET + 20 * ENCODER_WEIGHT_SIZE;

  // Cross entropy gradient
  Workload workload = getWorkloads({"classifier"}, false, 20).front();
  workload.params.INPUT_OFFSET += inputOffset;
  workload.params.WEIGHT_OFFSET = ACTIVATION_OFFSET - 16;
  workload.params.OUTPUT_OFFSET += inputOffset;
  backwardWorkloads.push_back(workload);

  // Classifier layer backprop
  workload = getWorkloads({"output_bottleneck_LayerNorm"}, false, 20).front();
  workload.params.INPUT_OFFSET += inputOffset;
  workload.params.WEIGHT_OFFSET += weightOffset;
  workload.params.OUTPUT_OFFSET += inputOffset;
  workload.loadWeight = false;
  backwardWorkloads.push_back(workload);

  auto encoderOrder = std::vector<std::string>(order.begin() + 2, order.end());

  for (int layer = 20; layer >= 0; layer--) {
    std::vector<Workload> workloads = getWorkloads(encoderOrder, false, layer);

    inputOffset = ACTIVATION_OFFSET;
    weightOffset = WEIGHT_OFFSET + layer * ENCODER_WEIGHT_SIZE;

    for (auto workload : workloads) {
      SimplifiedParams params = workload.params;

      workload.params.INPUT_OFFSET += inputOffset;
      workload.params.WEIGHT_OFFSET += weightOffset;
      workload.params.OUTPUT_OFFSET += inputOffset;
      workload.params.RESIDUAL_OFFSET += inputOffset;

      if (!workload.params.WEIGHT) {
        workload.params.WEIGHT_OFFSET = inputOffset + params.WEIGHT_OFFSET;
      }

      if (workload.params.RELU_GRAD || workload.params.SOFTMAX_GRAD) {
        workload.params.RESIDUAL_OFFSET = inputOffset + params.RESIDUAL_OFFSET;
      }

      if (workload.name.find("attention_self_value_layer") !=
          std::string::npos) {
        workload.params.INPUT_OFFSET = inputOffset + params.INPUT_OFFSET;
        workload.params.WEIGHT_OFFSET = inputOffset + params.WEIGHT_OFFSET;
      }

      workload.loadWeight = false;

      backwardWorkloads.push_back(workload);
    }
  }

  backwardWorkloads = std::vector<Workload>(backwardWorkloads.begin(),
                                            backwardWorkloads.end() - 3);

  return backwardWorkloads;
}

std::vector<Workload> MobileBERT::getBackwardWorkloads() {
  setTask("backward");
  std::vector<Workload> backwardWorkloads;

  int inputOffset = ACTIVATION_OFFSET + 20 * ENCODER_ACTIVATION_SIZE;
  int weightOffset = WEIGHT_OFFSET + 20 * ENCODER_WEIGHT_SIZE;

  // Cross entropy gradient
  Workload workload = getWorkloads({"classifier"}, false, 20).front();
  workload.params.INPUT_OFFSET += inputOffset;
  workload.params.WEIGHT_OFFSET = ACTIVATION_OFFSET - 16;
  workload.params.OUTPUT_OFFSET += ERROR_OFFSET;
  backwardWorkloads.push_back(workload);

  // Classifier layer backprop
  workload = getWorkloads({"output_bottleneck_LayerNorm"}, false, 20).front();
  workload.params.INPUT_OFFSET += ERROR_OFFSET;
  workload.params.WEIGHT_OFFSET += weightOffset;
  workload.params.OUTPUT_OFFSET += ERROR_OFFSET;
  workload.loadWeight = false;
  backwardWorkloads.push_back(workload);

  auto encoderOrder = std::vector<std::string>(order.begin() + 2, order.end());

  for (int layer = 20; layer >= 0; layer--) {
    std::vector<Workload> workloads = getWorkloads(encoderOrder, false, layer);

    inputOffset = ACTIVATION_OFFSET + layer * ENCODER_ACTIVATION_SIZE;
    weightOffset = WEIGHT_OFFSET + layer * ENCODER_WEIGHT_SIZE;

    for (auto workload : workloads) {
      SimplifiedParams params = workload.params;

      workload.params.INPUT_OFFSET += ERROR_OFFSET;
      workload.params.WEIGHT_OFFSET += weightOffset;
      workload.params.OUTPUT_OFFSET += ERROR_OFFSET;
      workload.params.RESIDUAL_OFFSET += ERROR_OFFSET;

      if (!workload.params.WEIGHT) {
        workload.params.WEIGHT_OFFSET = inputOffset + params.WEIGHT_OFFSET;
      }

      if (workload.params.RELU_GRAD || workload.params.SOFTMAX_GRAD) {
        workload.params.RESIDUAL_OFFSET = inputOffset + params.RESIDUAL_OFFSET;
      }

      if (workload.name.find("attention_self_value_layer") !=
          std::string::npos) {
        workload.params.INPUT_OFFSET = inputOffset + params.INPUT_OFFSET;
        workload.params.WEIGHT_OFFSET = ERROR_OFFSET + params.WEIGHT_OFFSET;
      }

      workload.loadWeight = false;

      backwardWorkloads.push_back(workload);

      // TODO: add gradient tests
      // std::cerr << workload.files.output_file << std::endl;

      // std::string operation = backOp;
      // if (backOp == "bottlenecked_hidden_states" && layer > 0) {
      //   operation = "output_bottleneck_LayerNorm";
      // } else if (backOp == "attention_self_query_layer_0") {
      //   operation = "attention_self_query";
      // } else if (backOp == "attention_self_key_layer_0") {
      //   operation = "attention_self_key";
      // } else if (backOp == "attention_self_value_layer_0") {
      //   operation = "attention_self_value";
      // } else if (backOp == "bottleneck_attention_LayerNorm_k") {
      //   operation = "bottleneck_attention_LayerNorm";
      // }
    }
  }

  // backwardWorkloads = std::vector<Workload>(backwardWorkloads.begin(),
  //                                           backwardWorkloads.begin() +
  //                                           35);
  backwardWorkloads = std::vector<Workload>(backwardWorkloads.begin(),
                                            backwardWorkloads.end() - 3);

  return backwardWorkloads;
}
