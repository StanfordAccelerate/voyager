#include "test/mobilebert/MobileBERT.h"

#include <algorithm>

#include "test/mobilebert/mobilebert_tiny2/gradient.h"
#include "test/mobilebert/mobilebert_tiny2/weight.h"
#include "test/mobilebert/training/backward.h"
#include "test/mobilebert/training/forward.h"

#if __has_include("test/mobilebert/paramsCodeGen.h")
#include "test/mobilebert/paramsCodeGen.h"
#else
const std::map<std::string, SimplifiedParams> paramsCodeGen;
const std::map<std::string, Files> filesCodeGen;
const std::vector<std::string> orderCodeGen;
#endif

MobileBERT::MobileBERT(const std::string modelName, const std::string task,
                       const std::string dataDir)
    : Network(modelName, dataDir), task(task) {
  // Make "codgen"-matching case insensitive
  std::string& modelNameLower = const_cast<std::string&>(this->modelName);
  std::transform(modelNameLower.begin(), modelNameLower.end(),
                 modelNameLower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (modelNameLower.find("codegen") != std::string::npos) {
    if (paramsCodeGen.empty() || filesCodeGen.empty() || orderCodeGen.empty()) {
      throw std::runtime_error(
          "Codegen files not found. Did you run the codegen script?");
    }
    this->order = ::orderCodeGen;
    this->params = ::paramsCodeGen;
    this->files = ::filesCodeGen;

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
  this->task = task;
  if (task == "inference" || task == "forward_with_weight_splitting") {
    order = forwardOrder;
    params = forwardParams;
    memOffsets = forwardMemOffsets;
    files = forwardTestFiles;
  } else if (task == "backward" || task == "backward_with_weight_splitting") {
    order = backwardOrder;
    params = backwardParams;
    memOffsets = backwardMemOffsets;
    files = backwardTestFiles;
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
    const std::vector<std::string>& layers, int encoderIndex = 0) const {
  std::vector<Workload> workloads;
  // Make "codgen"-matching case insensitive
  std::string& modelNameLower = const_cast<std::string&>(this->modelName);
  std::transform(modelNameLower.begin(), modelNameLower.end(),
                 modelNameLower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (modelNameLower.find("codegen") != std::string::npos) {
    for (const std::string& layer : layers) {
      Workload workload;
      workload.name = layer;
      workload.params = params.at(layer);
      workload.files = files.at(layer);
      workload.memoryMap = {SRAM, (workload.params.WEIGHT ? RRAM : SRAM), RRAM,
                            SRAM, SRAM};

      workloads.push_back(workload);
    }
  } else {
    for (const std::string& layer : layers) {
      std::string encoderPrefix =
          "mobilebert_encoder_layer_" + std::to_string(encoderIndex) + "_";

      Workload workload;
      workload.name = encoderPrefix + layer + "_" + task;
      workload.params = params.at(layer);
      workload.files = files.at(layer);

      // adjust files path
      std::string inputDataDir;
      std::string weightDataDir;
      std::string outputDataDir;
      std::string residualDataDir;
      std::string gradientDataDir;

      if (task == "inference") {
        inputDataDir = "step_51_activations/";
        weightDataDir = "step_51_weights/";
        outputDataDir = "step_51_activations/";
        residualDataDir = "step_51_activations/";

        if (!workload.params.WEIGHT) {
          weightDataDir = "step_51_activations/";
        }

        workload.memoryMap = {SRAM, workload.params.WEIGHT ? RRAM : SRAM, RRAM,
                              SRAM, SRAM};
      } else if (task == "backward") {
        inputDataDir = "step_51_activation_gradients/";
        weightDataDir = "step_51_weights/";
        outputDataDir = "step_51_activation_gradients/";
        residualDataDir = "step_51_activation_gradients/";

        if (!workload.params.WEIGHT) {
          weightDataDir = "step_51_activations/";
        }

        if (workload.params.SOFTMAX_GRAD || workload.params.RELU_GRAD) {
          residualDataDir = "step_51_activations/";
        }

        if (layer.find("attention_self_value_layer") != std::string::npos) {
          inputDataDir = "step_51_activations/";
          weightDataDir = "step_51_activation_gradients/";
        }

        if (workload.params.CROSS_ENTROPY_GRAD) {
          inputDataDir = "step_51_activations/";
          weightDataDir = "step_51_activations/";
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

        if (layer.find("classifier") != std::string::npos) {
          inputDataDir = "step_51_activation_gradients/";
          weightDataDir = "step_51_activations/";
        }

        workload.memoryMap = {SRAM, SRAM, RRAM, SRAM, SRAM};
      } else if (task == "weight_update") {
        workload.params.learningRate = 0.02995417748587139;

        inputDataDir = "step_51_weight_gradients/";
        weightDataDir = "step_51_weights/";
        outputDataDir = "step_52_weights/";

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

        if (!workload.params.WEIGHT) {
          weightDataDir = "step_52_activations/";
        }

        if (workload.params.SOFTMAX_GRAD || workload.params.RELU_GRAD) {
          residualDataDir = "step_52_activations/";
        }

        if (layer.find("attention_self_value_layer") != std::string::npos) {
          inputDataDir = "step_52_activations/";
          weightDataDir = "step_52_activation_gradients/";
        }

        if (workload.params.CROSS_ENTROPY_GRAD) {
          inputDataDir = "step_52_activations/";
          weightDataDir = "step_52_activations/";
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

      std::map<std::string, size_t> sramOffsets{
          {"activations", ACTIVATION_OFFSET},
          {"activation_gradients", ACTIVATION_OFFSET + 3 * INTERMEDIATE_SIZE},
          {"weight_gradients", ACTIVATION_OFFSET + 6 * INTERMEDIATE_SIZE},
      };

      const size_t rramOffsets =
          WEIGHT_OFFSET + encoderIndex * ENCODER_WEIGHT_SIZE;

      MemoryOffsets memoryOffsets = memOffsets.at(layer);
      workload.params.INPUT_OFFSET = memoryOffsets.INPUT_OFFSET;
      workload.params.WEIGHT_OFFSET = memoryOffsets.WEIGHT_OFFSET;
      workload.params.OUTPUT_OFFSET = memoryOffsets.OUTPUT_OFFSET;
      workload.params.BIAS_OFFSET = memoryOffsets.BIAS_OFFSET;
      workload.params.RESIDUAL_OFFSET = memoryOffsets.RESIDUAL_OFFSET;
      workload.params.WEIGHT_RESIDUAL_OFFSET =
          sramOffsets.at("weight_gradients");

      if (encoderIndex == 0 && workload.params.INPUT_OFFSET == 0 &&
          task == "inference") {
        workload.params.INPUT_OFFSET += STACK_SIZE;
      } else if (workload.memoryMap.inputs) {
        workload.params.INPUT_OFFSET += rramOffsets;
      } else {
        workload.params.INPUT_OFFSET +=
            sramOffsets.at(inputDataDir.substr(8, inputDataDir.length() - 9));
      }

      if (workload.params.CROSS_ENTROPY_GRAD) {
        workload.params.WEIGHT_OFFSET = ACTIVATION_OFFSET - 16;
      } else if (workload.memoryMap.weights) {
        workload.params.WEIGHT_OFFSET += rramOffsets;
      } else {
        workload.params.WEIGHT_OFFSET +=
            sramOffsets.at(weightDataDir.substr(8, weightDataDir.length() - 9));
      }

      if (!workload.params.WEIGHT_UPDATE) {
        workload.params.OUTPUT_OFFSET +=
            sramOffsets.at(outputDataDir.substr(8, outputDataDir.length() - 9));
      } else {
        workload.params.OUTPUT_OFFSET = workload.params.WEIGHT_OFFSET;
      }

      if (workload.params.BIAS &&
          workload.files.bias_file.find("mobilebert_attention_mask") ==
              std::string::npos) {
        workload.params.BIAS_OFFSET += rramOffsets;
      }

      if (encoderIndex == 0 && workload.params.RESIDUAL_OFFSET == 0 &&
          task == "inference") {
        workload.params.RESIDUAL_OFFSET += STACK_SIZE;
      } else if (!workload.params.WEIGHT_UPDATE) {
        workload.params.RESIDUAL_OFFSET += sramOffsets.at(
            residualDataDir.substr(8, residualDataDir.length() - 9));
      }

      workloads.push_back(workload);
    }
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

  // Training end to end
  if (layers.front() == "training") {
    return getBackwardWorkloads();
  }

  // Single layer
  if (layers.size() == 1) {
    layersInRange.push_back(layers.front());
    return getWorkloads(layersInRange);
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

  return getWorkloads(layersInRange);
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
    return getWorkloads(tests);
  }
  return getWorkloads(order);
}

std::vector<Workload> MobileBERT::getWorkloads(std::string start,
                                               std::string end, int startIndex,
                                               int endIndex) {
  if (startIndex == endIndex) {
    auto firstLayer = std::find(order.begin(), order.end(), start);
    auto lastLayer = std::find(order.begin(), order.end(), end);
    std::vector<std::string> layersInRange(firstLayer, lastLayer + 1);
    return getWorkloads(layersInRange, startIndex);
  } else {
    std::vector<Workload> workloads;
    for (int i = startIndex; i <= endIndex; i++) {
      std::vector<std::string> layersInRange;
      if (i == startIndex) {
        auto firstLayer = std::find(order.begin(), order.end(), start);
        layersInRange = std::vector<std::string>(firstLayer, order.end() - 1);
      } else if (i == endIndex) {
        auto lastLayer = std::find(order.begin(), order.end(), end);
        layersInRange = std::vector<std::string>(order.begin(), lastLayer + 1);
      } else {
        layersInRange =
            std::vector<std::string>(order.begin(), order.end() - 1);
      }

      std::vector<Workload> layers = getWorkloads(layersInRange, i);
      workloads.insert(workloads.end(), layers.begin(), layers.end());
    }
    return workloads;
  }
}

std::vector<Workload> MobileBERT::getBackwardWorkloads() {
  std::vector<Workload> workloads;

  // Forward pass
  setTask("inference");
  std::vector<Workload> layers =
      getWorkloads("bottleneck_attention_dense", "classifier", 0, 20);
  for (auto& item : layers) {
    item.loadWeightsAndBiases = true;
  }
  layers.back().checkOutputs = true;
  workloads.insert(workloads.end(), layers.begin(), layers.end());

  // Cross entropy gradient
  setTask("backward");
  std::vector<std::string> layersInRange{"classifier",
                                         "output_bottleneck_LayerNorm"};
  layers = getWorkloads(layersInRange, 20);
  layers.at(0).loadWeightsAndBiases = true;
  workloads.insert(workloads.end(), layers.begin(), layers.end());

  // Backpropagate encoder by encoder
  for (int i = 20; i >= 0; i--) {
    // FFN backward
    setTask("inference");
    layers =
        getWorkloads("bottleneck_attention_dense", "intermediate_dense", 0, i);
    workloads.insert(workloads.end(), layers.begin(), layers.end());

    setTask("backward");
    layers =
        getWorkloads("output_bottleneck_dense", "ffn_0_output_dense", i, i);
    workloads.insert(workloads.end(), layers.begin(), layers.end());

    setTask("inference");
    layers = getWorkloads({"ffn_0_intermediate_dense"}, i);
    workloads.insert(workloads.end(), layers.begin(), layers.end());

    setTask("backward");
    layers = getWorkloads("ffn_0_intermediate_dense",
                          "attention_self_context_layer", i, i);
    workloads.insert(workloads.end(), layers.begin(), layers.end());

    // MHA backward
    setTask("inference");
    layers = getWorkloads("bottleneck_attention_dense",
                          "attention_self_value_layer", i, i);
    workloads.insert(workloads.end(), layers.begin(), layers.end());

    for (int j = 0; j < 4; j++) {
      setTask("inference");
      layers = getWorkloads(
          "attention_self_attention_scores_" + std::to_string(j),
          "attention_self_attention_probs_" + std::to_string(j), i, i);
      workloads.insert(workloads.end(), layers.begin(), layers.end());

      setTask("backward");
      layers =
          getWorkloads("attention_self_value_layer_" + std::to_string(j),
                       "attention_self_key_layer_" + std::to_string(j), i, i);
      workloads.insert(workloads.end(), layers.begin(), layers.end());
    }

    setTask("backward");
    std::string endingPoint =
        i ? "bottlenecked_hidden_states" : "bottleneck_attention_dense";
    layers = getWorkloads("query_to_bottleneck_attention_LayerNorm",
                          endingPoint, i, i);
    layers.back().checkOutputs = true;
    workloads.insert(workloads.end(), layers.begin(), layers.end());
  }

  std::cerr << workloads.size() << std::endl;
  return workloads;
}
