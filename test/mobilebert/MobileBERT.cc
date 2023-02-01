#include "test/mobilebert/MobileBERT.h"

#include <algorithm>

#include "test/mobilebert/mobilebert_tiny2/backprop.h"
#include "test/mobilebert/mobilebert_tiny2/gradient.h"
#include "test/mobilebert/mobilebert_tiny2/inference.h"
#include "test/mobilebert/mobilebert_tiny2/weight.h"

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
  } else if (task == "weight_update") {
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
    const std::vector<std::string>& layers, int encoderIndex = 0,
    bool useOffsets = false) const {
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
      workload.name = layer;
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
        workload.params.RESIDUAL = true;
        workload.params.outputExpBias = -13;
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

      if (useOffsets) {
        MemoryOffsets offsets = memOffsets.at(layer);
        workload.params.INPUT_OFFSET = offsets.INPUT_OFFSET;
        workload.params.WEIGHT_OFFSET = offsets.WEIGHT_OFFSET;
        workload.params.OUTPUT_OFFSET = offsets.OUTPUT_OFFSET;
        workload.params.BIAS_OFFSET = offsets.BIAS_OFFSET;
        workload.params.RESIDUAL_OFFSET = offsets.RESIDUAL_OFFSET;
      } else {
        // Fake memory offsets used for unit tests
        workload.params.INPUT_OFFSET = STACK_SIZE;
        workload.params.WEIGHT_OFFSET = STACK_SIZE + INTERMEDIATE_SIZE;

        if (workload.params.WEIGHT_UPDATE) {
          // Offset used for comparing outputs in weight update unit tests
          workload.params.OUTPUT_OFFSET = workload.params.ERROR_FEEDBACK
                                              ? workload.params.WEIGHT_OFFSET
                                              : workload.params.WEIGHT_OFFSET;
        } else {
          workload.params.OUTPUT_OFFSET = STACK_SIZE + 2 * INTERMEDIATE_SIZE;
        }

        workload.params.BIAS_OFFSET = STACK_SIZE + 3 * INTERMEDIATE_SIZE;
        workload.params.RESIDUAL_OFFSET = STACK_SIZE + 4 * INTERMEDIATE_SIZE;
        workload.params.WEIGHT_RESIDUAL_OFFSET =
            STACK_SIZE + 5 * INTERMEDIATE_SIZE;
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
  // Expand layer vector to include intermediate layers, if necessary
  std::vector<std::string> layersInRange;
  if (layers.size() == 1) {  // Single layer
    if (layers.front() == "all") {
      std::vector<Workload> workloads;
      std::vector<Workload> forward = getForwardWorkloads();
      workloads.insert(workloads.end(), forward.begin(), forward.end());
      std::vector<Workload> backward = getBackwardWorkloads();
      workloads.insert(workloads.end(), backward.begin(), backward.end());
      return workloads;
    }
    layersInRange.push_back(layers.front());
  } else {  // Range of layers
    if (task == "gradient") {
      std::cerr << "Task gradient "
                << " does not have an order defined. Please define one before "
                   "attempting to run a sequence."
                << std::endl;
      std::abort();
    }

    auto firstLayer = std::find(order.begin(), order.end(), layers.at(0));
    auto lastLayer = std::find(order.begin(), order.end(), layers.at(1));
    layersInRange = std::vector<std::string>(firstLayer, lastLayer + 1);
  }

  if (layersInRange.empty()) {
    throw std::runtime_error("Layer list is empty.");
  }

  return getWorkloads(layersInRange);
}

std::vector<Workload> MobileBERT::getAllWorkloads() {
  return getWorkloads(order);
}

std::vector<Workload> MobileBERT::getForwardWorkloads() {
  setTask("inference");
  std::vector<Workload> inferenceWorkloads;

  int inputOffset;
  int weightOffset;

  auto encoderOrder = std::vector<std::string>(inferenceOrder.begin(),
                                               inferenceOrder.end() - 1);

  for (int layer = 0; layer < 21; layer++) {
    std::vector<Workload> workloads = getWorkloads(encoderOrder, layer, true);

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

  Workload classifier = getWorkloads({"classifier"}, 20, true).front();
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

std::vector<Workload> MobileBERT::getBackwardWorkloads() {
  setTask("backward");
  std::vector<Workload> backwardWorkloads;

  int inputOffset = ACTIVATION_OFFSET + 20 * ENCODER_ACTIVATION_SIZE;
  int weightOffset = WEIGHT_OFFSET + 20 * ENCODER_WEIGHT_SIZE;

  // Cross entropy gradient
  Workload workload = getWorkloads({"classifier"}, 20, true).front();
  workload.params.INPUT_OFFSET += inputOffset;
  workload.params.WEIGHT_OFFSET = ACTIVATION_OFFSET - 16;
  workload.params.OUTPUT_OFFSET += ERROR_OFFSET;
  backwardWorkloads.push_back(workload);

  // Classifier layer backprop
  workload = getWorkloads({"output_bottleneck_LayerNorm"}, 20, true).front();
  workload.params.INPUT_OFFSET += ERROR_OFFSET;
  workload.params.WEIGHT_OFFSET += weightOffset;
  workload.params.OUTPUT_OFFSET += ERROR_OFFSET;
  workload.loadWeight = false;
  backwardWorkloads.push_back(workload);

  auto encoderOrder =
      std::vector<std::string>(backpropOrder.begin() + 2, backpropOrder.end());

  for (int layer = 20; layer >= 0; layer--) {
    std::vector<Workload> workloads = getWorkloads(encoderOrder, layer, true);

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
  //                                           backwardWorkloads.begin() + 35);
  backwardWorkloads = std::vector<Workload>(backwardWorkloads.begin(),
                                            backwardWorkloads.end() - 3);

  return backwardWorkloads;
}
