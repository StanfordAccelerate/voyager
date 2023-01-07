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
    if (task == "inference" || task == "weight_splitting") {
      order = inferenceOrder;
      paramsMapping = inferenceParamsMapping;
      params = inferenceParams;
      memOffsets = inferenceMemOffsets;
      files = inferenceTestFiles;
    } else if (task == "gradient") {
      // NOTE: order is not defined for gradient
      paramsMapping = gradientParamsMapping;
      params = gradientParams;
      memOffsets = gradientMemOffsets;
      files = gradientTestFiles;
    } else if (task == "backprop") {
      order = backpropOrder;
      paramsMapping = backpropParamsMapping;
      params = backpropParams;
      memOffsets = backpropMemOffsets;
      files = backpropTestFiles;
    } else if (task == "weight_update" || task == "error_feedback") {
      paramsMapping = weightParamsMapping;
      params = weightParams;
      memOffsets = weightMemOffsets;

      for (auto it = weightParamsMapping.begin();
           it != weightParamsMapping.end(); it++) {
        Files file = {it->first, it->first, "", it->first};
        files.insert({it->first, file});
      }
    } else {
      std::cerr << "ERROR: Task must be one of \"inference\", \"gradient\", "
                   "\"backprop\", \"weight_update\", and \"weight_splitting\"."
                << std::endl;
      std::abort();
    }
  }
}

std::vector<Workload> MobileBERT::getWorkloads(
    const std::vector<std::string>& layers, int layerIndex = 0,
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
      const std::string& paramName = paramsMapping.at(layer);
      std::string encLayerName =
          "mobilebert_encoder_layer_" + std::to_string(layerIndex) + "_";

      Workload workload;
      workload.name = encLayerName + layer;
      workload.params = params.at(paramName);
      workload.files = files.at(layer);

      // adjust files path
      std::string inputDataDir;
      std::string weightDataDir;
      std::string outputDataDir;
      std::string residualDataDir;
      std::string gradientDataDir;

      if (task == "inference") {
        inputDataDir = "activations/";
        weightDataDir = "weights/";
        outputDataDir = "activations/";
        residualDataDir = "activations/";

        if (!workload.params.WEIGHT) {
          weightDataDir = "activations/";
        }
      } else if (task == "backprop") {
        inputDataDir = "activation_gradients/";
        weightDataDir = "weights/";
        outputDataDir = "activation_gradients/";
        residualDataDir = "activation_gradients/";

        if (!workload.params.WEIGHT) {
          weightDataDir = "activations/";
        }
        if (workload.params.SOFTMAX_GRAD || workload.params.RELU_GRAD) {
          residualDataDir = "activations/";
        }
        if (layer.find("attention_self_value_layer") != std::string::npos) {
          inputDataDir = "activations/";
          weightDataDir = "activation_gradients/";
        }
        if (workload.params.CROSS_ENTROPY_GRAD) {
          inputDataDir = "activations/";
          weightDataDir = "activations/";
        }
      } else if (task == "gradient") {
        inputDataDir = "activations/";
        weightDataDir = "activation_gradients/";
        outputDataDir = "weight_gradients/";
        residualDataDir = "weight_gradients/";

        if (layer.find("classifier") != std::string::npos) {
          inputDataDir = "activation_gradients/";
          weightDataDir = "activations/";
        }

        // TODO: accelerator doesn't support these functionalities yet
        // this results in FP32<->Pytorch failing for backprop and gradients
        workload.params.ACC_T_OUTPUT = false;
      } else if (task == "weight_update" || task == "error_feedback") {
        workload.params.ERROR_FEEDBACK = task == "error_feedback";

        inputDataDir = "quantized_weight_gradients/";
        weightDataDir = "quantized_weights/";
        outputDataDir = workload.params.ERROR_FEEDBACK
                            ? "updated_weight_gradients/"
                            : "updated_weights/";
      } else if (task == "weight_splitting") {
        workload.params.WEIGHT_SPLITTING = true;
        workload.params.learningRate = 3e-2;
        workload.files.weight_grad_file = workload.files.weights_file;
        workload.files.bias_grad_file = workload.files.bias_file;

        inputDataDir = "activations2/";
        weightDataDir = "weights/";
        outputDataDir = "activations2/";
        residualDataDir = "activations2/";
        gradientDataDir = "weight_gradients/";

        if (!workload.params.WEIGHT || workload.params.ATTENTION_MASK) {
          weightDataDir = "activations/";
        }
      }

      if (layer.find("classifier") != std::string::npos ||
          (task == "backprop" && layer == "output_bottleneck_LayerNorm")) {
        encLayerName = "";
      }

      if (!workload.files.inputs_file.empty()) {
        workload.files.inputs_file.insert(
            0, dataDir + inputDataDir + encLayerName);
      }

      if (!workload.files.weights_file.empty()) {
        workload.files.weights_file.insert(
            0, dataDir + weightDataDir + encLayerName);
      }

      if (workload.files.bias_file == "mobilebert_attention_mask") {
        workload.files.bias_file.insert(0, dataDir + inputDataDir);
      } else {
        workload.files.bias_file.insert(0,
                                        dataDir + weightDataDir + encLayerName);
      }

      workload.files.outputs_file.insert(
          0, dataDir + outputDataDir + encLayerName);
      workload.files.residual_file.insert(
          0, dataDir + residualDataDir + encLayerName);
      workload.files.weight_grad_file.insert(
          0, dataDir + gradientDataDir + encLayerName);
      workload.files.bias_grad_file.insert(
          0, dataDir + gradientDataDir + encLayerName);

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
                                              ? workload.params.INPUT_OFFSET
                                              : workload.params.WEIGHT_OFFSET;
        } else {
          workload.params.OUTPUT_OFFSET = STACK_SIZE + 2 * INTERMEDIATE_SIZE;
        }

        workload.params.BIAS_OFFSET = STACK_SIZE + 3 * INTERMEDIATE_SIZE;
        workload.params.RESIDUAL_OFFSET = STACK_SIZE + 4 * INTERMEDIATE_SIZE;
        workload.params.WEIGHT_GRADIENT_OFFSET =
            STACK_SIZE + 5 * INTERMEDIATE_SIZE;
        workload.params.BIAS_GRADIENT_OFFSET =
            STACK_SIZE + 6 * INTERMEDIATE_SIZE;
      }

      workload.memoryMap = {SRAM, (workload.params.WEIGHT ? RRAM : SRAM), RRAM,
                            SRAM, SRAM};

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
    const std::vector<std::string>& layers) const {
  // Expand layer vector to include intermediate layers, if necessary
  std::vector<std::string> layersInRange;
  if (layers.size() == 1) {  // Single layer
    if (layers.front() == "all" && task == "inference") {
      return getInferenceWorkloads();
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

std::vector<Workload> MobileBERT::getAllWorkloads() const {
  return getWorkloads(order);
}

std::vector<Workload> MobileBERT::getInferenceWorkloads() const {
  std::vector<Workload> inferenceWorkloads;
  auto encoderOrder = std::vector<std::string>(inferenceOrder.begin(),
                                               inferenceOrder.end() - 1);

  int inputOffset;
  int weightOffset;
  for (int layer = 0; layer < 24; layer++) {
    std::vector<Workload> workloads = getWorkloads(encoderOrder, layer, true);

    inputOffset = ACTIVATION_OFFSET + layer * ENCODER_ACTIVATION_SIZE;
    weightOffset = layer * ENCODER_WEIGHT_SIZE;

    for (auto workload : workloads) {
      workload.params.INPUT_OFFSET += inputOffset;
      workload.params.WEIGHT_OFFSET +=
          workload.params.WEIGHT ? weightOffset : inputOffset;
      workload.params.OUTPUT_OFFSET += inputOffset;
      workload.params.BIAS_OFFSET += weightOffset;
      workload.params.RESIDUAL_OFFSET += inputOffset;

      if (workload.files.bias_file.find("mobilebert_attention_mask") !=
          std::string::npos) {
        workload.params.BIAS_OFFSET = STACK_SIZE;
      }

      inferenceWorkloads.push_back(workload);
    }
  }

  std::vector<std::string> workloadInRange{"classifier"};
  Workload classifier = getWorkloads(workloadInRange, 23, true).front();
  classifier.params.INPUT_OFFSET += inputOffset;
  classifier.params.WEIGHT_OFFSET +=
      classifier.params.WEIGHT ? weightOffset : inputOffset;
  classifier.params.OUTPUT_OFFSET += inputOffset;
  classifier.params.BIAS_OFFSET += weightOffset;
  classifier.params.RESIDUAL_OFFSET += inputOffset;

  inferenceWorkloads.push_back(classifier);

  // auto workloads = std::vector<Workload>(inferenceWorkloads.begin(),
  //                                        inferenceWorkloads.begin() + 200);
  // return workloads;
  return inferenceWorkloads;
}

std::vector<Workload> MobileBERT::getBackpropWorkloads() const {
  std::vector<Workload> backpropWorkloads;
  return backpropWorkloads;
}
