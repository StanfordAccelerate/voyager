#include "test/mobilebert/MobileBERT.h"

#include <algorithm>

#include "test/mobilebert/mobilebert_tiny2/backprop.h"
#include "test/mobilebert/mobilebert_tiny2/gradient.h"
#include "test/mobilebert/mobilebert_tiny2/inference.h"

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
    if (this->task == "inference") {
      order = inferenceOrder;
      paramsMapping = inferenceParamsMapping;
      params = inferenceParams;
      memOffsets = inferenceMemOffsets;
      files = inferenceTestFiles;
    } else if (this->task == "gradient") {
      // NOTE: order is not defined for gradient
      paramsMapping = gradientParamsMapping;
      params = gradientParams;
      memOffsets = gradientMemOffsets;
      files = gradientTestFiles;
    } else if (this->task == "backprop") {
      order = backpropOrder;
      paramsMapping = backpropParamsMapping;
      params = backpropParams;
      memOffsets = backpropMemOffsets;
      files = backpropTestFiles;
    } else {
      std::cerr << "Task must be one of: inference, gradient, backprop"
                << std::endl;
      std::abort();
    }
  }
}

std::vector<Workload> MobileBERT::getWorkloads(
    const std::vector<std::string>& layers) const {
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

      Workload workload;
      workload.name = layer;
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
        gradientDataDir = "weight_gradients/";

        if (!workload.params.WEIGHT || workload.params.ATTENTION_MASK) {
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
      }

      std::string encLayerName = "mobilebert_encoder_layer_0_";
      if (layer.find("classifier") != std::string::npos ||
          (task == "backprop" && layer == "output_bottleneck_LayerNorm")) {
        encLayerName = "";
      }

      if (!workload.files.inputs_file.empty()) {
        workload.files.inputs_file.insert(
            0, dataDir + inputDataDir + encLayerName);
      }
      if (workload.params.ATTENTION_MASK) {
        workload.files.weights_file.insert(0, dataDir + weightDataDir);
      } else if (!workload.files.weights_file.empty()) {
        workload.files.weights_file.insert(
            0, dataDir + weightDataDir + encLayerName);
      }

      workload.files.outputs_file.insert(
          0, dataDir + outputDataDir + encLayerName);
      workload.files.bias_file.insert(0,
                                      dataDir + weightDataDir + encLayerName);
      workload.files.residual_file.insert(
          0, dataDir + residualDataDir + encLayerName);

      // Fake memory offsets used for unit tests
      workload.params.INPUT_OFFSET = STACK_SIZE;
      if (workload.params.SOFTMAX) {  // Softmax gets no weight address
        workload.params.WEIGHT_OFFSET = -1;
      } else {
        workload.params.WEIGHT_OFFSET = STACK_SIZE + INTERMEDIATE_SIZE;
      }
      workload.params.OUTPUT_OFFSET = STACK_SIZE + 2 * INTERMEDIATE_SIZE;
      if (workload.params.BIAS) {
        workload.params.BIAS_OFFSET = STACK_SIZE + 3 * INTERMEDIATE_SIZE;
      } else {
        workload.params.BIAS_OFFSET = -1;
      }
      if (workload.params.RESIDUAL) {
        workload.params.RESIDUAL_OFFSET = STACK_SIZE + 4 * INTERMEDIATE_SIZE;
      } else {
        workload.params.RESIDUAL_OFFSET = -1;
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
