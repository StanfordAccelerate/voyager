#include "test/resnet/ResNet.h"

#include <algorithm>

#include "test/resnet/params.h"
#if __has_include("test/resnet/paramsCodeGen.h")
#include "test/resnet/paramsCodeGen.h"
#else
const std::map<std::string, SimplifiedParams> paramsCodeGen;
const std::map<std::string, Files> filesCodeGen;
const std::vector<std::string> orderCodeGen;
#endif

ResNet::ResNet(const std::string modelName, const std::string dataDir)
    : Network(modelName, dataDir) {
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
  } else {
    this->order = ::order;
    this->params = ::params;
    this->files = ::files;
  }

  // Prepend dataDir to all files
  for (auto& it : files) {
    it.second.inputs_file.insert(0, this->dataDir);
    it.second.weights_file.insert(0, this->dataDir);
    it.second.bias_file.insert(0, this->dataDir);
    it.second.outputs_file.insert(0, this->dataDir);
    it.second.residual_file.insert(0, this->dataDir);
  }
}

std::vector<Workload> ResNet::getWorkloads(
    const std::vector<std::string>& layers) const {
  std::vector<Workload> workloads;
  // Make "codgen"-matching case insensitive
  std::string& modelNameLower = const_cast<std::string&>(this->modelName);
  std::transform(modelNameLower.begin(), modelNameLower.end(),
                 modelNameLower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Hacky way to check if we are using codegen
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

    // Otherwise, use handwritten models
  } else {
    for (const std::string& layer : layers) {
      Workload workload;
      workload.name = layer;
      workload.params = params.at(layer);
      workload.files = files.at(layer);
      workload.memoryMap = {SRAM, (workload.params.WEIGHT ? RRAM : SRAM), RRAM,
                            SRAM, SRAM};

      // Handwritten model offsets don't account for stack, thus offset here
      workload.params.INPUT_OFFSET += STACK_SIZE;
      workload.params.OUTPUT_OFFSET += STACK_SIZE;
      workload.params.RESIDUAL_OFFSET += STACK_SIZE;

      workloads.push_back(workload);
    }
  }
  return workloads;
}

/*
 * Returns a vector of workloads to run.
 * Layers specifies either a single layer or range of layers.
 */
std::vector<Workload> ResNet::getWorkloadsInRange(
    const std::vector<std::string>& layers) {
  // Expand layer vector to include intermediate layers, if necessary
  std::vector<std::string> layersInRange;
  if (layers.size() == 1) {  // Single layer
    layersInRange.push_back(layers.front());
  } else {  // Range of layers
    auto firstLayer = std::find(order.begin(), order.end(), layers.at(0));
    auto lastLayer = std::find(order.begin(), order.end(), layers.at(1));
    layersInRange = std::vector<std::string>(firstLayer, lastLayer + 1);
  }

  if (layersInRange.empty()) {
    throw std::runtime_error("Layer list is empty.");
  }

  return getWorkloads(layersInRange);
}

std::vector<Workload> ResNet::getAllWorkloads() { return getWorkloads(order); }
