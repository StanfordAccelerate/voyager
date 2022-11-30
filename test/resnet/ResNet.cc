#include "test/resnet/ResNet.h"

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
#endif

#include <algorithm>
#include <cassert>
#include <iostream>

#include "test/resnet/params.h"

ResNet::ResNet(const std::string& dataDir) {
  this->order = ::order;
  this->paramsMap = ::paramsMap;
  this->filesMap = ::filesMap;

  if (dataDir.empty()) {
    // TODO(fpedd): Clean this up
    std::string rawDataDir = "./models/resnet/binary_data/";
    this->dataDir = (*std::filesystem::begin(
                         std::filesystem::directory_iterator(rawDataDir)))
                        .path()
                        .string() +
                    '/';

  } else {
    this->dataDir = dataDir;
  }

  // prepend dataDir to all files
  for (auto& it : filesMap) {
    it.second.inputs_file.insert(0, this->dataDir);
    it.second.weights_file.insert(0, this->dataDir);
    it.second.bias_file.insert(0, this->dataDir);
    it.second.outputs_file.insert(0, this->dataDir);
    it.second.residual_file.insert(0, this->dataDir);
    // it.second.weight_grad_file.insert(0, this->dataDir);
    // it.second.bias_grad_file.insert(0, this->dataDir);
  }
}

std::vector<Workload> ResNet::getWorkloads(
    const std::vector<std::string>& layers) const {
  std::vector<Workload> workloads;

  for (const std::string& layer : layers) {
    Workload workload;
    workload.name = layer;
    workload.params = paramsMap.at(layer);
    workload.files = filesMap.at(layer);
    workload.memoryMap = {SRAM, RRAM, RRAM, SRAM, SRAM};

    workloads.push_back(workload);
  }

  return workloads;
}

/*
 * Returns a vector of workloads to run.
 * Layers specifies either a single layer or range of layers.
 */
std::vector<Workload> ResNet::getWorkloadsInRange(
    const std::vector<std::string>& layers) const {
  // Expand layer vector to include intermediate layers, if necessary
  std::vector<std::string> layersInRange;
  if (layers.size() == 1) {  // Single layer
    layersInRange.push_back(layers.front());

  } else {  // Range of layers
    auto firstLayer = std::find(order.begin(), order.end(), layers.at(0));
    auto lastLayer = std::find(order.begin(), order.end(), layers.at(1));

    layersInRange = std::vector<std::string>(firstLayer, lastLayer + 1);
  }

  assert(layersInRange.size() > 0 && "Layer list is empty.");

  return getWorkloads(layersInRange);
}

std::vector<Workload> ResNet::getAllWorkloads() const {
  return getWorkloads(order);
}
