#include "test/codegen/CodeGen.h"

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

#if __has_include("test/codegen/params_codegen.h")
#include "test/codegen/params_codegen.h"
#else
#pragma error \
    "Could not find generated params header. Make sure you ran Codegen (ZagZig)."
#endif

CodeGen::CodeGen(const std::string& dataDir) {
  this->order = ::order;
  this->paramsMap = ::paramsMap;
  this->filesMap = ::filesMap;

  if (dataDir.empty()) {
    this->tensorDir = "./test/codegen/gen_data";
    this->weightDir = "./test/codegen/gen_data";
  } else {
    // TODO(fpedd): This needs to be individually controlable at some point
    this->tensorDir = dataDir;
    this->weightDir = dataDir;
  }

  // prepend dataDir to all files
  for (auto& it : filesMap) {
    it.second.inputs_file.insert(0, this->tensorDir);
    it.second.weights_file.insert(0, this->weightDir);
    it.second.bias_file.insert(0, this->weightDir);
    it.second.outputs_file.insert(0, this->tensorDir);
    it.second.residual_file.insert(0, this->tensorDir);
    // it.second.weight_grad_file.insert(0, this->dataDir);
    // it.second.bias_grad_file.insert(0, this->dataDir);
  }
}

std::vector<Workload> CodeGen::getWorkloads(
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
std::vector<Workload> CodeGen::getWorkloadsInRange(
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

std::vector<Workload> CodeGen::getAllWorkloads() const {
  return getWorkloads(order);
}