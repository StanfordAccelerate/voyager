#include "test/resnet/ResNet18.h"

#ifdef SOC_COSIM
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
#else
#include <filesystem>
#endif

#include <algorithm>
#include <iostream>

#include "test/resnet/params.h"

ResNet18::ResNet18() {
  order = {"conv1",
           "layer1_0_conv1",
           "layer1_0_conv2",
           "layer1_1_conv1",
           "layer1_1_conv2",
           "layer2_0_downsample",
           "layer2_0_conv1",
           "layer2_0_conv2",
           "layer2_1_conv1",
           "layer2_1_conv2",
           "layer3_0_downsample",
           "layer3_0_conv1",
           "layer3_0_conv2",
           "layer3_1_conv1",
           "layer3_1_conv2",
           "layer4_0_downsample",
           "layer4_0_conv1",
           "layer4_0_conv2",
           "layer4_1_conv1",
           "layer4_1_conv2",
           "fc"};

  paramsMap = {{"conv1", conv1_params},
               {"layer1_0_conv1", layer1_0_conv1_params},
               {"layer1_0_conv2", layer1_0_conv2_params},
               {"layer1_1_conv1", layer1_1_conv1_params},
               {"layer1_1_conv2", layer1_1_conv2_params},
               {"layer2_0_downsample", layer2_0_downsample_params},
               {"layer2_0_conv1", layer2_0_conv1_params},
               {"layer2_0_conv2", layer2_0_conv2_params},
               {"layer2_1_conv1", layer2_1_conv1_params},
               {"layer2_1_conv2", layer2_1_conv2_params},
               {"layer3_0_downsample", layer3_0_downsample_params},
               {"layer3_0_conv1", layer3_0_conv1_params},
               {"layer3_0_conv2", layer3_0_conv2_params},
               {"layer3_1_conv1", layer3_1_conv1_params},
               {"layer3_1_conv2", layer3_1_conv2_params},
               {"layer4_0_downsample", layer4_0_downsample_params},
               {"layer4_0_conv1", layer4_0_conv1_params},
               {"layer4_0_conv2", layer4_0_conv2_params},
               {"layer4_1_conv1", layer4_1_conv1_params},
               {"layer4_1_conv2", layer4_1_conv2_params},
               {"fc", fc_params}};

  filesMap = {
      {"conv1", {"conv1_input", "conv1_weight", "conv1_bias", "conv1_comp"}},
      {"layer1_0_conv1",
       {"layer1_0_conv1_input", "layer1_0_conv1_weight", "layer1_0_conv1_bias",
        "layer1_0_conv1_comp"}},
      {"layer1_0_conv2",
       {"layer1_0_conv2_input", "layer1_0_conv2_weight", "layer1_0_conv2_bias",
        "layer1_0_conv2_comp", "conv1_comp"}},
      {"layer1_1_conv1",
       {"layer1_1_conv1_input", "layer1_1_conv1_weight", "layer1_1_conv1_bias",
        "layer1_1_conv1_comp"}},
      {"layer1_1_conv2",
       {"layer1_1_conv2_input", "layer1_1_conv2_weight", "layer1_1_conv2_bias",
        "layer1_1_conv2_comp", "layer1_0_conv2_comp"}},
      {"layer2_0_downsample",
       {"layer2_0_downsample_input", "layer2_0_downsample_weight",
        "layer2_0_downsample_bias", "layer2_0_downsample_comp"}},
      {"layer2_0_conv1",
       {"layer2_0_conv1_input", "layer2_0_conv1_weight", "layer2_0_conv1_bias",
        "layer2_0_conv1_comp"}},
      {"layer2_0_conv2",
       {"layer2_0_conv2_input", "layer2_0_conv2_weight", "layer2_0_conv2_bias",
        "layer2_0_conv2_comp", "layer2_0_downsample_comp"}},
      {"layer2_1_conv1",
       {"layer2_1_conv1_input", "layer2_1_conv1_weight", "layer2_1_conv1_bias",
        "layer2_1_conv1_comp"}},
      {"layer2_1_conv2",
       {"layer2_1_conv2_input", "layer2_1_conv2_weight", "layer2_1_conv2_bias",
        "layer2_1_conv2_comp", "layer2_0_conv2_comp"}},
      {"layer3_0_downsample",
       {"layer3_0_downsample_input", "layer3_0_downsample_weight",
        "layer3_0_downsample_bias", "layer3_0_downsample_comp"}},
      {"layer3_0_conv1",
       {"layer3_0_conv1_input", "layer3_0_conv1_weight", "layer3_0_conv1_bias",
        "layer3_0_conv1_comp"}},
      {"layer3_0_conv2",
       {"layer3_0_conv2_input", "layer3_0_conv2_weight", "layer3_0_conv2_bias",
        "layer3_0_conv2_comp", "layer3_0_downsample_comp"}},
      {"layer3_1_conv1",
       {"layer3_1_conv1_input", "layer3_1_conv1_weight", "layer3_1_conv1_bias",
        "layer3_1_conv1_comp"}},
      {"layer3_1_conv2",
       {"layer3_1_conv2_input", "layer3_1_conv2_weight", "layer3_1_conv2_bias",
        "layer3_1_conv2_comp", "layer3_0_conv2_comp"}},
      {"layer4_0_downsample",
       {"layer4_0_downsample_input", "layer4_0_downsample_weight",
        "layer4_0_downsample_bias", "layer4_0_downsample_comp"}},
      {"layer4_0_conv1",
       {"layer4_0_conv1_input", "layer4_0_conv1_weight", "layer4_0_conv1_bias",
        "layer4_0_conv1_comp"}},
      {"layer4_0_conv2",
       {"layer4_0_conv2_input", "layer4_0_conv2_weight", "layer4_0_conv2_bias",
        "layer4_0_conv2_comp", "layer4_0_downsample_comp"}},
      {"layer4_1_conv1",
       {"layer4_1_conv1_input", "layer4_1_conv1_weight", "layer4_1_conv1_bias",
        "layer4_1_conv1_comp"}},
      {"layer4_1_conv2",
       {"layer4_1_conv2_input", "layer4_1_conv2_weight", "layer4_1_conv2_bias",
        "layer4_1_conv2_comp", "layer4_0_conv2_comp"}},
      {"fc", {"fc_input", "fc_weight", "fc_bias", "fc_comp"}}};

  const char* dataDirEnv = std::getenv("DATA_DIR");
  std::string dataDir;
  if (dataDirEnv == NULL) {
    std::string rawDataDir = "./models/resnet/binary_data/";
    dataDir = (*std::filesystem::begin(
                   std::filesystem::directory_iterator(rawDataDir)))
                  .path()
                  .string() +
              '/';

  } else {
    dataDir = std::string(dataDirEnv);
  }

  // prepend dataDir to all files
  for (auto& it : filesMap) {
    it.second.inputs_file.insert(0, dataDir);
    it.second.weights_file.insert(0, dataDir);
    it.second.bias_file.insert(0, dataDir);
    it.second.outputs_file.insert(0, dataDir);
    it.second.residual_file.insert(0, dataDir);
    it.second.weight_grad_file.insert(0, dataDir);
    it.second.bias_grad_file.insert(0, dataDir);
  }
}

/*
 * Returns a vector of workloads to run.
 * Layers specifies either a single layer or range of layers.
 */
std::vector<Workload> ResNet18::getWorkloads(
    const std::vector<std::string>& layers) const {
  std::vector<Workload> workloads;

  std::vector<std::string> layersInRange;
  if (layers.size() == 1) {  // Single layer
    layersInRange.push_back(layers.front());
  } else {  // Range of layers
    auto firstLayer = std::find(order.begin(), order.end(), layers.at(0));
    auto lastLayer = std::find(order.begin(), order.end(), layers.at(1));
    layersInRange = std::vector<std::string>(firstLayer, lastLayer + 1);
  }

  for (const std::string& layer : layersInRange) {
    Workload workload;
    workload.name = layer;
    workload.params = paramsMap.at(layer);
    workload.files = filesMap.at(layer);
    workload.memoryMap = {SRAM, RRAM, RRAM, SRAM, SRAM};

    workloads.push_back(workload);
  }

  return workloads;
}
