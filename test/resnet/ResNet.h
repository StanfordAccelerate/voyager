#pragma once

#include <map>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class ResNet : public Network {
 public:
  ResNet() : ResNet("resnet"){};
  ResNet(const std::string modelName)
      : ResNet(modelName, "./models/resnet/binary_data/n01440764_00000293/"){};
  ResNet(const std::string modelName, const std::string dataDir);
  ~ResNet(void){};

  std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string> &) override;

  std::vector<Workload> getAllWorkloads() override;

 private:
  std::vector<std::string> order;
  std::map<std::string, SimplifiedParams> params;
  std::map<std::string, Files> files;

  std::vector<Workload> getWorkloads(const std::vector<std::string> &) const;
};
