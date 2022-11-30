#pragma once

#include <map>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class ResNet : public Network {
 public:
  ResNet(const std::string&);
  ~ResNet(void){};

  std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string>&) const override;

  std::vector<Workload> getAllWorkloads() const override;

 private:
  std::vector<std::string> order;
  std::map<std::string, SimplifiedParams> paramsMap;
  std::map<std::string, Files> filesMap;
  std::map<std::string, MemoryMap> memoryMap;

  std::vector<Workload> getWorkloads(const std::vector<std::string>&) const;
};
