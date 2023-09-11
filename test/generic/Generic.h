#pragma once

#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class Generic : public Network {
 public:
  Generic();

  std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string> &) override;
  std::vector<Workload> getAllWorkloads() override;

 private:
  std::vector<std::string> order;
  std::map<std::string, SimplifiedParams> params;
  std::map<std::string, MemoryMap> memoryMap;

  std::vector<Workload> getWorkloads(const std::vector<std::string> &) const;
};