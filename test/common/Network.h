#pragma once

#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

class Network {
 public:
  Network(const std::string modelName, const std::string dataDir)
      : modelName(modelName), dataDir(dataDir){};
  virtual ~Network(void){};

  virtual std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string> &) = 0;

  virtual std::vector<Workload> getAllWorkloads() = 0;

  const std::string getName() { return modelName; }
  const std::string getDataDir() { return dataDir; }

 protected:
  const std::string modelName;
  const std::string dataDir;
};