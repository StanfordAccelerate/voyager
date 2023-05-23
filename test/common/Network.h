#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

class Network {
 public:
  Network(const std::string modelName, const std::string dataDir)
      : modelName(modelName), dataDir(dataDir) {
    if (modelName.find("O1") != std::string::npos) {
      opt = O1;
    } else if (modelName.find("O2") != std::string::npos) {
      opt = O2;
    } else {
      opt = O0;
    }

    // Make "codegen"-matching case insensitive
    std::string& modelNameLower = const_cast<std::string&>(modelName);
    std::transform(modelNameLower.begin(), modelNameLower.end(),
                   modelNameLower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    codegen = modelNameLower.find("codegen") != std::string::npos;
  };
  virtual ~Network(void){};

  virtual std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string>&) = 0;

  virtual std::vector<Workload> getAllWorkloads() = 0;

  const std::string getName() { return modelName; }
  const std::string getDataDir() { return dataDir; }

  enum Optimization {
    O0,  // no optimizations are performed (no power gating)
    O1,  // minimal optimization (power gating for capacity)
    O2   // full optimization (power gating for bandwidth)
  };

 protected:
  const std::string modelName;
  const std::string dataDir;
  Optimization opt;
  bool codegen;
};