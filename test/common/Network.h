#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

class Network {
 public:
  Network(const std::string modelName, const std::string dataDir)
      : modelName(modelName), dataDir(dataDir) {
    if (modelName.find("O1") != std::string::npos) {
      this->opt = O1;
    } else if (modelName.find("O2") != std::string::npos) {
      this->opt = O2;
    } else {
      this->opt = O0;
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

  virtual std::vector<codegen::AcceleratorParam> get_params(
      const std::vector<std::string>& names) {
    return std::vector<codegen::AcceleratorParam>();
  };

  const std::string getName() { return modelName; }
  const std::string getDataDir() { return dataDir; }

  enum Optimization {
    O0,  // no optimizations are performed (no power gating)
    O1,  // minimal optimization (power gating for capacity)
    O2   // full optimization (power gating for bandwidth)
  };
  Optimization opt;

  const std::string optToString() {
    switch (opt) {
      case O0:
        return "O0";
      case O1:
        return "O1";
      case O2:
        return "O2";
      default:
        throw std::invalid_argument("Invalid option");
    }
  }

 protected:
  const std::string modelName;
  const std::string dataDir;
  bool codegen;
};