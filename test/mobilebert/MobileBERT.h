#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class MobileBERT : public Network {
 public:
  MobileBERT(const std::string &, const std::string &);
  ~MobileBERT(void){};

  std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string> &) const override;

  std::vector<Workload> getAllWorkloads() const override;

 private:
  std::string task;

  std::vector<std::string> order;
  std::map<std::string, std::string> paramsMapping;
  std::map<std::string, SimplifiedParams> params;
  std::map<std::string, MemoryOffsets> memOffsets;
  std::map<std::string, Files> testFiles;

  // inference
  std::vector<std::string> inferenceOrder;
  std::map<std::string, std::string> inferenceParamsMapping;
  std::map<std::string, SimplifiedParams> inferenceParams;
  std::map<std::string, MemoryOffsets> inferenceMemOffsets;
  std::map<std::string, Files> inferenceTestFiles;

  // gradient
  std::map<std::string, std::string> gradientParamsMapping;
  std::map<std::string, SimplifiedParams> gradientParams;
  std::map<std::string, MemoryOffsets> gradientMemOffsets;
  std::map<std::string, Files> gradientTestFiles;

  // backprop
  std::vector<std::string> backpropOrder;
  std::map<std::string, std::string> backpropParamsMapping;
  std::map<std::string, SimplifiedParams> backpropParams;
  std::map<std::string, MemoryOffsets> backpropMemOffsets;
  std::map<std::string, Files> backpropTestFiles;

  std::vector<Workload> getWorkloads(const std::vector<std::string> &) const;
};
