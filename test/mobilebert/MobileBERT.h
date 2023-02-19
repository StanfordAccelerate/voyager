#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

// TODO(fpedd): Once we have more than a single mobileBERT model, use the
// modelName parameter to select the correct one
// TODO(fpedd): It might make sense to ditch the task parameter and just use
// the modelName to select the correct task, ie mobilebert_inference,
// mobilebert_gradient, etc

class MobileBERT : public Network {
 public:
  MobileBERT() : MobileBERT("mobilebert", "inference"){};
  MobileBERT(const std::string modelName, const std::string task)
      : MobileBERT(modelName, task, "./data/mobilebert_tiny/datafile/step0/"){};
  MobileBERT(const std::string modelName, const std::string task,
             const std::string dataDir);
  ~MobileBERT(void){};

  std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string> &) override;
  std::vector<Workload> getAllWorkloads() override;

  void setTask(std::string);

 private:
  std::string task;

  std::vector<std::string> order;
  std::map<std::string, SimplifiedParams> params;
  std::map<std::string, Files> files;
  std::map<std::string, MemoryOffsets> memOffsets;

  std::vector<Workload> getWorkloads(const std::vector<std::string> &, bool,
                                     int) const;
  std::vector<Workload> getForwardWorkloads();
  std::vector<Workload> getBackwardWorkloads();
  std::vector<Workload> getWeightUpdateWorkloads();
};
