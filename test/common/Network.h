#pragma once

#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

class Network {
 public:
  Network(std::string& model);

  std::vector<codegen::AcceleratorParam> get_params();
  std::vector<codegen::AcceleratorParam> get_params(
      const std::vector<std::string>& names);

 private:
  std::string model;
  std::string project_root;
  codegen::ModelParams model_params;
};
