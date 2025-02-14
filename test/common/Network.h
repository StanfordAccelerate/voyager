#pragma once

#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

class Network {
 public:
  Network(std::string& model);

  std::vector<codegen::Operation> get_params(bool filter_nop = true);
  std::vector<codegen::Operation> get_params(
      const std::vector<std::string>& names, bool filter_nop = true);

  std::string project_root;
  codegen::Model model;
};
