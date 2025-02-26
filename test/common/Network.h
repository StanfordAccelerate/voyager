#pragma once

#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/compiler/proto/tiling.pb.h"

class Operation {
 public:
  Operation() {}
  Operation(const std::string& name, const codegen::Operator& param,
            const voyager::Tiling& tiling)
      : name(name), param(param), tiling(tiling), has_valid_tiling(true) {}

  Operation(const std::string& name, const codegen::Operator& param)
      : name(name), param(param), has_valid_tiling(false) {}

  std::string name;
  codegen::Operator param;
  voyager::Tiling tiling;
  bool has_valid_tiling;
};

class Network {
 public:
  Network(std::string& model_name);

  std::vector<Operation> get_operations(bool filter_nop = true);
  std::vector<Operation> get_operations(const std::vector<std::string>& names,
                                        bool filter_nop = true);

  std::string project_root;
  std::vector<Operation> operations;
  codegen::Model model;
};
