#pragma once

#include "test/compiler/proto/param.pb.h"
#include "test/compiler/proto/tiling.pb.h"

class Operation {
 public:
  Operation() {}
  Operation(const std::string& name, const codegen::Operation& param,
            const voyager::Tiling& tiling)
      : name(name),
        param(param),
        tiling(tiling),
        has_valid_tiling(true),
        has_shrunk_tiling(false) {}

  Operation(const std::string& name, const codegen::Operation& param,
            const voyager::Tiling& tiling, int shrink_factor)
      : name(name),
        param(param),
        tiling(tiling),
        has_valid_tiling(true),
        has_shrunk_tiling(true),
        shrink_factor(shrink_factor) {}

  Operation(const std::string& name, const codegen::Operation& param)
      : name(name),
        param(param),
        has_valid_tiling(false),
        has_shrunk_tiling(false) {}

  std::string name;
  codegen::Operation param;
  voyager::Tiling tiling;
  bool has_valid_tiling;
  bool has_shrunk_tiling;
  int shrink_factor;
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
