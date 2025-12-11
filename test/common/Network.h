#pragma once

#include "test/compiler/proto/param.pb.h"
#include "test/compiler/proto/tiling.pb.h"

class Operation {
 public:
  Operation() {}
  Operation(const std::string& name, const codegen::Operation& param,
            const voyager::Tiling& tiling)
      : name(name), param(param), tiling(tiling), has_valid_tiling(true) {}

  Operation(const std::string& name, const codegen::Operation& param)
      : name(name), param(param), has_valid_tiling(false) {}

  std::string name;
  codegen::Operation param;
  voyager::Tiling tiling;
  bool has_valid_tiling;
};

class Network {
 public:
  Network(std::string& model_name);

  std::vector<Operation> get_operations(bool filter_nop = true);
  std::vector<Operation> get_operations(const std::vector<std::string>& names,
                                        bool filter_nop = true);

  // Get the maximum DRAM address used across all tensors in the network
  uint64_t get_max_dram_address() const;

  // Get the maximum output size across all operations in the network
  uint64_t get_max_output_size() const;

  std::string project_root;
  std::vector<Operation> operations;
  codegen::Model model;
};
