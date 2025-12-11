#pragma once
#define NO_SYSC

#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"
#include "test/common/ArrayMemory.h"
#include "test/common/MemoryInterface.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"
#include "test/compiler/proto/param.pb.h"

struct ExtractedTensor {
  std::string key;         // the kwarg key
  codegen::Tensor tensor;  // main tensor
  bool needs_replication;  // whether this tensor is replicated
  std::string target;      // HLO operation
};

class DataLoader {
 public:
  DataLoader(MemoryInterface*, bool);

  MemoryInterface* memory;
  bool is_dut;  // special memory addressing for DUT (ex. replication)

  void load_tensor(const codegen::Tensor& tensor, std::string data_dir,
                   bool replication = false);
  void load_inputs(const std::vector<Operation>, std::string data_dir);
  void load_outputs(const codegen::Operation param, std::string data_dir);

  std::map<std::string, std::any> get_args(const codegen::Operation& param);
  std::vector<std::any> get_outputs(const codegen::Operation& param);
  std::vector<std::any> get_reference_outputs(const codegen::Operation& param);

  void copy_tile(const std::string& dtype, const std::vector<int>& full_shape,
                 const std::vector<int>& tiled_shape,
                 const std::vector<int>& tile_strides,
                 const std::vector<int>& tile_index, int src_partition,
                 uint64_t src_address, bool src_contiguous, int dst_partition,
                 uint64_t dst_address, bool dst_contiguous,
                 bool replication = false);
  void load_scratchpad(const codegen::Operation& param, const int tile_index,
                       const int offset = 0);
  void store_scratchpad(const codegen::Operation& param, const int tile_index,
                        const int offset = 0);

  std::vector<ExtractedTensor> extract_tensors_from_op(
      const codegen::Operation& param);
  float* read_tensor_from_file(const std::string& filename, int size);
};
