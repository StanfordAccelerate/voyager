#pragma once
#define NO_SYSC

// clang-format off
#include "src/datatypes/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/ArrayMemory.h"
#include "test/common/MemoryInterface.h"
#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

class DataLoader {
 public:
  DataLoader(MemoryInterface*, bool);

  MemoryInterface* memory_interface;
  bool is_dut;  // special memory addressing for DUT (ex. replication)

  void load_tensor(const codegen::Tensor& tensor, std::string data_dir,
                   bool replication = false);
  void load_inputs(const std::vector<Operation>, std::string data_dir);
  void load_outputs(const codegen::Operation param, std::string data_dir);

  void copy_tile(const std::string& dtype, const std::vector<int>& full_shape,
                 const std::vector<int>& tiled_shape,
                 const std::vector<int>& tile_index, int src_partition,
                 uint64_t src_address, bool src_contiguous, int dst_partition,
                 uint64_t dst_address, bool dst_contiguous);
  void load_scratchpad(const codegen::Operation& param, const int tile_index,
                       const int offset = 0);
  void store_scratchpad(const codegen::Operation& param, const int tile_index,
                        const int offset = 0);

  float* read_tensor_from_file(const std::string& filename, int size);
};
