#pragma once

#define NO_SYSC

#include <any>

#include "test/common/MemoryInterface.h"
#include "test/compiler/proto/param.pb.h"

class ArrayMemory : public MemoryInterface {
 public:
  ArrayMemory(std::vector<int>);
  ~ArrayMemory();

  std::vector<char*> memories;

  char* get_memory(const int partition);
  std::any get_tensor(const codegen::Tensor& tensor);
  std::vector<std::any> get_args(const codegen::AcceleratorParam& param);
  std::any get_output(const codegen::AcceleratorParam& param);
  std::any get_reference_output(const codegen::AcceleratorParam& param);

  template <typename T>
  void read_tensor_from_memory(const int address, const int partition,
                               const int size, T* tensor);

  void write_bytes_to_memory(const int address, const int partition,
                             const int size, const char* bytes) override;
};
