#pragma once

#define NO_SYSC

#include "test/common/MemoryInterface.h"
#include "test/compiler/proto/param.pb.h"

template <class T>
class ArrayMemory : public MemoryInterface {
 public:
  ArrayMemory(std::vector<int>);
  ~ArrayMemory();

  std::vector<T*> memories;

  T* get_memory(const int partition);
  T* get_tensor(const codegen::Tensor& tensor);
  std::vector<T*> get_args(const codegen::AcceleratorParam& param);
  T* get_output(const codegen::AcceleratorParam& param);

 private:
  void write_to_memory(const int address, const float value,
                       const int partition, bool double_precision) override;
};
