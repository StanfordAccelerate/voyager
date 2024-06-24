#ifndef PYTORCH_MEMORY_MODEL_IMPL_H
#define PYTORCH_MEMORY_MODEL_IMPL_H

#include "test/common/PytorchMemoryModel.h"

#define NO_SYSC

template <class T>
class PyTorchMemoryModelImpl : public PyTorchMemoryModel {
 public:
  PyTorchMemoryModelImpl(std::vector<int>, bool);
  ~PyTorchMemoryModelImpl();

  std::vector<T*> memories;

  T* get_memory(const int partition);
  T* get_tensor(const codegen::Tensor& tensor);
  std::vector<T*> get_args(const codegen::AcceleratorParam& param);
  T* get_output(const codegen::AcceleratorParam& param);

 private:
  void write_to_memory(const int address, const float value,
                       const int partition, bool double_precision) override;
};

template <class T>
inline PyTorchMemoryModelImpl<T>::PyTorchMemoryModelImpl(std::vector<int> sizes,
                                                         bool isDut)
    : PyTorchMemoryModel(isDut) {
  memories.reserve(sizes.size());
  try {
    for (int size : sizes) {
      T* memory = new T[size];
      memories.push_back(memory);
    }
  } catch (const std::bad_alloc& e) {
    // Clean up any allocated memory if an exception is thrown
    for (T* memory : memories) {
      delete[] memory;
    }
    memories.clear();
    throw std::runtime_error("Memory allocation failed: " +
                             std::string(e.what()));
  }
}

template <class T>
inline PyTorchMemoryModelImpl<T>::~PyTorchMemoryModelImpl() {
  for (T* memory : memories) {
    delete[] memory;
  }
}

template <class T>
inline T* PyTorchMemoryModelImpl<T>::get_memory(int partition) {
  if (partition < 0) {
    partition += memories.size();
  }
  if (partition < 0 or partition >= memories.size()) {
    throw std::runtime_error("Invalid memory partition: " +
                             std::to_string(partition));
  }
  return memories[partition];
}

template <class T>
inline T* PyTorchMemoryModelImpl<T>::get_tensor(const codegen::Tensor& tensor) {
  int partition = tensor.memory().partition();
  return get_memory(partition) + tensor.memory().offset();
}

template <class T>
inline std::vector<T*> PyTorchMemoryModelImpl<T>::get_args(
    const codegen::AcceleratorParam& param) {
  std::vector<T*> args;
  if (param.has_matrix_param()) {
    const codegen::MatrixParam& matrix_param = param.matrix_param();
    args.push_back(get_tensor(matrix_param.input()));
    args.push_back(get_tensor(matrix_param.weight()));
    if (matrix_param.has_bias()) {
      args.push_back(get_tensor(matrix_param.bias()));
    } else {
      args.push_back(nullptr);
    }
  }

  if (param.has_pooling_param()) {
    const codegen::PoolingParam& pooling_param = param.pooling_param();
    args.push_back(get_tensor(pooling_param.input()));
  }

  if (param.has_reduce_param()) {
    const codegen::ReduceParam& reduce_param = param.reduce_param();
    args.push_back(get_tensor(reduce_param.input()));
  }

  if (param.has_shape_param()) {
    const codegen::ShapeParam& shape_param = param.shape_param();
    args.push_back(get_tensor(shape_param.input()));
  }

  for (auto& vector_param : param.vector_params()) {
    if (vector_param.has_other()) {
      args.push_back(get_tensor(vector_param.other()));
    }
  }

  args.push_back(get_tensor(param.output()));
  return args;
}

/**
 * Retrieves the output tensor from the given accelerator parameter. The
 * output tensor is stored at the last partition with an offset of 0.
 */
template <class T>
inline T* PyTorchMemoryModelImpl<T>::get_output(
    const codegen::AcceleratorParam& param) {
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  memory->set_partition(-1);
  memory->set_offset(0);
  return get_tensor(output_tensor);
}

template <>
inline void PyTorchMemoryModelImpl<float>::write_to_memory(
    const int address, const float value, const int partition,
    bool double_precision) {
  auto memory = get_memory(partition);
  memory[address] = value;
}

template <>
inline void PyTorchMemoryModelImpl<INPUT_DATATYPE>::write_to_memory(
    const int address, const float value, const int partition,
    bool double_precision) {
  auto memory = get_memory(partition);
  if (double_precision) {
    ACCUM_DATATYPE posit16 = static_cast<ACCUM_DATATYPE>(value);
    posit16.storeAsLowerPrecision(&memory[address]);
  } else {
    memory[address] = static_cast<INPUT_DATATYPE>(value);
  }
}

#ifndef NO_UNIVERSAL
template <>
inline void PyTorchMemoryModelImpl<UniversalPosit>::write_to_memory(
    const int address, const float value, const int partition,
    bool double_precision) {
  auto memory = get_memory(partition);
  if (double_precision) {
    UniversalPositAccum posit16 = value;
    int bits = posit16.encoding();
    memory[address].setbits(bits & 0xFF);
    memory[address + 1].setbits((bits >> 8) & 0xFF);
  } else {
    memory[address] = value;
  }
}
#endif

#endif  // PYTORCH_MEMORY_MODEL_IMPL_H
