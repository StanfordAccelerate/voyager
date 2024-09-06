#include "test/common/ArrayMemory.h"

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"

template <class T>
ArrayMemory<T>::ArrayMemory(std::vector<int> sizes) : MemoryInterface() {
  memories.reserve(sizes.size());
  try {
    for (int size : sizes) {
      T* memory = new T[size];
      std::fill(memory, memory + size, 0);
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
ArrayMemory<T>::~ArrayMemory() {
  for (T* memory : memories) {
    delete[] memory;
  }
}

template <class T>
T* ArrayMemory<T>::get_memory(int partition) {
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
T* ArrayMemory<T>::get_tensor(const codegen::Tensor& tensor) {
  int partition = tensor.memory().partition();
  return get_memory(partition) + tensor.memory().offset();
}

template <class T>
std::vector<T*> ArrayMemory<T>::get_args(
    const codegen::AcceleratorParam& param) {
  std::vector<T*> args;
  std::string output_node = "";
  if (param.has_matrix_param()) {
    const codegen::MatrixParam& matrix_param = param.matrix_param();
    args.push_back(get_tensor(matrix_param.input()));
    args.push_back(get_tensor(matrix_param.weight()));
    if (matrix_param.has_bias()) {
      args.push_back(get_tensor(matrix_param.bias()));
    } else {
      args.push_back(nullptr);
    }
    output_node = matrix_param.name();
  } else if (param.has_pooling_param()) {
    const codegen::PoolingParam& pooling_param = param.pooling_param();
    args.push_back(get_tensor(pooling_param.input()));
  } else if (param.has_reduce_param()) {
    const codegen::ReduceParam& reduce_param = param.reduce_param();
    args.push_back(get_tensor(reduce_param.input()));
  } else if (param.has_reshape_param()) {
    const codegen::ReshapeParam& reshape_param = param.reshape_param();
    args.push_back(get_tensor(reshape_param.input()));
  } else if (param.vector_params_size() > 0) {
    const auto vector_param = param.vector_params(0);
    args.push_back(get_tensor(vector_param.input()));
  }

  for (auto& vector_param : param.vector_params()) {
    if (vector_param.has_other()) {
      // Check whether input or other is the output of the last operation.
      const auto input = vector_param.input();
      const auto other = vector_param.other();
      const auto tensor_to_load = other.node() == output_node ? input : other;
      args.push_back(get_tensor(tensor_to_load));
    }
    output_node = vector_param.name();
  }

  // Add the output tensor to the list of arguments
  args.push_back(get_tensor(param.output()));
  return args;
}

/**
 * Retrieves the output tensor from the given accelerator parameter. The
 * output tensor is stored at the last partition with an offset of 0.
 */
template <class T>
T* ArrayMemory<T>::get_output(const codegen::AcceleratorParam& param) {
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  memory->set_partition(-1);
  memory->set_offset(0);
  return get_tensor(output_tensor);
}

template <>
void ArrayMemory<float>::write_to_memory(const int address, const float value,
                                         const int partition,
                                         bool double_precision) {
  auto memory = get_memory(partition);
  memory[address] = value;
}

template <>
void ArrayMemory<INPUT_DATATYPE>::write_to_memory(const int address,
                                                  const float value,
                                                  const int partition,
                                                  bool double_precision) {
  auto memory = get_memory(partition);
  if (double_precision) {
    ACCUM_DATATYPE posit16 = static_cast<ACCUM_DATATYPE>(value);
    posit16.storeAsLowerPrecision(&memory[address]);
  } else {
    memory[address] = static_cast<INPUT_DATATYPE>(value);
  }
}

template class ArrayMemory<float>;
template class ArrayMemory<INPUT_DATATYPE>;