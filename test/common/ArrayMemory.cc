#include "test/common/ArrayMemory.h"

ArrayMemory::ArrayMemory(std::vector<long long> sizes) : MemoryInterface() {
  memories.reserve(sizes.size());
  try {
    for (const auto size : sizes) {
      char* memory = new char[size];
      std::fill(memory, memory + size, 0);
      memories.push_back(memory);
    }
  } catch (const std::bad_alloc& e) {
    // Clean up any allocated memory if an exception is thrown
    for (char* memory : memories) {
      delete[] memory;
    }
    memories.clear();
    throw std::runtime_error("Memory allocation failed: " +
                             std::string(e.what()));
  }
}

ArrayMemory::~ArrayMemory() {
  for (char* memory : memories) {
    delete[] memory;
  }
}

char* ArrayMemory::get_memory(int partition) {
  if (partition < 0) {
    partition += memories.size();
  }
  if (partition < 0 or partition >= memories.size()) {
    throw std::runtime_error("Invalid memory partition: " +
                             std::to_string(partition));
  }
  return memories[partition];
}

/**
 * Retrieves the reference output tensor from the given accelerator parameter.
 * The reference output tensor is stored at the last partition at address of
 * 0.
 */
std::vector<std::any> ArrayMemory::get_reference_outputs(
    const codegen::Operation& param) {
  const auto tensors = get_op_outputs(param);
  std::vector<std::any> outputs;

  uint64_t address = 0;

  for (const auto& tensor : tensors) {
    codegen::Tensor tensor_copy;
    tensor_copy.CopyFrom(tensor);
    auto memory = tensor_copy.mutable_memory();
    memory->set_partition(-1);
    memory->set_address(address);

    outputs.push_back(read_tensor(tensor_copy));
    address += get_size(tensor);
  }

  return outputs;
}

void ArrayMemory::read_bytes_from_memory(const long long address,
                                         const int partition,
                                         const int num_bytes, char* bytes) {
  auto memory = get_memory(partition);
  memcpy(bytes, memory + address, num_bytes);
}

void ArrayMemory::write_bytes_to_memory(const long long address,
                                        const int partition,
                                        const int num_bytes,
                                        const char* bytes) {
  auto memory = get_memory(partition);
  memcpy(memory + address, bytes, num_bytes);
}
