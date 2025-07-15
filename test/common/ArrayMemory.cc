#include "test/common/ArrayMemory.h"

// clang-format off
#include "src/datatypes/DataTypes.h"
// clang-format on

#include <fstream>

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

ArrayMemory::ArrayMemory(std::vector<uint64_t> sizes) : MemoryInterface() {
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
