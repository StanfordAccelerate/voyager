#include "test/common/ArrayMemory.h"

#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"
#include "test/common/Utils.h"

struct XorShift128Plus {
  uint64_t s[2];

  uint64_t next() {
    uint64_t x = s[0];
    uint64_t y = s[1];
    s[0] = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    s[1] = x;
    return x + y;
  }
};

void fill_random_fast(char* memory, uint64_t size) {
  std::random_device rd;
  XorShift128Plus rng = {rd(), rd()};

  uint64_t* ptr64 = reinterpret_cast<uint64_t*>(memory);
  uint64_t count64 = size / 8;

  for (uint64_t i = 0; i < count64; ++i) {
    ptr64[i] = rng.next();
  }

  uint64_t remaining = size % 8;
  if (remaining) {
    uint64_t tmp = rng.next();
    std::memcpy(memory + count64 * 8, &tmp, remaining);
  }
}

ArrayMemory::ArrayMemory(std::vector<uint64_t> sizes) : MemoryInterface() {
  memories.reserve(sizes.size());
  try {
    for (const auto size : sizes) {
      char* memory = new char[size];
#ifdef ZERO_INIT
      std::memset(memory, 0, size);
#else
      fill_random_fast(memory, size);
#endif
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
