#include "test/common/ArrayMemory.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
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

ArrayMemory::ArrayMemory(std::vector<uint64_t> sizes)
    : MemoryInterface(), sizes(sizes) {
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
  return memories[partition_index(partition)];
}

size_t ArrayMemory::partition_index(int partition) const {
  int64_t index = partition;
  if (index < 0) index += static_cast<int64_t>(memories.size());
  if (index < 0 || index >= static_cast<int64_t>(memories.size())) {
    throw std::runtime_error("Invalid memory partition: " +
                             std::to_string(partition));
  }
  return static_cast<size_t>(index);
}

size_t ArrayMemory::checked_range(int partition, long long address,
                                  int num_bytes) const {
  const size_t index = partition_index(partition);
  if (address < 0 || num_bytes < 0) {
    throw std::runtime_error("Invalid memory range: address " +
                             std::to_string(address) + ", size " +
                             std::to_string(num_bytes));
  }

  const uint64_t first = static_cast<uint64_t>(address);
  const uint64_t count = static_cast<uint64_t>(num_bytes);
  if (first > sizes[index] || count > sizes[index] - first) {
    throw std::runtime_error("Memory range exceeds partition " +
                             std::to_string(partition) + ": address " +
                             std::to_string(address) + ", size " +
                             std::to_string(num_bytes) + ", capacity " +
                             std::to_string(sizes[index]));
  }
  return index;
}

void ArrayMemory::read_bytes_from_memory(const long long address,
                                         const int partition,
                                         const int num_bytes, char* bytes) {
  const size_t index = checked_range(partition, address, num_bytes);
  memcpy(bytes, memories[index] + address, num_bytes);
}

void ArrayMemory::write_bytes_to_memory(const long long address,
                                        const int partition,
                                        const int num_bytes,
                                        const char* bytes) {
  const size_t index = checked_range(partition, address, num_bytes);
  memcpy(memories[index] + address, bytes, num_bytes);

  if (!written.empty() && num_bytes > 0) {
    auto& ranges = written.at(index);
    uint64_t first = static_cast<uint64_t>(address);
    const uint64_t count = static_cast<uint64_t>(num_bytes);
    uint64_t end = first + count;

    auto next = ranges.lower_bound(first);
    if (next != ranges.begin()) {
      auto previous = std::prev(next);
      if (previous->second >= first) {
        first = previous->first;
        end = std::max(end, previous->second);
        next = ranges.erase(previous);
      }
    }
    while (next != ranges.end() && next->first <= end) {
      end = std::max(end, next->second);
      next = ranges.erase(next);
    }
    ranges.emplace_hint(next, first, end);
  }
}

void ArrayMemory::track_writes() {
  written.clear();
  written.resize(memories.size());
}

bool ArrayMemory::was_written(int partition, uint64_t address) const {
  if (written.empty()) return false;
  const size_t index = partition_index(partition);
  if (address >= sizes[index]) return false;

  const auto& ranges = written[index];
  auto next = ranges.upper_bound(address);
  if (next == ranges.begin()) return false;
  return address < std::prev(next)->second;
}

bool ArrayMemory::any_written(int partition, uint64_t address,
                              uint64_t num_bytes) const {
  if (written.empty() || num_bytes == 0) return false;
  const size_t index = partition_index(partition);
  if (address >= sizes[index]) return false;

  const uint64_t count = std::min(num_bytes, sizes[index] - address);
  const uint64_t end = address + count;
  const auto& ranges = written[index];

  auto next = ranges.upper_bound(address);
  if (next != ranges.begin() && std::prev(next)->second > address) return true;
  return next != ranges.end() && next->first < end;
}
