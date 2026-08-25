#include "SoCMemory.h"

#include <algorithm>
#include <iostream>

#include "test/common/GraphUtils.h"

SoCMemory::SoCMemory(const std::vector<uint64_t>& sizes)
    : ArrayMemory(sizes),
      cache_size(get_env_int("SCRATCHPAD_SIZE", 8 * 1024 * 1024)),
      num_banks(get_env_int("NUM_BANKS", 8)),
      num_subbanks(get_env_int("NUM_SUBBANKS", 1)),
      bank_width(get_env_int("BANK_WIDTH", 8)),
      macro_width(get_env_int("MACRO_WIDTH", 1)),
      // Rows per SRAM macro: a subbank deeper than one macro stacks them as
      // mem_1_<w>, mem_2_<w>, ... Default keeps everything in depth 0.
      macro_depth(get_env_int("MACRO_DEPTH", 1 << 30)),
      memory_suffix(get_env("MEMORY_SUFFIX", "ram")) {}

vpiHandle SoCMemory::get_handle(const uint64_t address) {
  uint32_t bank_size = cache_size / num_banks;
  uint32_t subbank_size = bank_size / num_subbanks;

  int bank_index = address / bank_size;
  int subbank_index = (address % bank_size) / subbank_size;
  int row_index = (address % subbank_size) / bank_width;
  int macro_index = (address % bank_width) / macro_width;
  int depth_index = row_index / macro_depth;
  int macro_row = row_index % macro_depth;

  std::string bank_prefix =
      bank_index ? "bank_" + std::to_string(bank_index) : "bank";
  std::string subbank_prefix =
      subbank_index ? "ram_" + std::to_string(subbank_index) : "ram";
  std::string macro_prefix =
      std::to_string(depth_index) + "_" + std::to_string(macro_index);

  std::string chiptop_path = "TestDriver.testHarness.chiptop0.";
  std::string ram_path = "system." + bank_prefix + "." + subbank_prefix +
                         ".mem.mem_ext.mem_" + macro_prefix;

#ifndef GL_SIM
  std::string path_to_ram = chiptop_path + ram_path + "." + memory_suffix;
#else
  // TODO: replace this with your own logic to determine the correct path in
  // the GL netlist
  std::replace(ram_path.begin(), ram_path.end(), '.', '_');
  std::string path_to_ram = chiptop_path + ram_path + "." + memory_suffix;
#endif

  vpiHandle ram_handle =
      vpi_handle_by_name((PLI_BYTE8*)path_to_ram.c_str(), NULL);
  if (!ram_handle) {
    std::cerr << "Could not find memory handle: " << path_to_ram << std::endl;
    std::abort();
  }

  vpiHandle elem = vpi_handle_by_index(ram_handle, macro_row);
  if (!elem) {
    std::cerr << "Could not find element handle: " << macro_row << std::endl;
    std::abort();
  }

  return elem;
}

void SoCMemory::write_byte(uint64_t addr, uint8_t byte) {
  vpiHandle elem = get_handle(addr);

  int byte_offset = addr % macro_width;
  int chunk_index = byte_offset / 4;
  int bit_in_chunk = (byte_offset % 4) * 8;

  s_vpi_value value;
  value.format = vpiVectorVal;
  vpi_get_value(elem, &value);

  uint32_t chunk = value.value.vector[chunk_index].aval;
  chunk &= ~(0xFFu << bit_in_chunk);
  chunk |= (static_cast<uint32_t>(byte) << bit_in_chunk);
  value.value.vector[chunk_index].aval = chunk;
  // Clear the 4-state control bits too, or the deposited byte stays X.
  value.value.vector[chunk_index].bval &= ~(0xFFu << bit_in_chunk);

  vpi_put_value(elem, &value, NULL, vpiNoDelay);
}

uint8_t SoCMemory::read_byte(uint64_t addr) {
  vpiHandle elem = get_handle(addr);

  int byte_offset = addr % macro_width;
  int chunk_index = byte_offset / 4;         // which 32-bit word in the row
  int bit_in_chunk = (byte_offset % 4) * 8;  // bit offset inside the chunk

  s_vpi_value value;
  value.format = vpiVectorVal;
  vpi_get_value(elem, &value);

  uint32_t chunk = value.value.vector[chunk_index].aval;
  return static_cast<uint8_t>((chunk >> bit_in_chunk) & 0xFFu);
}

bool SoCMemory::was_written(int partition, uint64_t address) const {
  if (partition == SRAM_PARTITION) return true;
  return ArrayMemory::was_written(partition, address);
}

bool SoCMemory::any_written(int partition, uint64_t address,
                            uint64_t num_bytes) const {
  if (partition == SRAM_PARTITION) return true;
  return ArrayMemory::any_written(partition, address, num_bytes);
}

void SoCMemory::write_bytes_to_memory(const long long address,
                                      const int partition, const int num_bytes,
                                      const char* bytes) {
  if (partition == SRAM_PARTITION) {
    for (int i = 0; i < num_bytes; i++) {
      write_byte(address + i, static_cast<uint8_t>(bytes[i]));
    }
    // Mirror into the host array and remember the range so verify_shadow()
    // can detect the DUT overwriting testbench-loaded data mid-run.
    ArrayMemory::write_bytes_to_memory(address, partition, num_bytes, bytes);
    shadow_ranges_.emplace_back(address, address + num_bytes);
    return;
  }
  ArrayMemory::write_bytes_to_memory(address, partition, num_bytes, bytes);
}

uint64_t SoCMemory::verify_shadow() {
  // Merge the write ranges, skipping re-written overlaps (last write wins in
  // both worlds, so the shadow is already coherent byte-wise).
  std::sort(shadow_ranges_.begin(), shadow_ranges_.end());
  std::vector<std::pair<uint64_t, uint64_t>> merged;
  for (const auto& r : shadow_ranges_) {
    if (!merged.empty() && r.first <= merged.back().second) {
      merged.back().second = std::max(merged.back().second, r.second);
    } else {
      merged.push_back(r);
    }
  }

  uint64_t stomped = 0;
  for (const auto& [begin, end] : merged) {
    uint64_t run_start = 0;
    uint64_t run_len = 0;
    for (uint64_t addr = begin; addr < end; addr++) {
      const uint8_t sram = read_byte(addr);
      char expect;
      ArrayMemory::read_bytes_from_memory(addr, SRAM_PARTITION, 1, &expect);
      if (sram != static_cast<uint8_t>(expect)) {
        if (run_len == 0) run_start = addr;
        run_len++;
        stomped++;
      } else if (run_len > 0) {
        std::cerr << "[SHADOW] stomped [" << run_start << ", "
                  << run_start + run_len << ") len=" << run_len << std::endl;
        run_len = 0;
      }
    }
    if (run_len > 0) {
      std::cerr << "[SHADOW] stomped [" << run_start << ", "
                << run_start + run_len << ") len=" << run_len << std::endl;
    }
  }
  std::cerr << "[SHADOW] total testbench-written bytes checked across "
            << merged.size() << " ranges; stomped bytes: " << stomped
            << std::endl;
  return stomped;
}

void SoCMemory::read_bytes_from_memory(const long long address,
                                       const int partition, const int num_bytes,
                                       char* bytes) {
  if (partition == SRAM_PARTITION) {
    for (int i = 0; i < num_bytes; i++) {
      bytes[i] = static_cast<char>(read_byte(address + i));
    }
    return;
  }
  ArrayMemory::read_bytes_from_memory(address, partition, num_bytes, bytes);
}
