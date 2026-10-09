#pragma once

#include <google/protobuf/text_format.h>
#include <systemc.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test/compiler/proto/voyager_ir.pb.h"

// The allocator uses contiguous banks, not word-interleaved addresses.
// scratchpad_offset reserves low addresses; it is not a bank-address origin.
struct ScratchpadConfig {
  bool banked = false;
  uint64_t size = 0, banks = 0, word_bytes = 0, reserved = 0;
  double compiler_frequency_ghz = 0;
  std::string source = "legacy independent streams (no memory_config.txt)";

  void validate() const {
    if (!banked) return;
    if (!size || !banks || !word_bytes || size % banks ||
        (size / banks) % word_bytes || reserved >= size ||
        reserved % (size / banks)) {
      throw std::runtime_error("Invalid scratchpad bank geometry");
    }
  }

  static uint64_t number(const std::string& value) {
    size_t end = 0;
    if (value.empty() || value[0] == '-')
      throw std::runtime_error("Invalid scratchpad integer: " + value);
    const auto result = std::stoull(value, &end, 0);
    if (end != value.size())
      throw std::runtime_error("Invalid scratchpad integer: " + value);
    return result;
  }

  static ScratchpadConfig load(const std::string& path) {
    ScratchpadConfig config;
    std::ifstream input(path);
    const bool has_file = input.good();
    voyager::MemoryConfig fields;
    if (has_file) {
      std::stringstream buffer;
      buffer << input.rdbuf();
      if (!google::protobuf::TextFormat::ParseFromString(buffer.str(),
                                                         &fields) ||
          !fields.has_mode() ||
          !voyager::MemoryConfig_Mode_IsValid(fields.mode()))
        throw std::runtime_error("Malformed scratchpad config: " + path);
      config.source = path;
      config.compiler_frequency_ghz = fields.frequency_ghz();
    }

    std::string mode = fields.mode() == voyager::MemoryConfig::BANKED
                           ? "banked"
                           : "independent";
    if (const char* override = std::getenv("SCRATCHPAD_MODEL")) mode = override;
    if (mode != "banked" && mode != "independent")
      throw std::runtime_error(
          "SCRATCHPAD_MODEL must be banked or independent");
    config.banked = mode == "banked";
    const auto setting = [&](bool present, uint64_t value, const char* env,
                             bool required) {
      const char* override = std::getenv(env);
      if (present) {
        if (override && number(override) != value)
          throw std::runtime_error(std::string(env) + " disagrees with " +
                                   path);
        return value;
      }
      if (override) return number(override);
      if (required)
        throw std::runtime_error(std::string("Banked scratchpad requires ") +
                                 env + " or " + path);
      return uint64_t(0);
    };
    config.size =
        setting(fields.has_scratchpad_size(), fields.scratchpad_size(),
                "SCRATCHPAD_SIZE", config.banked);
    config.banks = setting(fields.has_num_banks(), fields.num_banks(),
                           "NUM_BANKS", config.banked);
    config.word_bytes = setting(fields.has_bank_width(), fields.bank_width(),
                                "BANK_WIDTH", config.banked);
    config.reserved =
        setting(fields.has_scratchpad_offset(), fields.scratchpad_offset(),
                "SCRATCHPAD_OFFSET", false);
    if (!has_file && config.banked) config.source = "explicit environment";
    config.validate();
    return config;
  }
};

// Testbench bandwidth model: one read OR write bank word per accelerator
// cycle per bank. Different banks operate concurrently. No extra SRAM access
// latency or DRAM/DMA timing is implied. Each calling stream has at most one
// outstanding word reservation; SystemC's cooperative execution orders ties.
class ScratchpadTiming {
 public:
  struct Counts {
    uint64_t reads = 0, writes = 0, wait_cycles = 0;
  };

  ScratchpadTiming(const ScratchpadConfig& config, sc_core::sc_time period)
      : config(config),
        counts(config.banked ? config.banks : 0),
        period_ticks(period.value()),
        next_cycle(config.banked ? config.banks : 0) {
    config.validate();
    if (!period_ticks)
      throw std::runtime_error("Invalid scratchpad clock period");
    if (config.banked) {
      if (const char* path = std::getenv("SCRATCHPAD_TRACE")) {
        trace.open(path);
        if (!trace) throw std::runtime_error("Cannot open SCRATCHPAD_TRACE");
        trace << "cycle bank write port address bytes\n";
      }
    }
  }

  // Do the actual byte access at its granted time, before another word can
  // reuse that bank. Splitting also handles unaligned beats, narrower banks,
  // partial final beats and accesses crossing a contiguous bank boundary.
  template <typename Access>
  void transfer(uint64_t address, uint64_t bytes, bool write,
                const std::string& port, Access access) {
    if (!config.banked) {
      access(address, uint64_t(0), bytes);
      return;
    }
    if (address > config.size || bytes > config.size - address)
      throw std::runtime_error(
          "Accelerator access exceeds configured scratchpad");
    const uint64_t bank_size = config.size / config.banks;
    uint64_t offset = 0;
    while (offset < bytes) {
      const uint64_t current = address + offset;
      const uint64_t bank = current / bank_size;
      const uint64_t chunk = std::min(
          bytes - offset, config.word_bytes - current % config.word_bytes);
      const uint64_t now = sc_core::sc_time_stamp().value() / period_ticks;
      const uint64_t slot = std::max(now, next_cycle[bank]);
      next_cycle[bank] = slot + 1;
      counts[bank].wait_cycles += slot - now;
      ports[port].wait_cycles += slot - now;
      const auto ready = sc_core::sc_time::from_value(slot * period_ticks);
      if (ready > sc_core::sc_time_stamp())
        sc_core::wait(ready - sc_core::sc_time_stamp());
      if (write) {
        ++counts[bank].writes;
        ++ports[port].writes;
      } else {
        ++counts[bank].reads;
        ++ports[port].reads;
      }
      if (trace)
        trace << slot << ' ' << bank << ' ' << write << ' ' << port << ' '
              << current << ' ' << chunk << '\n';
      access(current, offset, chunk);
      offset += chunk;
    }
  }

  void report(std::ostream& out) const {
    if (!config.banked) return;
    for (size_t bank = 0; bank < counts.size(); ++bank) {
      const auto& c = counts[bank];
      out << "ScratchpadBank: bank=" << bank << " read_words=" << c.reads
          << " write_words=" << c.writes << " wait_cycles=" << c.wait_cycles
          << '\n';
    }
    for (const auto& entry : ports) {
      const auto& c = entry.second;
      out << "ScratchpadPort: port=" << entry.first << " read_words=" << c.reads
          << " write_words=" << c.writes << " wait_cycles=" << c.wait_cycles
          << '\n';
    }
  }

  const ScratchpadConfig config;
  std::vector<Counts> counts;

 private:
  uint64_t period_ticks;
  std::vector<uint64_t> next_cycle;
  std::map<std::string, Counts> ports;
  std::ofstream trace;
};
