#define SC_INCLUDE_DYNAMIC_PROCESSES
#include <unistd.h>

#include <array>
#include <cassert>
#include <filesystem>
#include <functional>
#include <iostream>

#include "test/common/ScratchpadTiming.h"

struct ScratchpadTimingTb : sc_module {
  SC_CTOR(ScratchpadTimingTb) { SC_THREAD(run); }

  const sc_time period = sc_time(10, SC_NS);
  std::array<unsigned char, 256> memory{};

  ScratchpadConfig config(bool banked = true) {
    ScratchpadConfig c;
    c.banked = banked;
    c.size = 256;
    c.banks = 2;
    c.word_bytes = 16;
    return c;
  }

  void parallel(const std::vector<std::function<void()>>& clients) {
    sc_event done;
    unsigned remaining = clients.size();
    for (auto client : clients) {
      sc_spawn([&, client]() {
        client();
        --remaining;
        done.notify(SC_ZERO_TIME);
      });
    }
    while (remaining) wait(done);
  }

  void bandwidth(bool same_bank, bool write, bool banked) {
    wait(period * 3);
    ScratchpadTiming timing(config(banked), period);
    std::fill(memory.begin(), memory.end(), 0x5a);
    std::map<std::pair<uint64_t, uint64_t>, unsigned> grants;
    const auto start = sc_time_stamp();
    const auto stream = [&](unsigned base, bool stores, const char* port) {
      for (unsigned i = 0; i < 8; ++i) {
        timing.transfer(
            base, 16, stores, port,
            [&](uint64_t address, uint64_t offset, uint64_t count) {
              if (banked)
                assert(++grants[{sc_time_stamp().value() / period.value(),
                                 address / 128}] == 1);
              for (uint64_t j = 0; j < count; ++j) {
                if (stores)
                  memory[address + j] = i;
                else
                  assert(memory[address + j] == 0x5a);
              }
            });
      }
    };
    parallel({[&]() { stream(0, write, "weights"); },
              [&]() { stream(same_bank ? 64 : 128, false, "bias"); }});
    const auto cycles = (sc_time_stamp() - start) / period;
    assert(cycles == (banked ? (same_bank ? 15 : 7) : 0));
    if (write)
      for (int j = 0; j < 16; ++j) assert(memory[j] == 7);
    std::cout << "PASS " << (banked ? "banked" : "independent")
              << (same_bank ? " same-bank" : " separate-banks")
              << (write ? " read/write" : " read/read") << '\n';
  }

  void boundaries_and_backpressure() {
    wait(period * 3);
    ScratchpadTiming timing(config(), period);
    std::array<unsigned char, 40> result{};
    for (unsigned i = 0; i < memory.size(); ++i) memory[i] = i;
    timing.transfer(120, result.size(), false, "unaligned",
                    [&](uint64_t address, uint64_t offset, uint64_t count) {
                      std::copy_n(memory.begin() + address, count,
                                  result.begin() + offset);
                    });
    for (unsigned i = 0; i < result.size(); ++i) assert(result[i] == 120 + i);
    assert(timing.counts[0].reads == 1 && timing.counts[1].reads == 2);
    const auto start = sc_time_stamp();
    sc_time other_done;
    parallel({[&]() {
                wait(period * 20);  // no reservation while consumer is blocked
                timing.transfer(
                    0, 3, true, "small_store",
                    [&](uint64_t address, uint64_t, uint64_t count) {
                      for (uint64_t j = 0; j < count; ++j)
                        memory[address + j] = 0xee;
                    });
              },
              [&]() {
                for (unsigned i = 0; i < 8; ++i)
                  timing.transfer(32, 16, false, "unblocked",
                                  [](uint64_t, uint64_t, uint64_t) {});
                other_done = sc_time_stamp();
              }});
    assert((other_done - start) / period == 7);
    assert(memory[0] == 0xee && memory[2] == 0xee && memory[3] == 3);
    bool rejected = false;
    try {
      timing.transfer(250, 8, false, "invalid",
                      [](uint64_t, uint64_t, uint64_t) {});
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
    std::cout
        << "PASS bank/word boundaries, partial words, backpressure, bounds\n";
  }

  void run() {
    bandwidth(true, false, true);
    bandwidth(false, false, true);
    bandwidth(true, true, true);
    bandwidth(false, true, true);
    bandwidth(true, true, false);
    boundaries_and_backpressure();
    sc_stop();
  }
};

int sc_main(int argc, char** argv) {
  for (const char* key :
       {"SCRATCHPAD_MODEL", "SCRATCHPAD_SIZE", "NUM_BANKS", "BANK_WIDTH",
        "SCRATCHPAD_OFFSET", "SCRATCHPAD_TRACE"})
    unsetenv(key);
  if (argc == 2) {
    // Read the compiler smoke program's actual emitted protobuf text.
    const auto emitted = ScratchpadConfig::load(argv[1]);
    assert(emitted.banked && emitted.size == 12 * 1024 * 1024 &&
           emitted.banks == 8 && emitted.word_bytes == 64 &&
           emitted.reserved == 0 && emitted.compiler_frequency_ghz == 0.1);
  }
  const auto path = (std::filesystem::temp_directory_path() /
                     ("voyager-memory-config-" + std::to_string(getpid())))
                        .string();
  assert(!ScratchpadConfig::load(path).banked);
  setenv("SCRATCHPAD_MODEL", "banked", 1);
  bool rejected = false;
  try {
    ScratchpadConfig::load(path);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  unsetenv("SCRATCHPAD_MODEL");
  {
    std::ofstream f(path);
    f << "mode: BANKED\nscratchpad_size: 256\nnum_banks: 2\n"
         "bank_width: 16\nscratchpad_offset: 128\nfrequency_ghz: 0.1\n";
  }
  auto c = ScratchpadConfig::load(path);
  assert(c.banked && c.size == 256 && c.reserved == 128 &&
         c.compiler_frequency_ghz == 0.1);
  setenv("BANK_WIDTH", "32", 1);
  rejected = false;
  try {
    ScratchpadConfig::load(path);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  unsetenv("BANK_WIDTH");
  setenv("SCRATCHPAD_MODEL", "independent", 1);
  assert(!ScratchpadConfig::load(path).banked);
  unsetenv("SCRATCHPAD_MODEL");
  std::filesystem::remove(path);
  std::cout
      << "PASS config, missing geometry, conflicting override, legacy mode\n";
  ScratchpadTimingTb tb("tb");
  sc_start();
  return 0;
}
