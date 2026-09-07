#include "test/soc/jtag/JtagSimChecker.h"

#include <systemc.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "test/common/GraphUtils.h"
#include "test/common/Utils.h"
#include "test/soc/jtag/JtagPreload.h"

JtagSimChecker::JtagSimChecker(const std::string& dump_path)
    : Simulation(), dump_path_(dump_path) {}

ArrayMemory* JtagSimChecker::make_memory(const std::string& sim,
                                         const std::vector<uint64_t>& sizes) {
  if (sim != "accelerator") return Simulation::make_memory(sim, sizes);

  // The DUT wrote the scratchpad itself, so report it as covered.
  auto* memory = new JtagMemory(sizes, /*sram_covered=*/true);
  const uint64_t sram_size = sizes[SRAM_PARTITION];

  // Places `length` bytes of `path` at scratchpad offset `offset`.
  const auto load = [&](const std::string& path, uint64_t offset) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      throw std::runtime_error("cannot open scratchpad dump " + path);
    }
    const uint64_t dump_size = static_cast<uint64_t>(f.tellg());
    f.seekg(0);
    if (offset >= sram_size) {
      throw std::runtime_error(path + " lies beyond the scratchpad");
    }
    const uint64_t length = std::min(dump_size, sram_size - offset);
    f.read(memory->get_memory(SRAM_PARTITION) + offset,
           static_cast<std::streamsize>(length));
    std::cout << "Loaded " << length << " of " << dump_size << " bytes of "
              << path << " at scratchpad offset " << offset << std::endl;
  };

  // With a dump list (the emitter's <layer>_scratchpad_dump.txt), GDB wrote
  // one file per region as <dump>.<i>; otherwise one file starting at
  // SRAM_BASE + SOC_MEM_OFFSET.
  const char* list_path = std::getenv("SCRATCHPAD_DUMP_LIST");
  if (list_path != nullptr && *list_path != '\0') {
    std::ifstream list(list_path);
    if (!list.is_open()) {
      throw std::runtime_error(std::string("cannot open dump list ") +
                               list_path);
    }
    std::string phys_text;
    uint64_t length = 0;
    for (int i = 0; list >> phys_text >> length; i++) {
      const uint64_t phys = std::stoull(phys_text, nullptr, 0);
      if (phys < kScratchpadBase) {
        throw std::runtime_error("dump region below the scratchpad: " +
                                 phys_text);
      }
      load(dump_path_ + "." + std::to_string(i), phys - kScratchpadBase);
    }
  } else {
    load(dump_path_, getenv_int("SOC_MEM_OFFSET", 0));
  }
  return memory;
}

int JtagSimChecker::grade() {
  load_data();
  run_gold();

  ArrayMemory* memory = this->memory("accelerator");
  const std::vector<Step> steps = record_schedule(model, selection, memory);
  // What the testbench did to DRAM before the first dispatch (a DRAM
  // output's zero-fill, say) is part of gold's write mask, so it has to be
  // part of this memory's too -- without touching the scratchpad, which now
  // holds what the DUT produced.
  apply_host_steps(steps, JtagPhase::kDramPrologue, memory);
  apply_host_steps(steps, JtagPhase::kStores, memory);

  return check_outputs();
}

int sc_main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: jtag_sim_checker <scratchpad.bin>\n"
              << "\n"
              << "Environment (the RTL simulation's, plus the GDB dump's):\n"
              << "  NETWORK, TESTS, PROJECT_ROOT, CODEGEN_DIR, DATATYPE,\n"
              << "  MAX_TILES, SOC_MEM_OFFSET (dump start within the SRAM)\n";
    return 1;
  }
  // The SoC flow always grades the DUT against the gold walk.
  setenv("SIMS", "gold,accelerator", /*overwrite=*/0);

  JtagSimChecker checker(argv[1]);
  return checker.grade() == 0 ? 0 : 1;
}
