#include "JtagSimChecker.h"

#include <fstream>
#include <iostream>
#include <vector>

#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/common/Utils.h"
#include "test/common/operations/Common.h"
#include "test/toolchain/MapOperation.h"

// Stub for run_accelerator — this standalone binary does not run the
// accelerator
void run_accelerator(std::vector<Operation> params, DataLoader* dataloader) {
  // intentionally empty
}

JtagSimChecker::JtagSimChecker(const std::string& scratchpad_bin_path)
    : Simulation() {
  sims = {"gold", "soc"};

  // Read scratchpad dump from binary file
  std::ifstream f(scratchpad_bin_path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    std::cerr << "[JtagSimChecker] Failed to open: " << scratchpad_bin_path
              << std::endl;
    std::abort();
  }
  const size_t scratchpad_dump_size = static_cast<size_t>(f.tellg());
  f.seekg(0);

  // Create memory for the SoC simulation:
  //   partition 0 = DRAM (input tensors and output staging)
  //   partition 1 = scratchpad content from the GDB dump
  const uint64_t dram_size = network->get_max_dram_address();
  const uint64_t sram_size = 16 * 1024 * 1024;
  const uint64_t output_size = network->get_max_output_size();
  auto* soc_memory = new ArrayMemory({dram_size, sram_size, output_size});

  // Load scratchpad dump into partition 1
  const uint64_t soc_mem_offset = getenv_int("SOC_MEM_OFFSET", 0);
  f.read(soc_memory->get_memory(1) + soc_mem_offset,
         static_cast<std::streamsize>(scratchpad_dump_size));
  f.close();

  dataloaders["soc"] = new DataLoader(soc_memory, true);

  // Populate all dataloaders (gold + soc) with input tensor data from files,
  // and load expected reference outputs into the gold dataloader.
  load_data();

  // Copy output tensors from the scratchpad dump (partition 1) into DRAM
  // (partition 0) so that check_outputs() can read them via get_outputs().
  // Mirrors what SoCSimulation::tick() does via store_scratchpad().
  const int scratchpad_size = getenv_int("CACHE_SIZE", 8 * 1024 * 1024);
  const auto& last_op = operations.back();
  const int num_tiles = std::min(get_tile_count(last_op.param),
                                 2);  // JTAG sim supports max 2 tiles
  for (int t = 0; t < num_tiles; t++) {
    const int offset = (t % 2 == 0) ? 0 : scratchpad_size;
    dataloaders["soc"]->store_scratchpad(last_op.param, t, offset);
  }

  // Run the gold model to produce reference outputs (run_accelerator is a
  // no-op)
  run();

  // Compare gold model outputs vs. scratchpad dump outputs
  check_outputs();
}

int sc_main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: jtag_sim_checker <scratchpad.bin>\n"
              << "\n"
              << "Required environment variables (same as RTL simulation):\n"
              << "  NETWORK, TESTS, PROJECT_ROOT, CODEGEN_DIR, DATATYPE\n"
              << "  CACHE_SIZE (default: 8388608)\n"
              << "  OUT_DIR    (default: ./test_outputs/)\n";
    return 1;
  }
  JtagSimChecker sim(argv[1]);
  return 0;
}
