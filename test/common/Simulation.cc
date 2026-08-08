#include "test/common/Simulation.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include "test/common/GoldModel.h"
#include "test/common/Interpreter.h"
#include "test/common/Utils.h"

namespace {

std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> parts;
  std::istringstream stream(text);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    if (!item.empty()) parts.push_back(item);
  }
  return parts;
}

// Did the run write any byte of this tensor? Used to keep only the buffers a
// (possibly short) run actually produced.
bool wrote_any(ArrayMemory* memory, const Tensor& tensor) {
  if (!memory->tracking()) return false;
  return memory->any_written(tensor.partition, tensor.address,
                             get_num_bytes(tensor));
}

}  // namespace

Simulation::Simulation()
    : model_name(getenv("NETWORK", "resnet18")), model(model_name) {
  spdlog::cfg::load_env_levels();
  spdlog::set_pattern("%v");

  out_dir = getenv("OUT_DIR", "./test_outputs/");

  sims = split(getenv("SIMS", "gold,pytorch"), ',');
  if (sims.size() != 2) {
    throw std::runtime_error("SIMS must name exactly two simulators.");
  }

  // Simulator comparisons must agree exactly; only a comparison against the
  // float reference tensors gets tolerances.
  const bool vs_reference = uses("pytorch");
  rtol = std::stof(getenv("RTOL", vs_reference ? "0.05" : "0"));
  atol = std::stof(getenv("ATOL", vs_reference ? "0.1" : "0"));

  model.print_summary();

  // TESTS names a layer; its layers.txt extent is the smallest runnable thing
  // and is self-contained (allocs, credit fills, tile loop, drain), so the
  // selection runs as-is.
  selection = model.selected_ops(split(getenv("TESTS", ""), ','));
  spdlog::info("Running {} top-level operation(s)\n", selection.ops.size());

  partial_run = getenv_int("MAX_TILES", 0) > 0;
  check_all_writes = uses("gold") && uses("accelerator");
  model.live_sets(selection.ops, &live_in, &live_out, check_all_writes);
}

Simulation::~Simulation() {
  for (const auto& [sim, array] : memories) delete array;
}

void Simulation::load_data() {
  const auto sizes = model.memory_sizes();
  spdlog::info("Model memory requirements:\n");
  spdlog::info("> DRAM: {} bytes ({} MB)\n", sizes[0],
               sizes[0] / (1024 * 1024));
  spdlog::info("> SRAM: {} bytes ({} MB)\n", sizes[1],
               sizes[1] / (1024 * 1024));

  for (const auto& sim : sims) {
    auto* array = new ArrayMemory(sizes);
    memories[sim] = array;

    // Loads the reference output tensors from disk so there's something to
    // compare against.
    if (sim == "pytorch") {
      for (const auto* box : live_out) {
        load_tensor(to_tensor(*box), model.data_dir, array);
      }
      continue;
    }

    for (const auto* box : model.resident_inputs()) {
      load_tensor(to_tensor(*box), model.data_dir, array);
    }
    for (const auto* box : live_in) {
      load_tensor(to_tensor(*box), model.data_dir, array);
    }

    // Arm tracking after the preload so the write mask reflects the run's own
    // writes, not the input/parameter setup.
    // A full gold/reference run must prove that every expected byte was
    // produced. Partial runs and gold/accelerator comparisons also use the
    // masks to compare only corresponding writes.
    if (sim == "gold" || partial_run || check_all_writes) {
      array->track_writes();
    }
  }

  spdlog::info("Data loaded successfully\n");
}

bool Simulation::uses(const std::string& sim) const {
  return contains(sims, sim);
}

ArrayMemory* Simulation::memory(const std::string& sim) const {
  const auto found = memories.find(sim);
  if (found == memories.end()) {
    throw std::runtime_error("SIMS does not name the simulator " + sim);
  }
  return found->second;
}

void Simulation::run_gold() {
  ArrayMemory* array = memory("gold");

  GoldBackend backend(array);
  Interpreter interpreter(model, array, &backend);
  interpreter.run(selection);

  const long clock_period_ns = getenv_int("CLOCK_PERIOD", 5);
  const std::string name = getenv("TESTS", model_name);

  if (interpreter.matrix_cycles() > 0) {
    spdlog::info("{}, matrix unit ideal runtime: {} ns\n", name,
                 interpreter.matrix_cycles() * clock_period_ns);
  }
  if (interpreter.vector_cycles() > 0) {
    spdlog::info("{}, vector unit ideal runtime: {} ns\n", name,
                 interpreter.vector_cycles() * clock_period_ns);
  }
}

int Simulation::check_outputs() {
  // Drop buffers the run did not retire at all. A short MAX_TILES walk can
  // legitimately leave later pipeline checkpoints untouched; buffers that did
  // retire remain element-masked below. Gold's write mask is the source of
  // truth for both partial reference checks and gold/accelerator comparisons.
  if (uses("gold") && (check_all_writes || partial_run)) {
    ArrayMemory* gold = memory("gold");
    live_out.erase(std::remove_if(live_out.begin(), live_out.end(),
                                  [&](const voyager::TensorBox* box) {
                                    return !wrote_any(gold, to_tensor(*box));
                                  }),
                   live_out.end());
  }

  if (live_out.empty()) {
    spdlog::error(
        "No output of the selection has a reference to compare to.\n");
    return 1;
  }

  // Compare the more-real simulator against the more-trusted one; when grading
  // all writes, gold comes first because it owns the write mask.
  std::vector<std::string> ordered;
  if (check_all_writes) {
    ordered = {"gold", "accelerator"};
  } else {
    for (const auto& sim : {"accelerator", "gold", "pytorch"}) {
      if (uses(sim)) ordered.push_back(sim);
    }
  }

  const std::string tests = getenv("TESTS", "");
  const std::string prefix =
      model_name + "." + (tests.empty() ? "all" : tests) + ".";

  const bool require_complete = !partial_run && uses("gold") && uses("pytorch");
  return compare_memories(live_out, model, memory(ordered[0]), ordered[0],
                          memory(ordered[1]), ordered[1], out_dir, prefix, rtol,
                          atol, require_complete);
}
