#include "test/common/Checker.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

#include "spdlog/spdlog.h"
#include "test/common/Utils.h"

namespace {

const std::map<std::string, std::string> SIM_LABELS = {
    {"pytorch", "PyTorch"},
    {"gold", "Gold Model"},
    {"accelerator", "Accelerator"},
};

// Which elements of a buffer the run actually produced. Everything, unless the
// run was deliberately cut short.
std::vector<uint8_t> valid_mask(const Tensor& tensor, ArrayMemory* memory) {
  const int64_t size = get_size(tensor);
  if (!memory->tracking()) return std::vector<uint8_t>(size, 1);

  const size_t width = get_width(tensor);

  std::vector<uint8_t> valid(size, 0);
  for (int64_t i = 0; i < size; i++) {
    // An element counts as produced only if every byte it spans was written.
    const uint64_t first = tensor.address + (i * width) / 8;
    const uint64_t last = tensor.address + ((i + 1) * width - 1) / 8;
    bool complete = true;
    for (uint64_t b = first; b <= last && complete; b++) {
      complete = memory->was_written(tensor.partition, b);
    }
    valid[i] = complete;
  }
  return valid;
}

}  // namespace

void load_tensor(const Tensor& tensor, const std::string& data_dir,
                 MemoryInterface* memory) {
  const int64_t size = get_size(tensor);
  const std::string filename = data_dir + "/" + tensor.node + ".bin";

  std::ifstream input(filename, std::ios::binary);
  if (!input.good()) {
    throw std::runtime_error("Tensor \"" + filename + "\" does not exist");
  }
  std::vector<float> values(size);
  input.read(reinterpret_cast<char*>(values.data()), size * sizeof(float));
  if (!input) {
    throw std::runtime_error("Tensor \"" + filename + "\" is too short");
  }

  spdlog::debug("Loading {} {} into partition {} at {}\n", tensor.node,
                shape_str(tensor.shape), tensor.partition, tensor.address);

  for (int64_t i = 0; i < size; i++) {
    memory->write_value(tensor.partition, tensor.address, i, tensor.dtype,
                        values[i]);
  }
}

int compare_memories(const std::vector<const voyager::TensorBox*>& expected,
                     const Model& model, ArrayMemory* first,
                     const std::string& first_name, ArrayMemory* second,
                     const std::string& second_name, const std::string& out_dir,
                     const std::string& prefix, float rtol, float atol,
                     bool require_complete) {
  std::filesystem::create_directories(out_dir);

  spdlog::info("{} vs. {} (rtol: {}, atol: {})\n", SIM_LABELS.at(first_name),
               SIM_LABELS.at(second_name), rtol, atol);

  long mismatched = 0;
  int64_t compared = 0;
  bool missing_coverage = false;
  for (const auto* box : expected) {
    const Tensor tensor = to_tensor(*box);

    // Every worker of a sweep shares OUT_DIR, and the bufferized IR names most
    // buffers alloc_<n>, so the network and layer have to be in the name.
    const std::string filename = out_dir + "/" + prefix + tensor.node + "." +
                                 first_name + "_vs_" + second_name + ".txt";

    // A run bounded by MAX_TILES only fills part of each buffer. Grade the part
    // it filled; the rest was never computed.
    std::vector<uint8_t> valid = valid_mask(tensor, first);
    if (require_complete &&
        std::find(valid.begin(), valid.end(), 0) != valid.end()) {
      spdlog::error("{} did not write all of {}\n", SIM_LABELS.at(first_name),
                    tensor.node);
      missing_coverage = true;
    }

    // When both simulators expose write masks, every element produced by the
    // trusted first simulator must also have been produced by the second.
    if (second->tracking()) {
      const std::vector<uint8_t> second_valid = valid_mask(tensor, second);
      for (size_t i = 0; i < valid.size(); i++) {
        if (valid[i] && !second_valid[i]) {
          missing_coverage = true;
          valid[i] = 0;
        }
      }
    }
    for (const auto mark : valid) compared += mark;

    const std::any first_values = first->read_tensor(tensor);
    const std::any second_values = second->read_tensor(tensor);
    long diff = compare_arrays<SUPPORTED_TYPES>(
        tensor, first_values, first_name, second_values, second_name, filename,
        valid, rtol, atol);

    spdlog::info("{} {}: mismatched {}\n", tensor.node, shape_str(tensor.shape),
                 diff);
    mismatched += diff;
  }

  int error_count = static_cast<int>(
      std::min<long>(mismatched, std::numeric_limits<int>::max()));
  if (missing_coverage) {
    spdlog::error("One or more expected output regions were not written.\n");
    error_count = std::max(error_count, 1);
  }

  // Comparing nothing is not passing. A software-pipelined loop stores a tile's
  // result some iterations after it is computed, so a run cut shorter than the
  // pipeline depth retires nothing at all.
  if (compared == 0) {
    spdlog::error(
        "The run produced none of the buffers being graded, so nothing was "
        "compared. If MAX_TILES is set, raise it past the pipeline depth: a "
        "pipelined tile loop only stores a tile some iterations after it is "
        "computed.\n");
    error_count = std::max(error_count, 1);
  }

  spdlog::info("Error count: {}\n", error_count);
  return error_count;
}
