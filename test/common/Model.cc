#include "test/common/Model.h"

#include <google/protobuf/text_format.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "spdlog/spdlog.h"
#include "test/common/Utils.h"

using google::protobuf::TextFormat;

namespace {

std::string env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    throw std::runtime_error(std::string("Environment variable not set: ") +
                             name);
  }
  return std::string(value);
}

using Visitor = std::function<void(const voyager::Operation&)>;

// Visits an Operation and everything nested inside it, descending into loop
// bodies, loop conditions and both arms of every conditional. Allocations are
// model-scoped but are *declared* wherever the alloc happens to sit, which is
// often several regions deep, so a flat scan of model.ops() sees almost none of
// them.
void walk_op(const voyager::Operation& op, const Visitor& visit);

void walk_all(const google::protobuf::RepeatedPtrField<voyager::Operation>& ops,
              const Visitor& visit) {
  for (const auto& op : ops) walk_op(op, visit);
}

void walk_op(const voyager::Operation& op, const Visitor& visit) {
  visit(op);
  switch (op.op_type_case()) {
    case voyager::Operation::kLoop:
      if (op.loop().has_for_loop()) {
        walk_all(op.loop().for_loop().body().ops(), visit);
      } else if (op.loop().has_while_loop()) {
        walk_all(op.loop().while_loop().condition().ops(), visit);
        walk_all(op.loop().while_loop().body().ops(), visit);
      }
      break;
    case voyager::Operation::kCond:
      walk_all(op.cond().true_region().ops(), visit);
      walk_all(op.cond().false_region().ops(), visit);
      break;
    case voyager::Operation::kAsync:
      walk_all(op.async().body().ops(), visit);
      break;
    default:
      break;
  }
}

}  // namespace

Model::Model(const std::string& model_name) {
  const std::string dir = env("PROJECT_ROOT") + "/" + env("CODEGEN_DIR") +
                          "/networks/" + model_name + "/" + env("DATATYPE");
  data_dir = dir + "/tensor_files";

  const std::string filename = dir + "/model.txt";
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("Model does not exist: " + filename);
  }

  std::ifstream input(filename);
  std::stringstream buffer;
  buffer << input.rdbuf();
  if (!TextFormat::ParseFromString(buffer.str(), &model_)) {
    throw std::runtime_error("Failed to parse " + filename);
  }

  collect_boxes();
  load_layers(dir);
}

void Model::load_layers(const std::string& dir) {
  // Absent for whole-graph flows that never select by name; required to
  // resolve TESTS.
  const std::string filename = dir + "/layers.txt";
  if (!std::filesystem::exists(filename)) return;

  std::ifstream input(filename);
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string name, start, end, group, tiles;
    if (!(fields >> name)) continue;
    if (!(fields >> start >> end >> group >> tiles)) {
      throw std::runtime_error(
          filename +
          " is not the five-column layer table; regenerate the network with "
          "the current compiler.");
    }
    layers_[name] = {start, end};
  }
}

void Model::collect_boxes() {
  const auto declare = [&](const voyager::TensorBox& box) {
    const auto existing = boxes_.find(box.node());
    if (existing != boxes_.end()) {
      // slice_tensor is declared twice on purpose: once as a model output and
      // once as the box the slice op owns. Anything else is a real conflict.
      if (existing->second->SerializeAsString() != box.SerializeAsString()) {
        throw std::runtime_error("Conflicting declarations of TensorBox " +
                                 box.node());
      }
      return;
    }
    // A box with no memory names a value an earlier PrimOp of a fusion
    // produced: it never leaves the datapath and owns no storage.
    if (!box.has_memory()) return;
    boxes_[box.node()] = &box;
  };

  for (const auto& box : model_.inputs()) declare(box);
  for (const auto& box : model_.parameters()) declare(box);
  for (const auto& box : model_.outputs()) declare(box);
  walk_all(model_.ops(), [&](const voyager::Operation& op) {
    for (const auto& output : op.outputs()) {
      if (output.has_tensor_box()) declare(output.tensor_box());
    }
  });
}

const voyager::TensorBox& Model::box(const std::string& node) const {
  const auto found = boxes_.find(node);
  if (found == boxes_.end()) {
    throw std::runtime_error("No TensorBox named " + node);
  }
  return *found->second;
}

Model::Selection Model::selected_ops(
    const std::vector<std::string>& names) const {
  Selection selection;
  if (names.empty()) {
    // The whole graph: every outer tile loop is the run's own work, so the
    // bound applies to all of it.
    for (const auto& op : model_.ops()) {
      selection.ops.push_back(&op);
      selection.bounded.insert(&op);
    }
    return selection;
  }

  std::map<std::string, int> top_index;
  for (int i = 0; i < model_.ops_size(); i++) {
    top_index[model_.ops(i).name()] = i;
  }

  const auto extent_range = [&](const LayerExtent& extent) {
    const auto start = top_index.find(extent.start);
    const auto end = top_index.find(extent.end);
    if (start == top_index.end() || end == top_index.end()) {
      throw std::runtime_error("layers.txt extent [" + extent.start + ", " +
                               extent.end + "] names ops the model lacks.");
    }
    return std::pair<int, int>(start->second, end->second);
  };

  // A compiler-emitted drain has constants specialized for the complete loop
  // (for example, the final output-tile index). If MAX_TILES cuts the loop
  // short, that drain flushes a scratch tile the run never filled into the
  // real final-output slot -- uninitialized memory, at the tail of the output.
  // A sufficiently long partial loop already retires complete tiles through
  // its in-loop store, so stop the selection at the loop in that case.
  //
  // The bound only bites when it is actually below the trip count. A loop that
  // runs to completion drains legitimately, and stopping short of its drain
  // would instead leave the final tile -- for a single-output-tile layer, the
  // only tile -- never stored at all.
  const int64_t max_tiles = getenv_int("MAX_TILES", 0);

  // Bounds that are not compile-time constants cannot be evaluated here -- no
  // scalar environment exists yet -- so treat those as truncated, which is the
  // conservative choice and what the selection did before.
  const auto truncates = [&](const voyager::Operation& op) {
    if (max_tiles <= 0 || !op.has_loop()) return false;

    // A while loop has no trip count to compare the bound against, and the
    // interpreter stops it at MAX_TILES whatever its condition says, so it
    // always truncates.
    if (!op.loop().has_for_loop()) return true;

    const voyager::ForLoop& loop = op.loop().for_loop();
    if (!loop.start().has_int_value() || !loop.end().has_int_value() ||
        !loop.step().has_int_value()) {
      return true;
    }

    // The interpreter applies the bound only to a positive step, so anything
    // else runs to completion and drains legitimately.
    const int64_t step = loop.step().int_value();
    if (step <= 0) return false;

    const int64_t trips =
        (loop.end().int_value() - loop.start().int_value() + step - 1) / step;
    return max_tiles < trips;
  };

  for (const auto& name : names) {
    int start_index = -1;
    int end_index = -1;

    const auto layer = layers_.find(name);
    if (layer != layers_.end()) {
      const std::pair<int, int> range = extent_range(layer->second);
      start_index = range.first;
      end_index = range.second;
    } else {
      int found_index = -1;
      for (int i = 0; i < model_.ops_size() && found_index < 0; i++) {
        walk_op(model_.ops(i), [&](const voyager::Operation& op) {
          if (op.name() == name) found_index = i;
        });
      }
      if (found_index < 0) {
        throw std::runtime_error("No operation named " + name);
      }
      if (layers_.empty()) {
        throw std::runtime_error(
            "Selecting TESTS=" + name +
            " needs the layer extents in layers.txt; regenerate the network "
            "with the current compiler.");
      }
      // An extent that covers the op selects its whole layer; a host-side op
      // between two layers belongs to neither and is selected alone.
      start_index = end_index = found_index;
      for (const auto& [display, extent] : layers_) {
        const std::pair<int, int> range = extent_range(extent);
        if (range.first <= found_index && found_index <= range.second) {
          start_index = range.first;
          end_index = range.second;
          break;
        }
      }
    }

    for (int i = start_index; i <= end_index; i++) {
      const voyager::Operation& op = model_.ops(i);
      if (!contains(selection.ops, &op)) selection.ops.push_back(&op);
      if (truncates(op)) {
        selection.bounded.insert(&op);
        break;
      }
    }
  }
  return selection;
}

std::vector<uint64_t> Model::memory_sizes() const {
  uint64_t dram = 0;
  uint64_t sram = 0;
  uint64_t reference = 0;

  for (const auto& [node, box] : boxes_) {
    if (is_semaphore(*box) || is_constant(*box)) continue;

    const Tensor tensor = to_tensor(*box);
    const uint64_t end =
        tensor.address +
        banks_of(*box) *
            std::max<uint64_t>(bank_stride_of(*box), get_num_bytes(tensor));
    if (is_dram(*box)) {
      dram = std::max(dram, end);
      reference = std::max(reference, get_num_bytes(tensor));
    } else {
      sram = std::max(sram, end);
    }
  }

  // A little headroom: sub-byte writes round up to a whole byte at the tail.
  return {dram + 1024, sram + 1024, reference + 1024};
}

std::vector<const voyager::TensorBox*> Model::resident_inputs() const {
  std::vector<const voyager::TensorBox*> resident;
  for (const auto& box : model_.inputs()) resident.push_back(&box);
  for (const auto& box : model_.parameters()) {
    if (!is_constant(box)) resident.push_back(&box);
  }
  return resident;
}

void Model::live_sets(const std::vector<const voyager::Operation*>& ops,
                      std::vector<const voyager::TensorBox*>* live_in,
                      std::vector<const voyager::TensorBox*>* live_out,
                      bool include_scratchpad) const {
  std::set<std::string> read;
  std::set<std::string> written;
  // A scratchpad buffer a DMA writes into is a staged input or weight tile,
  // not a compute result, so it is never gradable.
  std::set<std::string> copy_dests;
  // A CSR's values and column indices are allocated for the worst-case
  // nonzero count, and only the entries the row pointers span mean anything.
  // Past them the buffer holds whatever the last tile that reached that far
  // left, which the two simulators need not agree on and no consumer reads --
  // an SpMM is bounded by the indptr. The row pointers themselves are defined
  // end to end, so they stay gradable.
  std::set<std::string> csr_payloads;

  const auto visit = [&](const voyager::Operation& op) {
    for (const auto* prim : get_prim_ops(op)) {
      const bool is_copy = prim->target() == "voyager::async_copy";
      for (const auto& [key, arg] : prim->kwargs()) {
        if (!arg.has_tensor_box()) continue;
        const std::string& node = arg.tensor_box().box().node();
        if (!has_box(node)) continue;  // a fusion intermediate
        if (is_copy && key == "dst") {
          copy_dests.insert(node);
          written.insert(node);
        } else if (prim->target() != "voyager::async_wait") {
          if (!written.count(node)) read.insert(node);
        }
      }
    }
    for (const auto& output : op.outputs()) {
      if (output.has_destination()) {
        written.insert(output.destination().box().node());
      } else if (output.has_tensor_box()) {
        written.insert(output.tensor_box().node());
      }
    }

    // set_quantize_mx_params reads the same three in this order: values,
    // column indices, row pointers.
    for (const auto* prim : get_prim_ops(op)) {
      if (strip_namespace(prim->target()) != "quantize_mx_outlier") continue;
      for (int i = 0; i < 2 && i < op.outputs_size(); i++) {
        const auto& output = op.outputs(i);
        if (output.has_destination()) {
          csr_payloads.insert(output.destination().box().node());
        } else if (output.has_tensor_box()) {
          csr_payloads.insert(output.tensor_box().node());
        }
      }
    }
  };

  for (const auto* op : ops) walk_op(*op, visit);

  if (live_in != nullptr) {
    for (const auto& node : read) {
      const voyager::TensorBox& b = box(node);
      if (is_dram(b) && !is_constant(b)) live_in->push_back(&b);
    }
  }
  if (live_out != nullptr) {
    for (const auto& node : written) {
      const voyager::TensorBox& b = box(node);
      // zeros declares the semaphore it posts as an output; never grade those.
      if (is_semaphore(b)) continue;

      if (csr_payloads.count(node) != 0) continue;

      if (is_dram(b)) {
        live_out->push_back(&b);
      } else if (include_scratchpad && partition_of(b) == SRAM_PARTITION &&
                 copy_dests.count(node) == 0) {
        live_out->push_back(&b);
      }
    }
  }
}

std::vector<std::string> Model::semaphore_nodes() const {
  std::vector<std::string> nodes;
  for (const auto& [node, box] : boxes_) {
    if (is_semaphore(*box)) nodes.push_back(node);
  }
  return nodes;
}

void Model::print_summary() const {
  int dram = 0, sram = 0, semaphores = 0, constants = 0, banked = 0;
  for (const auto& [node, box] : boxes_) {
    if (is_semaphore(*box))
      semaphores++;
    else if (partition_of(*box) == SRAM_PARTITION)
      sram++;
    else
      dram++;
    if (is_constant(*box)) constants++;
    if (banks_of(*box) > 1) banked++;
  }

  const auto sizes = memory_sizes();
  spdlog::info("Model: {} top-level ops, {} TensorBoxes\n", model_.ops_size(),
               boxes_.size());
  spdlog::info("> DRAM boxes: {} ({} file-backed constants)\n", dram,
               constants);
  spdlog::info("> Scratchpad boxes: {} ({} banked)\n", sram, banked);
  spdlog::info("> Semaphores: {}\n", semaphores);
  spdlog::info("> DRAM: {} B, SRAM: {} B\n", sizes[0], sizes[1]);
}
