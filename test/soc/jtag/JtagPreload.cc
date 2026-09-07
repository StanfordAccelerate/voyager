#include "test/soc/jtag/JtagPreload.h"

#include <algorithm>
#include <any>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>

#include "test/common/GoldModel.h"
#include "test/common/GraphUtils.h"
#include "test/common/Interpreter.h"

namespace {

using Regions = std::vector<std::pair<uint64_t, uint64_t>>;

Regions merge(Regions regions) {
  std::sort(regions.begin(), regions.end());
  Regions merged;
  for (const auto& r : regions) {
    if (!merged.empty() && r.first <= merged.back().second) {
      merged.back().second = std::max(merged.back().second, r.second);
    } else {
      merged.push_back(r);
    }
  }
  return merged;
}

}  // namespace

// ===========================================================================
// JtagMemory
// ===========================================================================

JtagMemory::JtagMemory(const std::vector<uint64_t>& sizes, bool sram_covered)
    : ArrayMemory(sizes), sram_covered_(sram_covered) {}

void JtagMemory::write_bytes_to_memory(const long long address,
                                       const int partition, const int num_bytes,
                                       const char* bytes) {
  ArrayMemory::write_bytes_to_memory(address, partition, num_bytes, bytes);
  if (recording_writes_ && partition == SRAM_PARTITION && num_bytes > 0) {
    ranges_.emplace_back(static_cast<uint64_t>(address),
                         static_cast<uint64_t>(address) + num_bytes);
    if (recording_copy_) copy_ranges_.push_back(ranges_.back());
  }
}

void JtagMemory::read_bytes_from_memory(const long long address,
                                        const int partition,
                                        const int num_bytes, char* bytes) {
  ArrayMemory::read_bytes_from_memory(address, partition, num_bytes, bytes);
  if (recording_reads_ && partition == SRAM_PARTITION && num_bytes > 0) {
    add_sram_read(static_cast<uint64_t>(address), num_bytes);
  }
}

void JtagMemory::add_sram_read(uint64_t address, uint64_t num_bytes) {
  if (num_bytes > 0) read_ranges_.emplace_back(address, address + num_bytes);
}

Regions JtagMemory::sram_regions() const { return merge(ranges_); }

Regions JtagMemory::sram_read_regions() const { return merge(read_ranges_); }

Regions JtagMemory::sram_writes_from(size_t first) const {
  if (first >= ranges_.size()) return {};
  return Regions(ranges_.begin() + first, ranges_.end());
}

Regions JtagMemory::sram_copy_regions() const { return merge(copy_ranges_); }

bool JtagMemory::was_written(int partition, uint64_t address) const {
  if (sram_covered_ && partition == SRAM_PARTITION) return true;
  return ArrayMemory::was_written(partition, address);
}

bool JtagMemory::any_written(int partition, uint64_t address,
                             uint64_t num_bytes) const {
  if (sram_covered_ && partition == SRAM_PARTITION) return true;
  return ArrayMemory::any_written(partition, address, num_bytes);
}

// ===========================================================================
// Schedule application
// ===========================================================================

std::vector<Step> record_schedule(const Model& model,
                                  const Model::Selection& selection,
                                  ArrayMemory* memory) {
  std::vector<Step> steps;
  ScheduleRecorder recorder(&steps);
  Interpreter interpreter(model, memory, &recorder);
  interpreter.run(selection);
  return steps;
}

// The non-init steps of a schedule, one per line; a copy also shows its
// source and destination as resolved under the recorded environment.
void print_schedule(const std::string& layer, const std::vector<Step>& steps) {
  static const char* kKind[] = {"COPY", "ZERO", "HOST", "SREAD", "SOP",
                                "INIT", "WAIT", "POST", "DISP"};
  std::cout << "  schedule of " << layer << " (non-init steps):" << std::endl;
  for (size_t i = 0; i < steps.size(); i++) {
    const Step& s = steps[i];
    if (s.kind == Step::kInit) continue;
    std::cout << "    " << i << " " << kKind[s.kind];
    if (s.kind == Step::kWait || s.kind == Step::kPost) {
      std::cout << " " << s.sem_node << "[" << s.sem_slot << "] amt "
                << s.amount << (s.retire_post ? " RETIRE" : "");
    } else if (s.kind == Step::kDispatch) {
      std::cout << " " << s.op_name << (s.sync ? " SYNC" : "");
    } else if (s.kind == Step::kCopy) {
      std::cout << " " << s.prim->name();
      try {
        const Tensor src = resolve(*s.prim, "src", s.env);
        const Tensor dst = resolve(*s.prim, "dst", s.env);
        std::cout << " " << src.node << "@p" << src.partition << ":"
                  << src.address << " -> " << dst.node << "@p" << dst.partition
                  << ":" << dst.address << "+" << get_num_bytes(dst);
      } catch (const std::exception&) {
      }
    } else if (s.prim != nullptr) {
      std::cout << " " << s.prim->name();
    } else if (s.op != nullptr) {
      std::cout << " " << s.op->name();
    }
    std::cout << std::endl;
  }
}

Regions dispatch_operand_regions(const Step& dispatch) {
  Regions regions;
  if (dispatch.op == nullptr) return regions;
  for (const auto* prim : get_prim_ops(*dispatch.op)) {
    for (const auto& [key, arg] : prim->kwargs()) {
      if (!arg.has_tensor_box()) continue;
      // A ref without memory is a datapath intermediate; an IMMEDIATE one a
      // constant. Neither has an address.
      if (!arg.tensor_box().box().has_memory()) continue;
      try {
        const Tensor t = resolve(*prim, key, dispatch.env);
        if (t.partition == SRAM_PARTITION) {
          regions.emplace_back(t.address, t.address + get_num_bytes(t));
        }
      } catch (const std::exception&) {
        // Not an addressable operand (a semaphore, a register-level box).
      }
    }
  }
  return merge(regions);
}

Regions load_regions(const Step& load) {
  Regions regions;
  if (load.kind == Step::kCopy) {
    const Tensor dst = resolve(*load.prim, "dst", load.env);
    if (dst.partition == SRAM_PARTITION) {
      regions.emplace_back(dst.address, dst.address + get_num_bytes(dst));
    }
  } else if (load.kind == Step::kZero) {
    const auto& box = load.op->outputs(0).tensor_box();
    if (partition_of(box) == SRAM_PARTITION) {
      for (uint32_t bank = 0; bank < banks_of(box); bank++) {
        const Tensor t = to_tensor(box, bank);
        regions.emplace_back(t.address, t.address + get_num_bytes(t));
      }
    }
  }
  return regions;
}

void apply_host_steps(const std::vector<Step>& steps, JtagPhase phase,
                      ArrayMemory* memory,
                      const std::function<bool(size_t)>& keep) {
  // Values of the deferred scalar reads, overlaid on each step's recorded
  // env exactly as SoCSimulation::live_env does.
  std::map<std::string, int64_t> deferred;
  const auto live_env = [&](const Step& step) {
    ScalarEnv env = step.env;
    for (const auto& [name, value] : deferred) env.define(name, value);
    return env;
  };

  auto* tracking = dynamic_cast<JtagMemory*>(memory);

  bool dispatched = false;
  for (size_t index = 0; index < steps.size(); index++) {
    const Step& step = steps[index];
    const bool prologue =
        phase == JtagPhase::kLoads || phase == JtagPhase::kDramPrologue;
    if (step.kind == Step::kDispatch) {
      if (prologue) return;
      dispatched = true;
      continue;
    }
    if (!prologue && !dispatched) continue;
    if (tracking) tracking->set_recording_copy(step.kind == Step::kCopy);

    switch (step.kind) {
      case Step::kCopy: {
        if (phase == JtagPhase::kDramPrologue) {
          // The grade's memory holds only the read-back regions, so a copy
          // whose window depends on a scratchpad cell may not resolve here;
          // such a copy is a scratchpad load anyway, never a DRAM write.
          try {
            const Tensor dst = resolve(*step.prim, "dst", step.env);
            if (dst.partition == SRAM_PARTITION) continue;
            run_async_copy(*step.prim, step.env, memory);
          } catch (const std::exception& error) {
            std::cout << "  prologue copy " << step.prim->name()
                      << " skipped: " << error.what() << std::endl;
          }
          break;
        }
        if (phase != JtagPhase::kLoads) {
          const Tensor dst = resolve(*step.prim, "dst", live_env(step));
          const bool load = dst.partition == SRAM_PARTITION;
          if (load != (phase == JtagPhase::kLateLoads)) continue;
          if (load && keep && !keep(index)) continue;
        }
        if (std::getenv("DUMP_SCHEDULE") != nullptr) {
          const Tensor src = resolve(*step.prim, "src", live_env(step));
          const Tensor dst = resolve(*step.prim, "dst", live_env(step));
          std::cout << "  apply " << index << " COPY " << step.prim->name()
                    << " " << src.node << "@p" << src.partition << ":"
                    << src.address << " -> " << dst.node << "@p"
                    << dst.partition << ":" << dst.address << "+"
                    << get_num_bytes(dst) << std::endl;
        }
        run_async_copy(*step.prim, live_env(step), memory);
        break;
      }

      case Step::kZero: {
        if (phase == JtagPhase::kStores) continue;
        if (phase == JtagPhase::kLateLoads && keep && !keep(index)) continue;
        const auto& box = step.op->outputs(0).tensor_box();
        if (phase == JtagPhase::kDramPrologue &&
            partition_of(box) == SRAM_PARTITION) {
          continue;  // the DUT's own results live there now
        }
        if (step.prim->target() == "voyager::alloc") {
          for (uint32_t bank = 0; bank < banks_of(box); bank++) {
            zero_buffer(to_tensor(box, bank), 1, 0, memory);
          }
        } else {
          zero_buffer(to_tensor(box), banks_of(box), bank_stride_of(box),
                      memory);
        }
        break;
      }

      case Step::kHostOp:
        if (phase == JtagPhase::kLateLoads ||
            phase == JtagPhase::kDramPrologue) {
          continue;
        }
        // In the store phase the firmware has already run the bookkeeping
        // on the chip (EmitC emits it in every mode), and its results are
        // in the readback; running it again here would apply the
        // running-base accumulation twice.
        if (phase == JtagPhase::kStores) continue;
        run_host_operation(*step.op, live_env(step), memory);
        break;

      case Step::kScalarRead: {
        // Not in the DRAM prologue: the cells it would read are scratchpad
        // loads the grade's memory never holds (only the read-back regions).
        if (phase == JtagPhase::kDramPrologue) continue;
        // Also in the late-load phase: a sparse tile's copies take their
        // extents from CSR cells that earlier loads placed in the image, so
        // the read is exact here -- only a cell the DUT itself produces would
        // not be, and a load depending on one cannot be preloaded anyway.
        const Tensor input = resolve(*step.prim, "input", live_env(step));
        if (get_size(input) != 1) {
          throw std::runtime_error("Scalar operation " + step.op->name() +
                                   " reads more than one element.");
        }
        std::any data = memory->read_tensor(input);
        int64_t value = 0;
        if (auto* cell =
                std::any_cast<std::shared_ptr<DataTypes::int32[]>>(&data)) {
          value = static_cast<int64_t>((*cell)[0].int_val.to_int64());
        } else if (auto* cell =
                       std::any_cast<std::shared_ptr<DataTypes::int64[]>>(
                           &data)) {
          value = static_cast<int64_t>((*cell)[0].int_val.to_int64());
        } else {
          throw std::runtime_error("Scalar operation " + step.op->name() +
                                   " reads a tensor of non-index dtype " +
                                   input.dtype + ".");
        }
        deferred[step.op->outputs(0).name()] = value;
        break;
      }

      case Step::kScalarOp: {
        if (phase == JtagPhase::kDramPrologue) continue;
        const Scalar value = eval_scalar_prim(*step.prim, live_env(step));
        deferred[step.op->outputs(0).name()] = to_int(value);
        break;
      }

      case Step::kInit:
      case Step::kWait:
      case Step::kPost:
      case Step::kDispatch:
        break;
    }
  }
}

// ===========================================================================
// Image emission
// ===========================================================================

void write_scratchpad_data(const std::string& base_path,
                           const std::string& layer, const JtagMemory& memory) {
  const auto regions = memory.sram_regions();
  const char* sram = const_cast<JtagMemory&>(memory).get_memory(SRAM_PARTITION);

  std::ofstream c_out(base_path + layer + "_scratchpad_data.c");
  std::ofstream ld_out(base_path + layer + "_scratchpad_data.ld");
  c_out << "#include <stdint.h>\n\n";
  ld_out << "SECTIONS {\n";

  uint64_t total = 0;
  for (size_t r = 0; r < regions.size(); r++) {
    const uint64_t start = regions[r].first;
    const uint64_t len = regions[r].second - start;
    total += len;

    const std::string sym = layer + "_input_" + std::to_string(r);
    const std::string section = ".spad_" + sym;

    c_out << "__attribute__((section(\"" << section << "\"), used))\n";
    c_out << "static const uint8_t " << sym << "[] = {\n\t";
    for (uint64_t i = 0; i < len; i++) {
      c_out << "0x" << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<unsigned>(
                   static_cast<unsigned char>(sram[start + i]));
      if (i + 1 < len) c_out << (((i + 1) % 12 == 0) ? ",\n\t" : ", ");
    }
    c_out << "\n};\n\n";

    ld_out << "  " << section << " (0x" << std::hex << (kScratchpadBase + start)
           << std::dec << ") : { *(" << section << ") }\n";
  }
  ld_out << "}\n";

  std::cout << "  scratchpad image: " << regions.size() << " region(s), "
            << total << " bytes" << std::endl;
}

Regions readback_regions(const Model& model, const Model::Selection& selection,
                         const std::vector<Step>& steps,
                         const ArrayMemory& gold, JtagMemory* memory) {
  // The outputs check_outputs grades in place: with gold and accelerator both
  // named, live_sets includes the scratchpad results, and compare_memories
  // reads each as to_tensor(*box) -- but compares only the bytes gold wrote
  // (a MAX_TILES-bounded run fills part of a buffer), so gold's write mask
  // is the extent worth reading back.
  std::vector<const voyager::TensorBox*> live_in;
  std::vector<const voyager::TensorBox*> live_out;
  model.live_sets(selection.ops, &live_in, &live_out,
                  /*include_scratchpad=*/true);
  for (const auto* box : live_out) {
    if (partition_of(*box) != SRAM_PARTITION) continue;
    const Tensor tensor = to_tensor(*box);
    const uint64_t begin = tensor.address;
    const uint64_t end = begin + get_num_bytes(tensor);
    if (!gold.tracking()) {
      memory->add_sram_read(begin, end - begin);
      continue;
    }
    uint64_t run_start = 0;
    bool in_run = false;
    for (uint64_t addr = begin; addr <= end; addr++) {
      const bool written = addr < end && gold.was_written(SRAM_PARTITION, addr);
      if (written && !in_run) {
        run_start = addr;
        in_run = true;
      } else if (!written && in_run) {
        memory->add_sram_read(run_start, addr - run_start);
        in_run = false;
      }
    }
  }

  // The store-backs' sources at full extent. A sparse store moves only
  // `count` elements, and count is a value the DUT produces.
  bool dispatched = false;
  for (const Step& step : steps) {
    if (step.kind == Step::kDispatch) dispatched = true;
    if (!dispatched || step.kind != Step::kCopy) continue;
    const Tensor src = resolve(*step.prim, "src", step.env);
    if (src.partition == SRAM_PARTITION) {
      memory->add_sram_read(src.address, get_num_bytes(src));
    }
  }

  // Everything else the store phase reads (a host op's operands, a scalar
  // cell), found by dry-running it. The scratchpad holds no results yet, so
  // a data-dependent step may well fail here; what it read before failing
  // is still a real read, and the full-extent sources above cover the rest.
  memory->record_sram_reads();
  try {
    apply_host_steps(steps, JtagPhase::kStores, memory);
  } catch (const std::exception& error) {
    std::cout << "  (store-phase dry run stopped: " << error.what() << ")"
              << std::endl;
  }
  return memory->sram_read_regions();
}

void write_scratchpad_dump_list(const std::string& base_path,
                                const std::string& layer,
                                const Regions& regions) {
  std::ofstream out(base_path + layer + "_scratchpad_dump.txt");
  uint64_t total = 0;
  for (const auto& [start, end] : regions) {
    out << "0x" << std::hex << (kScratchpadBase + start) << std::dec << " "
        << (end - start) << "\n";
    total += end - start;
  }
  std::cout << "  scratchpad readback: " << regions.size() << " region(s), "
            << total << " bytes" << std::endl;
}

// ===========================================================================
// JtagPreload
// ===========================================================================

ArrayMemory* JtagPreload::make_memory(const std::string& sim,
                                      const std::vector<uint64_t>& sizes) {
  if (sim == "accelerator")
    return new JtagMemory(sizes, /*sram_covered=*/false);
  return Simulation::make_memory(sim, sizes);
}

void JtagPreload::emit(const std::string& base_path, const std::string& layer) {
  load_data();

  auto* memory = dynamic_cast<JtagMemory*>(this->memory("accelerator"));
  if (memory == nullptr) {
    throw std::runtime_error("SIMS must name the accelerator simulator.");
  }

  // Only what the testbench would stage counts, not the DRAM preload.
  memory->record_sram_writes();
  const std::vector<Step> steps = record_schedule(model, selection, memory);
  if (std::getenv("DUMP_SCHEDULE") != nullptr) print_schedule(layer, steps);
  apply_host_steps(steps, JtagPhase::kLoads, memory);

  // The loads the testbench would stage while earlier tiles compute. A static
  // image holds them too -- unless one of them changes bytes an earlier
  // dispatch consumes, which no preload can represent. Two tiles never do
  // (the slots ping-pong); more would. Only copies count as consumed data:
  // every slot is zero-filled up front and filled later, and an operand both
  // tiles share is copied twice with the same bytes -- the replay performs
  // that second copy before the DUT starts, too.
  const Regions before = memory->sram_copy_regions();
  const size_t first_late = memory->sram_write_count();
  const char* sram = memory->get_memory(SRAM_PARTITION);
  std::vector<char> snapshot(sram, sram + model.memory_sizes()[SRAM_PARTITION]);

  // A software-pipelined loop prefetches tile i+1 while tile i computes,
  // and its guard compares against the untruncated trip count -- so under
  // MAX_TILES the last iteration still prefetches a tile that never runs,
  // into the slot the first tile just used. The replay tolerates that (the
  // dispatch is a barrier); a static image cannot. A late load is worth
  // holding only if a dispatch after it consumes what it writes.
  std::vector<Regions> operands(steps.size());
  for (size_t i = 0; i < steps.size(); i++) {
    if (steps[i].kind == Step::kDispatch) {
      operands[i] = dispatch_operand_regions(steps[i]);
    }
  }
  const auto consumed_later = [&](size_t index) {
    const Regions written = load_regions(steps[index]);
    for (size_t j = index + 1; j < steps.size(); j++) {
      for (const auto& [b, e] : operands[j]) {
        for (const auto& [s, t] : written) {
          if (s < e && b < t) return true;
        }
      }
    }
    return false;
  };
  size_t dead = 0;
  const auto keep = [&](size_t index) {
    const bool needed = consumed_later(index);
    if (!needed) dead++;
    return needed;
  };
  apply_host_steps(steps, JtagPhase::kLateLoads, memory, keep);
  if (dead > 0) {
    std::cout << "  " << dead
              << " late load(s) skipped: no later dispatch consumes them"
              << std::endl;
  }
  for (const auto& [start, end] : memory->sram_writes_from(first_late)) {
    for (const auto& [b, e] : before) {
      const uint64_t lo = std::max(start, b);
      const uint64_t hi = std::min(end, e);
      if (lo < hi && std::memcmp(snapshot.data() + lo, sram + lo, hi - lo)) {
        // Name the late loads whose destination covers the range, and show
        // the schedule so the gating around them can be read off.
        std::string culprits;
        bool dispatched = false;
        for (const Step& step : steps) {
          if (step.kind == Step::kDispatch) dispatched = true;
          if (!dispatched || step.kind != Step::kCopy) continue;
          const Tensor dst = resolve(*step.prim, "dst", step.env);
          const uint64_t d0 = dst.address;
          const uint64_t d1 = d0 + get_num_bytes(dst);
          if (dst.partition == SRAM_PARTITION && d0 < hi && lo < d1) {
            culprits += " " + step.prim->name() + "->" + dst.node + "[" +
                        std::to_string(d0) + "," + std::to_string(d1) + ")";
          }
        }
        print_schedule(layer, steps);
        throw std::runtime_error(
            "full JTAG mode cannot preload " + layer +
            ": a later tile's load [" + std::to_string(start) + ", " +
            std::to_string(end) +
            ") changes scratchpad an earlier dispatch consumes (" + culprits +
            " ); run with MAX_TILES=2");
      }
    }
  }
  write_scratchpad_data(base_path, layer, *memory);

  // Reading the whole scratchpad back over a bit-banged JTAG takes hours;
  // only what the grade needs is worth it, and gold's walk says what that is.
  run_gold();
  write_scratchpad_dump_list(
      base_path, layer,
      readback_regions(model, selection, steps, *this->memory("gold"), memory));
}
