#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "test/common/ArrayMemory.h"
#include "test/common/Simulation.h"
#include "test/soc/jtag/ScheduleRecorder.h"

// Host-side pieces of the full JTAG flow: program AND data go over the debug
// link and no testbench runs alongside the DUT.
//
// The firmware runs the whole program but its host requests compile to
// nothing, so the scratchpad image GDB writes before the firmware starts
// must hold every load the testbench would have staged: at MAX_TILES=2 the
// two ping-pong slots hold both tiles. The recorded schedule, split at the
// dispatches, says which loads those are. Afterwards GDB reads back the
// scratchpad-resident results and jtag_sim_checker compares them with
// gold's, which ran the same bounded program in place; nothing is replayed.

// An ArrayMemory that can remember which bytes of the scratchpad partition
// were written (to emit the preload image) and can report the scratchpad as
// always covered (to grade a dump the DUT produced, whose writes host-side
// tracking never saw).
class JtagMemory : public ArrayMemory {
 public:
  JtagMemory(const std::vector<uint64_t>& sizes, bool sram_covered);

  void record_sram_writes() { recording_writes_ = true; }

  // Merged, sorted [start, end) byte ranges of the scratchpad partition
  // written since record_sram_writes().
  std::vector<std::pair<uint64_t, uint64_t>> sram_regions() const;

  // The unmerged writes recorded so far, and those from index `first` on --
  // to tell the loads of one tile from the next.
  size_t sram_write_count() const { return ranges_.size(); }
  std::vector<std::pair<uint64_t, uint64_t>> sram_writes_from(
      size_t first) const;

  // Writes flagged as copies (as opposed to zero-fills), merged. A zero-fill
  // followed by a copy into the same slot is the normal pad-then-fill; a copy
  // over an earlier copy is a slot reuse.
  void set_recording_copy(bool copy) { recording_copy_ = copy; }
  std::vector<std::pair<uint64_t, uint64_t>> sram_copy_regions() const;

  bool was_written(int partition, uint64_t address) const override;
  bool any_written(int partition, uint64_t address,
                   uint64_t num_bytes) const override;

 protected:
  void write_bytes_to_memory(const long long address, const int partition,
                             const int num_bytes, const char* bytes) override;

 private:
  bool sram_covered_;
  bool recording_writes_ = false;
  bool recording_copy_ = false;
  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_ranges_;
};

// Physical SRAM address of a scratchpad-partition offset (SRAM_BASE in the
// firmware; the accelerator addresses the scratchpad from 0).
constexpr uint64_t kScratchpadBase = 0x40000000ULL;

// The scratchpad ranges the grade needs, i.e. what GDB has to read back after
// the run: the selection's scratchpad-resident outputs, bounded to the bytes
// gold wrote (a MAX_TILES-bounded run fills part of a buffer), which is what
// check_outputs compares.
std::vector<std::pair<uint64_t, uint64_t>> readback_regions(
    const Model& model, const Model::Selection& selection,
    const ArrayMemory& gold);

// Emits <base_path><layer>_scratchpad_dump.txt: one "0x<phys> <bytes>" line
// per region. run_jtag.gdb dumps exactly these (SCRATCHPAD_DUMP_LIST) and
// jtag_sim_checker loads them back.
void write_scratchpad_dump_list(
    const std::string& base_path, const std::string& layer,
    const std::vector<std::pair<uint64_t, uint64_t>>& regions);

// Emits <base_path><layer>_scratchpad_expected.bin.<i>: what gold left in
// each region, numbered as run_jtag.gdb numbers the dumps. Every byte of a
// region is graded (readback_regions walks gold's write mask) and gold and
// the accelerator are compared exactly, so a plain byte compare of a readback
// against these is the verdict jtag_sim_checker gives -- letting the chip
// flow grade itself with no gold, tensors or compiler.
void write_scratchpad_reference(
    const std::string& base_path, const std::string& layer,
    const std::vector<std::pair<uint64_t, uint64_t>>& regions,
    ArrayMemory& gold);

// Which part of the schedule to apply.
enum class JtagPhase {
  kLoads,      // every host-side step before the first dispatch
  kLateLoads,  // after it: copies into the scratchpad, zero-fills and the
               //   scalar reads their extents depend on -- the next tile's
               //   operands, staged while this one computes
};

// Applies the host-side steps of a recorded schedule to `memory` in program
// order, with the testbench's deferred-scalar overlay. Semaphore steps are
// ignored: with no testbench to pace against, the recorded order is the
// order. A late load's environment is the recorded one: a load whose address
// depends on a value the DUT produces cannot be preloaded anyway.
//
// `keep`, when given, is consulted for every late load (copy into the
// scratchpad or zero-fill) by step index; a false answer skips it.
void apply_host_steps(const std::vector<Step>& steps, JtagPhase phase,
                      ArrayMemory* memory,
                      const std::function<bool(size_t)>& keep = nullptr);

// The scratchpad byte ranges a dispatch's operands occupy (its prims'
// tensor arguments, resolved under the recorded environment).
std::vector<std::pair<uint64_t, uint64_t>> dispatch_operand_regions(
    const Step& dispatch);

// The scratchpad byte range(s) a load step writes, from its destination's
// extent (a copy's destination window; every bank of a zero-fill).
std::vector<std::pair<uint64_t, uint64_t>> load_regions(const Step& load);

// Records the selection's schedule the way SoCSimulation::start does.
std::vector<Step> record_schedule(const Model& model,
                                  const Model::Selection& selection,
                                  ArrayMemory* memory);

// Emits <base_path><layer>_scratchpad_data.c and .ld: one array and one
// section per written region, placed at the region's physical SRAM address.
// The firmware Makefile links them in under JTAG_SIM=1.
void write_scratchpad_data(const std::string& base_path,
                           const std::string& layer, const JtagMemory& memory);

// The codegen-side driver: one per layer (TESTS names it), stages the loads
// into a JtagMemory and emits the image.
class JtagPreload : public Simulation {
 public:
  JtagPreload() = default;

  void emit(const std::string& base_path, const std::string& layer);

 protected:
  ArrayMemory* make_memory(const std::string& sim,
                           const std::vector<uint64_t>& sizes) override;
};
