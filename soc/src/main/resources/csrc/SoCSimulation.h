#pragma once

#include <svdpi.h>

#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "SoCMemory.h"
#include "test/common/Simulation.h"
#include "test/soc/ScheduleRecorder.h"

// The SoC RTL testbench: replays the DMA/host/semaphore side of the layer's
// bufferized program against the DUT while the firmware on the Rocket core
// streams the accelerator dispatches.
//
// At start(): loads the model, preloads DRAM inputs, runs the gold walk for
// references, then records the schedule with ScheduleRecorder. Replay is
// event-driven from the DUT's per-unit start events (start_fired(unit)) and
// done events (tick(unit)): steps apply in program order and the walk stalls
// only at a kWait whose semaphore lacks credits. A retire kPost never stalls
// the walk: it rides the newest dispatch as a token and fires (credits its
// semaphore) when that dispatch -- and, FIFO, every earlier one -- has fully
// retired. This is what preserves the schedule's copy lookahead: the walk
// keeps executing copies for the next tile while the current dispatch
// computes, exactly one ping-pong slot ahead, paced by the program's own
// waits.
//
// A kDispatch does not credit the hardware directly: it queues one grant per
// unit start, in Harness::release_starts' order (groups in sequence; within
// a group the compute unit before the vector unit), and pump_grants() keeps
// at most one grant outstanding. Poking +1 into the unit's start semaphore
// is the grant; the unit consuming it -- its start firing, reported by the
// Verilog collateral through start_fired() -- completes the rendezvous,
// exactly like the harness's SyncPop. A group's vector start therefore
// cannot precede its own compute start, and no start can precede an earlier
// group's starts -- the ordering that keeps a fused pass's in-place
// accumulator read behind the previous dispatch's writeback. Between the
// groups of one multi-group dispatch the queue also reproduces
// dispatch_params' drain: the next group issues only once everything issued
// has retired.
class SoCSimulation : public Simulation {
 public:
  SoCSimulation();

  void start();
  void tick(int unit);
  void start_fired(int unit);

 protected:
  ArrayMemory* make_memory(const std::string& sim,
                           const std::vector<uint64_t>& sizes) override;

 private:
  void advance();
  bool apply(const Step& step);  // false = stalled, stop advancing
  void pump_grants();
  void poke_hw_semaphore(int unit, int credits);
  void finish();

  // Nothing dispatched is pending anywhere: no group in flight, no grant
  // queued or outstanding. The quiescence Harness::drain() reaches.
  bool quiescent() const {
    return inflight_.empty() && grants_.empty() && !grant_outstanding_;
  }

  std::vector<Step> steps_;
  size_t next_step_ = 0;

  // Index of a synchronous dispatch already issued but not yet retired; the
  // walk is parked on it (issue exactly once, pass only after retirement).
  size_t sync_issued_ = static_cast<size_t>(-1);

  // Testbench-side counters for the program's own semaphores.
  std::map<std::pair<std::string, int64_t>, int64_t> sems_;

  // Per-dispatch bookkeeping. groups_total counts the dispatch's invocation
  // groups; its retire-post tokens fire when the last of them retires.
  struct DispatchRec {
    size_t step_index;
    size_t groups_total;
    size_t groups_retired = 0;
    std::vector<const Step*> tokens;
  };
  std::vector<DispatchRec> dispatches_;

  // One queued unit-start grant, in Harness::release_starts' release order.
  struct Grant {
    int unit;               // Step::Unit index == hardware semaphore index
    size_t dispatch_index;  // into dispatches_
    Step::Group group;      // the invocation group this start belongs to
    bool leads_group;       // first grant of its group: creates the InFlight
    bool drain_before;      // dispatch_params' inter-group drain barrier
  };
  std::deque<Grant> grants_;
  bool grant_outstanding_ = false;

  // Issued invocation groups whose done events have not all arrived, oldest
  // first. The matrix unit pipelines up to two passes, so done events of
  // consecutive groups can interleave across units: each done is attributed
  // to the oldest entry still owing a pass on that unit. Entries retire only
  // at the front (FIFO), mirroring Harness::retire_dones.
  struct InFlight {
    size_t dispatch_index;
    int remaining[Step::kNumUnits];
    bool done() const {
      for (int u = 0; u < Step::kNumUnits; u++) {
        if (remaining[u] > 0) return false;
      }
      return true;
    }
  };
  std::deque<InFlight> inflight_;

  bool finished_ = false;
};
