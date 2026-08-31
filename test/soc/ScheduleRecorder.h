#pragma once

#include <deque>
#include <string>
#include <vector>

#include "src/Params.h"
#include "test/common/Backend.h"
#include "test/common/GraphUtils.h"
#include "test/common/Model.h"

// One entry of the testbench's replay schedule.
//
// The SoC firmware carries only the accelerator dispatches; everything the
// interpreter would otherwise do in place -- DMA copies, buffer zeroing,
// host-side tensor ops, and the semaphore bookkeeping that sequences them --
// is recorded here at initialization time and replayed by the testbench,
// paced by the DUT's done events.
struct Step {
  enum Kind {
    kCopy,     // voyager::async_copy: perform via run_async_copy(prim, env)
    kZero,     // voyager::zeros data buffer, or an integer alloc's zero-fill
    kHostOp,   // a "cpu"-tagged tensor op: run the gold kernel in place
    kInit,     // set a semaphore counter to `amount`
    kWait,     // consume `amount` credits; stalls the replay until available
    kPost,     // add `amount` credits; a retire post matures only once every
               //   dispatch recorded before it has fully retired
    kDispatch  // firmware sends the params; expect done events per unit
  };

  Kind kind;

  // kCopy / kZero / kHostOp: the op and the scalar env it executed under.
  const voyager::Operation* op = nullptr;
  const voyager::PrimOp* prim = nullptr;
  ScalarEnv env;

  // kInit / kWait / kPost
  std::string sem_node;
  int64_t sem_slot = 0;
  int64_t amount = 0;
  bool retire_post = false;

  // kDispatch: invocation groups per unit, in dispatch_params' chunking.
  // passes[u] is the number of done events unit `u` will produce.
  enum Unit { kMatrix = 0, kVector = 1, kMvm = 2, kSpmm = 3, kNumUnits = 4 };
  int passes[kNumUnits] = {0, 0, 0, 0};

  // kDispatch: the invocation groups in the order Harness::dispatch_params
  // would push them. The replay's start-release engine grants each group's
  // unit starts in this order (compute unit first, then vector), mirroring
  // Harness::release_starts.
  struct Group {
    int compute_unit = -1;  // kMatrix or kMvm; -1 for a vector-only group
    bool vector = false;
  };
  std::vector<Group> groups;
  std::string op_name;

  // kDispatch: a dispatch outside an async commit is synchronous -- a barrier
  // on both sides (Harness::execute drains before and after it), and the
  // interpreter posts the op's own semaphore the moment it "returns", i.e.
  // once the hardware has retired it. The replay must stall the walk on it.
  bool sync = false;
};

// Counts invocation groups per unit by replaying Harness::dispatch_params'
// chunking rules over a mapped params deque, and (when `groups` is non-null)
// records the group sequence itself. Throws on params sequences the MVP does
// not support (SpMM with fused dense, DwC).
void count_unit_passes(const std::deque<BaseParams*>& params,
                       int passes[Step::kNumUnits],
                       std::vector<Step::Group>* groups = nullptr);

// A Backend that records instead of executing. Drive it with the standard
// Interpreter over the layer selection; the walk's program order becomes the
// schedule. Semaphore counts are deliberately not enforced at record time:
// a commit's post is recorded as a retire post rather than performed, so
// later waits in the stream would underflow -- the replay enforces them.
class ScheduleRecorder : public Backend {
 public:
  explicit ScheduleRecorder(std::vector<Step>* steps) : steps_(steps) {}

  bool intercepts_data_ops() const override { return true; }

  void data_op(const voyager::Operation& op, const voyager::PrimOp& prim,
               const ScalarEnv& env) override;

  void execute(const voyager::Operation& op, const ScalarEnv& env) override;

  void init_semaphore(const std::string& node, int64_t slot,
                      int64_t value) override;
  void post_semaphore(const std::string& node, int64_t slot,
                      int64_t amount) override;
  void wait_semaphore(const std::string& node, int64_t slot,
                      int64_t amount) override;

  void begin_commit() override;
  void end_commit(bool has_post, const std::string& post_node,
                  int64_t post_slot, int64_t post_amount) override;

 private:
  std::vector<Step>* steps_;
  bool in_commit_ = false;
};
