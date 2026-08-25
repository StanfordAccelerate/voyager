#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "test/common/GraphUtils.h"
#include "test/compiler/proto/voyager_ir.pb.h"

// Where the interpreter hands an operation to whatever is actually computing.
//
// The interpreter resolves control flow, scalars and data movement identically
// in every environment; only this last step differs. The gold model runs the
// bit-accurate C++ kernel in place; the SystemC and RTL flows will instead
// lower the operation into MatrixParams / VectorParams and push it at the
// hardware.
class Backend {
 public:
  virtual ~Backend() = default;

  // `op` is a prim or a fusion whose targets run on the accelerator.
  virtual void execute(const voyager::Operation& op, const ScalarEnv& env) = 0;

  // --- Semaphore protocol -------------------------------------------------
  //
  // The interpreter resolves a semaphore TensorBoxRef to a (node, slot) and
  // drives its counter through these three. A slot is a bank for a scalar
  // semaphore and a flattened [bank, *dims] for a semaphore array; either way
  // it is only ever a key, never interpreted. The default is a plain integer
  // counter, which is exactly right for the sequential backends (gold, params):
  // the walk always reaches a wait after the op that posts it, so a wait never
  // blocks.
  //
  // The SystemC Harness overrides these with a blocking, cross-thread counting
  // semaphore, because there a `commit` retires on a second thread and a wait
  // can reach the interpreter before that thread has posted.
  virtual void init_semaphore(const std::string& node, int64_t slot,
                              int64_t value) {
    semaphores_[{node, slot}] = value;
  }
  virtual void post_semaphore(const std::string& node, int64_t slot,
                              int64_t amount) {
    semaphores_[{node, slot}] += amount;
  }
  virtual void wait_semaphore(const std::string& node, int64_t slot,
                              int64_t amount) {
    int64_t& count = semaphores_.at({node, slot});
    if (count < amount) {
      throw std::runtime_error(
          "Semaphore " + node + " slot " + std::to_string(slot) +
          " was waited before it was posted: the walk reached the wait before "
          "the op that signals it.");
    }
    count -= amount;
  }

  // --- Data-movement interception -----------------------------------------
  //
  // The interpreter itself performs the program's data movement: async_copy
  // transfers, the materialization of voyager::zeros data buffers, and the
  // zero-fill of integer allocations. A backend that does not execute the
  // program in place -- the SoC schedule recorder, which replays these
  // effects into the DUT's scratchpad later -- overrides
  // intercepts_data_ops() and receives each such op through data_op()
  // instead of the interpreter touching memory. Semaphore accounting is not
  // affected: posts and waits already flow through the backend.
  virtual bool intercepts_data_ops() const { return false; }
  virtual void data_op(const voyager::Operation& op,
                       const voyager::PrimOp& prim, const ScalarEnv& env) {}

  // --- Committed (async) regions ------------------------------------------
  //
  // A `commit` dispatches a region asynchronously so a matrix tile's ramp-up
  // (input/weight load, array fill) can overlap the previous tile's ramp-down
  // (output drain). Sequential backends have nothing to overlap, so the default
  // is fully synchronous: the interpreter waits the dependencies, runs the body
  // in place -- begin/end are no-ops, so the body's ops flow through execute()
  // exactly as top-level ops do -- and end_commit posts.
  //
  // The Harness overrides is_committed_async() to true; the interpreter still
  // walks the body identically, but the Harness routes the body's matrix op to
  // the datapath immediately (ramp-up) inside execute(), while deferring the
  // tail and the post to a completion thread it drains in end_commit().
  virtual bool is_committed_async() const { return false; }
  virtual void begin_commit() {}
  virtual void end_commit(bool has_post, const std::string& post_node,
                          int64_t post_slot, int64_t post_amount) {
    if (has_post) post_semaphore(post_node, post_slot, post_amount);
  }

 protected:
  std::map<std::pair<std::string, int64_t>, int64_t> semaphores_;
};
