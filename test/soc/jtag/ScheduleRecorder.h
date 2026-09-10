#pragma once

#include <set>
#include <string>
#include <vector>

#include "test/common/Backend.h"
#include "test/common/GraphUtils.h"
#include "test/common/Model.h"

// One host-side step of a layer's program, in program order.
//
// The firmware runs the whole program; what the full JTAG flow needs from
// the interpreter's walk is the sequence of loads it must bake into the
// scratchpad image and where the dispatches fall between them
// (JtagPreload). Semaphores are not recorded: with no testbench to pace
// against, the recorded order is the order.
struct Step {
  enum Kind {
    kCopy,        // voyager::async_copy: perform via run_async_copy(prim, env)
    kZero,        // voyager::zeros data buffer, or an integer alloc's zero-fill
    kHostOp,      // a "cpu"-tagged tensor op: run the gold kernel in place
    kScalarRead,  // aten::_local_scalar_dense: read the cell when applied
    kScalarOp,    // a scalar derived from one: recompute it when applied
    kDispatch     // an accelerator dispatch: a marker between the loads
  };
  Kind kind;
  // The op and the scalar env it executed under. A kScalarRead's value is
  // not recorded -- the walk continued on a placeholder -- so the step is
  // performed against live memory when the schedule is applied.
  const voyager::Operation* op = nullptr;
  const voyager::PrimOp* prim = nullptr;
  ScalarEnv env;
  std::string op_name;  // kDispatch / kHostOp
};

// A Backend that records instead of executing. Drive it with the standard
// Interpreter over the layer selection; the walk's program order becomes the
// schedule. Semaphores are neither enforced nor recorded: the walk reaches a
// commit's waits before anything could have posted them.
class ScheduleRecorder : public Backend {
 public:
  explicit ScheduleRecorder(std::vector<Step>* steps) : steps_(steps) {}

  bool intercepts_data_ops() const override { return true; }
  void data_op(const voyager::Operation& op, const voyager::PrimOp& prim,
               const ScalarEnv& env) override;
  bool intercepts_scalar_reads() const override { return true; }
  void scalar_read(const voyager::Operation& op, const voyager::PrimOp& prim,
                   const ScalarEnv& env) override;
  void scalar_op(const voyager::Operation& op, const voyager::PrimOp& prim,
                 const ScalarEnv& env) override;
  void execute(const voyager::Operation& op, const ScalarEnv& env) override;
  void init_semaphore(const std::string&, int64_t, int64_t) override {}
  void post_semaphore(const std::string&, int64_t, int64_t) override {}
  void wait_semaphore(const std::string&, int64_t, int64_t) override {}

 private:
  std::vector<Step>* steps_;
  // Scalars whose value descends from a read deferred to apply time. The
  // recording walk computed them from a placeholder, so they are recomputed
  // in this order before the copies that consume them.
  std::set<std::string> deferred_;
};
