#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "test/common/Backend.h"
#include "test/common/MemoryInterface.h"
#include "test/common/Model.h"
#include "test/compiler/proto/voyager_ir.pb.h"

// Interprets the bufferized graph.
//
// The IR spells out everything the backend used to invent: the tile loops, the
// reduction conditionals, the DMAs between memory levels, and the index
// arithmetic that drives them. So the interpreter executes exactly the
// structure it is given -- it never synthesizes a loop -- and hands off to the
// Backend only when it reaches an operation the accelerator actually runs.
class Interpreter {
 public:
  Interpreter(const Model& model, MemoryInterface* memory, Backend* backend);

  // Runs the selection's top-level operations in order.
  void run(const Model::Selection& selection);

  // Cycles the accelerator would need for the whole walk if it never stalled.
  //
  // Accumulated as the walk dispatches, so a tile loop multiplies it up by
  // itself: an operation's operands are one tile, and the loop runs the
  // operation once per tile. That is also why this is a total rather than a
  // per-operation figure -- a layer is no longer one operation. Its tile loop
  // splits the reduction across two arms of a conditional, so its cost is
  // spread over several operations and only the sum means anything.
  long matrix_cycles() const { return matrix_cycles_; }
  long vector_cycles() const { return vector_cycles_; }

 private:
  void execute_region(const voyager::Region& region);
  void execute_operation(const voyager::Operation& op);

  void execute_prim(const voyager::Operation& op, const voyager::PrimOp& prim);
  void execute_for(const voyager::Operation& op, const voyager::ForLoop& loop);
  void execute_while(const voyager::Operation& op,
                     const voyager::WhileLoop& loop);
  void execute_cond(const voyager::Operation& op, const voyager::CondOp& cond);

  void execute_scalar(const voyager::Operation& op,
                      const voyager::PrimOp& prim);
  void execute_zeros(const voyager::Operation& op, const voyager::PrimOp& prim);
  void execute_fill(const voyager::Operation& op, const voyager::PrimOp& prim);
  void execute_async_wait(const voyager::PrimOp& prim);
  void execute_async(const voyager::Operation& op,
                     const voyager::AsyncOp& async);

  int64_t resolve_semaphore_slot(const voyager::TensorBoxRef& ref,
                                 const std::string& op_name);

  // Dispatch an accelerator operation, charging its ideal cycles to its layer.
  void dispatch(const voyager::Operation& op);

  const Model& model_;
  MemoryInterface* memory_;
  Backend* backend_;

  ScalarEnv env_;
  int loop_depth_ = 0;
  bool bound_current_top_ = true;
  long matrix_cycles_ = 0;
  long vector_cycles_ = 0;
};
