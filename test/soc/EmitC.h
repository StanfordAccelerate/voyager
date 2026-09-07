#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "test/common/GraphUtils.h"
#include "test/common/Model.h"
#include "test/soc/HostRequests.h"

// Emits one layer's bufferized program as C for the control processor.
//
// The generated firmware executes the whole program: loops, scalar
// arithmetic, conditionals, dispatches, the index-cell bookkeeping, and the
// program's semaphores as counters it owns. The DRAM<->scratchpad transfers
// it cannot perform itself (Sphinx has no DMA engine) become requests to the
// SoC testbench (host_request.h): a voyager::async_copy or zero fill names
// its prim by ordinal in the host-request table and ships the run-time
// values of the scalars the prim references; a commit's retire post asks
// the testbench to credit the semaphore once the units have retired. Built
// with NO_TESTBENCH (full JTAG mode, the chip) those requests and the waits
// on them compile to nothing, as the image is preloaded.
//
// A dispatch's params are serialized once at generation time with the
// iteration-0 scalar environment and embedded as static byte arrays. Fields
// that depend on runtime scalars (software-pipeline slot addresses) are
// located by probing -- re-serializing with one scalar perturbed and diffing
// the bit stream -- and the generated C patches them with patch_bits() before
// each send.
class CEmitter {
 public:
  explicit CEmitter(const Model& model) : model_(model) {}

  // Emits one layer. Returns the complete C translation unit.
  std::string emit_layer(const Model::Selection& selection);

 private:
  // --- symbol table: SSA name -> C identifier, scoped like ScalarEnv ---
  std::string declare(const std::string& ssa_name, int indent,
                      bool emit_decl = true);
  std::string bind(const std::string& ssa_name, const std::string& c_name);
  std::string ref(const std::string& ssa_name) const;

  std::string scalar_expr(const voyager::ScalarValue& value) const;

  // --- emission over the operation tree ---
  void emit_ops(
      const google::protobuf::RepeatedPtrField<voyager::Operation>& ops,
      int indent);
  void emit_operation(const voyager::Operation& op, int indent);
  void emit_scalar_prim(const voyager::Operation& op,
                        const voyager::PrimOp& prim, int indent);
  void emit_delinearize(const voyager::Operation& op,
                        const voyager::PrimOp& prim, int indent);
  // aten::_local_scalar_dense: a volatile load from the scratchpad.
  void emit_scalar_load(const voyager::Operation& op,
                        const voyager::PrimOp& prim, int indent);

  // A window of byte-aligned integer cells in the scratchpad, as C: the
  // element type, its width, the constant part of the byte address, the
  // run-time offset terms, and the element count.
  struct CellWindow {
    std::string c_type;
    int64_t width = 0;
    int64_t base = 0;
    std::string runtime_terms;
    int64_t count = 0;
  };
  CellWindow cell_window(const voyager::TensorBoxRef& ref,
                         const std::string& who) const;

  // aten::clone of an index cell, or the integer aten::add that maintains a
  // CSR running base: bookkeeping the control processor runs in place, after
  // the dispatch whose results it reads has retired.
  void emit_host_bookkeeping(const voyager::Operation& op,
                             const voyager::PrimOp& prim, int indent);
  void emit_for(const voyager::Operation& op, const voyager::ForLoop& loop,
                int indent);
  void emit_while(const voyager::Operation& op, const voyager::WhileLoop& loop,
                  int indent);
  void emit_cond(const voyager::Operation& op, const voyager::CondOp& cond,
                 int indent);
  void emit_dispatch(const voyager::Operation& op, int indent);

  // --- the program's semaphores and its requests to the testbench ---
  // A semaphore node is two arrays of counters in the firmware, one per
  // slot: `_tb`, which the testbench increments (a copy's completion, a
  // commit's retirement), and `_fw`, the firmware's own posts minus its
  // consumptions. Declared on first use; returns the two C names.
  std::pair<std::string, std::string> sem_cells(const voyager::TensorBox& box);
  // The slot a semaphore ref selects, as a C expression (resolve_bank).
  std::string sem_slot_expr(const voyager::TensorBoxRef& ref,
                            const std::string& who) const;
  void emit_semaphore_zeros(const voyager::Operation& op, int indent);
  void emit_semaphore_fill(const voyager::Operation& op,
                           const voyager::PrimOp& prim, int indent);
  void emit_async_wait(const voyager::Operation& op,
                       const voyager::PrimOp& prim, int indent);
  void emit_async(const voyager::Operation& op, int indent);
  // A prim in the host-request table: a copy, a zero fill or a host tensor
  // op, issued to the testbench by ordinal.
  void emit_host_request(const voyager::Operation& op,
                         const voyager::PrimOp& prim, size_t ordinal,
                         int indent);

  bool contains_dispatch(const voyager::Operation& op) const;
  // Anything the firmware emits code for: a dispatch, a request, a
  // semaphore operation, bookkeeping. A loop with none is not emitted.
  bool contains_work(const voyager::Operation& op) const;
  bool cond_guards_dispatch(const voyager::Operation& op) const;

  // Concrete iteration-0 evaluation, mirroring the Interpreter's scalar
  // semantics; keeps env_ valid so dispatch sites can be probed.
  Scalar eval_scalar_prim(const voyager::PrimOp& prim) const;

  // Per-iteration advance of a scalar, followed back through constant
  // factors to a loop counter; 0 when it is not a scaled counter.
  int64_t derived_step(const std::string& ssa_name, int depth = 0) const;

  // Every scalar SSA name a dispatch's operand references (window offsets,
  // scalar kwargs) -- the probe set.
  std::set<std::string> collect_ref_scalars(const voyager::Operation& op) const;

  void line(int indent, const std::string& text);

  const Model& model_;
  ScalarEnv env_;
  std::vector<std::map<std::string, std::string>> scopes_;
  std::map<std::string, int> name_counts_;
  std::ostringstream decls_;  // file-scope statics: params arrays, semaphores
  std::ostringstream body_;   // statements inside main()

  // The layer's host-request table (HostRequests.h) and each prim's ordinal.
  std::vector<HostOp> host_table_;
  std::map<const voyager::PrimOp*, size_t> host_ordinals_;
  // Semaphore node -> C name stem of its counter arrays.
  std::map<std::string, std::string> sem_names_;
  const std::set<const voyager::Operation*>* bounded_ = nullptr;
  int loop_depth_ = 0;
  int max_tiles_ = 0;

  // Defining prim per scalar SSA value, recorded as the walk passes it.
  std::map<std::string, const voyager::PrimOp*> scalar_def_;

  // Counters whose advance is known: a for loop's induction variable, and a
  // carried value the body yields as add(itself, constant). Anything else is
  // not a counter -- assuming a step for it would let a rejected probe read
  // as invariance.
  std::map<std::string, int64_t> loop_counter_steps_;

  // Scalars sampled at their own step, so run-time values lie on the sampled
  // grid. Only these may carry a fractional coefficient.
  std::set<std::string> derived_scalars_;

  // True while concretely evaluating a cond arm the iteration-0 predicate
  // does not take: guarded arithmetic there (a divisor that is zero only on
  // the untaken path) yields a placeholder instead of aborting generation.
  bool speculative_ = false;

  // A first walk that emits nothing and maps nothing, run only to learn the
  // environments below. Both walks use the same bounds, so a dispatch absent
  // from concrete_env_ after it is one the firmware's own loop never reaches.
  bool surveying_ = false;

  // The first environment in which each dispatch was reached with its
  // guarding predicates actually true. A dispatch inside a cond arm the
  // iteration-0 predicate does not take carries only placeholder scalars at
  // its emission point, and params serialized from those describe nothing the
  // hardware will run -- so it is mapped and probed under this instead.
  std::map<const voyager::Operation*, ScalarEnv> concrete_env_;

  // Inside a commit region's body: its dispatches are asynchronous. A
  // dispatch emitted outside any commit is synchronous and gets
  // Harness::execute's drain-dispatch-drain brackets (Harness.cc:688-690).
  bool in_commit_ = false;
};
