#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "test/common/GraphUtils.h"
#include "test/common/Model.h"

// Emits the CPU side of one layer's bufferized program as C.
//
// The generated firmware mirrors the IR's control structure -- loops, scalar
// arithmetic, conditionals -- but carries none of the data movement: every
// voyager::async_copy / async_wait / zeros / fill is the testbench's job
// (ScheduleRecorder replays them), and the hardware start-gating semaphores
// pace the dispatches the firmware streams.
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
  void emit_for(const voyager::Operation& op, const voyager::ForLoop& loop,
                int indent);
  void emit_while(const voyager::Operation& op, const voyager::WhileLoop& loop,
                  int indent);
  void emit_cond(const voyager::Operation& op, const voyager::CondOp& cond,
                 int indent);
  void emit_dispatch(const voyager::Operation& op, int indent);

  bool contains_dispatch(const voyager::Operation& op) const;

  // Concrete iteration-0 evaluation, mirroring the Interpreter's scalar
  // semantics; keeps env_ valid so dispatch sites can be probed.
  Scalar eval_scalar_prim(const voyager::PrimOp& prim) const;

  // Every scalar SSA name a dispatch's operand references (window offsets,
  // scalar kwargs) -- the probe set.
  std::set<std::string> collect_ref_scalars(const voyager::Operation& op) const;

  void line(int indent, const std::string& text);

  const Model& model_;
  ScalarEnv env_;
  std::vector<std::map<std::string, std::string>> scopes_;
  std::map<std::string, int> name_counts_;
  std::ostringstream decls_;  // file-scope static params arrays
  std::ostringstream body_;   // statements inside main()
  const std::set<const voyager::Operation*>* bounded_ = nullptr;
  int loop_depth_ = 0;
  int max_tiles_ = 0;

  // True while concretely evaluating a cond arm the iteration-0 predicate
  // does not take: guarded arithmetic there (a divisor that is zero only on
  // the untaken path) yields a placeholder instead of aborting generation.
  bool speculative_ = false;

  // Inside a commit region's body: its dispatches are asynchronous. A
  // dispatch emitted outside any commit is synchronous and gets
  // Harness::execute's drain-dispatch-drain brackets (Harness.cc:688-690).
  bool in_commit_ = false;
};
