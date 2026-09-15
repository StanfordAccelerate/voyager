#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include "test/common/MemoryInterface.h"
#include "test/common/Tensor.h"
#include "test/compiler/proto/voyager_ir.pb.h"

// Reading the compiler's graph. Every function here takes a piece of the
// bufferized `voyager` IR -- an operation, a prim, an argument, an operand --
// and turns it into something the testbench can use: a scalar value, a
// concrete buffer, or a memory effect. Nothing here holds state; the loaded
// graph itself lives in Model.h.

// ===========================================================================
// Partitions of the ArrayMemory backing store
// ===========================================================================

// Partitions of the ArrayMemory backing store.
constexpr int DRAM_PARTITION = 0;
constexpr int SRAM_PARTITION = 1;
constexpr int REFERENCE_PARTITION = -1;

// ===========================================================================
// Scalars
// ===========================================================================

// Scalars are first-class SSA values in the bufferized IR: loop induction
// variables, loop-carried state, tile indices, and every branch predicate. They
// are host-side ("cpu") values -- the accelerator never produces one -- so the
// interpreter keeps them in a plain register bank.
using Scalar = std::variant<int64_t, double, bool>;

inline int64_t to_int(const Scalar& s) {
  if (std::holds_alternative<int64_t>(s)) return std::get<int64_t>(s);
  if (std::holds_alternative<bool>(s)) return std::get<bool>(s) ? 1 : 0;
  return static_cast<int64_t>(std::get<double>(s));
}

inline double to_double(const Scalar& s) {
  if (std::holds_alternative<double>(s)) return std::get<double>(s);
  if (std::holds_alternative<bool>(s)) return std::get<bool>(s) ? 1.0 : 0.0;
  return static_cast<double>(std::get<int64_t>(s));
}

inline bool to_bool(const Scalar& s) {
  if (std::holds_alternative<bool>(s)) return std::get<bool>(s);
  if (std::holds_alternative<int64_t>(s)) return std::get<int64_t>(s) != 0;
  return std::get<double>(s) != 0.0;
}

// A scope chain, not a flat map: loop induction variables and iter_args are
// named positionally (arg0, arg1, ...) and repeat across every loop in the
// model, so a single flat map would let one loop clobber another's state.
class ScalarEnv {
 public:
  ScalarEnv() { scopes_.emplace_back(); }

  void push() { scopes_.emplace_back(); }
  void pop() { scopes_.pop_back(); }

  void define(const std::string& name, Scalar value) {
    scopes_.back()[name] = value;
  }

  bool bound(const std::string& name) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->count(name)) return true;
    }
    return false;
  }

  Scalar lookup(const std::string& name) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      const auto found = it->find(name);
      if (found != it->end()) return found->second;
    }
    throw std::runtime_error("Unbound scalar SSA value: " + name);
  }

 private:
  std::vector<std::map<std::string, Scalar>> scopes_;
};

// A ScalarValue is either a literal baked into the proto or a reference to an
// SSA name defined earlier in an enclosing scope.
inline Scalar eval(const voyager::ScalarValue& value, const ScalarEnv& env) {
  switch (value.value_case()) {
    case voyager::ScalarValue::kNode:
      return env.lookup(value.node());
    case voyager::ScalarValue::kIntValue:
      return static_cast<int64_t>(value.int_value());
    case voyager::ScalarValue::kFloatValue:
      return value.float_value();
    case voyager::ScalarValue::kBoolValue:
      return value.bool_value();
    default:
      throw std::runtime_error("ScalarValue with no value set");
  }
}

inline int64_t eval_int(const voyager::ScalarValue& value,
                        const ScalarEnv& env) {
  return to_int(eval(value, env));
}

inline std::vector<int64_t> eval_int_list(const voyager::ScalarValueList& list,
                                          const ScalarEnv& env) {
  std::vector<int64_t> out;
  out.reserve(list.values_size());
  for (const auto& value : list.values()) out.push_back(eval_int(value, env));
  return out;
}

// ===========================================================================
// Operations
// ===========================================================================

// Classifying a target name. The IR qualifies targets ("aten::slice",
// "quantized_ops::linear_mx"), so these take a name already put through
// strip_namespace.
inline bool is_gemm_op(const std::string& op_target) {
  static const std::unordered_set<std::string> GEMM_OPS = {
      "conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx",
  };
  return GEMM_OPS.find(op_target) != GEMM_OPS.end();
}

inline bool is_reduction_op(const std::string& op_target) {
  static const std::unordered_set<std::string> REDUCTION_OPS = {
      "amax", "sum", "mean", "min", "max", "prod",
  };
  return REDUCTION_OPS.find(op_target) != REDUCTION_OPS.end();
}

inline bool is_dma_op(const std::string& op_target) {
  static const std::unordered_set<std::string> DMA_OPS = {
      "slice",
      "permute",
      "transpose",
  };
  return DMA_OPS.find(op_target) != DMA_OPS.end();
}

// Every PrimOp in an Operation: one for a `prim`, the whole chain for a
// `fused`.
std::vector<const voyager::PrimOp*> get_prim_ops(const voyager::Operation& op);

// Argument accessors. An argument is a ScalarValue that may name an SSA value,
// so reading one needs the environment that value was defined in.
inline const voyager::Argument& arg(const voyager::PrimOp& prim,
                                    const std::string& key) {
  const auto found = prim.kwargs().find(key);
  if (found == prim.kwargs().end()) {
    throw std::runtime_error("Operation " + prim.name() + " (" + prim.target() +
                             ") has no argument named " + key);
  }
  return found->second;
}

inline bool has_arg(const voyager::PrimOp& prim, const std::string& key) {
  return prim.kwargs().count(key) > 0;
}

inline bool has_tensor(const voyager::PrimOp& prim, const std::string& key) {
  const auto found = prim.kwargs().find(key);
  return found != prim.kwargs().end() && found->second.has_tensor_box();
}

inline int64_t arg_int(const voyager::PrimOp& prim, const std::string& key,
                       const ScalarEnv& env) {
  return to_int(eval(arg(prim, key).scalar(), env));
}

inline double arg_float(const voyager::PrimOp& prim, const std::string& key,
                        const ScalarEnv& env) {
  return to_double(eval(arg(prim, key).scalar(), env));
}

inline bool arg_bool(const voyager::PrimOp& prim, const std::string& key,
                     const ScalarEnv& env) {
  return to_bool(eval(arg(prim, key).scalar(), env));
}

inline std::vector<int> arg_ints(const voyager::PrimOp& prim,
                                 const std::string& key, const ScalarEnv& env) {
  const auto values = eval_int_list(arg(prim, key).scalar_list(), env);
  return std::vector<int>(values.begin(), values.end());
}

inline const std::string& arg_str(const voyager::PrimOp& prim,
                                  const std::string& key) {
  return arg(prim, key).str_value();
}

// Every scalar argument except `input`: the constants a few activation
// functions are parameterized by.
inline std::map<std::string, double> arg_scalars(const voyager::PrimOp& prim,
                                                 const ScalarEnv& env) {
  std::map<std::string, double> scalars;
  for (const auto& [key, value] : prim.kwargs()) {
    if (key == "input" || !value.has_scalar()) continue;
    scalars[key] = to_double(eval(value.scalar(), env));
  }
  return scalars;
}

// Match voyager_compiler.codegen.node_info.get_anchor_node: a GEMM is always
// the compute anchor of a fusion. Otherwise the first call_function, except
// that a leading dequantize is a prelude and the next call_function is the
// anchor.
const voyager::PrimOp& get_anchor_op(const voyager::Operation& op);

// CSR bookkeeping the control processor runs in place: a clone of an index
// cell, or the integer accumulator that maintains a running base (an
// aten::add of an integer cell and a scalar). The compiler tags both
// call_function, but no engine ever runs them.
bool is_host_bookkeeping(const voyager::PrimOp& prim);

// A datapath op runs on the accelerator; everything else is host-side control.
bool is_datapath(const voyager::Operation& op);

// "quantized_ops::conv2d_mx" -> "conv2d_mx", "aten::add" -> "add".
std::string strip_namespace(const std::string& target);

// ===========================================================================
// Tensor boxes and operand resolution
// ===========================================================================

// Every reference carries the box it names, so these are plain reads of the
// memory level the compiler assigned.

// A semaphore lives in a register, not in memory: the interpreter keeps a
// counter.
bool is_semaphore(const voyager::TensorBox& box);

// Datapath configuration -- codebooks, quantization maps, scalar dequant
// scales -- that kernels read straight out of tensor_files by node name
// instead of from ArrayMemory.
bool is_constant(const voyager::TensorBox& box);

bool is_dram(const voyager::TensorBox& box);

// DRAM_PARTITION or SRAM_PARTITION. Throws for a box with no memory level.
int partition_of(const voyager::TensorBox& box);

uint32_t banks_of(const voyager::TensorBox& box);

// A box says where it lives and how it is shaped, so this needs no context
// beyond the bank a software-pipelined operand selects.
Tensor to_tensor(const voyager::TensorBox& box, int64_t bank = 0);
uint64_t bank_stride_of(const voyager::TensorBox& box);

// Resolves a TensorBoxRef -- a box plus an optional bank window -- into the
// concrete buffer it denotes right now.
//
// A ref whose box has no memory is a datapath intermediate -- a value an
// earlier PrimOp of the same fusion produced -- and one at IMMEDIATE level is a
// constant the kernels read from tensor_files. Neither has an address.
//
// `op_name` is only used to make an assertion failure locatable.
Tensor resolve(const voyager::TensorBoxRef& ref, const ScalarEnv& env,
               const std::string& op_name);

// The operand `op` passes as `key`. This is how kernels ask for an operand;
// the overload above is for the few refs that are not kwargs, such as a copy's
// destination. Reports the op and its target when `key` is absent, which
// kwargs().at() cannot.
Tensor resolve(const voyager::PrimOp& op, const std::string& key,
               const ScalarEnv& env);

// Where an operation's results go. Destination-passing: these buffers exist
// already, and the op writes into them.
std::vector<Tensor> resolve_outputs(const voyager::Operation& op,
                                    const ScalarEnv& env);

// The slot a semaphore ref selects (a semaphore has no memory to resolve).
//
// A semaphore node may be declared with a shape, which makes it an array of
// independent counters per bank -- a GEMM epilogue tracks each of its chunks
// on its own counter. Such a ref indexes [bank, *dims], so flatten the two
// into one slot id: bank-major, then the dims row-major. A scalar semaphore
// has one slot per bank and this is just the bank.
//
// The whole point of flattening here is that a slot is only ever a key: the
// backends keep a counter per (node, slot) and never interpret it.
int64_t resolve_bank(const voyager::TensorBoxRef& ref, const ScalarEnv& env,
                     const std::string& op_name);

// How many slots a semaphore declaration has: banks times its shape.
int64_t semaphore_slots(const voyager::TensorBox& box);

// ===========================================================================
// Data movement
// ===========================================================================

// Executes a voyager::async_copy: the explicit tile transfer between two memory
// levels. Semantics follow voyager-compiler's async_copy, which is the
// executable spec.
//
// Does not touch the semaphore -- the interpreter posts Operation.semaphore
// once the op retires, which is where the IR puts it.
void run_async_copy(const voyager::PrimOp& prim, const ScalarEnv& env,
                    MemoryInterface* memory);

// Zeroes a buffer. (A REGISTER-level voyager::zeros is a semaphore and never
// reaches memory; the interpreter handles those.)
void zero_buffer(const Tensor& tensor, uint32_t banks, uint64_t bank_stride,
                 MemoryInterface* memory);

// A gemm whose sparse operands the compiler fused in.
inline bool has_fused_spmm(const voyager::PrimOp& op) {
  return has_arg(op, "A_indptr") && has_arg(op, "A_indices") &&
         has_arg(op, "A_data");
}

// A gemm whose input collapses to a single row, i.e. a matrix-vector multiply.
inline bool is_fc_layer(const voyager::PrimOp& op) {
  if (!is_gemm_op(strip_namespace(op.target()))) {
    return false;
  }

  // The operand states its own shape, so this needs no memory resolution.
  const auto& ref = op.kwargs().at("input").tensor_box();
  const auto& shape =
      ref.output_shape_size() > 0 ? ref.output_shape() : ref.box().shape();

  int dim = 1;
  for (int i = 0; i < shape.size() - 1; i++) {
    dim *= shape[i];
  }

  return dim == 1;
}
