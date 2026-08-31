// clang-format off
// The params serialization needs the SystemC marshalling side of Params.h,
// which test/common/Utils.h (reached through almost every toolchain header)
// disables by defining NO_SYSC. Include the marshalling world first --
// formatting must not resort these.
#include <systemc.h>

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "src/TypeToBits.h"

#include "test/soc/EmitC.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <stdexcept>

#include "test/common/Utils.h"
#include "test/soc/ScheduleRecorder.h"
#include "test/toolchain/MapOperation.h"
// clang-format on

namespace {

// ---------------------------------------------------------------------------
// Params serialization (shared with the firmware's wire format): TypeToBits'
// marshalled bit stream as little-endian bytes, padded to whole 64-bit words
// so the send loop never reads past the array.
// ---------------------------------------------------------------------------

std::vector<unsigned char> hex_to_bytes(const std::string& hex) {
  std::vector<unsigned char> bytes;
  for (int i = static_cast<int>(hex.length()) - 1; i >= 1; i -= 2) {
    bytes.push_back(static_cast<unsigned char>(
        strtol(hex.substr(i - 1, 2).c_str(), nullptr, 16)));
  }
  return bytes;
}

template <typename T>
std::vector<unsigned char> serialize_one(T& params) {
  std::string hex = TypeToBits(params).to_string(SC_HEX);
  hex = hex.substr(2, std::string::npos);  // strip 0x
  if (hex.size() % 2 != 0) hex = "0" + hex;
  auto bytes = hex_to_bytes(hex);
  const size_t padded = ((Wrapped<T>::width + 63) / 64) * 8;
  bytes.resize(std::max(bytes.size(), padded), 0);
  return bytes;
}

enum ParamKind { kMatrixParams, kVectorParams, kVectorConfig };

struct SerializedParam {
  ParamKind kind;
  bool is_fc = false;
  std::vector<unsigned char> bytes;
};

std::vector<SerializedParam> serialize_params(
    const std::deque<BaseParams*>& params) {
  std::vector<SerializedParam> out;
  for (auto* base : params) {
    SerializedParam sp;
    if (auto* mp = dynamic_cast<MatrixParams*>(base)) {
      sp.kind = kMatrixParams;
      sp.is_fc = mp->is_fc;
      sp.bytes = serialize_one(*mp);
    } else if (auto* vp = dynamic_cast<VectorParams*>(base)) {
      sp.kind = kVectorParams;
      sp.bytes = serialize_one(*vp);
    } else if (auto* vc = dynamic_cast<VectorInstructionConfig*>(base)) {
      sp.kind = kVectorConfig;
      sp.bytes = serialize_one(*vc);
    } else {
      throw std::runtime_error("Unknown BaseParams subtype in dispatch.");
    }
    out.push_back(std::move(sp));
  }
  return out;
}

// --- little-endian bit-stream helpers over the serialized byte vectors ---

bool get_bit(const std::vector<unsigned char>& bytes, size_t bit) {
  return (bytes[bit / 8] >> (bit % 8)) & 1;
}

uint64_t extract_bits(const std::vector<unsigned char>& bytes, size_t off,
                      size_t len) {
  uint64_t value = 0;
  for (size_t i = 0; i < len; i++) {
    value |= static_cast<uint64_t>(get_bit(bytes, off + i)) << i;
  }
  return value;
}

// A maximal run of bits that differ between two serializations.
struct BitRun {
  size_t param_idx;
  size_t off;
  size_t len;
};

std::vector<BitRun> diff_runs(const std::vector<SerializedParam>& a,
                              const std::vector<SerializedParam>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("Probe changed the params structure.");
  }
  std::vector<BitRun> runs;
  for (size_t p = 0; p < a.size(); p++) {
    if (a[p].bytes.size() != b[p].bytes.size()) {
      throw std::runtime_error("Probe changed a params blob's size.");
    }
    const size_t bits = a[p].bytes.size() * 8;
    size_t i = 0;
    while (i < bits) {
      if (get_bit(a[p].bytes, i) == get_bit(b[p].bytes, i)) {
        i++;
        continue;
      }
      size_t start = i;
      while (i < bits && get_bit(a[p].bytes, i) != get_bit(b[p].bytes, i)) i++;
      runs.push_back({p, start, i - start});
    }
  }
  return runs;
}

// One runtime-patched field of one params blob.
struct PatchField {
  size_t param_idx;
  size_t off;
  size_t len;
  int64_t base;                          // value in the baseline blob
  std::map<std::string, int64_t> coeff;  // ssa scalar -> per-unit delta
};

std::string sanitize(const std::string& name) {
  std::string out;
  for (char c : name) out += (isalnum(c) || c == '_') ? c : '_';
  if (out.empty() || isdigit(out[0])) out = "v_" + out;
  return out;
}

void collect_scalar_names(const voyager::ScalarValue& value,
                          std::set<std::string>* names) {
  if (value.value_case() == voyager::ScalarValue::kNode) {
    names->insert(value.node());
  }
}

void collect_ref_names(const voyager::TensorBoxRef& ref,
                       std::set<std::string>* names) {
  for (const auto& offset : ref.offsets()) collect_scalar_names(offset, names);
}

void collect_argument_names(const voyager::Argument& argument,
                            std::set<std::string>* names) {
  switch (argument.arg_type_case()) {
    case voyager::Argument::kTensorBox:
      collect_ref_names(argument.tensor_box(), names);
      break;
    case voyager::Argument::kTensorBoxList:
      for (const auto& ref : argument.tensor_box_list().values()) {
        collect_ref_names(ref, names);
      }
      break;
    case voyager::Argument::kScalar:
      collect_scalar_names(argument.scalar(), names);
      break;
    case voyager::Argument::kScalarList:
      for (const auto& value : argument.scalar_list().values()) {
        collect_scalar_names(value, names);
      }
      break;
    default:
      break;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Symbol table
// ---------------------------------------------------------------------------

std::string CEmitter::declare(const std::string& ssa_name, int indent,
                              bool emit_decl) {
  std::string c_name = sanitize(ssa_name);
  int& count = name_counts_[c_name];
  if (count > 0) c_name += "_x" + std::to_string(count);
  count++;
  scopes_.back()[ssa_name] = c_name;
  if (emit_decl) line(indent, "int64_t " + c_name + ";");
  return c_name;
}

std::string CEmitter::bind(const std::string& ssa_name,
                           const std::string& c_name) {
  scopes_.back()[ssa_name] = c_name;
  return c_name;
}

std::string CEmitter::ref(const std::string& ssa_name) const {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    const auto found = it->find(ssa_name);
    if (found != it->end()) return found->second;
  }
  throw std::runtime_error("C emitter: unbound scalar SSA value " + ssa_name +
                           " (defined by an op the CPU program skipped?)");
}

std::string CEmitter::scalar_expr(const voyager::ScalarValue& value) const {
  switch (value.value_case()) {
    case voyager::ScalarValue::kNode:
      return ref(value.node());
    case voyager::ScalarValue::kIntValue:
      return std::to_string(value.int_value()) + "LL";
    case voyager::ScalarValue::kFloatValue: {
      // The firmware computes scalars in int64 only; a fractional constant
      // would silently diverge from the interpreter's float semantics.
      const double v = value.float_value();
      if (v != static_cast<double>(static_cast<int64_t>(v))) {
        throw std::runtime_error(
            "C emitter: non-integral float scalar " + std::to_string(v) +
            " cannot be represented in the int64-only SoC firmware.");
      }
      return std::to_string(static_cast<int64_t>(v)) + "LL";
    }
    case voyager::ScalarValue::kBoolValue:
      return value.bool_value() ? "1" : "0";
    default:
      throw std::runtime_error("ScalarValue with no value set.");
  }
}

void CEmitter::line(int indent, const std::string& text) {
  for (int i = 0; i < indent; i++) body_ << "\t";
  body_ << text << "\n";
}

// ---------------------------------------------------------------------------
// Structure queries
// ---------------------------------------------------------------------------

bool CEmitter::contains_dispatch(const voyager::Operation& op) const {
  switch (op.op_type_case()) {
    case voyager::Operation::kPrim:
    case voyager::Operation::kFused:
      return is_datapath(op);
    case voyager::Operation::kLoop: {
      const auto& body = op.loop().has_for_loop()
                             ? op.loop().for_loop().body()
                             : op.loop().while_loop().body();
      for (const auto& child : body.ops()) {
        if (contains_dispatch(child)) return true;
      }
      if (op.loop().has_while_loop()) {
        for (const auto& child : op.loop().while_loop().condition().ops()) {
          if (contains_dispatch(child)) return true;
        }
      }
      return false;
    }
    case voyager::Operation::kCond:
      for (const auto& child : op.cond().true_region().ops()) {
        if (contains_dispatch(child)) return true;
      }
      for (const auto& child : op.cond().false_region().ops()) {
        if (contains_dispatch(child)) return true;
      }
      return false;
    case voyager::Operation::kAsync:
      for (const auto& child : op.async().body().ops()) {
        if (contains_dispatch(child)) return true;
      }
      return false;
    default:
      return false;
  }
}

std::set<std::string> CEmitter::collect_ref_scalars(
    const voyager::Operation& op) const {
  std::set<std::string> names;
  for (const auto* prim : get_prim_ops(op)) {
    for (const auto& [key, argument] : prim->kwargs()) {
      collect_argument_names(argument, &names);
    }
  }
  for (const auto& output : op.outputs()) {
    if (output.has_destination()) {
      collect_ref_names(output.destination(), &names);
    }
  }
  return names;
}

// ---------------------------------------------------------------------------
// Concrete iteration-0 scalar evaluation (mirrors Interpreter semantics)
// ---------------------------------------------------------------------------

Scalar CEmitter::eval_scalar_prim(const voyager::PrimOp& prim) const {
  const std::string target = strip_namespace(prim.target());

  if (target == "_local_scalar_dense") {
    throw std::runtime_error(
        "_local_scalar_dense reads memory at run time; the SoC MVP flow "
        "cannot emit it (sparse-CSR layers are out of scope).");
  }
  if (target == "sym_ite") {
    const bool predicate = to_bool(eval(prim.kwargs().at("b").scalar(), env_));
    return eval(prim.kwargs().at(predicate ? "t" : "f").scalar(), env_);
  }

  // int64-only firmware: refuse fractional operands loudly rather than
  // silently truncating away the interpreter's float semantics.
  auto as_int = [&](const char* key) -> int64_t {
    const Scalar value = eval(prim.kwargs().at(key).scalar(), env_);
    if (std::holds_alternative<double>(value)) {
      const double v = std::get<double>(value);
      if (v != static_cast<double>(static_cast<int64_t>(v))) {
        throw std::runtime_error("C emitter: non-integral float operand " +
                                 std::to_string(v) + " of " + prim.name() +
                                 " cannot be represented in int64 firmware.");
      }
    }
    return to_int(value);
  };
  const int64_t a = as_int("input");
  const int64_t b = as_int("other");

  if (target == "add") return a + b;
  if (target == "sub") return a - b;
  if (target == "mul") return a * b;
  if (target == "mod") {
    if (b == 0) {
      if (speculative_) return int64_t{0};
      throw std::runtime_error("mod by zero in " + prim.name());
    }
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
  }
  if (target == "floordiv") {
    if (b == 0) {
      if (speculative_) return int64_t{0};
      throw std::runtime_error("floordiv by zero in " + prim.name());
    }
    int64_t q = a / b;
    const int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
  }
  if (target == "eq") return static_cast<int64_t>(a == b);
  if (target == "ne") return static_cast<int64_t>(a != b);
  if (target == "lt") return static_cast<int64_t>(a < b);
  if (target == "le") return static_cast<int64_t>(a <= b);
  if (target == "gt") return static_cast<int64_t>(a > b);
  if (target == "ge") return static_cast<int64_t>(a >= b);
  if (target == "and_") return static_cast<int64_t>((a != 0) && (b != 0));
  if (target == "or_") return static_cast<int64_t>((a != 0) || (b != 0));

  throw std::runtime_error("C emitter: unsupported scalar op " + target);
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

void CEmitter::emit_ops(
    const google::protobuf::RepeatedPtrField<voyager::Operation>& ops,
    int indent) {
  for (const auto& op : ops) emit_operation(op, indent);
}

void CEmitter::emit_operation(const voyager::Operation& op, int indent) {
  switch (op.op_type_case()) {
    case voyager::Operation::kPrim: {
      const auto& prim = op.prim();
      const std::string& target = prim.target();
      if (target == "voyager::alloc" || target == "voyager::zeros" ||
          target == "voyager::fill" || target == "voyager::async_copy" ||
          target == "voyager::async_wait") {
        return;  // the testbench's job
      }
      if (target == "voyager::delinearize_index" ||
          target == "voyager::increment_indices") {
        emit_delinearize(op, prim, indent);
        return;
      }
      bool only_scalars = op.outputs_size() > 0;
      for (const auto& output : op.outputs()) {
        if (!output.has_scalar()) only_scalars = false;
      }
      if (only_scalars) {
        emit_scalar_prim(op, prim, indent);
        return;
      }
      if (is_datapath(op)) {
        emit_dispatch(op, indent);
      }
      // A host-side tensor op (slice/pad the control processor would run):
      // the testbench executes it against the DUT scratchpad.
      return;
    }
    case voyager::Operation::kFused:
      if (is_datapath(op)) emit_dispatch(op, indent);
      return;
    case voyager::Operation::kLoop:
      if (!contains_dispatch(op)) return;
      if (op.loop().has_for_loop()) {
        emit_for(op, op.loop().for_loop(), indent);
      } else {
        emit_while(op, op.loop().while_loop(), indent);
      }
      return;
    case voyager::Operation::kCond:
      if (contains_dispatch(op) || op.outputs_size() > 0) {
        emit_cond(op, op.cond(), indent);
      }
      return;
    case voyager::Operation::kAsync: {
      // Dependencies and post are hardware semaphores the testbench manages;
      // the CPU just streams the body's dispatches.
      const bool was_in_commit = in_commit_;
      in_commit_ = true;
      emit_ops(op.async().body().ops(), indent);
      in_commit_ = was_in_commit;
      return;
    }
    default:
      return;
  }
}

void CEmitter::emit_scalar_prim(const voyager::Operation& op,
                                const voyager::PrimOp& prim, int indent) {
  if (op.outputs_size() != 1) {
    throw std::runtime_error("Scalar op " + op.name() +
                             " with multiple outputs.");
  }
  const std::string target = strip_namespace(prim.target());

  std::string expr;
  if (target == "sym_ite") {
    expr = "(" + scalar_expr(prim.kwargs().at("b").scalar()) + " != 0) ? (" +
           scalar_expr(prim.kwargs().at("t").scalar()) + ") : (" +
           scalar_expr(prim.kwargs().at("f").scalar()) + ")";
  } else {
    const std::string a = scalar_expr(prim.kwargs().at("input").scalar());
    const std::string b = scalar_expr(prim.kwargs().at("other").scalar());
    if (target == "add")
      expr = "(" + a + ") + (" + b + ")";
    else if (target == "sub")
      expr = "(" + a + ") - (" + b + ")";
    else if (target == "mul")
      expr = "(" + a + ") * (" + b + ")";
    else if (target == "mod")
      expr = "vy_mod(" + a + ", " + b + ")";
    else if (target == "floordiv")
      expr = "vy_fdiv(" + a + ", " + b + ")";
    else if (target == "eq")
      expr = "(" + a + ") == (" + b + ")";
    else if (target == "ne")
      expr = "(" + a + ") != (" + b + ")";
    else if (target == "lt")
      expr = "(" + a + ") < (" + b + ")";
    else if (target == "le")
      expr = "(" + a + ") <= (" + b + ")";
    else if (target == "gt")
      expr = "(" + a + ") > (" + b + ")";
    else if (target == "ge")
      expr = "(" + a + ") >= (" + b + ")";
    else if (target == "and_")
      expr = "((" + a + ") != 0) && ((" + b + ") != 0)";
    else if (target == "or_")
      expr = "((" + a + ") != 0) || ((" + b + ") != 0)";
    else
      eval_scalar_prim(prim);  // throws the descriptive error
  }

  // Concrete value first (the expression references existing bindings), then
  // the C definition.
  const Scalar value = eval_scalar_prim(prim);
  const std::string c_name = declare(op.outputs(0).name(), indent, false);
  line(indent, "int64_t " + c_name + " = " + expr + ";");
  env_.define(op.outputs(0).name(), value);
}

void CEmitter::emit_delinearize(const voyager::Operation& op,
                                const voyager::PrimOp& prim, int indent) {
  if (strip_namespace(prim.target()) == "increment_indices") {
    throw std::runtime_error(
        "voyager::increment_indices is a legacy path the C emitter does not "
        "support; regenerate the network with the current compiler.");
  }
  const std::string linear = scalar_expr(prim.kwargs().at("linear").scalar());
  const int64_t linear_value =
      to_int(eval(prim.kwargs().at("linear").scalar(), env_));
  const auto basis =
      eval_int_list(prim.kwargs().at("basis").scalar_list(), env_);
  if (op.outputs_size() != static_cast<int>(basis.size())) {
    throw std::runtime_error("delinearize_index " + op.name() +
                             " output/basis mismatch.");
  }

  std::vector<int64_t> index(basis.size(), 0);
  int64_t remaining = linear_value;
  for (int d = static_cast<int>(basis.size()) - 1; d >= 0; d--) {
    index[d] = remaining % basis[d];
    remaining /= basis[d];
  }

  const std::string rem = declare("__rem_" + op.name(), indent, false);
  line(indent, "int64_t " + rem + " = " + linear + ";");
  for (int d = static_cast<int>(basis.size()) - 1; d >= 0; d--) {
    const std::string c_name = declare(op.outputs(d).name(), indent, false);
    line(indent, "int64_t " + c_name + " = " + rem + " % " +
                     std::to_string(basis[d]) + "LL;");
    line(indent, rem + " /= " + std::to_string(basis[d]) + "LL;");
    env_.define(op.outputs(d).name(), index[d]);
  }
}

void CEmitter::emit_for(const voyager::Operation& op,
                        const voyager::ForLoop& loop, int indent) {
  const int64_t start = eval_int(loop.start(), env_);
  int64_t end = eval_int(loop.end(), env_);
  const int64_t step = eval_int(loop.step(), env_);
  if (step == 0) throw std::runtime_error("Zero-step loop " + op.name());

  // The same MAX_TILES clamp the interpreter and the testbench apply, baked
  // into the emitted bound so all three walks agree.
  const bool outermost = loop_depth_ == 0;
  if (max_tiles_ > 0 && bounded_ && bounded_->count(&op) && outermost &&
      step > 0) {
    end = std::min(end, start + max_tiles_ * step);
  }

  // Iteration state lives in the enclosing scope: declare() already makes C
  // names unique, and the loop's outputs alias the iter variables after it.
  line(indent, "/* " + op.name() + " */");

  // Resolve every initial in the ENCLOSING scope before binding the iv or
  // any sibling iter_arg, exactly as the interpreter does -- positional
  // names (arg0, arg1, ...) repeat across loops, so a later initial naming
  // an outer loop's carried value must not capture this loop's.
  std::vector<std::string> init_exprs;
  std::vector<Scalar> init_vals;
  for (const auto& arg : loop.iter_args()) {
    init_exprs.push_back(scalar_expr(arg.initial()));
    init_vals.push_back(eval(arg.initial(), env_));
  }

  scopes_.push_back({});
  env_.push();

  const std::string iv = declare(loop.iv(), indent, false);
  env_.define(loop.iv(), start);

  std::vector<std::string> iter_vars;
  for (int i = 0; i < loop.iter_args_size(); i++) {
    const std::string c_name = declare(loop.iter_args(i).name(), indent, false);
    line(indent, "int64_t " + c_name + " = " + init_exprs[i] + ";");
    iter_vars.push_back(c_name);
    env_.define(loop.iter_args(i).name(), init_vals[i]);
  }

  const std::string cmp = step > 0 ? " < " : " > ";
  line(indent, "for (int64_t " + iv + " = " + std::to_string(start) + "LL; " +
                   iv + cmp + std::to_string(end) + "LL; " + iv +
                   " += " + std::to_string(step) + "LL) {");
  loop_depth_++;
  scopes_.push_back({});
  bind(loop.iv(), iv);
  for (int i = 0; i < loop.iter_args_size(); i++) {
    bind(loop.iter_args(i).name(), iter_vars[i]);
  }

  emit_ops(loop.body().ops(), indent + 1);

  if (loop.body().yields_size() != loop.iter_args_size()) {
    throw std::runtime_error("Loop " + op.name() + " yield/iter_arg mismatch.");
  }
  // Read every yield into a temporary before assigning, exactly like the
  // interpreter reads them before the scope closes.
  for (int i = 0; i < loop.body().yields_size(); i++) {
    line(indent + 1, "int64_t __y" + std::to_string(i) + " = " +
                         scalar_expr(loop.body().yields(i)) + ";");
  }
  for (int i = 0; i < loop.body().yields_size(); i++) {
    line(indent + 1, iter_vars[i] + " = __y" + std::to_string(i) + ";");
  }

  scopes_.pop_back();
  loop_depth_--;
  line(indent, "}");

  env_.pop();
  scopes_.pop_back();

  // The loop's outputs are the final iter values, visible to what follows.
  for (int i = 0;
       i < std::min<int>(op.outputs_size(), static_cast<int>(iter_vars.size()));
       i++) {
    bind(op.outputs(i).name(), iter_vars[i]);
    env_.define(op.outputs(i).name(), init_vals[i]);
  }
}

void CEmitter::emit_while(const voyager::Operation& op,
                          const voyager::WhileLoop& loop, int indent) {
  line(indent, "/* " + op.name() + " */");

  // Initials resolve in the enclosing scope (see emit_for).
  std::vector<std::string> init_exprs;
  std::vector<Scalar> init_vals;
  for (const auto& arg : loop.iter_args()) {
    init_exprs.push_back(scalar_expr(arg.initial()));
    init_vals.push_back(eval(arg.initial(), env_));
  }

  scopes_.push_back({});
  env_.push();

  std::vector<std::string> iter_vars;
  for (int i = 0; i < loop.iter_args_size(); i++) {
    const std::string c_name = declare(loop.iter_args(i).name(), indent, false);
    line(indent, "int64_t " + c_name + " = " + init_exprs[i] + ";");
    iter_vars.push_back(c_name);
    env_.define(loop.iter_args(i).name(), init_vals[i]);
  }

  const bool outermost = loop_depth_ == 0;
  const bool bounded =
      max_tiles_ > 0 && bounded_ && bounded_->count(&op) && outermost;
  const std::string trips = declare("__trips_" + op.name(), indent, false);
  if (bounded) line(indent, "int64_t " + trips + " = 0;");

  line(indent, "while (1) {");
  loop_depth_++;
  scopes_.push_back({});
  for (int i = 0; i < loop.iter_args_size(); i++) {
    bind(loop.iter_args(i).name(), iter_vars[i]);
  }

  if (bounded) {
    line(indent + 1,
         "if (" + trips + "++ >= " + std::to_string(max_tiles_) + "LL) break;");
  }

  if (loop.condition().yields_size() != 1) {
    throw std::runtime_error("While " + op.name() +
                             " condition must yield one value.");
  }
  emit_ops(loop.condition().ops(), indent + 1);
  line(indent + 1,
       "if (!(" + scalar_expr(loop.condition().yields(0)) + ")) break;");

  emit_ops(loop.body().ops(), indent + 1);

  if (loop.body().yields_size() != loop.iter_args_size()) {
    throw std::runtime_error("While " + op.name() +
                             " yield/iter_arg mismatch.");
  }
  for (int i = 0; i < loop.body().yields_size(); i++) {
    line(indent + 1, "int64_t __y" + std::to_string(i) + " = " +
                         scalar_expr(loop.body().yields(i)) + ";");
  }
  for (int i = 0; i < loop.body().yields_size(); i++) {
    line(indent + 1, iter_vars[i] + " = __y" + std::to_string(i) + ";");
  }

  scopes_.pop_back();
  loop_depth_--;
  line(indent, "}");

  env_.pop();
  scopes_.pop_back();
  for (int i = 0;
       i < std::min<int>(op.outputs_size(), static_cast<int>(iter_vars.size()));
       i++) {
    bind(op.outputs(i).name(), iter_vars[i]);
    env_.define(op.outputs(i).name(), init_vals[i]);
  }
}

void CEmitter::emit_cond(const voyager::Operation& op,
                         const voyager::CondOp& cond, int indent) {
  const std::string predicate = scalar_expr(cond.predicate());
  const bool taken = to_bool(eval(cond.predicate(), env_));

  // Scalar results are assigned by both branches into variables declared
  // before the if, exactly like the interpreter defines them in the
  // enclosing scope.
  std::vector<std::string> out_vars;
  for (const auto& output : op.outputs()) {
    const std::string c_name = declare(output.name(), indent, false);
    line(indent, "int64_t " + c_name + " = 0;");
    out_vars.push_back(c_name);
  }

  const ScalarEnv before = env_;
  ScalarEnv after_taken = env_;

  auto emit_region = [&](const voyager::Region& region, bool is_taken,
                         int region_indent) {
    scopes_.push_back({});
    env_.push();
    emit_ops(region.ops(), region_indent);
    std::vector<Scalar> yielded;
    for (int i = 0; i < std::min<int>(region.yields_size(),
                                      static_cast<int>(out_vars.size()));
         i++) {
      line(region_indent,
           out_vars[i] + " = " + scalar_expr(region.yields(i)) + ";");
      yielded.push_back(eval(region.yields(i), env_));
    }
    env_.pop();
    scopes_.pop_back();
    if (is_taken) {
      after_taken = env_;
      for (size_t i = 0; i < yielded.size(); i++) {
        after_taken.define(op.outputs(static_cast<int>(i)).name(), yielded[i]);
      }
    }
  };

  const bool outer_speculative = speculative_;
  line(indent, "if (" + predicate + ") { /* " + op.name() + " */");
  speculative_ = outer_speculative || !taken;
  emit_region(cond.true_region(), taken, indent + 1);
  line(indent, "} else {");
  env_ = before;
  speculative_ = outer_speculative || taken;
  emit_region(cond.false_region(), !taken, indent + 1);
  line(indent, "}");
  speculative_ = outer_speculative;

  // Continue concrete evaluation along the taken arm.
  env_ = after_taken;
  for (const auto& output : op.outputs()) {
    if (!env_.bound(output.name())) env_.define(output.name(), int64_t{0});
  }
}

// ---------------------------------------------------------------------------
// Dispatch: baseline serialization + probe-located runtime patches + sends
// ---------------------------------------------------------------------------

void CEmitter::emit_dispatch(const voyager::Operation& op, int indent) {
  // Baseline params under the concrete iteration-0 env.
  std::deque<BaseParams*> params;
  map_operation(op, env_, params);
  const auto baseline = serialize_params(params);
  for (auto* param : params) delete param;
  params.clear();

  // Locate env-dependent fields by probing each referenced scalar.
  const auto probe_names = collect_ref_scalars(op);
  std::vector<PatchField> fields;

  struct Probe {
    int64_t delta;
    std::vector<SerializedParam> serialized;
  };
  std::map<std::string, std::vector<Probe>> all_probes;

  auto try_probe = [&](const std::string& name, int64_t base_value,
                       int64_t delta, std::vector<Probe>* probes) -> bool {
    for (const auto& probe : *probes) {
      if (probe.delta == delta) return true;
    }
    ScalarEnv probe_env = env_;
    probe_env.define(name, base_value + delta);
    try {
      std::deque<BaseParams*> probe_params;
      map_operation(op, probe_env, probe_params);
      auto serialized = serialize_params(probe_params);
      for (auto* param : probe_params) delete param;
      probes->push_back({delta, std::move(serialized)});
      return true;
    } catch (const std::exception&) {
      return false;  // out of range for this scalar
    }
  };

  for (const auto& name : probe_names) {
    if (!env_.bound(name)) {
      throw std::runtime_error("Dispatch " + op.name() +
                               " references unbound scalar " + name);
    }
    const int64_t base_value = to_int(env_.lookup(name));
    auto& probes = all_probes[name];

    // A few small deltas establish affinity; resolve()'s range checks reject
    // the ones that leave the buffer.
    for (int64_t delta : {int64_t{1}, int64_t{-1}, int64_t{2}, int64_t{3}}) {
      try_probe(name, base_value, delta, &probes);
    }
    if (probes.empty()) {
      throw std::runtime_error("Dispatch " + op.name() +
                               ": no in-range probe " + "delta for scalar " +
                               name);
    }

    // Then probe the accepted EXTREMES in both directions (double until
    // rejected, then bisect to the boundary). Runtime values are themselves
    // range-checked by resolve(), so the union of bits flipped across the
    // extremes covers every bit a field can take at run time -- without this
    // a non-power-of-two extent would leave high field bits unpatched and
    // patch_bits would silently truncate.
    for (const int64_t direction : {int64_t{1}, int64_t{-1}}) {
      int64_t good = 0;
      int64_t step = direction;
      while (std::abs(step) <= (int64_t{1} << 22)) {
        ScalarEnv probe_env = env_;
        probe_env.define(name, base_value + step);
        try {
          std::deque<BaseParams*> probe_params;
          map_operation(op, probe_env, probe_params);
          for (auto* param : probe_params) delete param;
        } catch (const std::exception&) {
          break;
        }
        good = step;
        step *= 2;
      }
      if (good == 0) continue;
      int64_t bad = step;
      while (std::abs(bad - good) > 1) {
        const int64_t mid = good + (bad - good) / 2;
        ScalarEnv probe_env = env_;
        probe_env.define(name, base_value + mid);
        try {
          std::deque<BaseParams*> probe_params;
          map_operation(op, probe_env, probe_params);
          for (auto* param : probe_params) delete param;
          good = mid;
        } catch (const std::exception&) {
          bad = mid;
        }
      }
      try_probe(name, base_value, good, &probes);
    }

    // Union of differing bit-runs across all probes.
    std::map<std::pair<size_t, size_t>, size_t> bit_union;  // (param,bit)->1
    for (const auto& probe : probes) {
      for (const auto& run : diff_runs(baseline, probe.serialized)) {
        for (size_t b = 0; b < run.len; b++) {
          bit_union[{run.param_idx, run.off + b}] = 1;
        }
      }
    }
    // Merge into maximal runs.
    std::vector<BitRun> runs;
    for (auto it = bit_union.begin(); it != bit_union.end();) {
      const size_t p = it->first.first;
      const size_t start = it->first.second;
      size_t end = start;
      while (it != bit_union.end() && it->first.first == p &&
             it->first.second == end) {
        ++it;
        end++;
      }
      runs.push_back({p, start, end - start});
    }

    for (const auto& run : runs) {
      if (run.len > 64) {
        throw std::runtime_error("Dispatch " + op.name() + ": field wider " +
                                 "than 64 bits for scalar " + name);
      }
      const int64_t stored_base = static_cast<int64_t>(
          extract_bits(baseline[run.param_idx].bytes, run.off, run.len));

      // Per-unit coefficient from the first probe; verify every other probe
      // agrees (affine in the scalar).
      int64_t coefficient = 0;
      bool have = false;
      for (const auto& probe : probes) {
        const int64_t stored = static_cast<int64_t>(extract_bits(
            probe.serialized[run.param_idx].bytes, run.off, run.len));
        const int64_t diff = stored - stored_base;
        if (!have) {
          if (diff % probe.delta != 0) {
            throw std::runtime_error("Dispatch " + op.name() +
                                     ": non-affine field for " + name);
          }
          coefficient = diff / probe.delta;
          have = true;
        } else if (diff != coefficient * probe.delta) {
          throw std::runtime_error("Dispatch " + op.name() +
                                   ": non-affine field for " + name);
        }
      }
      if (coefficient == 0) continue;  // spurious (aliased) run

      // Merge with existing fields from other scalars. Runs that overlap but
      // do not coincide widen the field (an address affine in two scalars
      // with different strides flips different bit windows); coefficients
      // shift with the window.
      bool merged = false;
      for (auto& field : fields) {
        if (field.param_idx != run.param_idx) continue;
        const bool overlap =
            run.off < field.off + field.len && field.off < run.off + run.len;
        if (!overlap) continue;

        const size_t new_off = std::min(field.off, run.off);
        const size_t new_end =
            std::max(field.off + field.len, run.off + run.len);
        if (new_end - new_off > 64) {
          throw std::runtime_error("Dispatch " + op.name() +
                                   ": merged field wider than 64 bits.");
        }
        for (auto& [scalar, c] : field.coeff) {
          c *= int64_t{1} << (field.off - new_off);
        }
        field.coeff[name] += coefficient * (int64_t{1} << (run.off - new_off));
        field.off = new_off;
        field.len = new_end - new_off;
        field.base = static_cast<int64_t>(
            extract_bits(baseline[run.param_idx].bytes, new_off, field.len));
        merged = true;
        break;
      }
      if (!merged) {
        PatchField field;
        field.param_idx = run.param_idx;
        field.off = run.off;
        field.len = run.len;
        field.base = stored_base;
        field.coeff[name] = coefficient;
        fields.push_back(field);
      }
    }
  }

  // Generation-time self-verification: applying the patch formula to the
  // baseline must reproduce every probe's serialization bit-exactly. This
  // catches anything the affine model missed before it can ship as silently
  // wrong firmware.
  for (const auto& [name, probes] : all_probes) {
    for (const auto& probe : probes) {
      auto predicted = baseline;
      for (const auto& field : fields) {
        int64_t value = field.base;
        const auto found = field.coeff.find(name);
        if (found != field.coeff.end()) value += found->second * probe.delta;
        for (size_t b = 0; b < field.len; b++) {
          const size_t bit = field.off + b;
          unsigned char& byte = predicted[field.param_idx].bytes[bit / 8];
          const unsigned char mask =
              static_cast<unsigned char>(1u << (bit % 8));
          if ((static_cast<uint64_t>(value) >> b) & 1) {
            byte |= mask;
          } else {
            byte &= static_cast<unsigned char>(~mask);
          }
        }
      }
      for (size_t p = 0; p < baseline.size(); p++) {
        if (predicted[p].bytes != probe.serialized[p].bytes) {
          throw std::runtime_error(
              "Dispatch " + op.name() + ": patch formula fails to reproduce " +
              "the probe at " + name + " + " + std::to_string(probe.delta) +
              " (params blob " + std::to_string(p) + ") -- refusing to emit.");
        }
      }
    }
  }

  // --- static baseline blobs at file scope ---
  const std::string prefix = sanitize(op.name());
  std::vector<std::string> blob_names;
  for (size_t i = 0; i < baseline.size(); i++) {
    const std::string blob = prefix + "_params_" + std::to_string(i);
    blob_names.push_back(blob);
    decls_ << "static unsigned char " << blob << "[] = {";
    for (size_t j = 0; j < baseline[i].bytes.size(); j++) {
      if (j % 12 == 0) decls_ << "\n\t";
      decls_ << "0x" << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<unsigned>(baseline[i].bytes[j]) << std::dec;
      if (j + 1 != baseline[i].bytes.size()) decls_ << ", ";
    }
    decls_ << "\n};\n\n";
  }

  // --- runtime patches, then the sends, in deque order ---
  line(indent, "/* " + op.name() + " */");
  if (!in_commit_) {
    // Harness.cc:688, the pre-dispatch drain: a synchronous op waits on none
    // of the semaphores the in-flight commits will post, so only a drain
    // orders its fetches after their writes.
    line(indent, "wait_for_accelerator_done();");
  }
  for (const auto& field : fields) {
    std::string expr;
    int64_t constant = field.base;
    for (const auto& [name, coefficient] : field.coeff) {
      constant -= coefficient * to_int(env_.lookup(name));
      expr += " + " + std::to_string(coefficient) + "LL * " + ref(name);
    }
    line(indent, "patch_bits(" + blob_names[field.param_idx] + ", " +
                     std::to_string(field.off) + ", " +
                     std::to_string(field.len) + ", (uint64_t)(" +
                     std::to_string(constant) + "LL" + expr + "));");
  }
  // The synchronous post-drain below must observe each invocation group
  // actually run: ACCELERATOR_RUNNING alone cannot tell granted-but-unstarted
  // from finished, and it dips between the groups of a multi-pass dispatch.
  // Record each group's closing unit (the last one to start, in
  // Harness::dispatch_params' chunking) so the firmware can wait for its
  // inflight count to rise before draining.
  std::vector<std::string> group_close_regs;
  std::vector<size_t> group_ends;  // one past each group's last param set
  for (size_t i = 0; i < baseline.size();) {
    std::string reg;
    if (baseline[i].kind == kMatrixParams) {
      bool group_mvm = false;
#if SUPPORT_MVM
      group_mvm = baseline[i].is_fc;
#endif
      reg = group_mvm ? "MVM_UNIT_OP_INFLIGHT" : "MATRIX_UNIT_OP_INFLIGHT";
      i++;
    }
    if (i < baseline.size() && baseline[i].kind == kVectorParams) {
      reg = "VECTOR_UNIT_OP_INFLIGHT";
      i += 2;  // VectorParams + VectorInstructionConfig
    }
    group_close_regs.push_back(reg);
    group_ends.push_back(i);
  }

  size_t group_idx = 0;
  for (size_t i = 0; i < baseline.size(); i++) {
    // Routing mirrors Harness::dispatch_params: an is_fc MatrixParams goes to
    // the matrix-vector unit only when the build has one, else it falls back
    // to the plain matrix unit.
    bool to_mvm = false;
#if SUPPORT_MVM
    to_mvm = baseline[i].is_fc;
#endif
    switch (baseline[i].kind) {
      case kMatrixParams:
        line(indent, std::string(to_mvm ? "send_matrix_vector_unit_params("
                                        : "send_matrix_unit_params(") +
                         blob_names[i] + ");");
        break;
      case kVectorParams:
        line(indent, "send_vector_params(" + blob_names[i] + ");");
        break;
      case kVectorConfig:
        line(indent, "send_vector_instructions(" + blob_names[i] + ");");
        break;
    }
    // Each group's wait must follow its own sends, not trail the whole
    // dispatch: a later group's sends block on MMIO backpressure while
    // earlier groups run, so trailing waits would miss their starts and
    // spin forever. This mirrors dispatch_params exactly -- the
    // between-groups drain of Harness.cc:395, with the last group's wait
    // doubling as execute()'s post-dispatch drain (Harness.cc:690): nothing
    // later -- in particular the next dispatch's params, which arm the
    // units' fetch front-ends the moment they arrive -- may be sent until
    // this dispatch has fully retired.
    if (!in_commit_ && group_idx < group_ends.size() &&
        i + 1 == group_ends[group_idx]) {
      line(indent,
           "wait_for_dispatch_retired(" + group_close_regs[group_idx] + ");");
      group_idx++;
    }
  }
}

// ---------------------------------------------------------------------------
// Translation unit
// ---------------------------------------------------------------------------

std::string CEmitter::emit_layer(const Model::Selection& selection) {
  decls_.str("");
  body_.str("");
  env_ = ScalarEnv();
  scopes_.clear();
  scopes_.push_back({});
  name_counts_.clear();
  bounded_ = &selection.bounded;
  loop_depth_ = 0;
  max_tiles_ = getenv_int("MAX_TILES", 0);

  for (const auto* op : selection.ops) {
    // Selection.bounded gates the MAX_TILES clamp exactly as it does in the
    // interpreter; non-outermost loops are never clamped.
    emit_operation(*op, 1);
  }

  std::ostringstream out;
  out << "#include <stddef.h>\n";
  out << "#include <stdint.h>\n";
  out << "#include <stdio.h>\n\n";
  out << "#include \"mmio.h\"\n";
  out << "#include \"patch_bits.h\"\n";
  out << "#include \"run_voyager_operation.h\"\n";
  out << "#include \"traps.h\"\n";
  out << "#include \"voyager_address.h\"\n\n";
  out << decls_.str();
  out << "int main() {\n";
  out << "\tenable_interrupts();\n";
  out << "\tenable_semaphore_wait();\n\n";
  out << "\treg_write64(VOYAGER_BASE_ADDR, SRAM_BASE);\n\n";
  out << body_.str();
  out << "\n\tprintf(\"All params sent!\\n\");\n\n";
  out << "\twait_for_accelerator_done();\n\n";
  out << "\tprintf(\"Operation finished!\\n\");\n";
  out << "\tprintf(\"Matrix Unit Runtime     : %lu cycles\\n\", "
         "reg_read64(MATRIX_UNIT_CYCLE_COUNT));\n";
  out << "\tprintf(\"Vector Unit Runtime     : %lu cycles\\n\", "
         "reg_read64(VECTOR_UNIT_CYCLE_COUNT));\n";
  // Baked at generation time: the RISC-V firmware compile has no SUPPORT_*
  // defines, so an emitted #if would never be true there.
#if SUPPORT_MVM
  out << "\tprintf(\"MVM Unit Runtime        : %lu cycles\\n\", "
         "reg_read64(MVM_UNIT_CYCLE_COUNT));\n";
#endif
  out << "\tprintf(\"Accelerator Runtime     : %lu cycles\\n\", "
         "reg_read64(ACCELERATOR_CYCLE_COUNT));\n";
  out << "}\n";
  return out.str();
}
