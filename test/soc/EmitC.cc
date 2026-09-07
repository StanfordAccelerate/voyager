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
#include "test/soc/firmware/common/host_request.h"
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
  bool is_spmm = false;
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
      sp.is_spmm = mp->is_spmm;
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

// A field's movement per unit of one scalar, exactly. A whole number is
// den == 1; a field written in coarser units than the scalar counts in
// (slice_start is a column / OC_DIMENSION) needs the fraction.
struct Ratio {
  int64_t num = 0;
  int64_t den = 1;
};

int64_t gcd_i64(int64_t a, int64_t b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  while (b != 0) {
    const int64_t t = a % b;
    a = b;
    b = t;
  }
  return a == 0 ? 1 : a;
}

Ratio make_ratio(int64_t num, int64_t den) {
  if (den < 0) {
    num = -num;
    den = -den;
  }
  const int64_t g = gcd_i64(num, den);
  return Ratio{num / g, den / g};
}

Ratio add_ratio(const Ratio& a, const Ratio& b) {
  return make_ratio(a.num * b.den + b.num * a.den, a.den * b.den);
}

// Used when two runs merge into one wider field.
Ratio shift_ratio(const Ratio& a, size_t shift) {
  return make_ratio(a.num << shift, a.den);
}

// One runtime-patched field of one params blob.
struct PatchField {
  size_t param_idx;
  size_t off;
  size_t len;
  int64_t base;                        // value in the baseline blob
  std::map<std::string, Ratio> coeff;  // ssa scalar -> per-unit delta
};

std::string sanitize(const std::string& name) {
  std::string out;
  for (char c : name) out += (isalnum(c) || c == '_') ? c : '_';
  if (out.empty() || isdigit(out[0])) out = "v_" + out;
  return out;
}

// The prim inside `region`, at any nesting, whose output is `name`.
const voyager::PrimOp* find_scalar_prim(const voyager::Region& region,
                                        const std::string& name);

const voyager::PrimOp* find_scalar_prim_in_op(const voyager::Operation& op,
                                              const std::string& name) {
  if (op.op_type_case() == voyager::Operation::kPrim) {
    for (const auto& output : op.outputs()) {
      if (output.name() == name) return &op.prim();
    }
    return nullptr;
  }
  if (op.op_type_case() == voyager::Operation::kCond) {
    if (const auto* p = find_scalar_prim(op.cond().true_region(), name)) {
      return p;
    }
    return find_scalar_prim(op.cond().false_region(), name);
  }
  if (op.op_type_case() == voyager::Operation::kAsync) {
    return find_scalar_prim(op.async().body(), name);
  }
  return nullptr;
}

const voyager::PrimOp* find_scalar_prim(const voyager::Region& region,
                                        const std::string& name) {
  for (const auto& op : region.ops()) {
    if (const auto* p = find_scalar_prim_in_op(op, name)) return p;
  }
  return nullptr;
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
  if (surveying_) return;  // the survey walk's output is discarded
  for (int i = 0; i < indent; i++) body_ << "\t";
  body_ << text << "\n";
}

// ---------------------------------------------------------------------------
// Structure queries
// ---------------------------------------------------------------------------

// Whether any conditional in this subtree guards a dispatch. Only such a layer
// can have a dispatch the emitter reaches speculatively, and only such a layer
// is worth surveying -- the survey walks the tile loop for real, evaluating
// arms the single emitting walk never enters concretely, which is needless
// exposure anywhere it cannot pay off.
bool CEmitter::cond_guards_dispatch(const voyager::Operation& op) const {
  switch (op.op_type_case()) {
    case voyager::Operation::kCond:
      for (const auto& child : op.cond().true_region().ops()) {
        if (contains_dispatch(child)) return true;
      }
      for (const auto& child : op.cond().false_region().ops()) {
        if (contains_dispatch(child)) return true;
      }
      // A nested conditional may still guard one.
      for (const auto& child : op.cond().true_region().ops()) {
        if (cond_guards_dispatch(child)) return true;
      }
      for (const auto& child : op.cond().false_region().ops()) {
        if (cond_guards_dispatch(child)) return true;
      }
      return false;
    case voyager::Operation::kLoop: {
      const auto& body = op.loop().has_for_loop()
                             ? op.loop().for_loop().body()
                             : op.loop().while_loop().body();
      for (const auto& child : body.ops()) {
        if (cond_guards_dispatch(child)) return true;
      }
      if (op.loop().has_while_loop()) {
        for (const auto& child : op.loop().while_loop().condition().ops()) {
          if (cond_guards_dispatch(child)) return true;
        }
      }
      return false;
    }
    case voyager::Operation::kAsync:
      for (const auto& child : op.async().body().ops()) {
        if (cond_guards_dispatch(child)) return true;
      }
      return false;
    default:
      return false;
  }
}

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

bool CEmitter::contains_work(const voyager::Operation& op) const {
  switch (op.op_type_case()) {
    case voyager::Operation::kPrim: {
      const auto& prim = op.prim();
      const std::string& target = prim.target();
      if (host_ordinals_.count(&prim) > 0) return true;
      if (target == "voyager::zeros" || target == "voyager::fill" ||
          target == "voyager::async_wait") {
        return true;
      }
      return is_host_bookkeeping(prim) || is_datapath(op);
    }
    case voyager::Operation::kFused:
      return is_datapath(op);
    case voyager::Operation::kLoop: {
      const auto& body = op.loop().has_for_loop()
                             ? op.loop().for_loop().body()
                             : op.loop().while_loop().body();
      for (const auto& child : body.ops()) {
        if (contains_work(child)) return true;
      }
      if (op.loop().has_while_loop()) {
        for (const auto& child : op.loop().while_loop().condition().ops()) {
          if (contains_work(child)) return true;
        }
      }
      return false;
    }
    case voyager::Operation::kCond:
      for (const auto& child : op.cond().true_region().ops()) {
        if (contains_work(child)) return true;
      }
      for (const auto& child : op.cond().false_region().ops()) {
        if (contains_work(child)) return true;
      }
      return false;
    case voyager::Operation::kAsync:
      if (op.async().dependencies_size() > 0 || op.async().has_post()) {
        return true;
      }
      for (const auto& child : op.async().body().ops()) {
        if (contains_work(child)) return true;
      }
      return false;
    default:
      return false;
  }
}

std::set<std::string> CEmitter::collect_ref_scalars(
    const voyager::Operation& op) const {
  return request_scalar_names(op);
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
      // Data movement and host tensor ops: requests to the testbench.
      const auto host = host_ordinals_.find(&prim);
      if (host != host_ordinals_.end()) {
        emit_host_request(op, prim, host->second, indent);
        return;
      }
      // Address assignment is the compiler's; a float allocation's contents
      // are junk nothing reads before writing (the integer ones are zero
      // fills, in the table above).
      if (target == "voyager::alloc") return;
      if (target == "voyager::zeros") {
        emit_semaphore_zeros(op, indent);
        return;
      }
      if (target == "voyager::fill") {
        emit_semaphore_fill(op, prim, indent);
        return;
      }
      if (target == "voyager::async_wait") {
        emit_async_wait(op, prim, indent);
        return;
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
      if (is_host_bookkeeping(prim)) {
        emit_host_bookkeeping(op, prim, indent);
        return;
      }
      if (is_datapath(op)) {
        emit_dispatch(op, indent);
      }
      return;
    }
    case voyager::Operation::kFused:
      if (is_datapath(op)) emit_dispatch(op, indent);
      return;
    case voyager::Operation::kLoop:
      if (!contains_work(op)) return;
      if (op.loop().has_for_loop()) {
        emit_for(op, op.loop().for_loop(), indent);
      } else {
        emit_while(op, op.loop().while_loop(), indent);
      }
      return;
    case voyager::Operation::kCond:
      if (contains_work(op) || op.outputs_size() > 0) {
        emit_cond(op, op.cond(), indent);
      }
      return;
    case voyager::Operation::kAsync:
      emit_async(op, indent);
      return;
    default:
      return;
  }
}

// ---------------------------------------------------------------------------
// Semaphores and requests
// ---------------------------------------------------------------------------

std::pair<std::string, std::string> CEmitter::sem_cells(
    const voyager::TensorBox& box) {
  auto it = sem_names_.find(box.node());
  if (it == sem_names_.end()) {
    std::string stem = "__sem_" + sanitize(box.node());
    int& count = name_counts_[stem];
    if (count > 0) stem += "_x" + std::to_string(count);
    count++;
    const int64_t slots = semaphore_slots(box);
    // Cache-line apart from anything the firmware writes: the testbench's
    // increments are read-modify-writes of a whole macro row.
    decls_ << "static volatile int64_t " << stem << "_tb[" << slots
           << "] __attribute__((aligned(64)));\n";
    decls_ << "static int64_t " << stem << "_fw[" << slots << "];\n\n";
    it = sem_names_.emplace(box.node(), stem).first;
  }
  return {it->second + "_tb", it->second + "_fw"};
}

std::string CEmitter::sem_slot_expr(const voyager::TensorBoxRef& ref,
                                    const std::string& who) const {
  const voyager::TensorBox& box = ref.box();
  const int dims = box.shape_size();
  const bool banked = banks_of(box) > 1;
  if (dims == 0 || ref.offsets_size() == 0) {
    // select_bank: a scalar semaphore has one slot per bank.
    if (ref.offsets_size() == 0 || !banked) return "0LL";
    return "(" + scalar_expr(ref.offsets(0)) + ")";
  }
  // A semaphore array: [bank, *dims] flattened bank-major.
  const int bank_dims = banked ? 1 : 0;
  if (ref.offsets_size() != dims + bank_dims) {
    throw std::runtime_error(who + ": semaphore ref rank does not match " +
                             box.node());
  }
  std::string index = "0LL";
  int64_t elements = 1;
  for (int d = 0; d < dims; d++) {
    index = "(" + index + ") * " + std::to_string(box.shape(d)) + "LL + (" +
            scalar_expr(ref.offsets(d + bank_dims)) + ")";
    elements *= box.shape(d);
  }
  const std::string bank =
      banked ? "(" + scalar_expr(ref.offsets(0)) + ")" : "0LL";
  return bank + " * " + std::to_string(elements) + "LL + (" + index + ")";
}

void CEmitter::emit_semaphore_zeros(const voyager::Operation& op, int indent) {
  if (op.outputs_size() != 1 || !op.outputs(0).has_tensor_box() ||
      !is_semaphore(op.outputs(0).tensor_box())) {
    throw std::runtime_error("voyager::zeros " + op.name() +
                             " declares neither a buffer nor a semaphore.");
  }
  const auto& box = op.outputs(0).tensor_box();
  const auto [tb, fw] = sem_cells(box);
  const std::string slots = std::to_string(semaphore_slots(box)) + "LL";
  line(indent, "/* " + op.name() + ": semaphore " + box.node() + " */");
  line(indent, "for (int64_t __s = 0; __s < " + slots + "; __s++) {");
  line(indent + 1, tb + "[__s] = 0;");
  line(indent + 1, fw + "[__s] = 0;");
  line(indent, "}");
}

void CEmitter::emit_semaphore_fill(const voyager::Operation& op,
                                   const voyager::PrimOp& prim, int indent) {
  if (op.outputs_size() != 1 || !op.outputs(0).has_tensor_box() ||
      !is_semaphore(op.outputs(0).tensor_box())) {
    throw std::runtime_error("voyager::fill " + op.name() +
                             " targets a buffer that is not a semaphore.");
  }
  const auto& box = op.outputs(0).tensor_box();
  const auto [tb, fw] = sem_cells(box);
  const std::string value = scalar_expr(prim.kwargs().at("value").scalar());
  const std::string slots = std::to_string(semaphore_slots(box)) + "LL";
  // A fill seeds credits: it adds to every slot rather than setting it.
  line(indent, "/* " + op.name() + ": seed semaphore " + box.node() + " */");
  line(indent, "for (int64_t __s = 0; __s < " + slots + "; __s++) {");
  line(indent + 1, fw + "[__s] += (" + value + ");");
  line(indent, "}");
}

void CEmitter::emit_async_wait(const voyager::Operation& op,
                               const voyager::PrimOp& prim, int indent) {
  const auto& ref = prim.kwargs().at("semaphore").tensor_box();
  const auto [tb, fw] = sem_cells(ref.box());
  const std::string slot = sem_slot_expr(ref, "async_wait " + op.name());
  line(indent, "host_wait(&" + tb + "[" + slot + "], &" + fw + "[" + slot +
                   "]); /* " + op.name() + " */");
}

void CEmitter::emit_async(const voyager::Operation& op, int indent) {
  // A commit: wait every dependency, run the body's dispatches without
  // draining, and have the testbench credit the post once the units have
  // retired everything issued so far.
  const auto& async = op.async();
  for (const auto& dep : async.dependencies()) {
    const auto [tb, fw] = sem_cells(dep.box());
    const std::string slot = sem_slot_expr(dep, "commit " + op.name());
    line(indent, "host_wait(&" + tb + "[" + slot + "], &" + fw + "[" + slot +
                     "]); /* " + op.name() + " depends on " + dep.box().node() +
                     " */");
  }
  const bool was_in_commit = in_commit_;
  in_commit_ = true;
  emit_ops(async.body().ops(), indent);
  in_commit_ = was_in_commit;
  if (async.has_post()) {
    const auto [tb, fw] = sem_cells(async.post().box());
    const std::string slot = sem_slot_expr(async.post(), "commit " + op.name());
    line(indent, "host_post(&" + tb + "[" + slot + "], 1LL); /* " + op.name() +
                     " retires */");
  }
}

void CEmitter::emit_host_request(const voyager::Operation& op,
                                 const voyager::PrimOp& prim, size_t ordinal,
                                 int indent) {
  const HostOp& host = host_table_.at(ordinal);
  const std::string ord = std::to_string(ordinal) + "ULL";
  if (host.kind == HostOp::kZero) {
    line(indent, "host_zero(" + ord + "); /* " + op.name() + " */");
    return;
  }

  // The prim's scalar operands, by value, in request_scalar_names' order,
  // stored straight into the mailbox slot: this core has no data cache, so
  // staging them anywhere first costs a bus transaction per value.
  const std::set<std::string> names = request_scalar_names(op);
  if (names.size() > HOST_REQ_MAX_ARGS) {
    throw std::runtime_error(
        op.name() + " references " + std::to_string(names.size()) +
        " scalars; host_request.h allows " + std::to_string(HOST_REQ_MAX_ARGS));
  }
  std::string slot = "host_slot()";
  if (!names.empty()) {
    line(indent, "{");
    indent++;
    line(indent, "volatile host_request_t *__r = host_slot();");
    size_t i = 0;
    for (const auto& name : names) {
      line(indent,
           "__r->args[" + std::to_string(i++) + "] = " + ref(name) + ";");
    }
    slot = "__r";
  }
  const std::string count = std::to_string(names.size());

  if (host.kind == HostOp::kCopy) {
    // The copy's completion credits its slot semaphore post_count times.
    std::string cell = "(volatile int64_t *)0";
    const auto sem = prim.kwargs().find("semaphore");
    if (sem != prim.kwargs().end()) {
      const auto& ref = sem->second.tensor_box();
      const auto [tb, fw] = sem_cells(ref.box());
      cell =
          "&" + tb + "[" + sem_slot_expr(ref, "async_copy " + op.name()) + "]";
    }
    std::string amount = "1LL";
    const auto post_count = prim.kwargs().find("post_count");
    if (post_count != prim.kwargs().end()) {
      amount = scalar_expr(post_count->second.scalar());
    }
    line(indent, "host_copy(" + slot + ", " + ord + ", " + count + ", " + cell +
                     ", " + amount + "); /* " + op.name() + " */");
  } else {
    line(indent, "host_op(" + slot + ", " + ord + ", " + count + "); /* " +
                     op.name() + ": " + strip_namespace(prim.target()) + " */");
  }
  if (!names.empty()) line(indent - 1, "}");
}

void CEmitter::emit_scalar_prim(const voyager::Operation& op,
                                const voyager::PrimOp& prim, int indent) {
  if (op.outputs_size() != 1) {
    throw std::runtime_error("Scalar op " + op.name() +
                             " with multiple outputs.");
  }
  const std::string target = strip_namespace(prim.target());

  if (target == "_local_scalar_dense") {
    emit_scalar_load(op, prim, indent);
    return;
  }

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
  scalar_def_[op.outputs(0).name()] = &prim;
}

int64_t CEmitter::derived_step(const std::string& ssa_name, int depth) const {
  if (depth > 8) return 0;  // pathological chain; fall back to searching
  const auto counter = loop_counter_steps_.find(ssa_name);
  if (counter != loop_counter_steps_.end()) return counter->second;
  const auto found = scalar_def_.find(ssa_name);
  // Neither a known counter nor a scalar this walk defined: not derivable.
  // A guessed step the mapper rejects would be misread as invariance.
  if (found == scalar_def_.end()) return 0;

  const voyager::PrimOp& prim = *found->second;
  const std::string target = strip_namespace(prim.target());
  // A delinearized index wraps at its basis, but it only ever moves by whole
  // units, so its run-time values lie on the unit grid -- and a scalar scaled
  // off it (a column offset, getitem * 512) on that scale's grid, which is
  // what a fractional field coefficient needs to divide exactly.
  if (target == "delinearize_index") return 1;
  if (target != "mul" && target != "add" && target != "sub") return 0;
  if (prim.kwargs().count("input") == 0 || prim.kwargs().count("other") == 0) {
    return 0;
  }

  const voyager::ScalarValue& a = prim.kwargs().at("input").scalar();
  const voyager::ScalarValue& b = prim.kwargs().at("other").scalar();
  const bool a_node = a.value_case() == voyager::ScalarValue::kNode;
  const bool b_node = b.value_case() == voyager::ScalarValue::kNode;
  if (a_node == b_node) return 0;  // both varying or both constant

  const voyager::ScalarValue& varying = a_node ? a : b;
  const voyager::ScalarValue& constant = a_node ? b : a;
  const int64_t inner = derived_step(varying.node(), depth + 1);
  if (inner == 0) return 0;
  // Adding a constant shifts the sequence; only scaling changes its step.
  if (target != "mul") return inner;
  const int64_t factor = to_int(eval(constant, env_));
  return factor == 0 ? 0 : inner * factor;
}

CEmitter::CellWindow CEmitter::cell_window(const voyager::TensorBoxRef& ref,
                                           const std::string& who) const {
  const voyager::TensorBox& box = ref.box();
  if (!box.has_memory() ||
      box.memory().level() != voyager::MEMORY_LEVEL_SCRATCHPAD) {
    throw std::runtime_error(who + " touches a box outside the scratchpad.");
  }

  CellWindow cell;
  // Byte-aligned integer cells only.
  if (box.dtype() == "int32") {
    cell.c_type = "int32_t";
    cell.width = 4;
  } else if (box.dtype() == "int64") {
    cell.c_type = "int64_t";
    cell.width = 8;
  } else {
    throw std::runtime_error(who + " touches unsupported dtype " + box.dtype());
  }

  // resolve_window's addressing: an optional leading slot dimension strides
  // by the bank pitch, the rest row-major over the box's shape.
  const int rank = ref.offsets_size();
  const int bank_dims = box.bank_count() > 1 ? 1 : 0;
  // No offsets names the whole box, as select_bank reads it.
  if (rank != 0 && box.shape_size() != rank - bank_dims) {
    throw std::runtime_error(who + ": ref rank does not match the box.");
  }
  for (int d = 0; d < rank; d++) {
    if (ref.strides(d) != 1) {
      throw std::runtime_error(who + ": strided window.");
    }
  }
  std::vector<int64_t> byte_strides(rank, 0);
  if (bank_dims == 1) {
    byte_strides[0] = static_cast<int64_t>(box.bank_stride_bytes());
  }
  int64_t running = cell.width;
  for (int d = rank - 1; d >= bank_dims; d--) {
    byte_strides[d] = running;
    running *= box.shape(d - bank_dims);
  }

  cell.base = static_cast<int64_t>(box.memory().address());
  for (int d = 0; d < rank; d++) {
    const auto& offset = ref.offsets(d);
    if (offset.value_case() == voyager::ScalarValue::kIntValue) {
      cell.base += offset.int_value() * byte_strides[d];
    } else {
      cell.runtime_terms += " + " + std::to_string(byte_strides[d]) + "LL * (" +
                            scalar_expr(offset) + ")";
    }
  }

  // The window's extent: contiguous only, i.e. every dimension but the last
  // spans one element (a cell, or one row of cells).
  cell.count = 1;
  if (rank == 0) {
    for (int d = 0; d < box.shape_size(); d++) cell.count *= box.shape(d);
  } else {
    for (int d = 0; d < ref.sizes_size(); d++) {
      const int64_t size = ref.sizes(d);
      if (size != 1 && d != ref.sizes_size() - 1) {
        throw std::runtime_error(who + ": non-contiguous window.");
      }
      cell.count *= size;
    }
  }
  return cell;
}

void CEmitter::emit_host_bookkeeping(const voyager::Operation& op,
                                     const voyager::PrimOp& prim, int indent) {
  const std::string who = "host op " + op.name();
  const auto input = prim.kwargs().find("input");
  if (input == prim.kwargs().end() || !input->second.has_tensor_box()) {
    throw std::runtime_error(who + ": input is not a tensor box.");
  }
  const CellWindow src = cell_window(input->second.tensor_box(), who);

  if (op.outputs_size() != 1) {
    throw std::runtime_error(who + ": expected one output.");
  }
  const auto& output = op.outputs(0);
  CellWindow dst;
  if (output.has_destination()) {
    dst = cell_window(output.destination(), who);
  } else if (output.has_tensor_box()) {
    voyager::TensorBoxRef whole;
    *whole.mutable_box() = output.tensor_box();
    dst = cell_window(whole, who);
  } else {
    throw std::runtime_error(who + ": output has no box.");
  }
  if (src.count != dst.count || src.width != dst.width) {
    throw std::runtime_error(who + " must preserve dtype and size.");
  }

  const std::string target = strip_namespace(prim.target());
  const std::string s =
      "((volatile " + src.c_type + " *)(uintptr_t)(SRAM_BASE + " +
      std::to_string(src.base) + "LL" + src.runtime_terms + "))";
  const std::string d =
      "((volatile " + dst.c_type + " *)(uintptr_t)(SRAM_BASE + " +
      std::to_string(dst.base) + "LL" + dst.runtime_terms + "))";
  std::string rhs;
  if (target == "clone") {
    rhs = s + "[__i]";
  } else if (target == "add") {
    const std::string other = scalar_expr(prim.kwargs().at("other").scalar());
    std::string alpha = "1LL";
    const auto a = prim.kwargs().find("alpha");
    if (a != prim.kwargs().end()) alpha = scalar_expr(a->second.scalar());
    rhs = s + "[__i] + (" + dst.c_type + ")((" + alpha + ") * (" + other + "))";
  } else {
    throw std::runtime_error(who + ": unsupported host op " + target);
  }

  // In place, after the dispatch whose results it reads has retired.
  line(indent, "/* " + op.name() + ": " + target + " on " +
                   std::to_string(src.count) + " index cell(s) */");
  line(indent, "wait_for_accelerator_done();");
  line(indent, "for (int64_t __i = 0; __i < " + std::to_string(src.count) +
                   "LL; __i++) {");
  line(indent + 1, d + "[__i] = " + rhs + ";");
  line(indent, "}");
}

void CEmitter::emit_scalar_load(const voyager::Operation& op,
                                const voyager::PrimOp& prim, int indent) {
  const auto& argument = prim.kwargs().at("input");
  if (!argument.has_tensor_box()) {
    throw std::runtime_error("_local_scalar_dense " + op.name() +
                             ": input is not a tensor box.");
  }
  const CellWindow cell =
      cell_window(argument.tensor_box(), "_local_scalar_dense " + op.name());
  if (cell.count != 1) {
    throw std::runtime_error("_local_scalar_dense " + op.name() +
                             " reads more than one element.");
  }

  const std::string c_name = declare(op.outputs(0).name(), indent, false);
  line(indent, "int64_t " + c_name + " = (int64_t)*(volatile " + cell.c_type +
                   " *)(uintptr_t)(SRAM_BASE + " + std::to_string(cell.base) +
                   "LL" + cell.runtime_terms + ");");
  // The cells this op reads are zeroed before the first tile.
  env_.define(op.outputs(0).name(), int64_t{0});
  scalar_def_[op.outputs(0).name()] = &prim;
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
    // Not a scaled counter: it wraps at its basis.
    scalar_def_[op.outputs(d).name()] = &prim;
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
  loop_counter_steps_[loop.iv()] = step;
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

  if (surveying_ && outermost) {
    // The emitted C loop is written once, from the iteration-0 environment, so
    // a dispatch whose guard is false only at iteration 0 would never be seen.
    // Walk the tile loop for real here -- carrying iter_args across iterations
    // the way the interpreter does -- so concrete_env_ records those too. Only
    // the outermost loop, which MAX_TILES bounds; inner loops keep the single
    // walk rather than making the survey quadratic.
    if (loop.body().yields_size() != loop.iter_args_size()) {
      throw std::runtime_error("Loop " + op.name() +
                               " yield/iter_arg mismatch.");
    }
    // An unbounded tile loop can be hundreds of iterations; a guard that has
    // not come true in this many will fall back to the emission-point
    // environment rather than making generation quadratic.
    constexpr int kSurveyLimit = 64;
    int surveyed = 0;
    for (int64_t v = start;
         (step > 0 ? v < end : v > end) && surveyed < kSurveyLimit;
         v += step, surveyed++) {
      env_.define(loop.iv(), v);
      emit_ops(loop.body().ops(), indent + 1);
      std::vector<Scalar> yields;
      for (int i = 0; i < loop.body().yields_size(); i++) {
        yields.push_back(eval(loop.body().yields(i), env_));
      }
      for (int i = 0; i < loop.iter_args_size(); i++) {
        env_.define(loop.iter_args(i).name(), yields[i]);
      }
    }
    // Leave the environment as the single-walk path would, so what follows the
    // loop sees the same state in both walks.
    env_.define(loop.iv(), start);
    for (int i = 0; i < loop.iter_args_size(); i++) {
      env_.define(loop.iter_args(i).name(), init_vals[i]);
    }
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

  // A carried value the body yields as add(itself, constant) advances by
  // that constant. Derived before the body is walked, since the dispatches
  // inside it are probed during that walk.
  for (int i = 0; i < loop.iter_args_size(); i++) {
    const voyager::ScalarValue& yield = loop.body().yields(i);
    if (yield.value_case() != voyager::ScalarValue::kNode) continue;
    const voyager::PrimOp* def = find_scalar_prim(loop.body(), yield.node());
    if (def == nullptr || strip_namespace(def->target()) != "add") continue;
    if (def->kwargs().count("input") == 0 ||
        def->kwargs().count("other") == 0) {
      continue;
    }
    const voyager::ScalarValue& a = def->kwargs().at("input").scalar();
    const voyager::ScalarValue& b = def->kwargs().at("other").scalar();
    const std::string& carried = loop.iter_args(i).name();
    if (a.value_case() == voyager::ScalarValue::kNode && a.node() == carried &&
        b.value_case() == voyager::ScalarValue::kIntValue) {
      loop_counter_steps_[carried] = b.int_value();
    } else if (b.value_case() == voyager::ScalarValue::kNode &&
               b.node() == carried &&
               a.value_case() == voyager::ScalarValue::kIntValue) {
      loop_counter_steps_[carried] = a.int_value();
    }
  }

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
  if (surveying_) {
    // Record the first environment in which this dispatch's own guards hold.
    if (!speculative_) concrete_env_.emplace(&op, env_);
    return;
  }

  // Map and probe under an environment the program actually reaches. Inside a
  // cond arm the iteration-0 predicate does not take, env_ holds placeholders,
  // so the surveyed environment stands in; the patch expressions still name
  // the C variables in scope here, which carry the run-time values.
  struct EnvSwap {
    ScalarEnv* slot;
    ScalarEnv saved;
    bool active = false;
    ~EnvSwap() {
      if (active) *slot = saved;
    }
  } env_swap{&env_, env_};
  bool unsurveyed = false;
  if (speculative_) {
    // Absent means the survey -- which walks every iteration of the tile loop
    // the firmware runs -- never reached this dispatch with its guard true.
    const auto surveyed = concrete_env_.find(&op);
    if (surveyed != concrete_env_.end()) {
      env_ = surveyed->second;
      env_swap.active = true;
    } else {
      unsurveyed = true;
    }
  }

  // Baseline params under the concrete iteration-0 env.
  std::deque<BaseParams*> params;
  try {
    map_operation(op, env_, params);
  } catch (const std::exception& error) {
    if (unsurveyed) {
      // No iteration takes this arm and there is no environment that describes
      // it, so there are no honest params to send. The path should be dead;
      // say so loudly rather than shipping bytes that mean nothing, and let
      // the rest of the layer emit.
      line(indent, "printf(\"FATAL: unreachable dispatch " + op.name() +
                       "\\n\"); /* " + error.what() + " */");
      line(indent, "while (1) { }");
      return;
    }
    // Say which dispatch failed, and whether it was reached only
    // speculatively -- an arm the iteration-0 predicate does not take is
    // mapped under placeholder scalars, so a failure there says nothing
    // about the arm the hardware will run.
    throw std::runtime_error(std::string("mapping ") + op.name() +
                             (speculative_ ? " (speculative arm)" : "") + ": " +
                             error.what());
  }
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

    // A scalar scaled off a loop counter is sampled at its own advance,
    // which is legal by construction. Perturbing by 1 instead invents a
    // state the program never reaches and the mapper refuses.
    const int64_t step = derived_step(name);
    if (step != 0) {
      try_probe(name, base_value, step, &probes);
      try_probe(name, base_value, -step, &probes);
      if (!probes.empty()) derived_scalars_.insert(name);
      // An advance the mapper refuses is one the program never takes, so
      // the scalar holds base_value throughout and the baked bytes stand.
      if (probes.empty()) continue;
    }

    // A scalar read from memory at run time has no derivable step; search
    // for a delta the mapper accepts.
    if (probes.empty()) {
      for (int64_t delta : {int64_t{1}, int64_t{-1}, int64_t{2}, int64_t{3}}) {
        try_probe(name, base_value, delta, &probes);
      }
      for (int64_t g = 4; probes.empty() && g <= (int64_t{1} << 22); g <<= 1) {
        for (const int64_t direction : {int64_t{1}, int64_t{-1}}) {
          try_probe(name, base_value, direction * g, &probes);
        }
      }
    }
    if (probes.empty()) {
      throw std::runtime_error("Dispatch " + op.name() +
                               ": no in-range probe " + "delta for scalar " +
                               name);
    }
    // The extremes scan steps in these units to stay on the accepted grid.
    int64_t unit = 0;
    for (const auto& probe : probes) {
      const int64_t magnitude = std::abs(probe.delta);
      if (unit == 0 || magnitude < unit) unit = magnitude;
    }

    // Then probe the accepted EXTREMES in both directions (double until
    // rejected, then bisect to the boundary). Runtime values are themselves
    // range-checked by resolve(), so the union of bits flipped across the
    // extremes covers every bit a field can take at run time -- without this
    // a non-power-of-two extent would leave high field bits unpatched and
    // patch_bits would silently truncate.
    for (const int64_t direction : {int64_t{1}, int64_t{-1}}) {
      int64_t good = 0;
      int64_t step = direction * unit;
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
      while (std::abs(bad - good) > unit) {
        const int64_t half = (bad - good) / (2 * unit) * unit;
        if (half == 0) break;
        const int64_t mid = good + half;
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

      // Per-unit coefficient from the first probe; every other probe must
      // agree. The ratio need not be a whole number.
      Ratio coefficient;
      bool have = false;
      for (const auto& probe : probes) {
        const int64_t stored = static_cast<int64_t>(extract_bits(
            probe.serialized[run.param_idx].bytes, run.off, run.len));
        const int64_t diff = stored - stored_base;
        const auto detail = [&]() {
          return " (params blob " + std::to_string(run.param_idx) + " bits [" +
                 std::to_string(run.off) + ", " +
                 std::to_string(run.off + run.len) + "), delta " +
                 std::to_string(probe.delta) + ", stored " +
                 std::to_string(stored) + " vs base " +
                 std::to_string(stored_base) + ", coefficient " +
                 std::to_string(coefficient.num) + "/" +
                 std::to_string(coefficient.den) + ")";
        };
        if (!have) {
          coefficient = make_ratio(diff, probe.delta);
          have = true;
        } else if (diff * coefficient.den != coefficient.num * probe.delta) {
          throw std::runtime_error("Dispatch " + op.name() +
                                   ": non-affine field for " + name + detail());
        }
        // The emitted division truncates, so it is exact only where the
        // divided quantity is a multiple of the denominator -- guaranteed
        // when the samples came from the scalar's own step, not when the
        // delta was searched for.
        if (coefficient.den != 1 && derived_scalars_.count(name) == 0) {
          throw std::runtime_error(
              "Dispatch " + op.name() + ": field for " + name +
              " advances by " + std::to_string(coefficient.num) + "/" +
              std::to_string(coefficient.den) +
              " per unit, but the scalar's run-time step is not derivable, so "
              "the emitted integer division could truncate between samples" +
              detail());
        }
      }
      if (coefficient.num == 0) continue;  // spurious (aliased) run

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
          c = shift_ratio(c, field.off - new_off);
        }
        const Ratio shifted = shift_ratio(coefficient, run.off - new_off);
        const auto existing = field.coeff.find(name);
        field.coeff[name] = existing == field.coeff.end()
                                ? shifted
                                : add_ratio(existing->second, shifted);
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
        if (found != field.coeff.end()) {
          // Mirrors the emitted C exactly, truncating division included.
          value += found->second.num * probe.delta / found->second.den;
        }
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
    // Terms are written against each scalar's baseline so a fractional
    // coefficient divides a multiple of its denominator. That fails with
    // several scalars in one field: the toolchain divides their combined
    // contribution once where separate terms each truncate, and every probe
    // moves one scalar at a time so the self-check cannot see it.
    for (const auto& [name, coefficient] : field.coeff) {
      if (coefficient.den != 1 && field.coeff.size() > 1) {
        throw std::runtime_error(
            "Dispatch " + op.name() + ": field at bits [" +
            std::to_string(field.off) + ", " +
            std::to_string(field.off + field.len) + ") depends on " +
            std::to_string(field.coeff.size()) +
            " scalars with a fractional coefficient on " + name +
            "; the terms would truncate separately where the toolchain "
            "divides once -- refusing to emit.");
      }
    }
    std::string expr;
    for (const auto& [name, coefficient] : field.coeff) {
      std::string term = "(" + ref(name) + " - " +
                         std::to_string(to_int(env_.lookup(name))) + "LL)";
      if (coefficient.num != 1) {
        term = std::to_string(coefficient.num) + "LL * " + term;
      }
      if (coefficient.den != 1) {
        term = "(" + term + ") / " + std::to_string(coefficient.den) + "LL";
      }
      expr += " + " + term;
    }
    line(indent, "patch_bits(" + blob_names[field.param_idx] + ", " +
                     std::to_string(field.off) + ", " +
                     std::to_string(field.len) + ", (uint64_t)(" +
                     std::to_string(field.base) + "LL" + expr + "));");
  }
  // The synchronous post-drain below must observe each invocation group
  // actually run: ACCELERATOR_RUNNING alone cannot tell granted-but-unstarted
  // from finished, and it dips between the groups of a multi-pass dispatch.
  // Record each group's closing unit (the last one to start, in
  // Harness::dispatch_params' chunking) so the firmware can wait for its
  // inflight count to rise before draining.
  // Blob indices per invocation group, in the order dispatch_params sends
  // them, plus the register whose inflight count the group's closing unit
  // raises. release_starts opens a group matrix -> spmm -> vector, so the
  // closing unit is the last of those the group actually starts.
  struct EmitGroup {
    std::vector<size_t> sends;
    std::string close_reg;
  };
  std::vector<EmitGroup> emit_groups;
  for (size_t i = 0; i < baseline.size();) {
    EmitGroup group;
    if (baseline[i].kind == kMatrixParams) {
      bool group_mvm = false;
#if SUPPORT_MVM
      group_mvm = baseline[i].is_fc;
#endif
      bool group_spmm = false;
#if SUPPORT_SPMM
      group_spmm = !group_mvm && baseline[i].is_spmm;
#endif
      if (group_spmm) {
        const size_t sparse = i++;
        // A fused dense pass shares the group and is sent first, ahead of the
        // sparse params, inverting their order in the deque.
        if (i < baseline.size() && baseline[i].kind == kMatrixParams) {
          group.sends.push_back(i++);
        }
        group.sends.push_back(sparse);
        group.close_reg = "SPMM_UNIT_OP_INFLIGHT";
      } else {
        group.sends.push_back(i++);
        group.close_reg =
            group_mvm ? "MVM_UNIT_OP_INFLIGHT" : "MATRIX_UNIT_OP_INFLIGHT";
      }
    }
    if (i < baseline.size() && baseline[i].kind == kVectorParams) {
      group.sends.push_back(i++);  // VectorParams
      group.sends.push_back(i++);  // VectorInstructionConfig
      group.close_reg = "VECTOR_UNIT_OP_INFLIGHT";
    }
    emit_groups.push_back(std::move(group));
  }

  // The units arm their operand fetches the moment params arrive (the matrix
  // unit fans params out before taking its start credit), so the operands
  // must be in place by now. They are: the program's async_wait on each
  // load's semaphore precedes this dispatch, and the firmware performed it.

  for (size_t g = 0; g < emit_groups.size(); g++) {
    for (const size_t i : emit_groups[g].sends) {
      // Routing mirrors Harness::dispatch_params: an is_fc MatrixParams goes
      // to the matrix-vector unit and an is_spmm one to the SpMM unit, each
      // only when the build has that unit, else both fall back to the plain
      // matrix unit.
      bool to_mvm = false;
#if SUPPORT_MVM
      to_mvm = baseline[i].is_fc;
#endif
      bool to_spmm = false;
#if SUPPORT_SPMM
      to_spmm = !to_mvm && baseline[i].is_spmm;
#endif
      switch (baseline[i].kind) {
        case kMatrixParams:
          line(indent, std::string(to_spmm  ? "send_spmm_unit_params("
                                   : to_mvm ? "send_matrix_vector_unit_params("
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
    }
    // A synchronous dispatch mirrors Harness::execute's drain-dispatch-drain.
    // Each group's wait must follow its own sends, not trail the whole
    // dispatch: a later group's sends block on MMIO backpressure while
    // earlier groups run, so trailing waits would miss their starts and spin
    // forever. The last group's wait doubles as execute()'s post-dispatch
    // drain. An asynchronous dispatch gets no wait here at all: its
    // retirement is the commit's post, which the testbench observes.
    if (!in_commit_) {
      line(indent,
           "wait_for_dispatch_retired(" + emit_groups[g].close_reg + ");");
    }
  }
}

// ---------------------------------------------------------------------------
// Translation unit
// ---------------------------------------------------------------------------

std::string CEmitter::emit_layer(const Model::Selection& selection) {
  bounded_ = &selection.bounded;
  max_tiles_ = getenv_int("MAX_TILES", 0);
  // Survives the survey walk; everything else is rebuilt by it.
  concrete_env_.clear();

  // The same table the testbench builds from the same selection.
  host_table_ = enumerate_host_ops(selection);
  host_ordinals_.clear();
  for (size_t i = 0; i < host_table_.size(); i++) {
    host_ordinals_[host_table_[i].prim] = i;
  }

  auto walk = [&]() {
    decls_.str("");
    body_.str("");
    env_ = ScalarEnv();
    scopes_.clear();
    scopes_.push_back({});
    name_counts_.clear();
    sem_names_.clear();
    // These hold pointers into the previous layer's protobuf.
    scalar_def_.clear();
    loop_counter_steps_.clear();
    derived_scalars_.clear();
    loop_depth_ = 0;
    speculative_ = false;
    in_commit_ = false;

    for (const auto* op : selection.ops) {
      // Selection.bounded gates the MAX_TILES clamp exactly as it does in the
      // interpreter; non-outermost loops are never clamped.
      emit_operation(*op, 1);
    }
  };

  // Survey first, but only where it can pay off: the walk is deterministic, so
  // a dispatch's guard is true at the same iterations both times, and the
  // second walk can map a dispatch it reaches speculatively under the
  // environment the first one recorded. Where no conditional guards a
  // dispatch there is nothing to learn, and walking arms the emitting pass
  // never enters concretely only risks evaluating states the program does not
  // reach.
  bool survey = getenv_int("SOC_SURVEY", 1) != 0;
  if (survey) {
    survey = false;
    for (const auto* op : selection.ops) {
      if (cond_guards_dispatch(*op)) {
        survey = true;
        break;
      }
    }
  }
  if (survey) {
    // A refusal thrown mid-survey must not leave the flag set: this emitter
    // serves every layer of the network in turn, and a walk that believes it
    // is still surveying emits no dispatch at all.
    surveying_ = true;
    try {
      walk();
    } catch (...) {
      surveying_ = false;
      throw;
    }
    surveying_ = false;
  }
  walk();

  std::ostringstream out;
  out << "#include <stddef.h>\n";
  out << "#include <stdint.h>\n";
  out << "#include <stdio.h>\n\n";
  out << "#include \"host_request.h\"\n";
  out << "#include \"mmio.h\"\n";
  out << "#include \"patch_bits.h\"\n";
  out << "#include \"run_voyager_operation.h\"\n";
  out << "#include \"traps.h\"\n";
  out << "#include \"voyager_address.h\"\n\n";
  out << decls_.str();
  out << "int main() {\n";
  out << "\tenable_interrupts();\n";
  out << "\tenable_semaphore_wait();\n";
  out << "\thost_init();\n\n";
  out << "\treg_write64(VOYAGER_BASE_ADDR, SRAM_BASE);\n\n";
  out << body_.str();
  out << "\n\tprintf(\"All params sent!\\n\");\n\n";
  out << "\twait_for_accelerator_done();\n";
  // The testbench grades the outputs before the firmware goes on to exit.
  out << "\thost_finish();\n\n";
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
