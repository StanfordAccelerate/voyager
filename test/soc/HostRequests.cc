#include "test/soc/HostRequests.h"

#include "test/common/GraphUtils.h"

namespace {

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

// Mirrors Interpreter::execute_prim's routing: which prims move data or run
// a host kernel, as opposed to computing scalars, dispatching, or keeping a
// semaphore (which the firmware does itself).
bool classify(const voyager::Operation& op, HostOp::Kind* kind) {
  if (op.op_type_case() != voyager::Operation::kPrim) return false;
  const voyager::PrimOp& prim = op.prim();
  const std::string& target = prim.target();
  const bool declares_box =
      op.outputs_size() == 1 && op.outputs(0).has_tensor_box();
  if (target == "voyager::async_copy") {
    *kind = HostOp::kCopy;
    return true;
  }
  if (target == "voyager::alloc") {
    // The executable spec zero-fills integer allocations only.
    if (!declares_box) return false;
    const auto& box = op.outputs(0).tensor_box();
    const bool is_integer =
        box.dtype().rfind("int", 0) == 0 || box.dtype().rfind("uint", 0) == 0;
    if (!(is_integer && box.has_memory() && !is_semaphore(box))) return false;
    *kind = HostOp::kZero;
    return true;
  }
  if (target == "voyager::zeros") {
    if (!declares_box || is_semaphore(op.outputs(0).tensor_box())) return false;
    *kind = HostOp::kZero;
    return true;
  }
  if (target.rfind("voyager::", 0) == 0) return false;  // fill, wait, ...
  bool only_scalars = op.outputs_size() > 0;
  for (const auto& output : op.outputs()) {
    if (!output.has_scalar()) only_scalars = false;
  }
  if (only_scalars) return false;
  if (is_host_bookkeeping(prim) || is_datapath(op)) return false;
  *kind = HostOp::kHostOp;
  return true;
}

void walk(const voyager::Operation& op, std::vector<HostOp>* table);

void walk_all(const google::protobuf::RepeatedPtrField<voyager::Operation>& ops,
              std::vector<HostOp>* table) {
  for (const auto& op : ops) walk(op, table);
}

void walk(const voyager::Operation& op, std::vector<HostOp>* table) {
  HostOp::Kind kind;
  if (classify(op, &kind)) table->push_back(HostOp{kind, &op, &op.prim()});
  switch (op.op_type_case()) {
    case voyager::Operation::kLoop:
      if (op.loop().has_for_loop()) {
        walk_all(op.loop().for_loop().body().ops(), table);
      } else if (op.loop().has_while_loop()) {
        walk_all(op.loop().while_loop().condition().ops(), table);
        walk_all(op.loop().while_loop().body().ops(), table);
      }
      break;
    case voyager::Operation::kCond:
      walk_all(op.cond().true_region().ops(), table);
      walk_all(op.cond().false_region().ops(), table);
      break;
    case voyager::Operation::kAsync:
      walk_all(op.async().body().ops(), table);
      break;
    default:
      break;
  }
}

}  // namespace

std::vector<HostOp> enumerate_host_ops(const Model::Selection& selection) {
  std::vector<HostOp> table;
  for (const auto* op : selection.ops) walk(*op, &table);
  return table;
}

std::set<std::string> request_scalar_names(const voyager::Operation& op) {
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
