#include "test/soc/jtag/ScheduleRecorder.h"

#include <stdexcept>

void ScheduleRecorder::data_op(const voyager::Operation& op,
                               const voyager::PrimOp& prim,
                               const ScalarEnv& env) {
  Step step;
  step.kind =
      prim.target() == "voyager::async_copy" ? Step::kCopy : Step::kZero;
  step.op = &op;
  step.prim = &prim;
  step.env = env;
  steps_->push_back(std::move(step));
}

namespace {

// Whether any scalar the prim reads is one the replay will recompute.
bool reads_deferred(const voyager::PrimOp& prim,
                    const std::set<std::string>& deferred) {
  for (const auto& [key, argument] : prim.kwargs()) {
    if (argument.arg_type_case() != voyager::Argument::kScalar) continue;
    const voyager::ScalarValue& value = argument.scalar();
    if (value.value_case() == voyager::ScalarValue::kNode &&
        deferred.count(value.node()) > 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

void ScheduleRecorder::scalar_read(const voyager::Operation& op,
                                   const voyager::PrimOp& prim,
                                   const ScalarEnv& env) {
  Step step;
  step.kind = Step::kScalarRead;
  step.op = &op;
  step.prim = &prim;
  step.env = env;
  steps_->push_back(std::move(step));
  if (op.outputs_size() == 1) deferred_.insert(op.outputs(0).name());
}

void ScheduleRecorder::scalar_op(const voyager::Operation& op,
                                 const voyager::PrimOp& prim,
                                 const ScalarEnv& env) {
  // Only the cone below a deferred read needs recomputing; every other scalar
  // is a loop index the recording walk already got right.
  if (op.outputs_size() != 1 || !reads_deferred(prim, deferred_)) return;
  Step step;
  step.kind = Step::kScalarOp;
  step.op = &op;
  step.prim = &prim;
  step.env = env;
  steps_->push_back(std::move(step));
  deferred_.insert(op.outputs(0).name());
}

void ScheduleRecorder::execute(const voyager::Operation& op,
                               const ScalarEnv& env) {
  Step step;
  step.kind = is_datapath(op) ? Step::kDispatch : Step::kHostOp;
  step.op = &op;
  step.op_name = op.name();
  step.env = env;
  steps_->push_back(std::move(step));
}
