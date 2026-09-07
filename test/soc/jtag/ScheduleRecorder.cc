#include "test/soc/jtag/ScheduleRecorder.h"

#include <stdexcept>

#include "test/toolchain/MapOperation.h"

void count_unit_passes(const std::deque<BaseParams*>& params,
                       int passes[Step::kNumUnits],
                       std::vector<Step::Group>* groups) {
  // Mirrors Harness::dispatch_params' chunking: each loop iteration consumes
  // one invocation group off the deque and flags the units it starts. The
  // recorded group sequence is what the replay's start-release engine grants
  // in order, so it must match the chunking exactly.
  size_t idx = 0;
  while (idx < params.size()) {
    Step::Group group;
    if (auto* matrix = dynamic_cast<MatrixParams*>(params[idx])) {
      idx++;
      // Mirrors Harness::dispatch_params routing, including the fallback to
      // the plain matrix unit when the build has no matrix-vector unit.
      bool routed = false;
#if SUPPORT_MVM
      if (matrix->is_fc) {
        passes[Step::kMvm]++;
        group.compute_unit = Step::kMvm;
        routed = true;
      }
#endif
#if SUPPORT_SPMM
      if (!routed && matrix->is_spmm) {
        // A fused dense pass, when the deque holds one next, joins this group
        // rather than starting its own.
        if (idx < params.size() &&
            dynamic_cast<MatrixParams*>(params[idx]) != nullptr) {
          idx++;
          passes[Step::kMatrix]++;
          group.compute_unit = Step::kMatrix;
        }
        passes[Step::kSpmm]++;
        group.spmm = true;
        routed = true;
      }
#endif
      if (!routed) {
        passes[Step::kMatrix]++;
        group.compute_unit = Step::kMatrix;
      }
    }
    if (idx < params.size() &&
        dynamic_cast<VectorParams*>(params[idx]) != nullptr) {
      idx++;
      if (idx >= params.size() ||
          dynamic_cast<VectorInstructionConfig*>(params[idx]) == nullptr) {
        throw std::runtime_error(
            "VectorParams not followed by VectorInstructionConfig.");
      }
      idx++;
      passes[Step::kVector]++;
      group.vector = true;
    } else if (idx < params.size() &&
               dynamic_cast<MatrixParams*>(params[idx]) == nullptr) {
      throw std::runtime_error("Unrecognized params type in dispatch deque.");
    }
    if (groups != nullptr) groups->push_back(group);
  }
}

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
  step.op = &op;
  step.op_name = op.name();
  step.env = env;

  if (!is_datapath(op)) {
    step.kind = Step::kHostOp;
    steps_->push_back(std::move(step));
    return;
  }

  step.kind = Step::kDispatch;
  step.sync = !in_commit_;
  std::deque<BaseParams*> params;
  map_operation(op, env, params);
  count_unit_passes(params, step.passes, &step.groups);
  for (auto* param : params) delete param;

  steps_->push_back(std::move(step));
}

void ScheduleRecorder::begin_commit() { in_commit_ = true; }

void ScheduleRecorder::init_semaphore(const std::string& node, int64_t slot,
                                      int64_t value) {
  Step step;
  step.kind = Step::kInit;
  step.sem_node = node;
  step.sem_slot = slot;
  step.amount = value;
  steps_->push_back(std::move(step));
}

void ScheduleRecorder::post_semaphore(const std::string& node, int64_t slot,
                                      int64_t amount) {
  Step step;
  step.kind = Step::kPost;
  step.sem_node = node;
  step.sem_slot = slot;
  step.amount = amount;
  steps_->push_back(std::move(step));
}

void ScheduleRecorder::wait_semaphore(const std::string& node, int64_t slot,
                                      int64_t amount) {
  Step step;
  step.kind = Step::kWait;
  step.sem_node = node;
  step.sem_slot = slot;
  step.amount = amount;
  steps_->push_back(std::move(step));
}

void ScheduleRecorder::end_commit(bool has_post, const std::string& post_node,
                                  int64_t post_slot, int64_t post_amount) {
  in_commit_ = false;
  if (!has_post) return;
  Step step;
  step.kind = Step::kPost;
  step.sem_node = post_node;
  step.sem_slot = post_slot;
  step.amount = post_amount;
  step.retire_post = true;
  steps_->push_back(std::move(step));
}
