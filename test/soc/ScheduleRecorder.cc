#include "test/soc/ScheduleRecorder.h"

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
        throw std::runtime_error(
            "SpMM dispatches are not supported by the SoC MVP flow.");
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
