#include "SoCSimulation.h"

#include <vpi_user.h>

#include <cstdlib>
#include <iostream>

#include "test/common/GoldModel.h"
#include "test/common/GraphUtils.h"
#include "test/common/Interpreter.h"

static uint64_t vpi_time_now() {
  s_vpi_time t;
  t.type = vpiSimTime;
  vpi_get_time(nullptr, &t);
  return (uint64_t(t.high) << 32) | uint64_t(t.low);
}

SoCSimulation::SoCSimulation() : Simulation() {}

ArrayMemory* SoCSimulation::make_memory(const std::string& sim,
                                        const std::vector<uint64_t>& sizes) {
  if (sim == "accelerator") {
    return new SoCMemory(sizes);
  }
  return Simulation::make_memory(sim, sizes);
}

void SoCSimulation::start() {
  load_data();

  if (uses("gold")) {
    run_gold();
  }

  // Record the schedule: the same walk the gold model just made, but with a
  // backend that captures the data movement and semaphore protocol instead
  // of executing it.
  ScheduleRecorder recorder(&steps_);
  Interpreter interpreter(model, memory("accelerator"), &recorder);
  interpreter.run(selection);

  std::cerr << "[TB t=" << vpi_time_now() << "] Recorded " << steps_.size()
            << " schedule steps" << std::endl;
  if (std::getenv("DUMP_SCHEDULE")) {
    static const char* kKind[] = {"COPY", "ZERO", "HOST", "INIT",
                                  "WAIT", "POST", "DISP"};
    for (size_t i = 0; i < steps_.size(); i++) {
      const Step& s = steps_[i];
      std::cerr << "[SCHED " << i << "] " << kKind[s.kind];
      if (s.kind == Step::kInit || s.kind == Step::kWait ||
          s.kind == Step::kPost) {
        std::cerr << " " << s.sem_node << "[" << s.sem_slot << "] amt "
                  << s.amount << (s.retire_post ? " RETIRE" : "");
      } else if (s.kind == Step::kDispatch) {
        std::cerr << " " << s.op_name << " passes m" << s.passes[0] << " v"
                  << s.passes[1] << " g" << s.groups.size()
                  << (s.sync ? " SYNC" : "");
      } else if (s.op != nullptr) {
        std::cerr << " " << s.op->name();
      }
      std::cerr << std::endl;
    }
  }

  advance();
}

void SoCSimulation::poke_hw_semaphore(int unit, int credits) {
  const std::string path = "TestDriver.testHarness.chiptop0.system.voyager." +
                           std::string("semaphores_") + std::to_string(unit);
  vpiHandle handle = vpi_handle_by_name((PLI_BYTE8*)path.c_str(), NULL);
  if (!handle) {
    std::cerr << "Warning: could not find " << path << std::endl;
    return;
  }

  s_vpi_value current;
  current.format = vpiIntVal;
  vpi_get_value(handle, &current);

  s_vpi_value next;
  next.format = vpiIntVal;
  next.value.integer = current.value.integer + credits;
  std::cerr << "[TB t=" << vpi_time_now() << "] semaphores_" << unit
            << " += " << credits << " -> " << next.value.integer << std::endl;

  // CREDIT_DELAY_NS > 0 defers the credit by a transport delay so the next
  // pass cannot start until the previous pass's TileLink traffic has drained
  // (diagnostic for tail/head overlap hazards; default immediate).
  static const int64_t delay_ns = [] {
    const char* v = std::getenv("CREDIT_DELAY_NS");
    return v ? std::atoll(v) : 0LL;
  }();
  if (delay_ns > 0) {
    s_vpi_time when;
    when.type = vpiSimTime;
    // Timescale is 1ns/10ps: vpiSimTime counts 10ps units.
    const uint64_t units = static_cast<uint64_t>(delay_ns) * 100;
    when.high = static_cast<PLI_UINT32>(units >> 32);
    when.low = static_cast<PLI_UINT32>(units & 0xFFFFFFFFULL);
    vpi_put_value(handle, &next, &when, vpiTransportDelay);
    return;
  }
  vpi_put_value(handle, &next, NULL, vpiNoDelay);
}

bool SoCSimulation::apply(const Step& step) {
  auto* mem = memory("accelerator");

  switch (step.kind) {
    case Step::kInit:
      sems_[{step.sem_node, step.sem_slot}] = step.amount;
      return true;

    case Step::kPost:
      if (step.retire_post && !dispatches_.empty() &&
          dispatches_.back().groups_retired < dispatches_.back().groups_total) {
        // A commit's post is a deferred completion event, not a program
        // step: the walk moves on and the token rides the newest dispatch
        // (its own commit's -- everything later in the stream has not been
        // issued yet), firing when it and, FIFO, every earlier dispatch has
        // retired. Blocking the walk here instead would delay every
        // subsequent copy by one pass and let the DUT's double-buffered
        // prefetch read ping-pong slots before they are filled.
        dispatches_.back().tokens.push_back(&step);
        return true;
      }
      sems_[{step.sem_node, step.sem_slot}] += step.amount;
      return true;

    case Step::kWait: {
      int64_t& count = sems_[{step.sem_node, step.sem_slot}];
      if (count < step.amount) return false;
      count -= step.amount;
      return true;
    }

    case Step::kCopy:
      run_async_copy(*step.prim, step.env, mem);
      return true;

    case Step::kZero: {
      const auto& box = step.op->outputs(0).tensor_box();
      if (step.prim->target() == "voyager::alloc") {
        for (uint32_t bank = 0; bank < banks_of(box); bank++) {
          zero_buffer(to_tensor(box, bank), 1, 0, mem);
        }
      } else {
        zero_buffer(to_tensor(box), banks_of(box), bank_stride_of(box), mem);
      }
      return true;
    }

    case Step::kHostOp:
      // The harness quiesces the datapath before a host op; mirror that by
      // draining everything issued or queued first.
      if (!quiescent()) return false;
      std::cerr << "[TB t=" << vpi_time_now() << "] host op " << step.op_name
                << std::endl;
      run_host_operation(*step.op, step.env, mem);
      return true;

    case Step::kDispatch: {
      // A synchronous dispatch (outside any commit) is a barrier on both
      // sides, mirroring Harness::execute's drain-dispatch-drain: it may not
      // issue while anything is in flight or queued, and the walk may not
      // pass it -- in particular not the op's own inline semaphore post
      // recorded right after -- until it has retired.
      if (step.sync) {
        if (sync_issued_ == next_step_) {
          return quiescent();  // issued earlier; passable once retired
        }
        if (!quiescent()) return false;  // pre-dispatch drain
      }

      // Queue the dispatch's start grants in Harness::release_starts' order:
      // groups in sequence, the compute unit's start before the vector's.
      // pump_grants() releases them one rendezvous at a time.
      dispatches_.push_back(DispatchRec{next_step_, step.groups.size()});
      const size_t dispatch_index = dispatches_.size() - 1;
      int total = 0;
      for (size_t gi = 0; gi < step.groups.size(); gi++) {
        const Step::Group& group = step.groups[gi];
        bool leads = true;
        // dispatch_params drains between the invocation groups of one
        // dispatch: each pass reads what the previous one wrote, so group
        // gi may not issue until everything issued has retired.
        bool drain = gi > 0;
        if (group.compute_unit >= 0) {
          grants_.push_back(
              Grant{group.compute_unit, dispatch_index, group, leads, drain});
          leads = false;
          drain = false;
          total++;
        }
        if (group.vector) {
          grants_.push_back(
              Grant{Step::kVector, dispatch_index, group, leads, drain});
          total++;
        }
      }
      std::cerr << "[TB t=" << vpi_time_now() << "] dispatch " << step.op_name
                << " queued (" << step.groups.size() << " groups, " << total
                << " starts)" << (step.sync ? " [sync]" : "") << std::endl;
      pump_grants();
      if (step.sync) {
        sync_issued_ = next_step_;
        return false;  // post-dispatch drain: stall here until it retires
      }
      return true;
    }
  }
  return true;
}

// Releases queued unit starts one rendezvous at a time, in program order --
// the credit poke is the grant, and the unit's start firing (start_fired)
// completes it, exactly like release_starts' SyncPop: the harness thread
// only moves to the next start once the unit has actually accepted this one.
void SoCSimulation::pump_grants() {
  if (grant_outstanding_ || grants_.empty()) return;
  Grant& grant = grants_.front();
  // The inter-group drain: retried from tick() as groups retire.
  if (grant.drain_before && !inflight_.empty()) return;
  if (grant.leads_group) {
    InFlight entry;
    entry.dispatch_index = grant.dispatch_index;
    for (int u = 0; u < Step::kNumUnits; u++) entry.remaining[u] = 0;
    if (grant.group.compute_unit >= 0) {
      entry.remaining[grant.group.compute_unit] = 1;
    }
    if (grant.group.vector) entry.remaining[Step::kVector] = 1;
    inflight_.push_back(entry);
  }
  poke_hw_semaphore(grant.unit, 1);
  grant_outstanding_ = true;
}

void SoCSimulation::start_fired(int unit) {
  if (finished_) return;
  if (!grant_outstanding_ || grants_.empty() || grants_.front().unit != unit) {
    std::cerr << "[TB t=" << vpi_time_now() << "] Warning: start on unit "
              << unit << " with no matching grant outstanding" << std::endl;
    return;
  }
  grants_.pop_front();
  grant_outstanding_ = false;
  pump_grants();
}

void SoCSimulation::advance() {
  while (next_step_ < steps_.size()) {
    if (!apply(steps_[next_step_])) return;  // stalled on hardware progress
    next_step_++;
  }
  if (quiescent()) finish();
}

void SoCSimulation::tick(int unit) {
  if (finished_) return;

  // Attribute the done to the oldest dispatch still owing a pass on this
  // unit (matrix-unit pipelining lets consecutive dispatches' done events
  // interleave across units).
  InFlight* owner = nullptr;
  for (auto& entry : inflight_) {
    if (unit >= 0 && unit < Step::kNumUnits && entry.remaining[unit] > 0) {
      owner = &entry;
      break;
    }
  }
  if (owner == nullptr) {
    std::cerr << "[TB t=" << vpi_time_now() << "] Warning: done event on unit "
              << unit << " with no matching dispatch in flight" << std::endl;
    return;
  }
  owner->remaining[unit]--;

  // FIFO retirement: a group retires only at the front. A dispatch's retire
  // tokens fire when its last group leaves the queue.
  while (!inflight_.empty() && inflight_.front().done()) {
    DispatchRec& rec = dispatches_[inflight_.front().dispatch_index];
    inflight_.pop_front();
    rec.groups_retired++;
    if (rec.groups_retired == rec.groups_total) {
      for (const Step* token : rec.tokens) {
        sems_[{token->sem_node, token->sem_slot}] += token->amount;
      }
      rec.tokens.clear();
    }
  }

  pump_grants();  // an inter-group drain barrier may now be clear
  advance();
}

void SoCSimulation::finish() {
  if (finished_) return;
  finished_ = true;

  std::cerr << "[TB t=" << vpi_time_now() << "] All operations completed"
            << std::endl;
  if (auto* soc_mem = dynamic_cast<SoCMemory*>(memory("accelerator"))) {
    soc_mem->verify_shadow();
  }
  check_outputs();
}
