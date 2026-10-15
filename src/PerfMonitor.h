#pragma once

#include <ac_int.h>

#include "ArchitectureParams.h"

namespace MatrixPerformance {

using Counter = ac_int<32, false>;
using CounterIndex = ac_int<5, false>;
using SnapshotSequence = ac_int<32, false>;

// Stable indices for both matrix backends. Unsupported counters read zero.
// Counters wrap modulo 2^32 and accumulate until reset. Snapshots mark
// matrix-unit completion, including output drain; overlapping commands share
// each interval. Buffer counts are accesses, not bytes.
enum CounterId {
  SNAPSHOT_SEQUENCE = 0,
  PROCESSOR_ACTIVE_CYCLES,
  ARRAY_ISSUE_CYCLES,
  // Vector admission backpressure: CIM requests or the systolic input skewer.
  INPUT_BACKPRESSURE_CYCLES,
  RESULT_BACKPRESSURE_CYCLES,
  MAC_WAIT_WEIGHT_SET_LOAD_CYCLES,
  // One accepted physical B-port beat per counted cycle.
  CIM_WEIGHT_LOAD_CYCLES,
  INPUT_BUFFER_READS,
  INPUT_BUFFER_WRITES,
  ACCUM_BUFFER_READS,
  ACCUM_BUFFER_WRITES,
  WEIGHT_BUFFER_READS,
  WEIGHT_BUFFER_WRITES,
  COUNTER_COUNT
};

static constexpr int PROCESSOR_COUNTER_COUNT =
    INPUT_BUFFER_READS - PROCESSOR_ACTIVE_CYCLES;
static constexpr int COMMON_PROCESSOR_COUNTER_COUNT =
    MAC_WAIT_WEIGHT_SET_LOAD_CYCLES - PROCESSOR_ACTIVE_CYCLES;
static constexpr int PERFORMANCE_COUNTER_COUNT =
    COUNTER_COUNT - PROCESSOR_ACTIVE_CYCLES;

// Convert a public performance-counter ID to its compact storage index
static constexpr int storage_index(CounterId id) {
  return static_cast<int>(id) - static_cast<int>(PROCESSOR_ACTIVE_CYCLES);
}

// The backend supplies its vector admission and result handshakes. Pending
// commands measure activity through write-back, independently of output drain.
struct ProcessorCounters {
  Counter values[COMMON_PROCESSOR_COUNTER_COUNT];
  Counter active_commands;
  bool observed_completion;

  void reset() {
#pragma hls_unroll yes
    for (int i = 0; i < COMMON_PROCESSOR_COUNTER_COUNT; ++i) values[i] = 0;
    active_commands = 0;
    observed_completion = false;
  }

  void observe(bool started, bool completion_toggle, bool issue_valid,
               bool issue_ready, bool result_valid, bool result_ready) {
    const bool completed = completion_toggle != observed_completion;
    if (active_commands != 0 || started)
      ++values[storage_index(PROCESSOR_ACTIVE_CYCLES)];
    if (started && !completed)
      ++active_commands;
    else if (completed && !started && active_commands != 0)
      --active_commands;
    observed_completion = completion_toggle;
    if (issue_valid) {
      if (issue_ready)
        ++values[storage_index(ARRAY_ISSUE_CYCLES)];
      else
        ++values[storage_index(INPUT_BACKPRESSURE_CYCLES)];
    }
    if (result_valid && !result_ready)
      ++values[storage_index(RESULT_BACKPRESSURE_CYCLES)];
  }
};

}  // namespace MatrixPerformance
