#pragma once

#include <svdpi.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "SoCMemory.h"
#include "test/common/Simulation.h"
#include "test/soc/HostRequests.h"
#include "test/soc/firmware/common/host_request.h"

// The SoC RTL testbench: the DMA engine Sphinx does not have, plus the gold
// reference and the grading.
//
// The firmware on the Rocket core executes the whole of the layer's
// bufferized program. What it cannot do itself -- move data between the
// host-side DRAM and the DUT's scratchpad -- it asks for through a mailbox
// in its own memory (host_request.h): a copy or zero fill names its prim by
// ordinal in the host-request table (HostRequests.h) and carries the
// run-time values of the scalars the prim references, so the testbench runs
// the very same run_async_copy / zero_buffer the interpreter does. A
// request completes by adding to a counter cell the firmware spins on.
//
// The testbench is driven by three DPI events from the Verilog collateral:
// the doorbell (hardware semaphore 7 changed: read the mailbox), a unit's
// done pulse, and the same done `done_settle` cycles later, once the unit's
// last writes have landed. Requests that touch the scratchpad wait for that
// settle; a commit's retire post completes once every unit's settled done
// count has reached the count the firmware had issued when it posted.
class SoCSimulation : public Simulation {
 public:
  SoCSimulation();

  void start();
  void doorbell();
  void unit_started(int unit);
  void unit_done(int unit);
  void unit_retired(int unit);

 protected:
  ArrayMemory* make_memory(const std::string& sim,
                           const std::vector<uint64_t>& sizes) override;

 private:
  using Request = host_request_t;

  void drain_mailbox();
  void service();
  void execute(const Request& request);
  void complete(const Request& request);
  bool retired(const Request& post) const;
  ScalarEnv request_env(const HostOp& host, const Request& request) const;
  std::string describe(const Request& request) const;

  // Scratchpad cells by absolute (CPU) address.
  uint64_t read_u64(uint64_t address);
  void write_u64(uint64_t address, uint64_t value);
  void read_bytes(uint64_t address, uint64_t count, void* out);

  std::vector<HostOp> table_;
  uint64_t mailbox_ = 0;  // absolute address of host_mailbox
  uint64_t tail_ = 0;     // requests taken off the ring

  std::deque<Request> queue_;   // taken, not yet serviced, in order
  std::vector<Request> posts_;  // retire posts awaiting their done counts

  uint64_t done_count_[HOST_NUM_UNITS] = {0, 0, 0, 0};
  int settling_ = 0;  // done pulses whose settle has not elapsed

  bool finished_ = false;
  const bool trace_;
};
