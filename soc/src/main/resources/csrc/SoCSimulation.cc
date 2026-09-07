#include "SoCSimulation.h"

#include <vpi_user.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "test/common/GoldModel.h"
#include "test/common/GraphUtils.h"
#include "test/soc/firmware/common/voyager_address.h"

static_assert(sizeof(host_request_t) == 200, "host_request_t layout");
static_assert(offsetof(host_mailbox_t, tail) == 64, "host_mailbox_t layout");
static_assert(offsetof(host_mailbox_t, slots) == 128, "host_mailbox_t layout");

static uint64_t vpi_time_now() {
  s_vpi_time t;
  t.type = vpiSimTime;
  vpi_get_time(nullptr, &t);
  return (uint64_t(t.high) << 32) | uint64_t(t.low);
}

namespace {

// Indexed by request kind (host_request.h).
const char* kKind[] = {"?", "COPY", "ZERO", "HOST_OP", "POST", "FINISH"};
// Indexed by HostOp::Kind.
const char* kHostKind[] = {"copy", "zero", "host op"};

}  // namespace

SoCSimulation::SoCSimulation()
    : Simulation(), trace_(std::getenv("DUMP_REQUESTS") != nullptr) {}

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

  const char* mailbox = std::getenv("HOST_MAILBOX");
  if (mailbox == nullptr) {
    throw std::runtime_error(
        "HOST_MAILBOX is not set: run_voyager.py reads the firmware's "
        "host_mailbox symbol from the ELF and exports its address.");
  }
  mailbox_ = std::strtoull(mailbox, nullptr, 0);

  table_ = enumerate_host_ops(selection);
  std::cerr << "[TB t=" << vpi_time_now() << "] " << table_.size()
            << " host ops in the request table; mailbox at 0x" << std::hex
            << mailbox_ << std::dec << std::endl;
  if (trace_) {
    for (size_t i = 0; i < table_.size(); i++) {
      std::cerr << "[HOSTOP " << i << "] " << kHostKind[table_[i].kind] << " "
                << table_[i].op->name() << std::endl;
    }
  }
}

// --- scratchpad cells ------------------------------------------------------

void SoCSimulation::read_bytes(uint64_t address, uint64_t count, void* out) {
  MemoryInterface* mem = memory("accelerator");
  mem->read_bytes_from_memory(static_cast<long long>(address - SRAM_BASE),
                              SRAM_PARTITION, static_cast<int>(count),
                              static_cast<char*>(out));
}

uint64_t SoCSimulation::read_u64(uint64_t address) {
  uint64_t value = 0;
  read_bytes(address, sizeof(value), &value);
  return value;
}

void SoCSimulation::write_u64(uint64_t address, uint64_t value) {
  MemoryInterface* mem = memory("accelerator");
  mem->write_bytes_to_memory(static_cast<long long>(address - SRAM_BASE),
                             SRAM_PARTITION, sizeof(value),
                             reinterpret_cast<const char*>(&value));
}

// --- the mailbox -------------------------------------------------------------

void SoCSimulation::drain_mailbox() {
  const uint64_t head = read_u64(mailbox_ + offsetof(host_mailbox_t, head));
  if (head == tail_) return;
  while (tail_ < head) {
    Request request;
    const uint64_t slot = tail_ % HOST_MAILBOX_SLOTS;
    read_bytes(mailbox_ + offsetof(host_mailbox_t, slots) +
                   slot * sizeof(host_request_t),
               sizeof(host_request_t), &request);
    queue_.push_back(request);
    tail_++;
  }
  write_u64(mailbox_ + offsetof(host_mailbox_t, tail), tail_);
}

std::string SoCSimulation::describe(const Request& request) const {
  std::string text =
      request.kind <= HOST_REQ_FINISH ? kKind[request.kind] : "?";
  if (request.kind == HOST_REQ_COPY || request.kind == HOST_REQ_ZERO ||
      request.kind == HOST_REQ_HOST_OP) {
    text += " #" + std::to_string(request.ordinal);
    if (request.ordinal < table_.size()) {
      text += " " + table_[request.ordinal].op->name();
    }
  }
  if (request.kind == HOST_REQ_POST) {
    text += " after";
    for (int u = 0; u < HOST_NUM_UNITS; u++) {
      text += " " + std::to_string(request.retired[u]);
    }
  }
  if (request.cell != 0) {
    char cell[32];
    std::snprintf(cell, sizeof(cell), " cell 0x%llx",
                  static_cast<unsigned long long>(request.cell));
    text += std::string(cell) + " +" + std::to_string(request.amount);
  }
  if (trace_ && request.nargs > 0) {
    text += " args";
    for (uint64_t i = 0; i < request.nargs && i < HOST_REQ_MAX_ARGS; i++) {
      text += " " + std::to_string(request.args[i]);
    }
  }
  return text;
}

ScalarEnv SoCSimulation::request_env(const HostOp& host,
                                     const Request& request) const {
  const std::set<std::string> names = request_scalar_names(*host.op);
  if (names.size() != request.nargs) {
    throw std::runtime_error(
        host.op->name() + ": the firmware sent " +
        std::to_string(request.nargs) + " scalars, the prim references " +
        std::to_string(names.size()) + " (emitter/testbench mismatch).");
  }
  ScalarEnv env;
  size_t i = 0;
  for (const auto& name : names) env.define(name, request.args[i++]);
  return env;
}

bool SoCSimulation::retired(const Request& post) const {
  for (int u = 0; u < HOST_NUM_UNITS; u++) {
    if (done_count_[u] < post.retired[u]) return false;
  }
  return true;
}

void SoCSimulation::complete(const Request& request) {
  if (request.cell == 0) return;
  write_u64(request.cell, read_u64(request.cell) + request.amount);
}

void SoCSimulation::execute(const Request& request) {
  auto* mem = memory("accelerator");
  std::cerr << "[TB t=" << vpi_time_now() << "] " << describe(request)
            << std::endl;

  if (request.kind == HOST_REQ_FINISH) {
    finished_ = true;
    if (auto* soc_mem = dynamic_cast<SoCMemory*>(mem)) {
      soc_mem->verify_shadow();
    }
    check_outputs();
    return;
  }

  if (request.ordinal >= table_.size()) {
    throw std::runtime_error(
        "Request names ordinal " + std::to_string(request.ordinal) +
        " but the table has " + std::to_string(table_.size()) + " entries.");
  }
  const HostOp& host = table_[request.ordinal];

  switch (request.kind) {
    case HOST_REQ_COPY:
      if (host.kind != HostOp::kCopy) {
        throw std::runtime_error("Ordinal " + std::to_string(request.ordinal) +
                                 " is not a copy: " + host.op->name());
      }
      run_async_copy(*host.prim, request_env(host, request), mem);
      return;

    case HOST_REQ_ZERO: {
      if (host.kind != HostOp::kZero) {
        throw std::runtime_error("Ordinal " + std::to_string(request.ordinal) +
                                 " is not a zero fill: " + host.op->name());
      }
      const auto& box = host.op->outputs(0).tensor_box();
      if (host.prim->target() == "voyager::alloc") {
        // Each slot's own elements only; the span between slots belongs to
        // a buffer with a disjoint lifetime.
        for (uint32_t bank = 0; bank < banks_of(box); bank++) {
          zero_buffer(to_tensor(box, bank), 1, 0, mem);
        }
      } else {
        zero_buffer(to_tensor(box), banks_of(box), bank_stride_of(box), mem);
      }
      return;
    }

    case HOST_REQ_HOST_OP:
      if (host.kind != HostOp::kHostOp) {
        throw std::runtime_error("Ordinal " + std::to_string(request.ordinal) +
                                 " is not a host op: " + host.op->name());
      }
      run_host_operation(*host.op, request_env(host, request), mem);
      return;

    default:
      throw std::runtime_error("Unknown request kind " +
                               std::to_string(request.kind));
  }
}

void SoCSimulation::service() {
  if (finished_) return;
  drain_mailbox();

  // Retire posts complete on done counts, independently of the queue.
  for (auto it = posts_.begin(); it != posts_.end();) {
    if (retired(*it)) {
      std::cerr << "[TB t=" << vpi_time_now() << "] " << describe(*it)
                << " completed" << std::endl;
      complete(*it);
      it = posts_.erase(it);
    } else {
      ++it;
    }
  }

  while (!queue_.empty() && !finished_) {
    const Request request = queue_.front();
    if (request.kind == HOST_REQ_POST) {
      queue_.pop_front();
      if (retired(request)) {
        std::cerr << "[TB t=" << vpi_time_now() << "] " << describe(request)
                  << " completed" << std::endl;
        complete(request);
      } else {
        posts_.push_back(request);
      }
      continue;
    }
    // Everything else touches the scratchpad. A unit's done does not mean
    // its last writes have landed (the up_proj lesson, +done_settle in
    // VerificationCollateral.v), so hold until every done has settled.
    if (settling_ > 0) return;
    queue_.pop_front();
    execute(request);
    complete(request);
  }
}

// --- DPI events
// ----------------------------------------------------------------

void SoCSimulation::doorbell() { service(); }

void SoCSimulation::unit_started(int unit) {
  // A request held for the settle window while the next dispatch already
  // runs: if that dispatch writes what a held store-back reads, the store
  // reads the new tile. The program's own ordering normally keeps the two
  // apart (double-buffered slots), so this is a diagnostic, not a stall.
  if (!queue_.empty() && settling_ > 0) {
    std::cerr << "[TB t=" << vpi_time_now() << "] Warning: unit " << unit
              << " started with " << queue_.size()
              << " request(s) held for the settle window: "
              << describe(queue_.front()) << std::endl;
  }
}

void SoCSimulation::unit_done(int unit) {
  (void)unit;
  settling_++;
}

void SoCSimulation::unit_retired(int unit) {
  settling_--;
  if (unit >= 0 && unit < HOST_NUM_UNITS) done_count_[unit]++;
  service();
}
