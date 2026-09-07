#include <assert.h>
#include <stdlib.h>

#include "SoCSimulation.h"

SoCSimulation* sim = NULL;

bool running_a_network() {
  /* In full JTAG mode (JTAG_SIM=1) GDB performs the data loading and output
   * checking, so the DPI testbench must stay out of the way. */
  if (std::getenv("JTAG_SIM")) return false;

  /* If the TESTS and NETWORK environment variables are set, then we are
   * running a network. Otherwise, we are running a simple testbench. */
  const char* tests = std::getenv("TESTS");
  const char* network = std::getenv("NETWORK");
  return tests && network;
}

extern "C" void load_memory() {
  if (running_a_network()) {
    assert(sim == NULL && "load_memory() called twice");

    // The SoC flow always grades the DUT against the gold walk.
    setenv("SIMS", "gold,accelerator", /*overwrite=*/0);

    printf("Loading memory\n");
    sim = new SoCSimulation();
    sim->start();
  }
}

/* Randomized register init can pulse a unit's done_vld, or flip the doorbell
 * semaphore, before reset even asserts; nothing real happens before
 * load_memory() has run, so events that arrive first are ignored. */

extern "C" void host_doorbell() {
  if (running_a_network()) {
    if (sim == NULL) return;
    sim->doorbell();
  }
}

extern "C" void unit_started(int unit) {
  if (running_a_network()) {
    if (sim == NULL) return;
    sim->unit_started(unit);
  }
}

extern "C" void unit_done(int unit) {
  if (running_a_network()) {
    if (sim == NULL) return;
    sim->unit_done(unit);
  }
}

extern "C" void unit_retired(int unit) {
  if (running_a_network()) {
    if (sim == NULL) return;
    sim->unit_retired(unit);
  }
}

extern "C" int sc_main(int argc, char* argv[]) { return 0; }
