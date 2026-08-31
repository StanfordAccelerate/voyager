// This file is the main entry point to running event-level SystemC simulations.
// It can run any model, such as ResNet and mobileBert-Tiny.
// Parameters are passed via environment variables, not arguments.
//
// The graph walk is the same whichever simulators SIMS names. Only what
// executes an accelerator operation differs: the gold model runs a bit-accurate
// C++ kernel, the accelerator lowers the operation into params and hands it to
// the DUT.

#include "test/common/Harness.h"
#include "test/common/Simulation.h"

extern "C" int sc_main(int argc, char* argv[]) {
  Simulation sim;
  sim.load_data();

  if (sim.uses("gold")) {
    sim.run_gold();
  }

  if (sim.uses("accelerator")) {
    run_accelerator(sim.model, sim.selection, sim.memory("accelerator"));
  }

  return sim.check_outputs();
}
