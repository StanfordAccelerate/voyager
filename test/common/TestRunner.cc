// This file is the main entry point to running event-level SystemC simulations.
// It can run any model, such as ResNet and mobileBert-Tiny
// Parameters are passed via environment variables, not arguments.

#include "test/common/Simulation.h"

extern "C" int sc_main(int argc, char* argv[]) {
  Simulation sim;
  sim.load_data();
  sim.run();
  return sim.check_outputs();
}
