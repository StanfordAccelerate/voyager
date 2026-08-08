#pragma once
#define NO_SYSC

#include <map>
#include <string>
#include <vector>

#include "test/common/ArrayMemory.h"
#include "test/common/Checker.h"
#include "test/common/Model.h"

// One simulation, configured from the environment.
//
//   NETWORK    the model to run           (default resnet18)
//   TESTS      the layer to run           (default: the whole graph)
//   SIMS       the pair to compare        (default gold,pytorch)
//   RTOL       isclose relative tolerance (default 0.05 vs pytorch, else 0)
//   ATOL       isclose absolute tolerance (default 0.1 vs pytorch, else 0)
//   OUT_DIR    where diffs are written    (default ./test_outputs/)
//   MAX_TILES  bound outer tile loops     (default: run them all)
//
// SIMS names two of {gold, accelerator, pytorch}. Each gets its own memory, is
// run independently over the same graph, and the two are compared at the end:
//
//   gold,pytorch      the C++ kernels against the compiler's reference tensors
//                     (the run's DRAM checkpoints)
//   gold,accelerator  the C++ kernels against the DUT, grading the selection's
//                     DRAM outputs and scratchpad compute results
//
// "pytorch" is not executed -- its memory is just filled with the reference
// dumps -- so gold,pytorch needs no hardware at all.
//
// The walk is the same whichever simulators are named; only what executes an
// accelerator operation differs. The accelerator run therefore lives in the
// caller, because it is the only part that needs SystemC.
class Simulation {
 public:
  Simulation();
  ~Simulation();

  void load_data();

  // Runs the graph with the bit-accurate C++ kernels.
  void run_gold();

  // Compares the two simulators named by SIMS and prints "Error count: N",
  // which is what the regression greps for.
  int check_outputs();

  bool uses(const std::string& sim) const;
  ArrayMemory* memory(const std::string& sim) const;

  std::string model_name;
  Model model;

  // The top-level operations to run, and which of them MAX_TILES bounds.
  Model::Selection selection;

 private:
  std::vector<std::string> sims;
  std::map<std::string, ArrayMemory*> memories;
  std::vector<const voyager::TensorBox*> live_in;
  std::vector<const voyager::TensorBox*> live_out;
  std::string out_dir;
  float rtol;
  float atol;
  bool check_all_writes = false;
  bool partial_run = false;
};
