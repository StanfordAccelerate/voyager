#include "test/generic/Generic.h"

Generic::Generic() : Network("generic", "") {
  SimplifiedParams nop;
  nop.INPUT_OFFSET = 0;
  for (int j = 0; j < 6; j++) {
    nop.loops[0][j] = 1;
  }
  nop.loops[1][0] = 1;
  nop.loops[1][1] = 1;
  nop.loops[1][2] = 1;
  nop.loops[1][3] = 1023;
  nop.loops[1][4] = 1023;
  nop.loops[1][5] = 1023;
  nop.inputXLoopIndex[0] = 0;
  nop.inputXLoopIndex[1] = 5;
  nop.inputYLoopIndex[0] = 1;
  nop.inputYLoopIndex[1] = 4;
  nop.reductionLoopIndex[0] = 3;
  nop.reductionLoopIndex[1] = 0;
  nop.weightLoopIndex[0] = 2;
  nop.weightLoopIndex[1] = 1;
  nop.fxIndex = 3;
  nop.fyIndex = 2;
  nop.weightReuseIndex[0] = 4;
  nop.weightReuseIndex[1] = 5;
  nop.STRIDE = 1;
  nop.NOP = true;
  // force all banks to be on
  for (int i = 0; i < NUM_SRAM_BANKS; i++) {
    nop.sram_banks[i] = ON;
  }
  for (int i = 0; i < NUM_RRAM_BANKS; i++) {
    nop.rram_banks[i] = ON;
  }
  nop.bandwidth_mode = QUAD;

  params["sram_access"] = nop;
  order = {"sram_access"};
  memoryMap["sram_access"] = {SRAM, SRAM, SRAM, SRAM, SRAM};

  params["rram_access"] = nop;
  order.push_back("rram_access");
  memoryMap["rram_access"] = {RRAM, RRAM, RRAM, RRAM, RRAM};
}

std::vector<Workload> Generic::getWorkloads(
    const std::vector<std::string> &layers) const {
  std::vector<Workload> workloads;
  for (const std::string &layer : layers) {
    Workload workload;
    workload.name = layer;
    workload.params = params.at(layer);
    workload.memoryMap = memoryMap.at(layer);
    workloads.push_back(workload);
  }
  return workloads;
}

std::vector<Workload> Generic::getWorkloadsInRange(
    const std::vector<std::string> &workloads) {
  return getWorkloads(workloads);
}

std::vector<Workload> Generic::getAllWorkloads() { return getWorkloads(order); }