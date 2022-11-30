#include "test/common/Simulation.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "test/common/GoldModel.h"
#include "test/common/UniversalPosit.h"
#include "test/common/Utils.h"
#include "test/mobilebert/MobileBERT.h"
#include "test/resnet/ResNet.h"

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
#endif

Simulation::Simulation() {
  model = get_env_var("NETWORK");
  if (model.empty()) model = "resnet";

  std::string tests(get_env_var("TESTS"));
  if (tests.empty()) tests = "fc";

  std::string simsEnv(get_env_var("SIMS"));
  if (simsEnv.empty()) simsEnv = "accelerator,customposit";

  // Only applicable when NETWORK=mobilebert
  std::string task(get_env_var("TASK"));
  if (task.empty()) task = "inference";

  std::string tolerance_str(get_env_var("TOLERANCE"));
  if (!tolerance_str.empty()) tolerance = std::stof(tolerance_str);

  // Paths are relative to Makefile
  std::string data_dir(get_env_var("DATA_DIR"));

  out_dir = get_env_var("OUT_DIR");
  if (out_dir.empty()) out_dir = "./test_outputs/";

  // Parse tests to run
  std::vector<std::string> tests_list;
  split_string(tests, ',', std::back_inserter(tests_list));

  if (tests_list.size() > 2) {
    std::cerr << "ERROR: Supply at max two TESTS." << std::endl;
    print_help();
    std::abort();
  }

  // Parse sims to run
  split_string(simsEnv, ',', std::back_inserter(sims));
  if (sims.size() & 0x01) {
    std::cerr << "ERROR: Need to supply even number of sim pairs." << std::endl;
    print_help();
    std::abort();
  }

  // Construct required network
  std::unique_ptr<Network> network;
  if (model == "resnet") {
    network = std::make_unique<ResNet>(data_dir);
  } else if (model == "mobilenetv2") {
    throw std::runtime_error("Not implemented.");
    // network = new MobileNetV2(data_dir);
  } else if (model == "mobilebert") {
    network = std::make_unique<MobileBERT>(data_dir, task);
  } else {
    throw std::runtime_error("Unsupported model.");
  }

  // Collect workloads (aka. layers) from Network
  workloads = network->getWorkloadsInRange(tests_list);

  std::cout << "Starting new simulation with config:";
  std::cout << "\n> Model: " << model;
  std::cout << "\n> Tests: ";
  for (const std::string& l : tests_list) std::cout << l << ' ';
  std::cout << "\n> Sims: ";
  for (const std::string& s : sims) std::cout << s << ' ';
  if (model == "mobilebert") std::cout << "\n> Task: " << task;
  std::cout << "\n> Tolerance: " << tolerance;
  std::cout << "\n> Data dir: " << network->getDataDir();
  std::cout << "\n> Out dir: " << out_dir << "\n";
  std::cout << "> SRAM: " << SRAM_MEMORY_SIZE / 1024 << " KB\n";
  std::cout << "> RRAM: " << RRAM_MEMORY_SIZE / 1024 << " KB\n" << std::endl;
}

void Simulation::loadMemory() {
  std::vector<MemoryModel*> memories;
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    acceleratorMemory = new SimpleMemoryModel<INPUT_DATATYPE>(true);
    memories.push_back(acceleratorMemory);
  }
  if (std::find(sims.begin(), sims.end(), "customposit") != sims.end()) {
    positMemory = new SimpleMemoryModel<INPUT_DATATYPE>(false);
    memories.push_back(positMemory);
  }
  if (std::find(sims.begin(), sims.end(), "universal") != sims.end()) {
    universalPositMemory = new SimpleMemoryModel<UniversalPosit>(false);
    memories.push_back(universalPositMemory);
  }
  if (std::find(sims.begin(), sims.end(), "fp32") != sims.end()) {
    floatMemory = new SimpleMemoryModel<float>(false);
    memories.push_back(floatMemory);
  }

  // Load first tests input
  for (MemoryModel* memModel : memories) {
    memModel->loadModelActivations(workloads.front().params,
                                   workloads.front().files,
                                   workloads.front().memoryMap, true);
  }

  // Load weights, biases for all layers
  for (const Workload& workload : workloads) {
    for (MemoryModel* memModel : memories) {
      memModel->loadModelParams(workload.params, workload.files,
                                workload.memoryMap, true);
    }
  }

  // load last layer reference outputs
  for (MemoryModel* memModel : memories) {
    memModel->loadReferenceOutput(workloads.back().params,
                                  workloads.back().files);
  }
}

void Simulation::run() {
  // Run tests in sequence
  int X, Y, C, K, FX, FY, STRIDE;
  for (const Workload& workload : workloads) {
    SimplifiedParams currentParams = workload.params;

    // Check if mapping valid
    if (validateMapping(currentParams) != 0) {
      std::abort();
    }

    X = currentParams.loops[0][currentParams.inputXLoopIndex[0]] *
        currentParams.loops[1][currentParams.inputXLoopIndex[1]];
    Y = currentParams.loops[0][currentParams.inputYLoopIndex[0]] *
        currentParams.loops[1][currentParams.inputYLoopIndex[1]];
    C = currentParams.loops[1][currentParams.reductionLoopIndex[1]] * DIMENSION;
    K = currentParams.loops[0][currentParams.weightLoopIndex[0]] *
        currentParams.loops[1][currentParams.weightLoopIndex[1]] * DIMENSION;
    FX = currentParams.loops[1][currentParams.fxIndex];
    FY = currentParams.loops[1][currentParams.fyIndex];
    STRIDE = currentParams.STRIDE;

    if (currentParams.REPLICATION) {
      FX = 7;
      C = 3;
    }

    if (currentParams.MAXPOOL) {
      X /= 2;
      Y /= 2;
    }

    if (currentParams.AVGPOOL) {
      X = 1;
      Y = 1;
    }

    std::cout << "Performing " + workload.name + ":" << std::endl;
    std::cout << "(" << X << "x" << Y << "x" << C << ")"
              << " * "
              << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
              << std::endl;

    // Run gold models
    if (std::find(sims.begin(), sims.end(), "customposit") != sims.end()) {
      run_custom_posit_gold_model(
          currentParams, positMemory->sram + currentParams.INPUT_OFFSET,
          positMemory->rram + currentParams.WEIGHT_OFFSET,
          positMemory->sram + currentParams.OUTPUT_OFFSET,
          positMemory->rram + currentParams.BIAS_OFFSET,
          positMemory->sram + currentParams.RESIDUAL_OFFSET, nullptr, nullptr);
    }
    if (std::find(sims.begin(), sims.end(), "universal") != sims.end()) {
      run_universal_posit_gold_model(
          currentParams,
          universalPositMemory->sram + currentParams.INPUT_OFFSET,
          universalPositMemory->rram + currentParams.WEIGHT_OFFSET,
          universalPositMemory->sram + currentParams.OUTPUT_OFFSET,
          universalPositMemory->rram + currentParams.BIAS_OFFSET,
          universalPositMemory->sram + currentParams.RESIDUAL_OFFSET, nullptr,
          nullptr);
    }
    if (std::find(sims.begin(), sims.end(), "fp32") != sims.end()) {
      run_fp_gold_model(
          currentParams, floatMemory->sram + currentParams.INPUT_OFFSET,
          floatMemory->rram + currentParams.WEIGHT_OFFSET,
          floatMemory->sram + currentParams.OUTPUT_OFFSET,
          floatMemory->rram + currentParams.BIAS_OFFSET,
          floatMemory->sram + currentParams.RESIDUAL_OFFSET, nullptr, nullptr);
    }
  }

  // Run accelerator
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    std::vector<SimplifiedParams> params_list;
    for (const Workload& workload : workloads) {
      params_list.push_back(workload.params);
    }
    // TODO: currently assumes that all layers have the same memory map
    run_op(params_list, acceleratorMemory->sram, acceleratorMemory->rram,
           workloads.front().memoryMap);
  }
}

int Simulation::checkOutput() {
  SimplifiedParams currentParams = workloads.back().params;
  int X, Y, C, K, FX, FY, STRIDE;
  X = currentParams.loops[0][currentParams.inputXLoopIndex[0]] *
      currentParams.loops[1][currentParams.inputXLoopIndex[1]];
  Y = currentParams.loops[0][currentParams.inputYLoopIndex[0]] *
      currentParams.loops[1][currentParams.inputYLoopIndex[1]];
  C = currentParams.loops[1][currentParams.reductionLoopIndex[1]] * DIMENSION;
  K = currentParams.loops[0][currentParams.weightLoopIndex[0]] *
      currentParams.loops[1][currentParams.weightLoopIndex[1]] * DIMENSION;
  FX = currentParams.loops[1][currentParams.fxIndex];
  FY = currentParams.loops[1][currentParams.fyIndex];
  STRIDE = currentParams.STRIDE;

  if (currentParams.REPLICATION) {
    FX = 7;
    C = 3;
  }

  if (currentParams.MAXPOOL) {
    X /= 2;
    Y /= 2;
  }

  if (currentParams.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  size_t size = X * Y * K;

  int error_count = 0;
  // Go over every combination of sims and cross-check results
  for (int i = 0; i < sims.size(); i += 2) {
    std::string diff_file = out_dir + model + '.' + workloads.front().name +
                            "_to_" + workloads.back().name + '.' + sims[i] +
                            "_vs_" + sims[i + 1];

    float rel_err = 0;
    if ((sims[i] == "accelerator" && sims[i + 1] == "customposit") ||
        (sims[i + 1] == "accelerator" && sims[i] == "customposit")) {
      rel_err += compare_arrays(
          acceleratorMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "accelerator",
          positMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "customposit", size, diff_file, false);
    } else if ((sims[i] == "accelerator" && sims[i + 1] == "file") ||
               (sims[i + 1] == "accelerator" && sims[i] == "file")) {
      rel_err += compare_arrays(
          acceleratorMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "accelerator", positMemory->reference, "file", size, diff_file,
          false);
    } else if ((sims[i] == "customposit" && sims[i + 1] == "file") ||
               (sims[i + 1] == "customposit" && sims[i] == "file")) {
      rel_err += compare_arrays(
          positMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "customposit", positMemory->reference, "file", size, diff_file,
          false);
    } else if ((sims[i] == "universal" && sims[i + 1] == "customposit") ||
               (sims[i + 1] == "universal" && sims[i] == "customposit")) {
      rel_err += compare_arrays(
          positMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "universal",
          universalPositMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "customposit", size, diff_file, false);
    } else if ((sims[i] == "universal" && sims[i + 1] == "file") ||
               (sims[i + 1] == "universal" && sims[i] == "file")) {
      rel_err += compare_arrays(
          universalPositMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "universal", universalPositMemory->reference, "file", size, diff_file,
          false);
    } else if ((sims[i] == "fp32" && sims[i + 1] == "file") ||
               (sims[i + 1] == "fp32" && sims[i] == "file")) {
      rel_err += compare_arrays(
          floatMemory->sram + workloads.back().params.OUTPUT_OFFSET, "fp32",
          floatMemory->reference, "file", size, diff_file, false);
    } else if ((sims[i] == "customposit" && sims[i + 1] == "fp32") ||
               (sims[i] == "fp32" && sims[i + 1] == "customposit")) {
      rel_err += compare_arrays(
          positMemory->sram + workloads.back().params.OUTPUT_OFFSET,
          "customposit",
          floatMemory->sram + workloads.back().params.OUTPUT_OFFSET, "fp32",
          size, diff_file, false);
    } else {
      std::cerr << "ERROR: Comparison between " + sims[i] + " and "
                << sims[i + 1] << " not supported." << std::endl;
      std::abort();
    }

    if (rel_err > tolerance) error_count += rel_err < 1.0 ? 1 : (int)rel_err;
    std::cout << "Rela. error: " << rel_err << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
  }

  return error_count;
}

// Iterate over string and return vector of substrings cut at delim
template <typename T>
void Simulation::split_string(const std::string& in_string, char delim,
                              T result) {
  std::istringstream ss(in_string);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *result++ = item;
  }
}

// Return environment variable
std::string Simulation::get_env_var(std::string const& name) {
  const char* val = std::getenv(name.c_str());
  return val == NULL ? std::string() : std::string(val);
}

void Simulation::print_help() {
  std::cout
      << "\nConfigure simulator by using environment variables."
      << "\n NETWORK - Type of network to run {mobilebert, resnet}"
      << "\n TESTS - Layers in network to run. Either single or tuple: "
         "<first>,<last>."
      << "\n SIMS - Simulators / models to compare {accelerator, "
         "customposit, universal, fp32, file}."
      << "\n TASK - MobileBERT run time (forward, backward, e2e)."
      << "\n TOLERANCE - Relative normalized error in % we allow (default 10)."
      << "\n DATA_DIR - Path to binary input data."
      << "\n OUT_DIR - Path to output data." << std::endl;
}