#include "test/common/Simulation.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"

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
  if (model.empty()) {
    model = "resnet18";
  }

  tests = get_env_var("TESTS");
  if (tests.empty()) {
    tests = "submodule_0";
  }

  std::vector<std::string> tests_list;
  split_string(tests, ',', std::back_inserter(tests_list));
  if (tests_list.size() > 2) {
    throw std::runtime_error("Incorrect number of tests specified.");
  }

  std::string sims_str(get_env_var("SIMS"));
  if (sims_str.empty()) {
    sims_str = "fp32,file";
  }

  split_string(sims_str, ',', std::back_inserter(sims));
  if (sims.size() != 2) {
    throw std::runtime_error("Incorrect number of simulators specified.");
  }

  std::string tolerance_str(get_env_var("TOLERANCE"));
  if (!tolerance_str.empty()) {
    tolerance = std::stof(tolerance_str);
  }

  out_dir = get_env_var("OUT_DIR");
  if (out_dir.empty()) {
    out_dir = "./test_outputs/";
  }

  // Get list of params to run
  auto network = new Network(model);
  params = network->get_params(tests_list);

  std::cout << "Starting new simulation with config:";
  std::cout << "\n> Model: " << model;
  std::cout << "\n> Tests: ";
  for (const std::string& t : tests_list) std::cout << t << ' ';
  std::cout << "\n> Sims: ";
  for (const std::string& s : sims) std::cout << s << ' ';
  std::cout << "\n> Tolerance: " << tolerance;
  std::cout << "\n> Output dir: " << out_dir << "\n";
  std::cout << "> SRAM: " << SRAM_MEMORY_SIZE / 1024 << " KB\n";
  std::cout << "> RRAM: " << RRAM_MEMORY_SIZE / 1024 << " KB\n";
}

void Simulation::load_data() {
  std::vector<int> memory_sizes{SRAM_MEMORY_SIZE, RRAM_MEMORY_SIZE,
                                REFERENCE_MEMORY_SIZE};
  if (std::find(sims.begin(), sims.end(), "fp32") != sims.end()) {
    memories["fp32"] = new ArrayMemory<float>(memory_sizes);
    dataLoaders["fp32"] = new DataLoader(memories["fp32"], false);
  }
  if (std::find(sims.begin(), sims.end(), "systemc") != sims.end()) {
    memories["systemc"] = new ArrayMemory<INPUT_DATATYPE>(memory_sizes);
    dataLoaders["systemc"] = new DataLoader(memories["systemc"], false);
  }
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    memories["accelerator"] = new ArrayMemory<INPUT_DATATYPE>(memory_sizes);
    dataLoaders["accelerator"] = new DataLoader(memories["accelerator"], true);
  }

  std::string project_root = std::string(getenv("PROJECT_ROOT"));
  std::string data_dir =
      project_root + "/test/compiler/networks/" + model + "/tensor_files";
  for (const auto& [key, dataLoader] : dataLoaders) {
    dataLoader->load_inputs(params.front(), data_dir);
    dataLoader->load_outputs(params.back(), data_dir);
    for (const auto& param : params) {
      dataLoader->load_weights(param, data_dir);
    }
  }
}

void Simulation::run() {
  // Run gold models
  for (const auto& param : params) {
    if (std::find(sims.begin(), sims.end(), "fp32") != sims.end()) {
      auto memory = (ArrayMemory<float>*)(memories["fp32"]);
      auto args = memory->get_args(param);
      run_gold_model(param, args);
    }

    if (std::find(sims.begin(), sims.end(), "systemc") != sims.end()) {
      auto memory = (ArrayMemory<INPUT_DATATYPE>*)(memories["systemc"]);
      auto args = memory->get_args(param);
      run_gold_model(param, args);
    }
  }

  // Run accelerator
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    auto memory = (ArrayMemory<INPUT_DATATYPE>*)(memories["accelerator"]);
    run_gold_model(params, memory->memories[0], memory->memories[1]);
  }
}

int Simulation::check_outputs() {
  std::string prefix;
  if (params.size() == 1) {
    prefix = out_dir + model + '.' + params.front().name() + '.';
  } else {
    prefix = out_dir + model + '.' + params.front().name() + "_to_" +
             params.back().name() + '.';
  }

  bool has_valid_comp = false;
  double rel_err = 0.0;

  auto param = params.back();
  int size = 1;
  for (const auto& dim : param.output().shape()) size *= dim;

  bool file = std::find(sims.begin(), sims.end(), "file") != sims.end();

  auto fp32_memory = (ArrayMemory<float>*)memories["fp32"];
  auto systemc_memory = (ArrayMemory<INPUT_DATATYPE>*)memories["systemc"];
  auto accelerator_memory =
      (ArrayMemory<INPUT_DATATYPE>*)memories["accelerator"];

  if (fp32_memory && file) {
    std::cout << "FP32 PyTorch Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    std::string filename = prefix + "fp32_vs_pytorch.txt";

    rel_err += compare_arrays(fp32_memory->get_args(param).back(), "fp32",
                              fp32_memory->get_output(param), "file", size,
                              filename, false);
    has_valid_comp = true;
  }

  if (systemc_memory && file) {
    std::cout << "System C Gold Model vs. Pytorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    std::string filename = prefix + "systemc_vs_pytorch.txt";

    rel_err += compare_arrays(systemc_memory->get_args(param).back(), "systemc",
                              systemc_memory->get_output(param), "file", size,
                              filename, false);
    has_valid_comp = true;
  }

  if (accelerator_memory && file) {
    std::cout << "Accelerator vs. System C Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    std::string filename = prefix + "accelerator_vs_pytorch.txt";

    rel_err += compare_arrays(
        accelerator_memory->get_args(param).back(), "accelerator",
        accelerator_memory->get_output(param), "file", size, filename, false);
    has_valid_comp = true;
  }

  if (accelerator_memory && systemc_memory) {
    std::cout << "Accelerator vs. System C Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    std::string filename = prefix + "accelerator_vs_systemc.txt";

    rel_err +=
        compare_arrays(accelerator_memory->get_args(param).back(),
                       "accelerator", systemc_memory->get_args(param).back(),
                       "systemc", size, filename, false);
    has_valid_comp = true;
  }

  if (!has_valid_comp) {
    std::cout << "No valid comparisons specified" << std::endl;
    std::abort();
  }

  int error_count = 0;
  if (rel_err > tolerance) {
    error_count += rel_err < 1.0 ? 1 : static_cast<int>(rel_err);
  }

  std::cout << "Rela. error: " << rel_err << std::endl;
  std::cout << "Error count: " << error_count << std::endl;

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
  std::cout << "\nConfigure simulator by using environment variables."
            << "\n NETWORK - Type of network to run {mobilebert, resnet}"
            << "\n TESTS - Layers in network to run. Either single or tuple: "
               "<first>,<last>."
            << "\n SIMS - Simulators / models to compare {accelerator, "
               "customposit, universal, fp32, file}."
            << "\n TASK - MobileBERT run time (forward, backward, e2e)."
            << "\n TOLERANCE - Relative normalized error in % we allow "
               "(default 10)."
            << "\n DATA_DIR - Path to binary input data."
            << "\n OUT_DIR - Path to output data." << std::endl;
}
