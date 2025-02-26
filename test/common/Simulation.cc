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

  std::string tests_str(get_env_var("TESTS"));
  if (tests_str.empty()) {
    tests_str = "submodule_0";
  }

  split_string(tests_str, ',', std::back_inserter(tests));
  if (tests.size() > 2) {
    throw std::runtime_error("Incorrect number of tests specified.");
  }

  std::string sims_str(get_env_var("SIMS"));
  if (sims_str.empty()) {
    sims_str = "gold,pytorch";
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

  // Get list of operations to run
  network = new Network(model);
  operations = network->get_operations(tests);

  std::cout << "Starting new simulation with config:";
  std::cout << "\n> Model: " << model;
  std::cout << "\n> Tests: ";
  for (const std::string& t : tests) std::cout << t << ' ';
  std::cout << "\n> Sims: ";
  for (const std::string& s : sims) std::cout << s << ' ';
  std::cout << "\n> Tolerance: " << tolerance;
  std::cout << "\n> Output dir: " << out_dir << "\n";
  std::cout << "> SRAM: " << SRAM_MEMORY_SIZE / 1024 << " KB\n";
}

Simulation::~Simulation() {
  for (const auto& [key, memory] : memories) {
    delete memory;
  }
  for (const auto& [key, dataLoader] : dataLoaders) {
    delete dataLoader;
  }
}

void Simulation::load_data() {
  std::vector<long long> memory_sizes{SRAM_MEMORY_SIZE, REFERENCE_MEMORY_SIZE};

  if (std::find(sims.begin(), sims.end(), "gold") != sims.end()) {
    memories["gold"] = new ArrayMemory(memory_sizes);
    dataLoaders["gold"] = new DataLoader(memories["gold"], false);
  }
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    memories["accelerator"] = new ArrayMemory(memory_sizes);
    dataLoaders["accelerator"] = new DataLoader(memories["accelerator"], true);
  }

  std::string project_root = std::string(getenv("PROJECT_ROOT"));
  std::string datatype = std::string(getenv("DATATYPE"));
  std::string data_dir = project_root + "/" +
                         std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                         model + "/" + datatype + "/tensor_files";

  const auto params_to_load = network->get_operations(tests, false);

  for (const auto& [key, dataLoader] : dataLoaders) {
    dataLoader->load_inputs(operations.front().param, data_dir);
    dataLoader->load_outputs(operations.back().param, data_dir);
    for (const auto& operation : operations) {
      dataLoader->load_weights(operation.param, data_dir);
    }
  }

  std::cout << "Data loaded successfully" << std::endl;
}

void Simulation::print_ideal_runtime(const codegen::Operator& param) {
  long cycles;
  if (param.has_matrix_op()) {
    // the total number of operations is X*Y*C*FX*FY*K.
    long num_ops = 1;

    for (const auto& dim : param.output().shape()) num_ops *= dim;  // X * Y * K

    // skip the first dimension (K) since it is already accounted for
    for (int i = 1; i < param.matrix_op().weight().shape_size(); i++) {
      num_ops *= param.matrix_op().weight().shape(i);  // FX * FY * C
    }

    cycles = num_ops / (IC_DIMENSION * OC_DIMENSION);
    std::cout << param.name() << ", matrix unit ideal runtime: ";
  } else {
    long num_ops = 1;
    for (const auto& dim : param.output().shape()) num_ops *= dim;

    cycles = num_ops / OC_DIMENSION;
    std::cout << param.name() << ", vector unit ideal runtime: ";
  }
  // read CLOCK_PERIOD from environment
  int clock_period =
      std::getenv("CLOCK_PERIOD") ? std::stoi(std::getenv("CLOCK_PERIOD")) : 1;
  std::cout << cycles * clock_period << " ns" << std::endl;
}

void Simulation::run() {
  // Run gold models
  for (const auto& operation : operations) {
    print_ideal_runtime(operation.param);

    if (std::find(sims.begin(), sims.end(), "gold") != sims.end()) {
      auto memory = (ArrayMemory*)(memories["gold"]);
      auto args = memory->get_args(operation.param);
      run_gold_model(operation, args);
    }
  }

  // Run accelerator test
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    auto memory = (ArrayMemory*)(memories["accelerator"]);
    run_accelerator(operations, memory->memories[0]);
  }
}

int Simulation::check_outputs() {
  std::string prefix;
  if (operations.size() == 1) {
    prefix = out_dir + model + '.' + operations.front().name + '.';
  } else {
    prefix = out_dir + model + '.' + operations.front().name + "_to_" +
             operations.back().name + '.';
  }

  bool has_valid_comp = false;
  double rel_err = 0.0;

  auto param = operations.back().param;
  int size = 1;
  for (const auto& dim : param.output().shape()) size *= dim;

  bool pytorch = std::find(sims.begin(), sims.end(), "pytorch") != sims.end();

  auto gold_memory = (ArrayMemory*)memories["gold"];
  auto accel_memory = (ArrayMemory*)memories["accelerator"];

  std::string filename;
  std::any output1, output2;
  std::string output_names[2];

  if (gold_memory && pytorch) {
    std::cout << "Gold Model vs. PyTorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    filename = prefix + "gold_vs_pytorch.txt";

    output_names[0] = "gold_model";
    output1 = gold_memory->get_output(param);

    output_names[1] = "pytorch";
    output2 = gold_memory->get_reference_output(param);

    has_valid_comp = true;
  }

  if (accel_memory && pytorch) {
    std::cout << "Accelerator vs. PyTorch" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    filename = prefix + "accelerator_vs_pytorch.txt";

    output_names[0] = "accelerator";
    output1 = accel_memory->get_output(param);

    output_names[1] = "pytorch";
    output2 = accel_memory->get_reference_output(param);

    has_valid_comp = true;
  }

  if (accel_memory && gold_memory) {
    std::cout << "Accelerator vs. Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    filename = prefix + "accelerator_vs_gold.txt";

    output_names[0] = "accelerator";
    output1 = accel_memory->get_output(param);

    output_names[1] = "gold_model";
    output2 = gold_memory->get_output(param);

    has_valid_comp = true;
  }

  if (has_valid_comp) {
    if (param.output().dtype() == "int8") {
      rel_err += compare_arrays<DataTypes::int8, DataTypes::int8>(
          output1, output_names[0], output2, output_names[1], size, filename,
          false);
    } else if (param.output().dtype() == "e8m0") {
      rel_err += compare_arrays<DataTypes::e8m0, DataTypes::e8m0>(
          output1, output_names[0], output2, output_names[1], size, filename,
          false);
    } else if (param.output().dtype() == "bfloat16") {
      rel_err += compare_arrays<DataTypes::bfloat16, DataTypes::bfloat16>(
          output1, output_names[0], output2, output_names[1], size, filename,
          false);
    } else {
      // if unspecified, we will assume it's INPUT_DATATYPE
      rel_err += compare_arrays<INPUT_DATATYPE, INPUT_DATATYPE>(
          output1, output_names[0], output2, output_names[1], size, filename,
          false);
    }
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
