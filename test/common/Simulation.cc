#include "test/common/Simulation.h"

#include <iostream>
#include <string>

#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
#include "test/common/UniversalPosit.h"
#include "test/common/Utils.h"
#include "test/mobilebert/MobileBERT.h"
#include "test/resnet/ResNet18.h"

// TODO(fpedd): These defines get overwritten from other files...

// By default we have 2MB of SRAM per MINOTAUR SoC
// organized as 8x 256KB Banks with 2x 128KB Macros each
#ifndef SRAM_MEMORY_SIZE
#define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
#endif

// By default we have 12MB of RRAM per MINOTAUR SoC
// organized as 12x 1MB Banks with 4x 256KB Macros each
#ifndef RRAM_MEMORY_SIZE
#define RRAM_MEMORY_SIZE (12 * 1024 * 1024)
#endif

#ifdef SOC_COSIM
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
#else
#include <filesystem>
#endif

// Iterate over string and return vector of substrings
template <typename T>
void split_string(const std::string& in_string, char delim, T result) {
  std::istringstream ss(in_string);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *result++ = item;
  }
}

Simulation::Simulation(int argc, char* argv[]) {
#ifndef SOC_COSIM
  if (argc > 1) {
    std::cerr
        << "ERROR: Don't supply command line arguments, instead use env vars."
        << std::endl;
    print_help();
    std::abort();
  }
#endif

  // TODO(fpedd): Implement more cmd line arg tests
  model = get_env_var("NETWORK");
  if (model.empty()) model = "simple";

  std::string tests(get_env_var("TESTS"));
  if (tests.empty()) tests = "simple";

  std::string simsEnv(get_env_var("SIMS"));
  if (simsEnv.empty()) simsEnv = "accelerator,customposit";

  // Only applicable when NETWORK=mobilebert
  std::string task(get_env_var("TASK"));
  if (task.empty()) task = "inference";

  std::string tolerance_str(get_env_var("TOLERANCE"));
  tolerance = 0.1;
  if (!tolerance_str.empty()) tolerance = std::stof(tolerance_str);

  // Paths are relative to Makefile
  std::string data_dir(get_env_var("DATA_DIR"));
  if (data_dir.empty()) {
    if (model == "resnet") {
      std::string raw_data_dir = "./models/resnet/binary_data/";
      data_dir = (*std::filesystem::begin(
                      std::filesystem::directory_iterator(raw_data_dir)))
                     .path()
                     .string() +
                 '/';
    } else if (model == "mobilebert") {
      data_dir = "./data/mobilebert_tiny/datafile/step0/";
    }
  }

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

  std::cout << "Starting new simulation with config:";
  std::cout << "\n> Model: " << model;
  std::cout << "\n> Tests: ";
  for (const std::string& l : tests_list) std::cout << l << ' ';
  std::cout << "\n> Sims: ";
  for (const std::string& s : sims) std::cout << s << ' ';
  if (model == "mobilebert") std::cout << "\n> Task: " << task;
  std::cout << "\n> Tolerance: " << tolerance;
  std::cout << "\n> Data dir: " << data_dir;
  std::cout << "\n> Out dir: " << out_dir << "\n";
  std::cout << "> SRAM: " << SRAM_MEMORY_SIZE / 1024 << " KB\n";
  std::cout << "> RRAM: " << RRAM_MEMORY_SIZE / 1024 << " KB\n" << std::endl;

  if (model == "resnet") {
    ResNet18 resnet18;
    workloads = resnet18.getWorkloads(tests_list);
  } else if (model == "mobilebert") {
    MobileBERT mobileBERT(task);
    workloads = mobileBERT.getWorkloads(tests_list);
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

int Simulation::run() {
  // Set data parameters
  bool use_data_file = true;

  // TODO: clean this up, and enable more reuse from the SoC simulation
  // Memory allocation
  INPUT_DATATYPE* acc_sram_memory = nullptr;
  INPUT_DATATYPE* acc_rram_memory = nullptr;
  INPUT_DATATYPE* hls_gold_sram_memory = nullptr;
  INPUT_DATATYPE* hls_gold_rram_memory = nullptr;
  UniversalPosit* uni_gold_sram_memory = nullptr;
  UniversalPosit* uni_gold_rram_memory = nullptr;
  float* float_gold_sram_memory = nullptr;
  float* float_gold_rram_memory = nullptr;
  uint64_t* trash = nullptr;
  try {
    acc_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
    acc_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
    hls_gold_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
    hls_gold_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
    uni_gold_sram_memory = new UniversalPosit[SRAM_MEMORY_SIZE];
    uni_gold_rram_memory = new UniversalPosit[RRAM_MEMORY_SIZE];
    float_gold_sram_memory = new float[SRAM_MEMORY_SIZE];
    float_gold_rram_memory = new float[RRAM_MEMORY_SIZE];
    trash = new uint64_t[RRAM_MEMORY_SIZE];
  } catch (const std::bad_alloc&) {
    throw std::runtime_error("ERROR: Failed to allocate simulation memory");
  }

  // Load first tests input
  load_memory(workloads.front().params, workloads.front().files,
              workloads.front().memoryMap, use_data_file, acc_sram_memory,
              acc_rram_memory,
              hls_gold_sram_memory + workloads.front().params.INPUT_OFFSET,
              (INPUT_DATATYPE*)trash, (INPUT_DATATYPE*)trash,
              hls_gold_sram_memory + workloads.front().params.RESIDUAL_OFFSET,
              (INPUT_DATATYPE*)trash, (INPUT_DATATYPE*)trash,
              uni_gold_sram_memory + workloads.front().params.INPUT_OFFSET,
              (UniversalPosit*)trash, (UniversalPosit*)trash,
              uni_gold_sram_memory + workloads.front().params.RESIDUAL_OFFSET,
              (UniversalPosit*)trash, (UniversalPosit*)trash,
              float_gold_sram_memory + workloads.front().params.INPUT_OFFSET,
              (float*)trash, (float*)trash,
              float_gold_sram_memory + workloads.front().params.RESIDUAL_OFFSET,
              (float*)trash, (float*)trash);

  // Load weights, biases, and sims
  for (const Workload& workload : workloads) {
    load_wb(workload.params, workload.files, workload.memoryMap, use_data_file,
            acc_sram_memory, acc_rram_memory, (INPUT_DATATYPE*)trash,
            hls_gold_rram_memory + workload.params.WEIGHT_OFFSET,
            hls_gold_rram_memory + workload.params.BIAS_OFFSET,
            (INPUT_DATATYPE*)trash, (INPUT_DATATYPE*)trash,
            (INPUT_DATATYPE*)trash, (UniversalPosit*)trash,
            uni_gold_rram_memory + workload.params.WEIGHT_OFFSET,
            uni_gold_rram_memory + workload.params.BIAS_OFFSET,
            (UniversalPosit*)trash, (UniversalPosit*)trash,
            (UniversalPosit*)trash, (float*)trash,
            float_gold_rram_memory + workload.params.WEIGHT_OFFSET,
            float_gold_rram_memory + workload.params.BIAS_OFFSET, (float*)trash,
            (float*)trash, (float*)trash);
  }

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
          currentParams, hls_gold_sram_memory + currentParams.INPUT_OFFSET,
          hls_gold_rram_memory + currentParams.WEIGHT_OFFSET,
          hls_gold_sram_memory + currentParams.OUTPUT_OFFSET,
          hls_gold_rram_memory + currentParams.BIAS_OFFSET,
          hls_gold_sram_memory + currentParams.RESIDUAL_OFFSET, nullptr,
          nullptr);
    }
    if (std::find(sims.begin(), sims.end(), "universal") != sims.end()) {
      run_universal_posit_gold_model(
          currentParams, uni_gold_sram_memory + currentParams.INPUT_OFFSET,
          uni_gold_rram_memory + currentParams.WEIGHT_OFFSET,
          uni_gold_sram_memory + currentParams.OUTPUT_OFFSET,
          uni_gold_rram_memory + currentParams.BIAS_OFFSET,
          uni_gold_sram_memory + currentParams.RESIDUAL_OFFSET, nullptr,
          nullptr);
    }
    if (std::find(sims.begin(), sims.end(), "fp32") != sims.end()) {
      run_fp_gold_model(currentParams,
                        float_gold_sram_memory + currentParams.INPUT_OFFSET,
                        float_gold_rram_memory + currentParams.WEIGHT_OFFSET,
                        float_gold_sram_memory + currentParams.OUTPUT_OFFSET,
                        float_gold_rram_memory + currentParams.BIAS_OFFSET,
                        float_gold_sram_memory + currentParams.RESIDUAL_OFFSET,
                        nullptr, nullptr);
    }
  }

  // Run accelerator
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    std::vector<SimplifiedParams> params_list;
    for (const Workload& workload : workloads) {
      params_list.push_back(workload.params);
    }
    // TODO: currently assumes that all layers have the same memory map
    run_op(params_list, acc_sram_memory, acc_rram_memory,
           workloads.front().memoryMap);
  }

  // Allocate comparison
  INPUT_DATATYPE* hls_comp = nullptr;
  UniversalPosit* uni_comp = nullptr;
  float* fp_comp = nullptr;
  size_t size = X * Y * K;

  try {
    hls_comp = new INPUT_DATATYPE[size];
    uni_comp = new UniversalPosit[size];
    fp_comp = new float[size];
  } catch (const std::bad_alloc&) {
    throw std::runtime_error("ERROR: Failed to allocate comparison memory");
  }

  // Load reference values from file
  if (use_data_file) {
    load_datafile_outputs(workloads.back().params,
                          workloads.back().files.outputs_file, hls_comp,
                          uni_comp, fp_comp);
  }

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
          acc_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "accelerator",
          hls_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "customposit", size, diff_file, false);
    } else if ((sims[i] == "accelerator" && sims[i + 1] == "file") ||
               (sims[i + 1] == "accelerator" && sims[i] == "file")) {
      rel_err += compare_arrays(
          acc_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "accelerator", hls_comp, "file", size, diff_file, false);
    } else if ((sims[i] == "customposit" && sims[i + 1] == "file") ||
               (sims[i + 1] == "customposit" && sims[i] == "file")) {
      rel_err += compare_arrays(
          hls_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "customposit", hls_comp, "file", size, diff_file, false);
    } else if ((sims[i] == "universal" && sims[i + 1] == "customposit") ||
               (sims[i + 1] == "universal" && sims[i] == "customposit")) {
      rel_err += compare_arrays(
          hls_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "universal",
          uni_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "customposit", size, diff_file, false);
    } else if ((sims[i] == "universal" && sims[i + 1] == "file") ||
               (sims[i + 1] == "universal" && sims[i] == "file")) {
      rel_err += compare_arrays(
          uni_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "universal", uni_comp, "file", size, diff_file, false);
    } else if ((sims[i] == "fp32" && sims[i + 1] == "file") ||
               (sims[i + 1] == "fp32" && sims[i] == "file")) {
      rel_err += compare_arrays(
          float_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "fp32", fp_comp, "file", size, diff_file, false);
    } else if ((sims[i] == "customposit" && sims[i + 1] == "fp32") ||
               (sims[i] == "fp32" && sims[i + 1] == "customposit")) {
      rel_err += compare_arrays(
          hls_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "customposit",
          float_gold_sram_memory + workloads.back().params.OUTPUT_OFFSET,
          "fp32", size, diff_file, false);
    } else {
      std::cerr << "ERROR: Comparison between " + sims[i] + " and "
                << sims[i + 1] << " not supported." << std::endl;
      std::abort();
    }

    if (rel_err > tolerance) error_count += rel_err < 1.0 ? 1 : (int)rel_err;
    std::cout << "Rela. error: " << rel_err << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
  }

  delete[] acc_sram_memory;
  delete[] acc_rram_memory;
  delete[] hls_gold_sram_memory;
  delete[] hls_gold_rram_memory;
  delete[] uni_gold_sram_memory;
  delete[] uni_gold_rram_memory;
  delete[] float_gold_sram_memory;
  delete[] float_gold_rram_memory;
  delete[] trash;
  delete[] hls_comp;
  delete[] uni_comp;
  delete[] fp_comp;

  return error_count;
}
