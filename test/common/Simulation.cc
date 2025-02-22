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

  network = new Network(model);
  params = network->get_params(tests);

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

  bool is_cnn = model == "resnet18" || model == "resnet50";

  if (std::find(sims.begin(), sims.end(), "gold") != sims.end()) {
    memories["gold"] = new ArrayMemory(memory_sizes);
    dataLoaders["gold"] = new DataLoader(memories["gold"], false, is_cnn);
  }
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    memories["accelerator"] = new ArrayMemory(memory_sizes);
    dataLoaders["accelerator"] =
        new DataLoader(memories["accelerator"], true, is_cnn);
  }

  std::string project_root = std::string(getenv("PROJECT_ROOT"));
  std::string datatype = std::string(getenv("DATATYPE"));
  std::string data_dir = project_root + "/" +
                         std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                         model + "/" + datatype + "/tensor_files";

  const auto params_to_load = network->get_params(tests);

  // Fully connected layer, or linear ops with input of a 1D tensor, is run on
  // the vector unit. Its weight does not need to be transposed. We check if an
  // op is a FC by checking its output dimension. This should be improved in the
  // future.
  int num_classes;
  if (model == "mobilebert" || model == "bert") {
    num_classes = 2;
  } else if (model == "resnet18" || model == "resnet50") {
    num_classes = 1000;
  }

  for (const auto& [key, dataloader] : dataLoaders) {
    dataloader->load_inputs(params_to_load.front(), data_dir);
    dataloader->load_outputs(params_to_load.back(), data_dir);

    for (const auto& tensor : network->model.parameters()) {
      bool has_transpose = tensor.shape(0) != num_classes;
      dataloader->load_tensor(tensor, data_dir, has_transpose);
    }
  }

  std::cout << "Data loaded successfully" << std::endl;
}

void Simulation::print_ideal_runtime(const codegen::Operation& param) {
  const auto op_list = get_op_list(param);
  const auto first_op = op_list.front();

  long cycles;

  if (GEMM_OPS.find(first_op.target()) != GEMM_OPS.end()) {
    const auto input = first_op.kwargs().at("input").tensor();

    bool is_matmul = first_op.target().find("matmul") != std::string::npos;
    std::string weight_key = is_matmul ? "other" : "weight";
    const auto weight = first_op.kwargs().at(weight_key).tensor();

    // the total number of operations is X * Y * C * FX * FY * K.
    long num_ops = get_size(input) * get_size(weight) / weight.shape(0);
    cycles = num_ops / (IC_DIMENSION * OC_DIMENSION);
    std::cout << get_op_name(param) << ", matrix unit ideal runtime: ";
  } else {
    long num_ops;
    if (param.has_output()) {
      num_ops = get_size(param.output());
    } else if (op_list.back().target() == "quantize_mx") {
      num_ops = get_size(param.outputs().tensors(1));
    } else {
      std::cerr << "Unrecognized operation: " << get_op_name(param)
                << std::endl;
    }

    cycles = num_ops / OC_DIMENSION;
    std::cout << get_op_name(param) << ", vector unit ideal runtime: ";
  }

  char* clock_period = std::getenv("CLOCK_PERIOD");
  long clock_period_ns = clock_period ? std::stoi(clock_period) : 1;
  std::cout << cycles * clock_period_ns << " ns" << std::endl;
}

void Simulation::run() {
  // Run gold models
  for (const auto& op : params) {
    print_ideal_runtime(op);

    if (std::find(sims.begin(), sims.end(), "gold") != sims.end()) {
      const auto memory = (ArrayMemory*)(memories["gold"]);
      const auto kwargs = memory->get_args(op);
      const auto outputs = run_gold_model(op, kwargs);
      const auto tensors = get_op_outputs(op);

      assert(outputs.size() == tensors.size());

      for (int i = 0; i < outputs.size(); i++) {
        memory->write_tensor(tensors[i], outputs[i]);
      }
    }
  }

  // Run accelerator test
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    auto memory = (ArrayMemory*)(memories["accelerator"]);
    run_accelerator(params, memory->memories[0]);
  }
}

int Simulation::check_outputs() {
  std::string prefix;
  if (params.size() == 1) {
    prefix = out_dir + model + '.' + get_op_name(params.front()) + '.';
  } else {
    prefix = out_dir + model + '.' + get_op_name(params.front()) + "_to_" +
             get_op_name(params.back()) + '.';
  }

  const auto param = params.back();

  bool pytorch = std::find(sims.begin(), sims.end(), "pytorch") != sims.end();
  auto gold_memory = (ArrayMemory*)memories["gold"];
  auto accel_memory = (ArrayMemory*)memories["accelerator"];

  std::string filename;
  std::vector<std::any> outputs1, outputs2;
  std::string output_names[2];

  bool has_valid_comp = false;

  if (gold_memory && pytorch) {
    std::cout << "Gold Model vs. PyTorch" << std::endl;
    std::cout << "(reveals issues in data loading or mapping)" << std::endl;
    filename = prefix + "gold_vs_pytorch";

    output_names[0] = "gold_model";
    outputs1 = gold_memory->get_outputs(param);

    output_names[1] = "pytorch";
    outputs2 = gold_memory->get_reference_outputs(param);

    has_valid_comp = true;
  }

  if (accel_memory && pytorch) {
    std::cout << "Accelerator vs. PyTorch" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    filename = prefix + "accelerator_vs_pytorch";

    output_names[0] = "accelerator";
    outputs1 = accel_memory->get_outputs(param);

    output_names[1] = "pytorch";
    outputs2 = accel_memory->get_reference_outputs(param);

    has_valid_comp = true;
  }

  if (accel_memory && gold_memory) {
    std::cout << "Accelerator vs. Gold Model" << std::endl;
    std::cout << "(reveals bugs in accelerator or memory placement)"
              << std::endl;
    filename = prefix + "accelerator_vs_gold";

    output_names[0] = "accelerator";
    outputs1 = accel_memory->get_outputs(param);

    output_names[1] = "gold_model";
    outputs2 = gold_memory->get_outputs(param);

    has_valid_comp = true;
  }

  if (!has_valid_comp) {
    std::cout << "No valid comparisons specified" << std::endl;
    std::abort();
  }

  double rel_err = 0.0;
  const auto output_tensors = get_op_outputs(param);

  for (int i = 0; i < outputs1.size(); i++) {
    const auto output1 = outputs1[i];
    const auto output2 = outputs2[i];

    const auto tensor = output_tensors[i];
    const int size = get_size(tensor);

    std::string suffix = ".txt";
    if (outputs1.size() > 1) {
      suffix = "_" + std::to_string(i) + ".txt";
    }

    if (tensor.dtype() == "bfloat16") {
      rel_err += compare_arrays<DataTypes::bfloat16, DataTypes::bfloat16>(
          output1, output_names[0], output2, output_names[1], size,
          filename + suffix, false);
    } else if (tensor.dtype() == "fp8_e8m0") {
      rel_err += compare_arrays<DataTypes::fp8_e8m0, DataTypes::fp8_e8m0>(
          output1, output_names[0], output2, output_names[1], size,
          filename + suffix, false);
    } else if (tensor.dtype() == "fp8_e5m3") {
      rel_err += compare_arrays<DataTypes::fp8_e5m3, DataTypes::fp8_e5m3>(
          output1, output_names[0], output2, output_names[1], size,
          filename + suffix, false);
    } else {
      // if unspecified, we will assume it's INPUT_DATATYPE
      rel_err += compare_arrays<INPUT_DATATYPE, INPUT_DATATYPE>(
          output1, output_names[0], output2, output_names[1], size,
          filename + suffix, false);
    }
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
