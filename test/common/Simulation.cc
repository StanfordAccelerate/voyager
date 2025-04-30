#include "test/common/Simulation.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define SPDLOG_EOL ""
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"

Simulation::Simulation() {
  spdlog::cfg::load_env_levels();
  spdlog::set_pattern("%v");

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

  spdlog::info("Starting new simulation with config:");
  spdlog::info("\n> Model: {}", model);
  spdlog::info("\n> Tests: ");
  for (const std::string& t : tests) spdlog::info("{} ", t);
  spdlog::info("\n> Sims: ");
  for (const std::string& s : sims) spdlog::info("{} ", s);
  spdlog::info("\n> Tolerance: {}", tolerance);
  spdlog::info("\n> Output dir: {}", out_dir);
  spdlog::info("\n> SRAM: {} KB\n", SRAM_MEMORY_SIZE / 1024);
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
  std::vector<uint64_t> memory_sizes{SRAM_MEMORY_SIZE, REFERENCE_MEMORY_SIZE};

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

  const auto operations = network->get_operations(tests, false);

  for (const auto& [key, dataloader] : dataLoaders) {
    // Load the inputs to the entire model first, e.g. attention mask
    for (const auto& tensor : network->model.inputs()) {
      dataloader->load_tensor(tensor, data_dir);
    }

    for (const auto& tensor : network->model.parameters()) {
      bool has_transpose =
          tensor.shape(0) != num_classes &&
          tensor.node().find("_param_constant") != std::string::npos;
      dataloader->load_tensor(tensor, data_dir, has_transpose);
    }

    // Load the layer's input and output last
    dataloader->load_inputs(operations.front().param, data_dir);
    dataloader->load_outputs(operations.back().param, data_dir);
  }

  spdlog::info("Data loaded successfully\n");
}

void Simulation::print_ideal_runtime(const Operation operation) {
  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto first_op = op_list.front();
  const auto output = get_op_outputs(param).back();

  long cycles;

  char* clock_period = std::getenv("CLOCK_PERIOD");
  long clock_period_ns = clock_period ? std::stoi(clock_period) : 1;

  if (GEMM_OPS.find(first_op.target()) != GEMM_OPS.end()) {
    bool is_matmul = first_op.target().find("matmul") != std::string::npos;
    std::string weight_key = is_matmul ? "other" : "weight";
    const auto weight = first_op.kwargs().at(weight_key).tensor();
    const auto weight_shape = get_shape(weight);
    const auto output_shape = get_shape(output);

    int K = is_matmul ? weight_shape[weight_shape.size() - 1] : weight_shape[0];

    // the total number of operations is X * Y * C * FX * FY * K.
    long num_macs = get_size(output) * get_size(weight) / K;
    cycles = num_macs / (IC_DIMENSION * OC_DIMENSION);

    if (operation.has_shrunk_tiling) {
      cycles *= operation.shrink_factor;
    }

    spdlog::info("{}, matrix unit ideal runtime: {} ns\n", get_op_name(param),
                 cycles * clock_period_ns);
  } else {
    long num_ops = get_size(output);
    cycles = num_ops / OC_DIMENSION;
    spdlog::info("{}, vector unit ideal runtime: {} ns\n", get_op_name(param),
                 cycles * clock_period_ns);
  }
}

void Simulation::run() {
  // Run gold models
  for (const auto& operation : operations) {
    const auto param = operation.param;
    print_ideal_runtime(operation);

    if (std::find(sims.begin(), sims.end(), "gold") != sims.end()) {
      const auto memory = (ArrayMemory*)(memories["gold"]);
      const auto kwargs = memory->get_args(param);
      const auto outputs = run_gold_model(operation, kwargs);
      const auto tensors = get_op_outputs(param);

      assert(outputs.size() == tensors.size());

      for (int i = 0; i < outputs.size(); i++) {
        memory->write_tensor(tensors[i], outputs[i]);
      }
    }
  }

  // Run accelerator test
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    auto memory = (ArrayMemory*)(memories["accelerator"]);
    run_accelerator(operations, memory->memories[0]);
  }
}

template <typename T>
bool compare_arrays(codegen::Tensor tensor, const std::any& output1,
                    const std::string& name1, const std::any& output2,
                    const std::string& name2, const std::string& filename,
                    int& error_count) {
  if (tensor.dtype() == DataTypes::TypeName<T>::name()) {
    const auto size = get_size(tensor);
    error_count += compare_arrays<T, T>(output1, name1, output2, name2, size,
                                        filename, false);
    return true;
  }
  return false;
}

template <typename... Ts>
int compare_arrays_helper(codegen::Tensor tensor, const std::any& output1,
                          const std::string& name1, const std::any& output2,
                          const std::string& name2,
                          const std::string& filename) {
  int error_count = 0;
  bool matched = (compare_arrays<Ts>(tensor, output1, name1, output2, name2,
                                     filename, error_count) ||
                  ...);

  if (!matched) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
  }

  return error_count;
}

int Simulation::check_outputs() {
  std::string prefix;
  if (operations.size() == 1) {
    prefix = out_dir + model + '.' + operations.front().name + '.';
  } else {
    prefix = out_dir + model + '.' + operations.front().name + "_to_" +
             operations.back().name + '.';
  }

  const auto param = operations.back().param;

  bool pytorch = std::find(sims.begin(), sims.end(), "pytorch") != sims.end();
  auto gold_memory = memories["gold"];
  auto accel_memory = memories["accelerator"];

  std::string filename;
  std::vector<std::any> outputs1, outputs2;
  std::string output_names[2];

  bool has_valid_comp = false;

  if (gold_memory && pytorch) {
    spdlog::info("Gold Model vs. PyTorch\n");
    spdlog::info("(reveals issues in data loading or mapping)\n");
    filename = prefix + "gold_vs_pytorch";

    output_names[0] = "gold_model";
    outputs1 = gold_memory->get_outputs(param);

    output_names[1] = "pytorch";
    outputs2 = gold_memory->get_reference_outputs(param);

    has_valid_comp = true;
  }

  if (accel_memory && pytorch) {
    spdlog::info("Accelerator vs. PyTorch\n");
    spdlog::info("(reveals bugs in accelerator or memory placement)\n");
    filename = prefix + "accelerator_vs_pytorch";

    output_names[0] = "accelerator";
    outputs1 = accel_memory->get_outputs(param);

    output_names[1] = "pytorch";
    outputs2 = accel_memory->get_reference_outputs(param);

    has_valid_comp = true;
  }

  if (accel_memory && gold_memory) {
    spdlog::info("Accelerator vs. Gold Model\n");
    spdlog::info("(reveals bugs in accelerator or memory placement)\n");
    filename = prefix + "accelerator_vs_gold";

    output_names[0] = "accelerator";
    outputs1 = accel_memory->get_outputs(param);

    output_names[1] = "gold_model";
    outputs2 = gold_memory->get_outputs(param);

    has_valid_comp = true;
  }

  if (!has_valid_comp) {
    spdlog::info("No valid comparisons specified\n");
    std::abort();
  }

  double rel_err = 0.0;
  const auto output_tensors = get_op_outputs(param);

  for (int i = 0; i < outputs1.size(); i++) {
    std::string suffix = ".txt";
    if (outputs1.size() > 1) {
      suffix = "_" + std::to_string(i) + ".txt";
    }

    rel_err += compare_arrays_helper<SUPPORTED_TYPES>(
        output_tensors[i], outputs1[i], output_names[0], outputs2[i],
        output_names[1], filename + suffix);
  }

  int error_count = 0;
  if (rel_err > tolerance) {
    error_count += rel_err < 1.0 ? 1 : static_cast<int>(rel_err);
  }
  spdlog::info("Rela. error: {}\n", rel_err);
  spdlog::info("Error count: {}\n", error_count);

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
  spdlog::info(
      "\nConfigure simulator by using environment variables."
      "\n NETWORK - Type of network to run {mobilebert, resnet}"
      "\n TESTS - Layers in network to run. Either single or tuple: "
      "<first>,<last>."
      "\n SIMS - Simulators / models to compare {accelerator, "
      "customposit, universal, fp32, file}."
      "\n TASK - MobileBERT run time (forward, backward, e2e)."
      "\n TOLERANCE - Relative normalized error in % we allow "
      "(default 10)."
      "\n DATA_DIR - Path to binary input data."
      "\n OUT_DIR - Path to output data.\n");
}
