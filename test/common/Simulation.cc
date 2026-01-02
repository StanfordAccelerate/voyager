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

  spdlog::info("Starting new simulation with config:\n");
  spdlog::info("> Model: {}\n", model);
  spdlog::info("> Tests: ");
  for (const std::string& t : tests) spdlog::info("{} ", t);
  spdlog::info("\n");
  spdlog::info("> Sims: ");
  for (const std::string& s : sims) spdlog::info("{} ", s);
  spdlog::info("\n");
  spdlog::info("> Tolerance: {}\n", tolerance);
  spdlog::info("> Output dir: {}\n", out_dir);
}

Simulation::~Simulation() {
  for (const auto& [key, memory] : memories) {
    delete memory;
  }

  for (const auto& [key, dataloader] : dataloaders) {
    delete dataloader;
  }
}

void Simulation::load_data() {
  // Calculate dynamic memory sizes based on network parameters
  uint64_t dram_size = network->get_max_dram_address();
  uint64_t sram_size = 16 * 1024 * 1024;
  uint64_t output_size = network->get_max_output_size();

  spdlog::info("Model memory requirements:\n");
  spdlog::info("> Max DRAM address: {} bytes ({} MB)\n", dram_size,
               dram_size / (1024 * 1024));
  spdlog::info("> Max output size: {} bytes ({} MB)\n", output_size,
               output_size / (1024 * 1024));

  // Add some padding to ensure we don't run out of memory (10% extra)
  dram_size = static_cast<uint64_t>(dram_size * 1.1);
  output_size = static_cast<uint64_t>(output_size * 1.1);

  spdlog::info("Memory sizes with 10% margin:\n");
  spdlog::info("> DRAM: {} bytes ({} MB)\n", dram_size,
               dram_size / (1024 * 1024));
  spdlog::info("> SRAM: {} bytes ({} MB)\n", sram_size,
               sram_size / (1024 * 1024));
  spdlog::info("> Reference: {} bytes ({} MB)\n", output_size,
               output_size / (1024 * 1024));

  std::vector<uint64_t> memory_sizes{dram_size, sram_size, output_size};

  if (std::find(sims.begin(), sims.end(), "gold") != sims.end()) {
    memories["gold"] = new ArrayMemory(memory_sizes);
    dataloaders["gold"] = new DataLoader(memories["gold"], false);
  }

  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    memories["accelerator"] = new ArrayMemory(memory_sizes);
    dataloaders["accelerator"] = new DataLoader(memories["accelerator"], true);
  }

  std::string project_root = std::string(getenv("PROJECT_ROOT"));
  std::string datatype = std::string(getenv("DATATYPE"));
  std::string data_dir = project_root + "/" +
                         std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                         model + "/" + datatype + "/tensor_files";

  const auto operations = network->get_operations(tests, false);

  for (const auto& [key, dataloader] : dataloaders) {
    dataloader->load_inputs(operations, data_dir);
    dataloader->load_outputs(operations.back().param, data_dir);
  }

  spdlog::info("Data loaded successfully\n");
}

void Simulation::print_ideal_runtime(const Operation operation) {
  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto& first_op = op_list.front();
  const auto& kwargs = first_op.kwargs();
  const auto output = get_op_outputs(param).back();

  long cycles;

  const long clock_period_ns = getenv_int("CLOCK_PERIOD", 1);
  const int num_tiles = is_soc_sim() ? get_tile_count(param) : 1;

  if (is_gemm_op(first_op.target())) {
    bool is_matmul = first_op.target().find("matmul") != std::string::npos;
    std::string weight_key = is_matmul ? "other" : "weight";
    const auto weight = first_op.kwargs().at(weight_key).tensor();
    const auto weight_shape = get_shape(weight);
    const auto output_shape = get_shape(output);

    // the total number of operations is X * Y * C * FX * FY * K.
    long num_macs = get_size(output) * get_size(weight) * num_tiles;

    if (is_fc_layer(first_op)) {
      auto input = first_op.kwargs().at("input").tensor();
      int K = weight_shape[0];
      if (input.dtype() == "bfloat16") {
        cycles = num_macs / K / VECTOR_UNIT_WIDTH;
      } else {
        cycles = num_macs / K / MV_UNIT_WIDTH;
      }
    } else {
      int K = weight_shape[weight_shape.size() - 1];
      cycles = num_macs / K / (IC_DIMENSION * OC_DIMENSION);
    }

    spdlog::info("{}, matrix unit ideal runtime: {} ns\n", get_op_name(param),
                 cycles * clock_period_ns);
  } else {
    long num_ops = get_size(output);
    cycles = num_ops / VECTOR_UNIT_WIDTH * num_tiles;

    if (first_op.target() == "softmax") {
      cycles *= 3;
    } else if (first_op.target() == "layer_norm") {
      cycles *= 4;
    } else if (first_op.target() == "calculate_mx_qparam") {
      const int block_size = first_op.kwargs().at("block_size").int_value();
      cycles *= block_size;
    } else if (first_op.target() == "max_pool2d") {
      const auto kernel_size = kwargs.at("kernel_size").int_list().values();
      for (const auto& k : kernel_size) {
        cycles *= k;
      }
    } else if (first_op.target() == "adaptive_avg_pool2d") {
      const auto input_shape = get_shape(kwargs.at("input").tensor());
      const auto output_shape = get_shape(output);
      int kernel_y = input_shape[1] / output_shape[1];
      int kernel_x = input_shape[2] / output_shape[2];
      cycles *= (kernel_y * kernel_x);
    }

    spdlog::info("{}, vector unit ideal runtime: {} ns\n", get_op_name(param),
                 cycles * clock_period_ns);
  }
}

void Simulation::run() {
  // Run gold model
  if (std::find(sims.begin(), sims.end(), "gold") != sims.end()) {
    for (const auto& operation : operations) {
      const auto param = operation.param;
      print_ideal_runtime(operation);

      const auto tensors = get_op_outputs(param);
      const auto dataloader = dataloaders["gold"];
      const auto memory = (ArrayMemory*)dataloader->memory;

      if (is_soc_sim()) {
        const int num_tiles = get_tile_count(param);
        std::cerr << "Number of tiles to run: " << num_tiles << std::endl;

        // for SoC simulation, run operation num_tiles times
        for (int i = 0; i < num_tiles; i++) {
          dataloader->load_scratchpad(param, i);

          const auto kwargs = dataloader->get_args(param);
          const auto outputs = run_gold_model(operation, kwargs);

          assert(outputs.size() == tensors.size());

          for (int i = 0; i < outputs.size(); i++) {
            memory->write_tensor(tensors[i], outputs[i]);
          }

          dataloader->store_scratchpad(param, i);
        }
      } else {
        const auto kwargs = dataloader->get_args(param);
        const auto outputs = run_gold_model(operation, kwargs);

        assert(outputs.size() == tensors.size());

        for (int i = 0; i < outputs.size(); i++) {
          memory->write_tensor(tensors[i], outputs[i]);
        }
      }
    }
  }

  // Run accelerator
  if (std::find(sims.begin(), sims.end(), "accelerator") != sims.end()) {
    run_accelerator(operations, dataloaders["accelerator"]);
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

  const auto param = operations.back().param;

  bool has_pytorch = contains(sims, "pytorch");
  bool has_gold = contains(sims, "gold");
  bool has_accelerator = contains(sims, "accelerator");

  std::string filename;
  std::vector<std::any> outputs1, outputs2;
  std::string name1, name2;

  if (has_gold && has_pytorch) {
    spdlog::info("Gold Model vs. PyTorch\n");
    spdlog::info("(reveals issues in data loading or mapping)\n");
    filename = prefix + "gold_vs_pytorch";

    name1 = "gold_model";
    outputs1 = dataloaders["gold"]->get_outputs(param);

    name2 = "pytorch";
    outputs2 = dataloaders["gold"]->get_reference_outputs(param);
  } else if (has_accelerator && has_pytorch) {
    spdlog::info("Accelerator vs. PyTorch\n");
    spdlog::info("(reveals bugs in accelerator or memory placement)\n");
    filename = prefix + "accelerator_vs_pytorch";

    name1 = "accelerator";
    outputs1 = dataloaders["accelerator"]->get_outputs(param);

    name2 = "pytorch";
    outputs2 = dataloaders["accelerator"]->get_reference_outputs(param);
  } else if (has_accelerator && has_gold) {
    spdlog::info("Accelerator vs. Gold Model\n");
    spdlog::info("(reveals bugs in accelerator or memory placement)\n");
    filename = prefix + "accelerator_vs_gold";

    name1 = "accelerator";
    outputs1 = dataloaders["accelerator"]->get_outputs(param);

    name2 = "gold_model";
    outputs2 = dataloaders["gold"]->get_outputs(param);
  } else {
    spdlog::info("No valid comparisons specified\n");
    std::abort();
  }

  double rel_err = 0.0;
  const auto output_tensors = get_op_outputs(param);
  const int num_tiles_executed = get_tile_count(param);

  // HACK For sparse outputs, we only want to compare the non-zero values
  const int num_outputs = output_tensors.size();
  int csr_data_size;
  if (num_outputs > 2) {
    const int indptr_size = get_size(output_tensors[2]);
    const int last_index = (indptr_size - 1) * num_tiles_executed;
    std::cerr << "indptr size: " << indptr_size << " last index: " << last_index
              << "\n";

    // Try to get the index of last CSR column indices
    const auto indptr = std::any_cast<SPMM_META_DATATYPE*>(outputs2[2]);
    csr_data_size = indptr[last_index];
    std::cerr << "num_data: " << csr_data_size << "\n";
  }

  for (int i = 0; i < output_tensors.size(); i++) {
    std::string suffix = ".txt";
    if (output_tensors.size() > 1) {
      suffix = "_" + std::to_string(i) + ".txt";
    }

    const auto full_shape = get_shape(output_tensors[i], true, false);

    std::vector<uint8_t> valid;
    if (num_outputs > 2 && i < 2) {
      valid.resize(get_size(full_shape), 0);
      std::fill_n(valid.begin(), csr_data_size, 1);
    } else if (is_soc_sim()) {
      const auto tile_shape = get_shape(output_tensors[i], true, true);
      build_valid_mask(full_shape, tile_shape, num_tiles_executed, valid);
    } else {
      valid.resize(get_size(full_shape), 1);
    }

    rel_err += compare_arrays<SUPPORTED_TYPES>(output_tensors[i], outputs1[i],
                                               name1, outputs2[i], name2,
                                               filename + suffix, valid);
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
