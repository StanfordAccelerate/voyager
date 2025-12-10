#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define NO_SYSC
#include "src/ArchitectureParams.h"
#include "src/datatypes/DataTypes.h"
#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"
#include "test/compiler/proto/param.pb.h"

/* Run inference on a single sample and return correct classification. */
bool run_sample(std::string model_name, std::string data_dir,
                std::string sample, Network network) {
  uint64_t dram_size = network.get_max_dram_address();
  std::vector<uint64_t> memory_sizes{dram_size};
  auto memory = std::make_unique<ArrayMemory>(memory_sizes);

  const auto model = network.model;

  auto data_loader = std::make_unique<DataLoader>(memory.get(), false);

  // Load inputs and parameters
  std::string inputs_dir = data_dir + "/" + sample;
  for (const auto& tensor : model.inputs()) {
    data_loader->load_tensor(tensor, inputs_dir, false);
  }

  std::string params_dir = std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                           model_name + "/" + std::getenv("DATATYPE") +
                           "/tensor_files";
  for (const auto& tensor : model.parameters()) {
    data_loader->load_tensor(tensor, params_dir, false);
  }

  // Run inference
  const auto operations = network.get_operations(true);
  for (const auto& operation : operations) {
    const auto param = operation.param;
    const auto kwargs = data_loader->get_args(param);
    const auto outputs = run_gold_model(operation, kwargs);
    const auto tensors = get_op_outputs(param);

    assert(outputs.size() == tensors.size());

    for (int i = 0; i < outputs.size(); i++) {
      memory->write_tensor(tensors[i], outputs[i]);
    }
  }

  // Extract final output
  const auto last_op = model.ops(model.ops_size() - 1);
  const auto output = data_loader->get_outputs(last_op).back();
  auto output_ptr = std::any_cast<VECTOR_DATATYPE*>(output);

  const int num_classes = get_size(last_op.output());

  int max_index = 0;
  for (int i = 1; i < num_classes; i++) {
    if (output_ptr[i] > output_ptr[max_index]) {
      max_index = i;
    }
  }

  // Get ground truth from folder name
  int ground_truth;
  if (model_name == "mobilebert" || model_name == "bert") {
    // SST2: the last character of folder name is 0 or 1
    ground_truth = sample.back() - '0';
  } else if (model_name == "resnet18" || model_name == "resnet50" ||
             model_name == "vit" || model_name == "mobilenet_v2") {
    // ImageNet: split folder name by underscore, second part is ground truth
    ground_truth = std::stoi(sample.substr(sample.find("_") + 1));
  } else {
    throw std::runtime_error("Error: Model not supported.");
  }

  return max_index == ground_truth;
}

/* Test accuracy of model on a dataset using the C++ gold model. */
int main(int argc, char* argv[]) {
  // model name is first argument
  std::string model_name = std::string(argv[1]);
  // path to dataset is second argument
  std::string data_dir = std::string(argv[2]);

  int num_threads = 1;
  if (argc > 3) {
    num_threads = std::stoi(argv[3]);
  }

  int num_samples = -1;
  if (argc > 4) {
    num_samples = std::stoi(argv[4]);
  }

  // Print out model name, path, and number of threads
  std::cout << "*** Accuracy Tester ***" << std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout << "Model name: " << model_name << std::endl;
  std::cout << "Dataset path: " << data_dir << std::endl;
  std::cout << "Number of threads: " << num_threads << std::endl;

  Network network(model_name);

  if (!std::filesystem::exists(data_dir)) {
    throw std::runtime_error("Error: Path does not exist.");
  }

  if (!std::filesystem::is_directory(data_dir)) {
    throw std::runtime_error("Error: Path is not a directory.");
  }

  // iterate through all folders in data_dir
  // each folder consists of two inputs
  std::vector<std::string> dataset;
  std::filesystem::directory_iterator dir_it(data_dir);
  for (const auto& entry : dir_it) {
    if (entry.is_directory()) {
      // get folder name
      std::string directory = entry.path().filename().string();

      // skip params folder
      if (directory == "params") {
        continue;
      }
      dataset.push_back(directory);
    }
  }
  if (num_samples == -1) {
    num_samples = dataset.size();
  } else {
    num_samples = std::min(num_samples, (int)dataset.size());
  }
  std::cout << "Number of samples: " << num_samples << std::endl;

  // Run num_threads samples in parallel
  int num_batches = num_samples / num_threads;
  num_batches += num_samples % num_threads == 0 ? 0 : 1;
  int num_correct = 0;
  int num_finished = 0;

  for (int batch = 0; batch < num_batches; batch++) {
    int start_index = batch * num_threads;
    int end_index = std::min(start_index + num_threads, num_samples);

    if (num_threads == 1) {
      // Direct call for easier debugging
      int sample_index = start_index;
      bool result =
          run_sample(model_name, data_dir, dataset[sample_index], network);
      if (result) {
        num_correct++;
      }
      num_finished++;

    } else {
      // Parallel execution
      std::vector<std::future<bool>> results;
      for (int sample_index = start_index; sample_index < end_index;
           ++sample_index) {
        results.push_back(std::async(std::launch::async, run_sample, model_name,
                                     data_dir, dataset[sample_index], network));
      }

      for (auto& result : results) {
        if (result.get()) {
          num_correct++;
        }
        num_finished++;
      }
    }

    std::cout << "Accuracy: " << num_correct << "/" << num_finished << " ("
              << std::fixed << std::setprecision(2)
              << (static_cast<float>(num_correct) / num_finished * 100) << "%)"
              << std::endl;
  }
}
