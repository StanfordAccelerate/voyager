#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define NO_SYSC
// clang-format off
#include "src/datatypes/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

/* Run inference on a single sample and return correct classification. */
bool run_sample(std::string model_name, std::string data_dir,
                std::string sample, Network network) {
  std::vector<uint64_t> memory_sizes{SRAM_MEMORY_SIZE};
  auto memory = std::make_unique<ArrayMemory>(memory_sizes);

  const auto model = network.model;

  bool is_cnn = model_name == "resnet18" || model_name == "resnet50";
  auto data_loader = std::make_unique<DataLoader>(memory.get(), false, is_cnn);

  int num_classes;
  if (model_name == "mobilebert" || model_name == "bert") {
    num_classes = 2;
  } else if (model_name == "resnet18" || model_name == "resnet50") {
    num_classes = 1000;
  } else {
    throw std::runtime_error("Error: Model not supported.");
  }

  float logits[num_classes];

  // Load inputs and parameters
  std::string inputs_dir = data_dir + "/" + sample;
  for (const auto& tensor : model.inputs()) {
    data_loader->load_tensor(tensor, inputs_dir, is_cnn);
  }

  // Fully connected layer, or linear ops with input of a 1D tensor, is run
  // on the vector unit. Its weight does not need to be transposed. We check
  // if an op is a FC by checking its output dimension. This should be
  // improved in the future.
  std::string params_dir = std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                           model_name + "/" + std::getenv("DATATYPE") +
                           "/tensor_files";
  for (const auto& tensor : model.parameters()) {
    bool transpose_weight = tensor.shape(0) != num_classes;
    data_loader->load_tensor(tensor, params_dir, transpose_weight);
  }

  // Run inference
  const auto operations = network.get_operations(true);
  for (const auto& operation : operations) {
    const auto param = operation.param;
    const auto kwargs = memory->get_args(param);
    const auto outputs = run_gold_model(operation, kwargs);
    const auto tensors = get_op_outputs(param);

    assert(outputs.size() == tensors.size());

    for (int i = 0; i < outputs.size(); i++) {
      memory->write_tensor(tensors[i], outputs[i]);
    }
  }

  // Extract final output
  const auto last_op = model.ops(model.ops_size() - 1);
  const auto output = memory->get_outputs(last_op).back();
  if (last_op.output().dtype() == "bfloat16") {
    auto output_ptr = std::any_cast<DataTypes::bfloat16*>(output);
    for (int i = 0; i < num_classes; i++) {
      logits[i] = output_ptr[i];
    }
  } else {
    // if unspecified, we will assume it's INPUT_DATATYPE
    auto output_ptr = std::any_cast<INPUT_DATATYPE*>(output);
    for (int i = 0; i < num_classes; i++) {
      logits[i] = output_ptr[i];
    }
  }

  // Find the maximum logit index
  int max_index = 0;
  for (int i = 1; i < num_classes; i++) {
    if (logits[i] > logits[max_index]) {
      max_index = i;
    }
  }

  // Get ground truth from folder name
  int ground_truth;
  if (model_name == "mobilebert") {
    // last character of folder name is 0 or 1
    ground_truth = sample.back() - '0';
  } else if (model_name == "resnet18" || model_name == "resnet50") {
    // split folder name by underscore, first part is ground truth
    ground_truth = std::stoi(sample.substr(0, sample.find("_")));
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
