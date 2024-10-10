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
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/ArrayMemory.h"
#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
#include "test/common/Network.h"
#include "test/common/Utils.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

/* Run inference on a single sample and return correct classification. */
bool run_sample(std::string model_name, std::string data_dir,
                std::string sample,
                std::vector<codegen::AcceleratorParam> params) {
  std::vector<int> memory_sizes{SRAM_MEMORY_SIZE};
  auto memory = std::make_unique<ArrayMemory>(memory_sizes);
  auto data_loader = std::make_unique<DataLoader>(memory.get(), false);

  int num_classes;
  auto matrix_param = params.front().matrix_param();
  codegen::Tensor input;
  if (params.front().has_matrix_param()) {
    input = params.front().matrix_param().input();
  } else {
    input = params.front().vector_params(0).input();
  }
  std::string sample_dir = data_dir + "/" + sample;
  if (model_name == "mobilebert") {
    num_classes = 2;
    data_loader->load_tensor(input, sample_dir);

    // Load attention mask
    bool found_param = false;
    for (const auto& param : params) {
      for (const auto& vector_param : param.vector_params()) {
        if (vector_param.has_other() &&
            vector_param.other().node() == "arg1_1") {
          data_loader->load_tensor(vector_param.other(), sample_dir);
          found_param = true;
          break;
        }
      }
      if (found_param) {
        break;
      }
    }

    if (!found_param) {
      throw std::runtime_error("Error: Could not find attention mask.");
    }
  } else if (model_name == "resnet18" || model_name == "resnet50") {
    num_classes = 1000;
    data_loader->load_tensor(input, sample_dir, true);
  }

  std::string params_dir = std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                           model_name + "/" + std::getenv("DATATYPE") +
                           "/tensor_files";

  for (const auto& param : params) {
    data_loader->load_weights(param, params_dir);
  }

  // Run inference
  for (const auto& param : params) {
    auto args = memory->get_args(param);
    run_gold_model(param, args);
  }

  // Extract final output
  float logits[num_classes];
  auto output = memory->get_output(params.back());
  if (params.back().output().dtype() == "bfloat16") {
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
  auto params = network.get_params();
  std::cout << "Number of params: " << params.size() << std::endl;

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
    std::vector<std::future<bool>> results;

    for (int i = 0; i < num_threads; i++) {
      int sample_index = batch * num_threads + i;
      if (sample_index >= num_samples) {
        break;
      }
      results.push_back(std::async(std::launch::async, run_sample, model_name,
                                   data_dir, dataset[sample_index], params));
    }

    for (int i = 0; i < results.size(); i++) {
      if (results[i].get()) {
        num_correct++;
      }
      num_finished++;
    }

    std::cout << "Accuracy: " << num_correct << "/" << num_finished << " ("
              << (float)num_correct / num_finished * 100 << "%)" << std::endl;
  }
}
