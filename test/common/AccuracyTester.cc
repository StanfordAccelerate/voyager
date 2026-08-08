// Full-network accuracy on a dataset, using the C++ gold model.
//
//   AccuracyTester <model> <dataset_dir> [<num_threads>] [<num_samples>]
//
// Each sample is one folder under <dataset_dir>, holding that sample's input
// tensors; the ground truth is encoded in the folder name. A sample is a whole
// independent inference, so samples run in parallel -- each on its own memory.
//
// This is the same graph walk as every other flow, with the gold-model backend.
// It just runs it once per sample and reads the argmax instead of grading
// buffers.

#define NO_SYSC

#include <algorithm>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/ArchitectureParams.h"
#include "test/common/ArrayMemory.h"
#include "test/common/Checker.h"
#include "test/common/GoldModel.h"
#include "test/common/Interpreter.h"
#include "test/common/Model.h"
#include "test/common/Utils.h"

namespace {

// The label the dataset encodes in a sample's folder name.
int ground_truth(const std::string& model_name, const std::string& sample) {
  if (model_name == "mobilebert" || model_name == "bert") {
    // SST2: the last character of the folder name is 0 or 1.
    return sample.back() - '0';
  }
  if (model_name == "resnet18" || model_name == "resnet50" ||
      model_name == "vit" || model_name == "mobilenet_v2") {
    // ImageNet: the folder name is <index>_<label>.
    return std::stoi(sample.substr(sample.find("_") + 1));
  }
  throw std::runtime_error("Accuracy is not supported for model " + model_name);
}

// Runs inference on one sample and reports whether it classified correctly.
//
// `model` is read-only and shared; every sample gets its own memory,
// interpreter and scalar state, so samples are independent.
bool run_sample(const Model& model, const std::string& model_name,
                const std::string& dataset_dir, const std::string& sample) {
  ArrayMemory memory(model.memory_sizes());

  // This sample's inputs, then the weights the graph reads out of DRAM.
  const std::string sample_dir = dataset_dir + "/" + sample;
  for (const auto& box : model.proto().inputs()) {
    load_tensor(to_tensor(box), sample_dir, &memory);
  }
  for (const auto& box : model.proto().parameters()) {
    if (is_constant(box)) continue;
    load_tensor(to_tensor(box), model.data_dir, &memory);
  }

  GoldBackend backend(&memory);
  Interpreter interpreter(model, &memory, &backend);
  interpreter.run(model.selected_ops({}));

  // The network's own output, read back from where the graph put it.
  if (model.proto().outputs_size() == 0) {
    throw std::runtime_error("Model declares no output to classify.");
  }
  const Tensor output = to_tensor(model.proto().outputs(0));
  auto* scores = std::any_cast<VECTOR_DATATYPE*>(memory.read_tensor(output));

  const int num_classes = get_size(output);
  int best = 0;
  for (int i = 1; i < num_classes; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  delete[] scores;

  return best == ground_truth(model_name, sample);
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: AccuracyTester <model> <dataset_dir> "
                 "[<num_threads>] [<num_samples>]\n";
    return 1;
  }

  const std::string model_name = argv[1];
  const std::string dataset_dir = argv[2];
  const int num_threads = argc > 3 ? std::stoi(argv[3]) : 1;
  int num_samples = argc > 4 ? std::stoi(argv[4]) : -1;

  std::cout << "*** Accuracy Tester ***" << std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout << "Model name: " << model_name << std::endl;
  std::cout << "Dataset path: " << dataset_dir << std::endl;
  std::cout << "Number of threads: " << num_threads << std::endl;

  if (!std::filesystem::is_directory(dataset_dir)) {
    throw std::runtime_error("Dataset path is not a directory: " + dataset_dir);
  }

  Model model(model_name);

  std::vector<std::string> dataset;
  for (const auto& entry : std::filesystem::directory_iterator(dataset_dir)) {
    if (!entry.is_directory()) continue;
    const std::string name = entry.path().filename().string();
    if (name == "params") continue;
    dataset.push_back(name);
  }
  std::sort(dataset.begin(), dataset.end());

  num_samples = num_samples < 0
                    ? static_cast<int>(dataset.size())
                    : std::min(num_samples, static_cast<int>(dataset.size()));
  std::cout << "Number of samples: " << num_samples << std::endl;

  int correct = 0;
  int finished = 0;

  for (int start = 0; start < num_samples; start += num_threads) {
    const int end = std::min(start + num_threads, num_samples);

    if (num_threads == 1) {
      // Run inline, so a failing sample is debuggable.
      correct += run_sample(model, model_name, dataset_dir, dataset[start]);
      finished++;
    } else {
      std::vector<std::future<bool>> results;
      for (int i = start; i < end; i++) {
        results.push_back(std::async(std::launch::async, run_sample,
                                     std::cref(model), model_name, dataset_dir,
                                     dataset[i]));
      }
      for (auto& result : results) {
        correct += result.get();
        finished++;
      }
    }

    std::cout << "Accuracy: " << correct << "/" << finished << " ("
              << std::fixed << std::setprecision(2)
              << (static_cast<float>(correct) / finished * 100) << "%)"
              << std::endl;
  }

  return 0;
}
