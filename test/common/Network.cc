#include "test/common/Network.h"

#include <google/protobuf/text_format.h>

#include <fstream>

using namespace std;
using namespace google::protobuf;

Network::Network(std::string& model) : model(model) {
  project_root = std::string(getenv("PROJECT_ROOT"));
  std::string datatype = std::string(getenv("DATATYPE"));

  // Open the file
  std::string filename = project_root + "/" +
                         std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                         model + "/" + datatype + "/params.txt";

  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("Error: File " + filename + " does not exist.");
  }
  std::fstream input(filename, ios::in);
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string text_str = buffer.str();

  if (!TextFormat::ParseFromString(text_str, &model_params)) {
    std::cerr << "Failed to parse text file." << std::endl;
  }
}

std::vector<codegen::AcceleratorParam> Network::get_params() {
  std::vector<codegen::AcceleratorParam> params;
  for (const auto& param : model_params.params()) {
    if (!param.has_nop()) {
      params.push_back(param);
    }
  }
  return params;
}

std::vector<codegen::AcceleratorParam> Network::get_params(
    const std::vector<std::string>& names) {
  std::vector<codegen::AcceleratorParam> params;
  std::vector<codegen::AcceleratorParam> all_params(
      model_params.params().begin(), model_params.params().end());

  if (names.size() == 1) {
    for (const auto& param : all_params) {
      if (param.name() == names[0]) {
        params.push_back(param);
        break;
      }
    }
  } else if (names.size() == 2) {
    auto first_it = std::find_if(
        all_params.begin(), all_params.end(),
        [&names](const auto& param) { return param.name() == names[0]; });
    auto last_it = std::find_if(all_params.rbegin(), all_params.rend(),
                                [&names](const auto& param) {
                                  return param.name() == names[1];
                                })
                       .base();

    if (first_it != all_params.end() && last_it != all_params.begin() &&
        first_it < last_it) {
      params.assign(first_it, last_it);
    }
  } else {
    std::cerr << "Invalid number of names provided" << std::endl;
    exit(1);
  }

  if (params.empty()) {
    std::cerr << "Param not found" << std::endl;
    exit(1);
  }

  return params;
}
