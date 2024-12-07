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

std::vector<codegen::AcceleratorParam> Network::get_all_params() {
  return {model_params.params().begin(), model_params.params().end()};
}

std::vector<codegen::AcceleratorParam> Network::get_params() {
  std::vector<codegen::AcceleratorParam> params;
  for (const auto& param : model_params.params()) {
    if (!param.has_nop_param()) {
      params.push_back(param);
    }
  }
  return params;
}

std::vector<codegen::AcceleratorParam> Network::get_params(
    const std::vector<std::string>& names) {
  std::vector<codegen::AcceleratorParam> params;
  if (names.size() == 1) {
    for (const auto& param : model_params.params()) {
      if (param.name() == names[0]) {
        params.push_back(param);
        break;
      }
    }
  } else if (names.size() == 2) {
    bool found_first = false;
    bool found_second = false;
    for (const auto& param : model_params.params()) {
      if (param.name() == names[0]) {
        found_first = true;
      }
      if (found_first) {
        params.push_back(param);
      }
      if (param.name() == names[1]) {
        found_second = true;
        break;
      }
    }

    if (!found_first || !found_second) {
      std::cerr << "Invalid names provided" << std::endl;
      exit(1);
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
