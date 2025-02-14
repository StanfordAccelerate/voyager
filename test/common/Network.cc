#include "test/common/Network.h"

#include <google/protobuf/text_format.h>

#include <fstream>

using namespace std;
using namespace google::protobuf;

Network::Network(std::string& model_path) {
  project_root = std::string(getenv("PROJECT_ROOT"));
  std::string datatype = std::string(getenv("DATATYPE"));

  // Open the file
  std::string filename = project_root + "/" +
                         std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                         model_path + "/" + datatype + "/model.txt";

  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("Error: File " + filename + " does not exist.");
  }
  std::fstream input(filename, ios::in);
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string text_str = buffer.str();

  if (!TextFormat::ParseFromString(text_str, &model)) {
    std::cerr << "Failed to parse text file." << std::endl;
  }
}

std::vector<codegen::Operation> Network::get_params(bool filter_nop) {
  if (!filter_nop) {
    return {model.ops().begin(), model.ops().end()};
  }

  std::vector<codegen::Operation> params;
  for (const auto& op : model.ops()) {
    if (op.op().target() != "nop") {
      params.push_back(op);
    }
  }
  return params;
}

std::vector<codegen::Operation> Network::get_params(
    const std::vector<std::string>& names, bool filter_nop) {
  const auto all_params = get_params(filter_nop);

  std::vector<codegen::Operation> params;

  if (names.size() == 1) {
    for (const auto& param : all_params) {
      if (get_op_name(param) == names[0]) {
        params.push_back(param);
        break;
      }
    }
  } else if (names.size() == 2) {
    bool found_first = false;
    bool found_second = false;
    for (const auto& param : all_params) {
      if (get_op_name(param) == names[0]) {
        found_first = true;
      }
      if (found_first) {
        params.push_back(param);
      }
      if (get_op_name(param) == names[1]) {
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
