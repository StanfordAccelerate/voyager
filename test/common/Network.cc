#include "test/common/Network.h"

#include <google/protobuf/text_format.h>

#include <fstream>

using namespace std;
using namespace google::protobuf;

Network::Network(std::string& model_name) {
  project_root = std::string(getenv("PROJECT_ROOT"));
  std::string datatype = std::string(getenv("DATATYPE"));

  // Open the file
  std::string filename = project_root + "/" +
                         std::string(getenv("CODEGEN_DIR")) + "/networks/" +
                         model_name + "/" + datatype + "/model.txt";

  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("Error: File " + filename + " does not exist.");
  }
  std::ifstream input(filename);
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string text_str = buffer.str();

  if (!TextFormat::ParseFromString(text_str, &model)) {
    std::cerr << "Failed to parse text file." << std::endl;
  }

  std::map<std::string, voyager::Tiling> tiling_map;
  filename = project_root + "/" + std::string(getenv("CODEGEN_DIR")) +
             "/networks/" + model_name + "/" + datatype + "/" +
             std::string(getenv("IC_DIMENSION")) + "x" +
             std::string(getenv("OC_DIMENSION")) + "_" +
             std::string(getenv("INPUT_BUFFER_SIZE")) + "x" +
             std::string(getenv("WEIGHT_BUFFER_SIZE")) + "x" +
             std::string(getenv("ACCUM_BUFFER_SIZE")) + "/" + "/tilings.txtpb";

  bool tilings_exist = std::filesystem::exists(filename);
  if (tilings_exist) {
    std::ifstream input2(filename);
    std::string content = std::string((std::istreambuf_iterator<char>(input2)),
                                      std::istreambuf_iterator<char>());
    voyager::ModelTiling model_tiling;
    if (!TextFormat::ParseFromString(content, &model_tiling)) {
      std::cerr << "Failed to parse text file." << std::endl;
    }

    for (const auto& tiling : model_tiling.tilings()) {
      tiling_map[tiling.name()] = tiling;
    }
  } else {
    std::cerr << "Tilings file does not exist." << std::endl;
  }

  for (const auto& param : model.ops()) {
    if (tiling_map.find(param.name()) != tiling_map.end()) {
      operations.push_back(
          Operation(param.name(), param, tiling_map.at(param.name())));
    } else {
      operations.push_back(Operation(param.name(), param));
    }
  }
}

std::vector<Operation> Network::get_operations(bool filter_nop) {
  if (!filter_nop) {
    return operations;
  }

  std::vector<Operation> ops;
  for (const auto& op : operations) {
    if (!op.param.has_nop()) {
      ops.push_back(op);
    }
  }
  return ops;
}

std::vector<Operation> Network::get_operations(
    const std::vector<std::string>& names, bool filter_nop) {
  const auto all_params = get_operations(filter_nop);

  std::vector<Operation> params;

  if (names.size() == 1) {
    for (const auto& operation : operations) {
      if (operation.name == names[0]) {
        params.push_back(operation);
        break;
      }
    }
  } else if (names.size() == 2) {
    auto first_it = std::find_if(
        operations.begin(), operations.end(),
        [&names](const auto& operation) { return operation.name == names[0]; });
    auto last_it = std::find_if(operations.rbegin(), operations.rend(),
                                [&names](const auto& operation) {
                                  return operation.name == names[1];
                                })
                       .base();

    if (first_it != operations.end() && last_it != operations.begin() &&
        first_it < last_it) {
      params.assign(first_it, last_it);
    }
  } else {
    std::cerr << "Invalid number of names provided" << std::endl;
    exit(1);
  }

  // if filter_nop, remove nop operations
  if (filter_nop) {
    std::vector<Operation> filtered_params;
    for (const auto& param : params) {
      if (!param.param.has_nop()) {
        filtered_params.push_back(param);
      }
    }
    params = filtered_params;
  }

  if (params.empty()) {
    std::cerr << "Param not found" << std::endl;
    exit(1);
  }

  return params;
}
