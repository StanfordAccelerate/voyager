#include "test/common/Network.h"

#include <google/protobuf/text_format.h>

#include <fstream>

#include "test/common/VerificationTypes.h"

using namespace std;
using namespace google::protobuf;

#include "spdlog/spdlog.h"

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
    spdlog::error("Failed to parse text file.\n");
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
      spdlog::error("Failed to parse text file.\n");
    }

    for (const auto& tiling : model_tiling.tilings()) {
      tiling_map[tiling.name()] = tiling;
    }
  } else {
    spdlog::error("Tilings file does not exist.\n");
  }

  for (const auto& op : model.ops()) {
    const std::string name = get_op_name(op);
    if (tiling_map.find(name) != tiling_map.end()) {
      operations.push_back(Operation(name, op, tiling_map.at(name)));
    } else {
      operations.push_back(Operation(name, op));
    }
  }
}

std::vector<Operation> Network::get_operations(bool filter_nop) {
  if (!filter_nop) {
    return operations;
  }

  std::vector<Operation> ops;
  for (const auto& op : operations) {
    if (op.param.op().op() != "nop") {
      ops.push_back(op);
    }
  }
  return ops;
}

std::vector<Operation> Network::get_operations(
    const std::vector<std::string>& names, bool filter_nop) {
  const auto operations = get_operations(filter_nop);

  std::vector<Operation> filtered_ops;

  if (names.size() == 1) {
    for (const auto& operation : operations) {
      if (operation.name == names[0]) {
        filtered_ops.push_back(operation);
        break;
      }
    }
  } else if (names.size() == 2) {
    bool found_first = false;
    bool found_second = false;
    for (const auto& operation : operations) {
      const auto param = operation.param;
      if (get_op_name(param) == names[0]) {
        found_first = true;
      }
      if (found_first) {
        filtered_ops.push_back(operation);
      }
      if (get_op_name(param) == names[1]) {
        found_second = true;
        break;
      }
    }

    if (!found_first || !found_second) {
      spdlog::error("Invalid names provided\n");
      exit(1);
    }
  } else {
    spdlog::error("Invalid number of names provided\n");
    exit(1);
  }

  if (filtered_ops.empty()) {
    spdlog::error("Param not found\n");
    exit(1);
  }

  return filtered_ops;
}
