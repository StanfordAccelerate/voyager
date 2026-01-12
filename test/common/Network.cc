#include "test/common/Network.h"

#include <google/protobuf/text_format.h>

#include <fstream>

#include "spdlog/spdlog.h"
#include "test/common/Utils.h"

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
    spdlog::error("Failed to parse text file.\n");
  }

  std::map<std::string, voyager::Tiling> tiling_map;
  filename = project_root + "/" + std::string(getenv("CODEGEN_DIR")) +
             "/networks/" + model_name + "/" + datatype + "/" +
             std::string(getenv("IC_DIMENSION")) + "x" +
             std::string(getenv("OC_DIMENSION")) + "_" +
             std::string(getenv("INPUT_BUFFER_SIZE", "1024")) + "x" +
             std::string(getenv("WEIGHT_BUFFER_SIZE", "1024")) + "x" +
             std::string(getenv("ACCUM_BUFFER_SIZE", "1024")) + "_" +
             std::string(getenv("DOUBLE_BUFFERED_ACCUM_BUFFER", "false")) +
             "/tilings.txtpb";

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
    spdlog::error("Tilings file does not exist: {} \n", filename);
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

uint64_t get_max_address(const codegen::Tensor& tensor) {
  if (!tensor.has_memory()) {
    return 0;
  }

  uint64_t addr = tensor.memory().address();
  uint64_t size = get_size(tensor, false, false);  // untiled and unreshaped

  int index = get_index_from_type_name<SUPPORTED_TYPES>(tensor.dtype());
  size_t width = get_type_width<SUPPORTED_TYPES>(index);

  uint64_t end_addr = addr + (size * width / 8);
  return end_addr;
}

uint64_t Network::get_max_dram_address() const {
  uint64_t max_address = 0;

  // Check all input tensors
  for (const auto& input : model.inputs()) {
    max_address = std::max(max_address, get_max_address(input));
  }

  // Check all parameter tensors
  for (const auto& param : model.parameters()) {
    max_address = std::max(max_address, get_max_address(param));
  }

  // Check all operation outputs
  for (const auto& op : model.ops()) {
    const auto outputs = get_op_outputs(op);
    for (const auto& output : outputs) {
      max_address = std::max(max_address, get_max_address(output));
    }
  }

  return max_address;
}

uint64_t Network::get_max_output_size() const {
  uint64_t max_size = 0;

  for (const auto& op : model.ops()) {
    const auto outputs = get_op_outputs(op);

    uint64_t num_bytes = 0;
    for (const auto& output : outputs) {
      auto size = get_size(output, false, false);
      auto index = get_index_from_type_name<SUPPORTED_TYPES>(output.dtype());
      auto width = get_type_width<SUPPORTED_TYPES>(index);
      num_bytes += size * width / 8;
    }

    max_size = std::max(max_size, num_bytes);
  }

  return max_size;
}
