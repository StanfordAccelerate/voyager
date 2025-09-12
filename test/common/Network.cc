#include "test/common/Network.h"

#include <google/protobuf/text_format.h>

#include <fstream>

#include "spdlog/spdlog.h"
#include "test/common/VerificationTypes.h"

using namespace std;
using namespace google::protobuf;

bool is_shrinkable_operation(const codegen::Operation& operation,
                             const voyager::Tiling& tiling) {
  // only shrink the operation if there aren't reshapes.
  // changing shape sizes in operations with reshapes complicates things.
  // operations with reshape also typically aren't the largest
  // operations in the network, so it's not worth the effort to shrink them

  auto op_list = get_op_list(operation);
  const auto first_op = op_list.front();

  // check for input reshape
  if (first_op.kwargs().at("input").tensor().has_reshape()) {
    return false;
  }

  // check for weight reshape
  bool is_matmul = first_op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  if (first_op.kwargs().at(weight_key).tensor().has_reshape()) {
    return false;
  }

  // check for output reshape
  if (operation.output().has_reshape()) {
    return false;
  }

  return true;
}

int shrink_operation(codegen::Operation& operation, voyager::Tiling& tiling) {
  std::string tiling_print;
  google::protobuf::TextFormat::PrintToString(tiling, &tiling_print);
  spdlog::debug("Original Tiling: {}", tiling_print);

  int shrink_factor = 1;
  int X_shrink_factor = 1;
  int Y_shrink_factor = 1;
  int C_shrink_factor = 1;
  int K_shrink_factor = 1;

  // look at tiling and see which loops can be shrunk
  for (int loop_index = 0;
       loop_index < tiling.level_tilings(1).loop_bounds_size(); loop_index++) {
    int min_loop_factor = 1;
    if (shrink_factor == 1) {
      min_loop_factor = 2;
    }

    if (tiling.level_tilings(1).loop_bounds(loop_index).bound() >
        min_loop_factor) {
      int loop_shrink_factor =
          tiling.level_tilings(1).loop_bounds(loop_index).bound() /
          min_loop_factor;

      tiling.mutable_level_tilings(1)
          ->mutable_loop_bounds(loop_index)
          ->set_bound(min_loop_factor);

      if (tiling.level_tilings(1).loop_bounds(loop_index).loop() ==
          voyager::Loop::IC) {
        C_shrink_factor = loop_shrink_factor;
      } else if (tiling.level_tilings(1).loop_bounds(loop_index).loop() ==
                 voyager::Loop::OC) {
        K_shrink_factor = loop_shrink_factor;
      } else if (tiling.level_tilings(1).loop_bounds(loop_index).loop() ==
                 voyager::Loop::OX) {
        X_shrink_factor = loop_shrink_factor;
      } else if (tiling.level_tilings(1).loop_bounds(loop_index).loop() ==
                 voyager::Loop::OY) {
        Y_shrink_factor = loop_shrink_factor;
      }
      shrink_factor *= loop_shrink_factor;
    }
  }

  google::protobuf::TextFormat::PrintToString(tiling, &tiling_print);
  spdlog::debug("Shrunk Tiling: {}", tiling_print);

  if (shrink_factor != 1) {
    std::vector<codegen::OpOverload*> ops_list;
    if (operation.has_op()) {
      ops_list.push_back(operation.mutable_op());
    } else {
      for (auto& op : *operation.mutable_fused_op()->mutable_op_list()) {
        ops_list.push_back(&op);
      }
    }

    google::protobuf::TextFormat::PrintToString(operation, &tiling_print);
    spdlog::debug("Original Operation: {}", tiling_print);

    for (auto& op : ops_list) {
      if (GEMM_OPS.find(op->target()) != GEMM_OPS.end()) {
        auto input_tensor = op->mutable_kwargs()->at("input").mutable_tensor();
        int input_channels;
        if (input_tensor->shape_size() == 2) {
          input_channels = input_tensor->shape(1);
          input_tensor->set_shape(0, input_tensor->shape(0) / X_shrink_factor);
          input_tensor->set_shape(1, input_tensor->shape(1) / C_shrink_factor);
        } else if (input_tensor->shape_size() == 3) {
          // 3d tensor, where the first dimension should be 1
          input_channels = input_tensor->shape(2);
          assert(input_tensor->shape(0) == 1);
          input_tensor->set_shape(1, input_tensor->shape(1) / X_shrink_factor);
          input_tensor->set_shape(2, input_tensor->shape(2) / C_shrink_factor);
        } else if (input_tensor->shape_size() == 4) {
          // 4d tensor, where the first two dimensions should be 1
          assert(input_tensor->shape(0) == 1);
          assert(input_tensor->shape(1) == 1);
          input_channels = input_tensor->shape(3);
          input_tensor->set_shape(2, input_tensor->shape(2) / X_shrink_factor);
          input_tensor->set_shape(3, input_tensor->shape(3) / C_shrink_factor);

        } else {
          spdlog::error("Unsupported shape\n");
          std::abort();
        }

        bool is_matmul = op->target().find("matmul") != std::string::npos;
        std::string weight_key = is_matmul ? "other" : "weight";
        const auto weight_tensor =
            op->mutable_kwargs()->at(weight_key).mutable_tensor();

        if (weight_tensor->shape(0) == input_channels) {
          // weight tensor is in ICxOC format
          weight_tensor->set_shape(0,
                                   weight_tensor->shape(0) / C_shrink_factor);
          weight_tensor->set_shape(1,
                                   weight_tensor->shape(1) / K_shrink_factor);

        } else {
          weight_tensor->set_shape(0,
                                   weight_tensor->shape(0) / K_shrink_factor);
          weight_tensor->set_shape(1,
                                   weight_tensor->shape(1) / C_shrink_factor);
        }

        if (op->kwargs().contains("bias")) {
          auto bias = op->mutable_kwargs()->at("bias").mutable_tensor();
          bias->set_shape(0, bias->shape(0) / K_shrink_factor);
        }
      } else {
        // adjust dimensions of fused vector operations
        auto input_tensor = op->mutable_kwargs()->at("input").mutable_tensor();
        if (input_tensor->shape_size() == 2) {
          input_tensor->set_shape(0, input_tensor->shape(0) / X_shrink_factor);
          input_tensor->set_shape(1, input_tensor->shape(1) / K_shrink_factor);
        } else if (input_tensor->shape_size() == 3) {
          assert(input_tensor->shape(0) == 1);
          input_tensor->set_shape(1, input_tensor->shape(1) / X_shrink_factor);
          input_tensor->set_shape(2, input_tensor->shape(2) / K_shrink_factor);
        } else if (input_tensor->shape_size() == 4) {
          assert(input_tensor->shape(0) == 1);
          assert(input_tensor->shape(1) == 1);
          input_tensor->set_shape(2, input_tensor->shape(2) / X_shrink_factor);
          input_tensor->set_shape(3, input_tensor->shape(3) / K_shrink_factor);
        } else {
          spdlog::error("Unsupported shape\n");
          std::abort();
        }

        if (op->kwargs().contains("other")) {
          if (op->kwargs().at("other").tensor().shape_size() == 3) {
            auto other_tensor =
                op->mutable_kwargs()->at("other").mutable_tensor();
            assert(other_tensor->shape(0) == 1);
            other_tensor->set_shape(1,
                                    other_tensor->shape(1) / X_shrink_factor);
            other_tensor->set_shape(2,
                                    other_tensor->shape(2) / K_shrink_factor);
          } else if (op->kwargs().at("other").tensor().shape_size() == 4) {
            assert(op->kwargs().at("other").tensor().shape(0) == 1);
            assert(op->kwargs().at("other").tensor().shape(1) == 1);
            auto other_tensor =
                op->mutable_kwargs()->at("other").mutable_tensor();
            other_tensor->set_shape(2,
                                    other_tensor->shape(2) / X_shrink_factor);
            other_tensor->set_shape(3,
                                    other_tensor->shape(3) / K_shrink_factor);
          } else {
            spdlog::error("Unsupported shape\n");
            std::abort();
          }
        }
      }
    }

    // Adjust shape of output tensor
    auto output_tensor = operation.mutable_output();

    if (output_tensor->shape_size() == 2) {
      output_tensor->set_shape(0, output_tensor->shape(0) / X_shrink_factor);
      output_tensor->set_shape(1, output_tensor->shape(1) / K_shrink_factor);
    } else if (output_tensor->shape_size() == 3) {
      assert(output_tensor->shape(0) == 1);
      output_tensor->set_shape(1, output_tensor->shape(1) / X_shrink_factor);
      output_tensor->set_shape(2, output_tensor->shape(2) / K_shrink_factor);
    } else if (output_tensor->shape_size() == 4) {
      assert(output_tensor->shape(0) == 1);
      assert(output_tensor->shape(1) == 1);
      output_tensor->set_shape(2, output_tensor->shape(2) / X_shrink_factor);
      output_tensor->set_shape(3, output_tensor->shape(3) / K_shrink_factor);
    } else {
      spdlog::error("Unsupported shape\n");
      std::abort();
    }

    google::protobuf::TextFormat::PrintToString(operation, &tiling_print);
    spdlog::debug("Shrunk operation: {}", tiling_print);
  }

  return shrink_factor;
}

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
  std::cerr << "Reading tilings from file: " << filename << std::endl;
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

  // Scale down operation if environment variable is set
  const char* env_var = std::getenv("SCALE_DOWN_OPERATION");
  bool scale_down_operation = env_var ? std::stoi(env_var) : false;

  for (const auto& op : model.ops()) {
    const std::string name = get_op_name(op);
    if (tiling_map.find(name) != tiling_map.end()) {
      if (scale_down_operation &&
          is_shrinkable_operation(op, tiling_map.at(name))) {
        codegen::Operation shrunk_operation = op;
        int shrink_factor =
            shrink_operation(shrunk_operation, tiling_map.at(name));
        operations.push_back(Operation(name, shrunk_operation,
                                       tiling_map.at(name), shrink_factor));
      } else {
        operations.push_back(Operation(name, op, tiling_map.at(name)));
      }

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
      std::cerr << get_op_name(operation.param) << std::endl;
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
