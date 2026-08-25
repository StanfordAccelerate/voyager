#include <systemc.h>

#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/Model.h"
#include "test/common/Utils.h"
#include "test/soc/EmitC.h"

namespace {

std::string require_env(const std::string& name) {
  const char* value = std::getenv(name.c_str());
  if (!value) {
    std::cerr << "Environment variable " << name << " not set" << std::endl;
    exit(1);
  }
  return std::string(value);
}

// The firmware's send loops need each struct's marshalled bit width.
void write_params_width() {
  std::stringstream ss;
  ss << "#ifndef VOYAGER_PARAMS_H\n";
  ss << "#define VOYAGER_PARAMS_H\n\n";
  ss << "static const unsigned int matrix_params_width = "
     << Wrapped<MatrixParams>::width << ";\n";
  ss << "static const unsigned int vector_params_width = "
     << Wrapped<VectorParams>::width << ";\n";
  ss << "static const unsigned int vector_instruction_config_width = "
     << Wrapped<VectorInstructionConfig>::width << ";\n";
  ss << "#endif\n";

  std::ofstream header("firmware/common/voyager_params.h", std::ios::out);
  header << ss.str();
}

std::vector<std::string> split_names(const std::string& csv) {
  std::vector<std::string> names;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) names.push_back(item);
  }
  return names;
}

}  // namespace

int sc_main(int argc, char* argv[]) {
  const std::string network = require_env("NETWORK");
  const std::string datatype = require_env("DATATYPE");
  const std::string tests = require_env("TESTS");

  const std::string base_path =
      "firmware/networks/" + network + "/" + datatype + "/" +
      require_env("IC_DIMENSION") + "x" + require_env("OC_DIMENSION") + "_" +
      require_env("INPUT_BUFFER_SIZE") + "x" +
      require_env("WEIGHT_BUFFER_SIZE") + "x" +
      require_env("ACCUM_BUFFER_SIZE") + "_" +
      require_env("DOUBLE_BUFFERED_ACCUM_BUFFER") + "_" +
      require_env("SUPPORT_MVM") + "_" + require_env("SUPPORT_SPMM") + "/";
  std::experimental::filesystem::create_directories(base_path);

  std::cout << "Generating SoC firmware for " << network << "/" << datatype
            << " into " << base_path << std::endl;

  write_params_width();

  Model model(network);
  CEmitter emitter(model);

  bool ok = true;
  for (const auto& name : split_names(tests)) {
    std::cout << "Emitting " << name << std::endl;
    try {
      const auto selection = model.selected_ops({name});
      const std::string program = emitter.emit_layer(selection);
      const std::string path = base_path + name + ".c";
      {
        std::ofstream out(path, std::ios::out);
        out << program;
      }
      // Emitted firmware follows the repo's formatting rule too; best-effort.
      std::system(("clang-format -i --style=file " + path + " 2>/dev/null")
                      .c_str());
    } catch (const std::exception& error) {
      std::cerr << "Skipping " << name << ": " << error.what() << std::endl;
      ok = false;
    }
  }

  return ok ? 0 : 1;
}
