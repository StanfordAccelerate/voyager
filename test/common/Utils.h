#pragma once

#define NO_SYSC

// IWYU pragma: begin_exports
#include <algorithm>
#include <any>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "src/Params.h"
#include "test/common/Tensor.h"
// IWYU pragma: end_exports

template <typename Container, typename T>
inline bool contains(const Container& container, const T& value) {
  return std::find(std::begin(container), std::end(container), value) !=
         std::end(container);
}

inline bool contain_matrix_param(
    const std::deque<BaseParams*> accelerator_params) {
  for (int i = 0; i < accelerator_params.size(); i++) {
    if (dynamic_cast<MatrixParams*>(accelerator_params[i])) return true;
  }
  return false;
}

inline void print_shape(const std::vector<int>& shape) {
  spdlog::error("{}\n", shape_str(shape));
}

inline int pad_shape_to_ndim(std::vector<int>& shape, const int ndim) {
  const int padding = ndim - shape.size();
  if (padding < 0) {
    throw std::invalid_argument("Number of dimensions exceeds the limit!");
  }

  for (int i = 0; i < padding; i++) {
    shape.insert(shape.begin(), 1);
  }
  return padding;
}

inline std::string getenv(std::string const& name,
                          std::string const& default_value) {
  const char* val = std::getenv(name.c_str());
  return val == NULL ? default_value : std::string(val);
}

inline int getenv_int(const std::string& name, int default_value = 0) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    return default_value;
  }

  try {
    return std::stoi(val);
  } catch (const std::invalid_argument&) {
    return default_value;
  } catch (const std::out_of_range&) {
    return default_value;
  }
}

// Datapath configuration the compiler never maps into memory: codebooks,
// quantization maps, scalar dequantization scales. The accelerator takes them
// as immediates, so they are read straight off disk by node name. Reads a
// file-backed constant (codebook, quantization map, scalar scale).
//
// The size comes from the *file*, not from the tensor's shape. Codebooks and
// qmaps are not declared as TensorBoxes -- they are bare kwarg names -- so a
// resolved codebook operand has no real shape (it inherits the datapath tile
// shape, which is unrelated to the codebook's entry count). The file holds
// exactly the constant's values, so its length is the authority. `count`, when
// given, receives that entry count so a caller can bound its copy loop by it.
inline float* read_constant_param(const Tensor& tensor, int* count = nullptr) {
  const std::string filename =
      std::string(std::getenv("PROJECT_ROOT")) + "/" +
      std::string(getenv("CODEGEN_DIR", "test/compiler")) + "/networks/" +
      std::string(std::getenv("NETWORK")) + "/" +
      std::string(std::getenv("DATATYPE")) + "/tensor_files/" + tensor.node +
      ".bin";

  std::ifstream input_stream(filename, std::ios::binary | std::ios::ate);
  if (!input_stream.good()) {
    throw std::runtime_error("Constant \"" + filename + "\" does not exist");
  }
  const int size = static_cast<int>(input_stream.tellg() / sizeof(float));
  input_stream.seekg(0, std::ios::beg);

  float* data = new float[size];
  input_stream.read(reinterpret_cast<char*>(data), size * sizeof(float));

  if (count != nullptr) *count = size;
  return data;
}
