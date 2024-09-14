#pragma once

#include <map>
#include <set>

#include "src/Params.h"
#include "test/common/VerificationTypes.h"

std::map<int, std::set<std::string>> vector_ops = {
    {0, {"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_"}},
    {1, {"exp"}},
    {2, {}},
    {3, {"add", "add_", "mul", "mul_", "div", "div_", "square"}},
    {4, {"relu", "relu_"}}};

std::map<std::string, unsigned int> get_vector_instruction_mapping() {
  std::map<std::string, unsigned int> mapping;
  mapping["add"] = VectorInstructions::vadd;
  mapping["add_"] = VectorInstructions::vadd;
  mapping["sub"] = VectorInstructions::vsub;
  mapping["sub_"] = VectorInstructions::vsub;
  mapping["mul"] = VectorInstructions::vmult;
  mapping["mul_"] = VectorInstructions::vmult;
  mapping["div"] = VectorInstructions::vmult;
  mapping["div_"] = VectorInstructions::vmult;
  mapping["relu"] = VectorInstructions::vrelu;
  mapping["relu_"] = VectorInstructions::vrelu;
  mapping["exp"] = VectorInstructions::vexp;
  mapping["square"] = VectorInstructions::vsquare;
  return mapping;
}

inline MemorySource get_partition(const int &partition) {
  return partition == 0 ? SRAM : RRAM;
}

inline float read_constant_param(const codegen::Tensor tensor) {
  const char *env_var = std::getenv("NETWORK");
  std::string model_name(env_var);
  std::string project_root = std::string(std::getenv("PROJECT_ROOT"));
  std::string datatype = std::string(std::getenv("DATATYPE"));
  std::string filename = project_root + "/test/compiler/networks/" +
                         model_name + "/" + datatype + "/tensor_files/" +
                         tensor.node() + ".bin";

  float *array_ptr = new float[1];
  std::ifstream input_stream(filename, std::ios::binary);
  input_stream.read(reinterpret_cast<char *>(array_ptr), sizeof(float));
  return array_ptr[0];
}

int getLargestFactor(const int dim) {
  int largestFactor = 1;
  for (int i = 2; i <= 1024; i++) {  // 1024 is the maximum dimension
    if (dim % i == 0) {
      largestFactor = i;
    }
  }

  return largestFactor;
}

void factorizeForAddressGen(const int dim, int *factors) {
  if (dim > 1024) {
    int largestFactor = getLargestFactor(dim);
    factors[0] = dim / largestFactor;
    factors[1] = largestFactor;
  } else {
    factors[0] = 1;
    factors[1] = dim;
  }
}
