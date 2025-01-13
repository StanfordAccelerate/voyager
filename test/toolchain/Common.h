#pragma once

#include <map>
#include <set>

#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

std::map<int, std::set<std::string>> vector_unit_stages = {
    {0, {"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_"}},
    {1, {"exp"}},
    {2, {}},
    {3, {"add", "add_", "mul", "mul_", "div", "div_", "square"}},
    {4, {"relu", "relu_", "vmap"}},
};

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
  mapping["vmap"] = VectorInstructions::vmap;
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
  std::string filename =
      project_root + "/" + std::string(getenv("CODEGEN_DIR")) + "/networks/" +
      model_name + "/" + datatype + "/tensor_files/" + tensor.node() + ".bin";

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

inline std::vector<int> get_tensor_shape(const codegen::Tensor &tensor) {
  return std::vector<int>(tensor.shape().begin(), tensor.shape().end());
}

inline std::vector<int> get_input_shape(const codegen::Tensor &tensor) {
  if (tensor.has_reshape()) {
    const auto &param = tensor.reshape();
    return {param.output_sizes().begin(), param.output_sizes().end()};
  }

  if (tensor.has_slicing()) {
    const auto &param = tensor.slicing();
    return {param.output_sizes().begin(), param.output_sizes().end()};
  }

  const auto repeated_field = tensor.shape();
  return {repeated_field.begin(), repeated_field.end()};
}

inline std::vector<int> squeeze_shape(const std::vector<int> &input) {
  std::vector<int> result;
  for (int value : input) {
    if (value > 1) {
      result.push_back(value);
    }
  }
  return result;
}

inline int get_size(const std::vector<int> &shape) {
  int size = 1;
  for (const auto &dim : shape) size *= dim;
  return size;
}

inline int get_size(const codegen::Tensor &tensor) {
  const auto shape = get_input_shape(tensor);
  return get_size(shape);
}

std::vector<int> split_loops(std::vector<int> loops, int max_value) {
  if (max_value <= 1) {
    throw std::invalid_argument("max_value must be greater than 1.");
  }

  std::vector<int> result;

  for (int value : loops) {
    if (value <= max_value) {
      result.push_back(value);
    } else {
      // Find two factors of value such that both are <= max_value
      bool split_found = false;
      for (int factor = std::sqrt(value); factor > 1; --factor) {
        if (value % factor == 0) {
          int other_factor = value / factor;
          if (factor <= max_value && other_factor <= max_value) {
            result.push_back(factor);
            result.push_back(other_factor);
            split_found = true;
            break;
          }
        }
      }

      if (!split_found) {
        throw std::runtime_error("Unable to split value " +
                                 std::to_string(value) +
                                 " into two factors <= max_value.");
      }
    }
  }

  return result;
}

// Function to compute the prime factors of a given number
std::vector<int> prime_factors(int n) {
  std::vector<int> factors;
  for (int i = 2; i <= n / i; ++i) {
    while (n % i == 0) {
      factors.push_back(i);
      n /= i;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

/**
 * Adjusts loop indices such that the prime factors of the target factor are
 * distributed among the indices, starting from the rightmost index.
 *
 * @param loops A vector of integers representing the loop indices.
 * @param target_factor An integer whose prime factors are to be distributed.
 * @return A vector of adjusted loop indices.
 * @throws std::runtime_error if the prime factors cannot be fully distributed.
 */
std::vector<int> adjust_loop_indices(const std::vector<int> &loops,
                                     int target_factor) {
  // Prime factorization of the target factor
  std::vector<int> remaining_factors = prime_factors(target_factor);

  // Result vector
  std::vector<int> result = loops;

  // Process from right to left
  for (int i = result.size() - 1; i >= 0; --i) {
    for (auto it = remaining_factors.begin(); it != remaining_factors.end();) {
      int factor = *it;
      if (result[i] % factor == 0) {
        result[i] /= factor;  // Divide the current index by the factor
        it = remaining_factors.erase(it);  // Remove the used factor
      } else {
        ++it;  // Move to the next factor
      }
    }
  }

  // If any factors remain unused, error out
  if (!remaining_factors.empty()) {
    throw std::runtime_error("Unused prime factors remain.");
  }

  // Multiply the rightmost index by the target factor
  result[result.size() - 1] *= target_factor;

  return result;
}

int pad_shape_to_ndim(std::vector<int> &shape, const int ndim) {
  const int padding = ndim - shape.size();
  if (padding < 0) {
    throw std::invalid_argument("Number of dimensions exceeds the limit!");
  }

  for (int i = 0; i < padding; i++) {
    shape.insert(shape.begin(), 1);
  }
  return padding;
}
