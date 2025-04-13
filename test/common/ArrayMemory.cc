#include "test/common/ArrayMemory.h"

// clang-format off
#include "src/datatypes/DataTypes.h"
// clang-format on

#include <fstream>

#include "spdlog/spdlog.h"
#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

ArrayMemory::ArrayMemory(std::vector<uint64_t> sizes) {
  memories.reserve(sizes.size());
  try {
    for (const auto size : sizes) {
      char* memory = new char[size];
      std::fill(memory, memory + size, 0);
      memories.push_back(memory);
    }
  } catch (const std::bad_alloc& e) {
    // Clean up any allocated memory if an exception is thrown
    for (char* memory : memories) {
      delete[] memory;
    }
    memories.clear();
    throw std::runtime_error("Memory allocation failed: " +
                             std::string(e.what()));
  }
}

ArrayMemory::~ArrayMemory() {
  for (char* memory : memories) {
    delete[] memory;
  }
}

char* ArrayMemory::get_memory(int partition) {
  if (partition < 0) {
    partition += memories.size();
  }
  if (partition < 0 or partition >= memories.size()) {
    throw std::runtime_error("Invalid memory partition: " +
                             std::to_string(partition));
  }
  return memories[partition];
}

std::map<std::string, std::any> ArrayMemory::get_args(
    const codegen::Operation& param) {
  std::map<std::string, std::any> kwargs;

  const auto op_list = get_op_list(param);

  for (const auto op : op_list) {
    for (const auto [key, value] : op.kwargs()) {
      if (value.has_tensor() && value.tensor().has_memory()) {
        spdlog::debug("Pushing tensor: {}\n", value.tensor().node());
        kwargs[value.tensor().node()] = read_tensor(value.tensor());
      }
    }
  }

  return kwargs;
}

std::vector<std::any> ArrayMemory::get_outputs(
    const codegen::Operation& param) {
  const auto tensors = get_op_outputs(param);
  std::vector<std::any> outputs;
  for (const auto& tensor : tensors) {
    outputs.push_back(read_tensor(tensor));
  }
  return outputs;
}

/**
 * Retrieves the reference output tensor from the given accelerator parameter.
 * The reference output tensor is stored at the last partition at address of
 * 0.
 */
std::vector<std::any> ArrayMemory::get_reference_outputs(
    const codegen::Operation& param) {
  const auto tensors = get_op_outputs(param);
  std::vector<std::any> outputs;

  uint64_t address = 0;

  for (const auto& tensor : tensors) {
    codegen::Tensor tensor_copy;
    tensor_copy.CopyFrom(tensor);
    auto memory = tensor_copy.mutable_memory();
    memory->set_partition(-1);
    memory->set_address(address);

    outputs.push_back(read_tensor(tensor_copy));
    address += get_size(tensor);
  }

  return outputs;
}

template <typename T>
bool read_tensor(ArrayMemory* mem, codegen::Tensor tensor, std::any& output) {
  if (tensor.dtype() != DataTypes::TypeName<T>::name()) {
    return false;
  }

  const auto& memory = tensor.memory();
  const int size = get_size(tensor, false);

  T* data = new T[size];
  mem->read_tensor_from_memory<T>(memory.address(), memory.partition(), size,
                                  data);
  output = data;
  return true;
}

template <typename... Ts>
std::any read_tensor_helper(ArrayMemory* mem, codegen::Tensor tensor) {
  std::any output;
  bool matched = (read_tensor<Ts>(mem, tensor, output) || ...);
  if (!matched) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
  }
  return output;
}

std::any ArrayMemory::read_tensor(const codegen::Tensor& tensor) {
  int partition = tensor.memory().partition();

  int size = get_size(tensor, false);

  if (size == 1) {  // for scalar, we get the arg from the file, not from memory
    const char* env_var = std::getenv("NETWORK");
    std::string model_name(env_var);
    std::string project_root = std::string(std::getenv("PROJECT_ROOT"));
    std::string datatype = std::string(std::getenv("DATATYPE"));
    std::string filename =
        project_root + "/" + std::string(getenv("CODEGEN_DIR")) + "/networks/" +
        model_name + "/" + datatype + "/tensor_files/" + tensor.node() + ".bin";

    float scalar;
    std::ifstream input_stream(filename, std::ios::binary);
    input_stream.read(reinterpret_cast<char*>(&scalar), sizeof(float));

    if (tensor.dtype() == "bfloat16" || tensor.dtype() == "float32") {
      VECTOR_DATATYPE* data = new VECTOR_DATATYPE[1];
      data[0] = scalar;
      return data;
    } else {
      spdlog::debug("Unsupported data type for scalar tensor: {}\n",
                    tensor.dtype());
      std::abort();
    }
  }

  return read_tensor_helper<SUPPORTED_TYPES>(this, tensor);
}

template <typename T>
bool write_tensor(ArrayMemory* mem, codegen::Tensor tensor, std::any data) {
  if (tensor.dtype() != DataTypes::TypeName<T>::name()) {
    return false;
  }

  const auto& memory = tensor.memory();
  const auto size = get_size(tensor, false);

  mem->write_tensor_to_memory<T>(memory.address(), memory.partition(), size,
                                 std::any_cast<T*>(data));
  return true;
}

template <typename... Ts>
void write_tensor_helper(ArrayMemory* mem, codegen::Tensor tensor,
                         std::any data) {
  bool matched = (write_tensor<Ts>(mem, tensor, data) || ...);
  if (!matched) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype());
  }
}

void ArrayMemory::write_tensor(const codegen::Tensor& tensor,
                               const std::any data) {
  write_tensor_helper<SUPPORTED_TYPES>(this, tensor, data);
}
