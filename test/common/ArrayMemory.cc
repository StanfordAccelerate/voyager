#include "test/common/ArrayMemory.h"

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include <fstream>

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

ArrayMemory::ArrayMemory(std::vector<long long> sizes) {
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
        kwargs[value.tensor().node()] = read_tensor(value.tensor());
      }
    }
  }

  return kwargs;
}

std::any ArrayMemory::get_output(const codegen::Operation& param) {
  codegen::Tensor output_tensor = param.output();
  return read_tensor(output_tensor);
}

/**
 * Retrieves the reference output tensor from the given accelerator parameter.
 * The reference output tensor is stored at the last partition at address of
 * 0.
 */
std::any ArrayMemory::get_reference_output(const codegen::Operation& param) {
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  memory->set_partition(-1);
  memory->set_address(0);
  return read_tensor(output_tensor);
}

std::any ArrayMemory::read_tensor(const codegen::Tensor& tensor) {
  int partition = tensor.memory().partition();

  int size = 1;
  for (const auto& dim : tensor.shape()) {
    size *= dim;
  }

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
      std::cerr << "Unsupported data type for scalar tensor: " << tensor.dtype()
                << std::endl;
      std::abort();
    }
  }

  if (tensor.dtype() == "bfloat16") {
    DataTypes::bfloat16* data = new DataTypes::bfloat16[size];
    read_tensor_from_memory<DataTypes::bfloat16>(tensor.memory().address(),
                                                 partition, size, data);
    return data;
  } else if (tensor.dtype() == "int8") {
    DataTypes::int8* data = new DataTypes::int8[size];
    read_tensor_from_memory<DataTypes::int8>(tensor.memory().address(),
                                             partition, size, data);
    return data;
  } else if (tensor.dtype() == "int24") {
    DataTypes::int24* data = new DataTypes::int24[size];
    read_tensor_from_memory<DataTypes::int24>(tensor.memory().address(),
                                              partition, size, data);
    return data;
  } else if (tensor.dtype() == "int32") {
    DataTypes::int32* data = new DataTypes::int32[size];
    read_tensor_from_memory<DataTypes::int32>(tensor.memory().address(),
                                              partition, size, data);
    return data;
  } else if (tensor.dtype() == "fp8_e8m0") {
    DataTypes::fp8_e8m0* data = new DataTypes::fp8_e8m0[size];
    read_tensor_from_memory<DataTypes::fp8_e8m0>(tensor.memory().address(),
                                             partition, size, data);
    return data;
  } else if (tensor.dtype() == "fp8_e5m3") {
    DataTypes::fp8_e5m3* data = new DataTypes::fp8_e5m3[size];
    read_tensor_from_memory<DataTypes::fp8_e5m3>(tensor.memory().address(),
                                                 partition, size, data);
    return data;
  } else {
    INPUT_DATATYPE* data = new INPUT_DATATYPE[size];
    read_tensor_from_memory<INPUT_DATATYPE>(tensor.memory().address(),
                                            partition, size, data);
    return data;
  }
}

void ArrayMemory::write_tensor(const codegen::Tensor& tensor,
                               const std::any data) {
  const auto& tensor_memory = tensor.memory();
  const uint64_t address = tensor_memory.address();
  const int partition = tensor_memory.partition();

  int size = 1;
  for (const auto& dim : tensor.shape()) {
    size *= dim;
  }

  const auto dtype = tensor.dtype();
  if (dtype == "bfloat16") {
    write_tensor_to_memory<DataTypes::bfloat16>(
        address, partition, size, std::any_cast<DataTypes::bfloat16*>(data));
  } else if (dtype == "int8") {
    write_tensor_to_memory<DataTypes::int8>(
        address, partition, size, std::any_cast<DataTypes::int8*>(data));
  } else if (dtype == "int24") {
    write_tensor_to_memory<DataTypes::int24>(
        address, partition, size, std::any_cast<DataTypes::int24*>(data));
  } else if (dtype == "int32") {
    write_tensor_to_memory<DataTypes::int32>(
        address, partition, size, std::any_cast<DataTypes::int32*>(data));
  } else if (dtype == "fp8_e8m0") {
    write_tensor_to_memory<DataTypes::fp8_e8m0>(
        address, partition, size, std::any_cast<DataTypes::fp8_e8m0*>(data));
  } else if (dtype == "fp8_e5m3") {
    write_tensor_to_memory<DataTypes::fp8_e5m3>(
        address, partition, size, std::any_cast<DataTypes::fp8_e5m3*>(data));
  } else {
    // Default to INPUT_DATATYPE
    write_tensor_to_memory<INPUT_DATATYPE>(
        address, partition, size, std::any_cast<INPUT_DATATYPE*>(data));
  }
}
