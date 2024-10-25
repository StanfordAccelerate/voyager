#include "test/common/ArrayMemory.h"

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include <fstream>

#include "src/ArchitectureParams.h"

ArrayMemory::ArrayMemory(std::vector<int> sizes) : MemoryInterface() {
  memories.reserve(sizes.size());
  try {
    for (int size : sizes) {
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

std::vector<std::any> ArrayMemory::get_args(
    const codegen::AcceleratorParam& param) {
  std::vector<std::any> args;
  std::string output_node = "";
  if (param.has_matrix_param()) {
    const codegen::MatrixParam& matrix_param = param.matrix_param();

    if (matrix_param.has_mx_input()) {
      args.push_back(get_tensor(matrix_param.mx_input().input()));
      args.push_back(get_tensor(matrix_param.mx_input().scale()));
    } else {
      args.push_back(get_tensor(matrix_param.input()));
    }

    if (matrix_param.has_mx_weight()) {
      args.push_back(get_tensor(matrix_param.mx_weight().input()));
      args.push_back(get_tensor(matrix_param.mx_weight().scale()));
    } else {
      args.push_back(get_tensor(matrix_param.weight()));
    }

    if (matrix_param.has_bias()) {
      args.push_back(get_tensor(matrix_param.bias()));
    } else {
      args.push_back(nullptr);
    }
    output_node = matrix_param.name();
  } else if (param.has_pooling_param()) {
    const codegen::PoolingParam& pooling_param = param.pooling_param();
    args.push_back(get_tensor(pooling_param.input()));
  } else if (param.has_reduce_param()) {
    const codegen::ReduceParam& reduce_param = param.reduce_param();
    args.push_back(get_tensor(reduce_param.input()));
  } else if (param.has_reshape_param()) {
    const codegen::ReshapeParam& reshape_param = param.reshape_param();
    args.push_back(get_tensor(reshape_param.input()));
  } else if (param.vector_params_size() > 0) {
    const auto vector_param = param.vector_params(0);
    args.push_back(get_tensor(vector_param.input()));
  }

  for (auto& vector_param : param.vector_params()) {
    if (vector_param.has_other()) {
      // Check whether input or other is the output of the last operation.
      const auto input = vector_param.input();
      const auto other = vector_param.other();
      const auto tensor_to_load = other.node() == output_node ? input : other;
      args.push_back(get_tensor(tensor_to_load));
    }
    output_node = vector_param.name();
  }

  // Add the output tensor to the list of arguments
  const codegen::Tensor& output = param.output();
  char* outputPtr =
      get_memory(output.memory().partition()) + output.memory().offset();
  args.push_back(outputPtr);

  return args;
}

/**
 * Retrieves the reference output tensor from the given accelerator parameter.
 * The reference output tensor is stored at the last partition with an offset of
 * 0.
 */
std::any ArrayMemory::get_reference_output(
    const codegen::AcceleratorParam& param) {
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  memory->set_partition(-1);
  memory->set_offset(0);
  return get_tensor(output_tensor);
}

std::any ArrayMemory::get_output(const codegen::AcceleratorParam& param) {
  codegen::Tensor output_tensor = param.output();
  return get_tensor(output_tensor);
}

template <typename T>
void ArrayMemory::read_tensor_from_memory(const int address,
                                          const int partition, const int size,
                                          T* tensor) {
  char* memory = get_memory(partition) + address;

  for (int i = 0; i < size; i++) {
    ac_int<T::width> bits;
    assert(T::width % 8 == 0);
    for (int j = 0; j < T::width / 8; j++) {
      bits.set_slc(
          j * 8, static_cast<ac_int<8, false> >(memory[i * T::width / 8 + j]));
    }
    tensor[i].setbits(bits);
  }
}

// TODO: clean this up. currently this is causing namespace conflicts. we need
// to make this file .cc instead of .h
inline std::vector<int> get_shape_2(const codegen::Tensor& tensor) {
  auto repeated_field = tensor.shape();
  return std::vector<int>(repeated_field.begin(), repeated_field.end());
}

inline int get_size_2(const std::vector<int>& shape) {
  int size = 1;
  for (const auto& dim : shape) size *= dim;
  return size;
}

inline int get_size_2(const codegen::Tensor& tensor) {
  const auto shape = get_shape_2(tensor);
  return get_size_2(shape);
}

std::any ArrayMemory::get_tensor(const codegen::Tensor& tensor) {
  int partition = tensor.memory().partition();
  int size = get_size_2(tensor);

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
    read_tensor_from_memory<DataTypes::bfloat16>(tensor.memory().offset(),
                                                 partition, size, data);
    return data;
  } else if (tensor.dtype() == "int8") {
    DataTypes::int8* data = new DataTypes::int8[size];
    read_tensor_from_memory<DataTypes::int8>(tensor.memory().offset(),
                                             partition, size, data);
    return data;
  } else if (tensor.dtype() == "int24") {
    DataTypes::int24* data = new DataTypes::int24[size];
    read_tensor_from_memory<DataTypes::int24>(tensor.memory().offset(),
                                              partition, size, data);
    return data;
  } else if (tensor.dtype() == "int32") {
    DataTypes::int32* data = new DataTypes::int32[size];
    read_tensor_from_memory<DataTypes::int32>(tensor.memory().offset(),
                                              partition, size, data);
    return data;
  } else if (tensor.dtype() == "e8m0") {
    DataTypes::e8m0* data = new DataTypes::e8m0[size];
    read_tensor_from_memory<DataTypes::e8m0>(tensor.memory().offset(),
                                             partition, size, data);
    return data;
  } else {
    INPUT_DATATYPE* data = new INPUT_DATATYPE[size];
    read_tensor_from_memory<INPUT_DATATYPE>(tensor.memory().offset(), partition,
                                            size, data);
    return data;
  }
}

void ArrayMemory::write_bytes_to_memory(const int address, const int partition,
                                        const int size, const char* bytes) {
  auto memory = get_memory(partition);
  for (int i = 0; i < size; i++) {
    memory[address + i] = bytes[i];
  }
}
