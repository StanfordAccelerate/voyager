#include "test/common/ArrayMemory.h"

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include <fstream>

#include "src/ArchitectureParams.h"

ArrayMemory::ArrayMemory(std::vector<long long> sizes) : MemoryInterface() {
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

std::vector<std::any> ArrayMemory::get_args(const codegen::Operator& param) {
  std::vector<std::any> args;
  std::string output_node = "";
  if (param.has_matrix_op()) {
    const auto& matrix_op = param.matrix_op();

    if (matrix_op.has_mx_input()) {
      args.push_back(get_tensor(matrix_op.mx_input().input()));
      args.push_back(get_tensor(matrix_op.mx_input().scale()));
    } else {
      args.push_back(get_tensor(matrix_op.input()));
    }

    if (matrix_op.has_mx_weight()) {
      args.push_back(get_tensor(matrix_op.mx_weight().input()));
      args.push_back(get_tensor(matrix_op.mx_weight().scale()));
    } else {
      args.push_back(get_tensor(matrix_op.weight()));
    }

    if (matrix_op.has_bias()) {
      args.push_back(get_tensor(matrix_op.bias()));
    } else {
      args.push_back(nullptr);
    }
    output_node = matrix_op.name();
  } else if (param.has_pooling_op()) {
    const auto& pooling_op = param.pooling_op();
    args.push_back(get_tensor(pooling_op.input()));
  } else if (param.has_reduce_op()) {
    const auto& reduce_op = param.reduce_op();
    args.push_back(get_tensor(reduce_op.input()));
  } else if (param.has_reshape_op()) {
    const auto& reshape_op = param.reshape_op();
    args.push_back(get_tensor(reshape_op.input()));
  } else if (param.has_slicing_op()) {
    const auto& slicing_op = param.slicing_op();
    args.push_back(get_tensor(slicing_op.input()));
  } else if (param.vector_ops_size() > 0) {
    const auto vector_op = param.vector_ops(0);
    args.push_back(get_tensor(vector_op.input()));
  }

  for (auto& vector_op : param.vector_ops()) {
    if (vector_op.has_other()) {
      // Check whether input or other is the output of the last operation.
      const auto input = vector_op.input();
      const auto other = vector_op.other();
      const auto tensor_to_load = other.node() == output_node ? input : other;
      args.push_back(get_tensor(tensor_to_load));
    }
    output_node = vector_op.name();
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
std::any ArrayMemory::get_reference_output(const codegen::Operator& param) {
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  memory->set_partition(-1);
  memory->set_offset(0);
  return get_tensor(output_tensor);
}

std::any ArrayMemory::get_output(const codegen::Operator& param) {
  codegen::Tensor output_tensor = param.output();
  return get_tensor(output_tensor);
}

template <typename T>
void ArrayMemory::read_tensor_from_memory(const long long address,
                                          const int partition, const int size,
                                          T* tensor) {
  char* memory = get_memory(partition) + address;
  ac_int<(T::width / 8 + 2) * 8> bits;

  for (int i = 0; i < size; i++) {
    // Data may be unaligned and span multiple bytes. We calculate the start
    // and end byte indices and the offset within the first byte. We then read
    // the bytes into a temporary ac_int and shift it to the correct position.
    int start = i * T::width / 8;
    int end = (i + 1) * T::width / 8;
    int offset = (i * T::width) % 8;

    for (int j = start; j <= end; j++) {
      bits.set_slc((j - start) * 8, static_cast<ac_int<8>>(memory[j]));
    }

    bits = bits >> offset;
    tensor[i].set_bits(bits.template slc<T::width>(0));
  }
}

std::any ArrayMemory::get_tensor(const codegen::Tensor& tensor) {
  int partition = tensor.memory().partition();

  int size = 1;
  for (const auto& dim : tensor.shape()) size *= dim;

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

void ArrayMemory::write_bytes_to_memory(const long long address,
                                        const int partition, const int size,
                                        const char* bytes, const char* masks) {
  auto memory = get_memory(partition);
  for (int i = 0; i < size; i++) {
    char new_data = bytes[i] & masks[i];
    char orig_data = memory[address + i] & ~masks[i];
    memory[address + i] = new_data | orig_data;
  }
}
