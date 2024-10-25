#include "test/common/DataLoader.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

DataLoader::DataLoader(MemoryInterface* memory_interface, bool is_dut)
    : memory_interface(memory_interface), is_dut(is_dut) {}

void DataLoader::load_tensor(const codegen::Tensor& tensor,
                             std::string data_dir, bool transpose,
                             bool replication, bool double_precision_ow,
                             bool is_output, bool random_data) {
  auto repeated_field = tensor.shape();
  std::vector<size_t> shape(repeated_field.begin(), repeated_field.end());
  int size = 1;
  for (int dim : shape) size *= dim;

  std::string filename = data_dir + "/" + tensor.node() + ".bin";
  auto array_ptr = read_tensor_from_file(filename, size, random_data);
  auto array = xt::adapt(array_ptr, size, xt::no_ownership(), shape);

  // Accelerator expect the data to be layed out in a different order
  if (shape.size() == 4) {
    array = xt::transpose(array, {2, 3, 1, 0});
  } else if (transpose && shape.size() == 2) {
    array = xt::transpose(array, {1, 0});
  }

  auto memory = tensor.memory();
  int partition = memory.partition();
  int offset = memory.offset();
  bool double_precision = double_precision_ow || is_double_precision(tensor);
  int address_multiplier = double_precision ? 2 : 1;

  // if size is 1, then it is a scalar, so it should not be
  // written to memory
  if (size == 1) {
    return;
  }

  // std::cerr << "Loading tensor file " << filename << std::endl;
  // std::cerr << "Datatype: " << tensor.dtype() << std::endl;
  // std::cerr << "Offset: " << offset << std::endl;

  // number of elements packed into a single word for replication
  const int packing_factor = IC_DIMENSION / 4 * 3;
  if (replication) {
    std::cerr << "packing factor: " << packing_factor << std::endl;
  }

  int address = 0;
  for (auto it = array.begin(); it != array.end(); ++it) {
    if (tensor.dtype() == "int8") {
      memory_interface->write_to_memory<DataTypes::int8>(offset, address, *it,
                                                         partition);
    } else if (tensor.dtype() == "bfloat16") {
      memory_interface->write_to_memory<DataTypes::bfloat16>(offset, address,
                                                             *it, partition);
    } else if (tensor.dtype() == "int24") {
      memory_interface->write_to_memory<DataTypes::int24>(offset, address, *it,
                                                          partition);
    } else if (tensor.dtype() == "int32") {
      memory_interface->write_to_memory<DataTypes::int32>(offset, address, *it,
                                                          partition);
    } else if (tensor.dtype() == "e8m0") {
      memory_interface->write_to_memory<DataTypes::e8m0>(offset, address, *it,
                                                         partition);
    } else {
      // if unspecified, we will assume it's INPUT_DATATYPE
      memory_interface->write_to_memory<INPUT_DATATYPE>(offset, address, *it,
                                                        partition);
    }

    address++;
    if (replication && address % IC_DIMENSION == packing_factor) {
      address += IC_DIMENSION - packing_factor;
    }
  }

  delete[] array_ptr;
}

void DataLoader::load_inputs(const codegen::AcceleratorParam param,
                             std::string data_dir, bool random_data) {
  // convolution layer inputs/outputs need to be permuted. If the matrix
  // operation is a convolution, the following fused vector operations will
  // need to be permuted as well. This logic should be refined in the future.
  bool is_conv2d = param.matrix_param().opcode() == "conv2d" ||
                   param.matrix_param().opcode() == "conv2d_mx";
  std::string output_node = "";
  if (param.has_matrix_param()) {
    const codegen::MatrixParam& matrix_param = param.matrix_param();

    auto input = matrix_param.has_mx_input() ? matrix_param.mx_input().input()
                                             : matrix_param.input();
    bool replication = input.shape(1) == 3 && is_dut;
    load_tensor(input, data_dir, is_conv2d, replication);

    if (matrix_param.has_mx_input()) {
      load_tensor(matrix_param.mx_input().scale(), data_dir, is_conv2d);
    }

    if (matrix_param.opcode() == "matmul") {
      auto weight = matrix_param.has_mx_weight()
                        ? matrix_param.mx_weight().input()
                        : matrix_param.weight();
      load_tensor(weight, data_dir);

      if (matrix_param.has_mx_weight()) {
        load_tensor(matrix_param.mx_weight().scale(), data_dir);
      }
    }
    output_node = matrix_param.name();
  } else if (param.has_pooling_param()) {
    const codegen::PoolingParam& pooling_param = param.pooling_param();
    load_tensor(pooling_param.input(), data_dir, true);
  } else if (param.has_reduce_param()) {
    const codegen::ReduceParam& reduce_param = param.reduce_param();
    load_tensor(reduce_param.input(), data_dir);
  } else if (param.has_reshape_param()) {
    const codegen::ReshapeParam& reshape_param = param.reshape_param();
    load_tensor(reshape_param.input(), data_dir);
  } else if (param.vector_params_size() > 0) {
    const auto vector_param = param.vector_params(0);
    load_tensor(vector_param.input(), data_dir);
  }

  for (const auto& vector_param : param.vector_params()) {
    if (vector_param.has_other()) {
      // Load the other tensor if it is not the output of last operation and
      // it is a constant tensor. Might fail if input or other tensor is a nop.
      const auto input = vector_param.input();
      const auto other = vector_param.other();
      const auto tensor_to_load = other.node() == output_node ? input : other;
      if (tensor_to_load.node().find("constant") == std::string::npos) {
        load_tensor(tensor_to_load, data_dir, is_conv2d);
      }
    }
    output_node = vector_param.name();
  }
}

void DataLoader::load_weights(const codegen::AcceleratorParam param,
                              std::string data_dir, bool random_data) {
  if (param.has_matrix_param() && param.matrix_param().opcode() != "matmul") {
    const auto matrix_param = param.matrix_param();
    const auto input = matrix_param.has_mx_input()
                           ? matrix_param.mx_input().input()
                           : matrix_param.input();
    const auto weight = matrix_param.has_mx_weight()
                            ? matrix_param.mx_weight().input()
                            : matrix_param.weight();

    // Transpose linear weights except for matrix vector multiply
    int dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      dim *= input.shape(i);
    }
    bool transpose = dim > 1;
    load_tensor(weight, data_dir, transpose);

    if (matrix_param.has_mx_weight()) {
      load_tensor(matrix_param.mx_weight().scale(), data_dir, transpose);
    }

    if (matrix_param.has_bias()) {
      // bias is hardcoded to double precision right now
      load_tensor(matrix_param.bias(), data_dir, false, false, true);
    }
  }

  for (const auto& vector_param : param.vector_params()) {
    if (vector_param.has_other()) {
      // Check both input and other tensors to see if they are parameters.
      const auto input = vector_param.input();
      if (input.node().find("constant") != std::string::npos) {
        load_tensor(input, data_dir);
      }
      const auto other = vector_param.other();
      if (other.node().find("constant") != std::string::npos) {
        load_tensor(other, data_dir);
      }
    }
  }
}

void DataLoader::load_outputs(const codegen::AcceleratorParam param,
                              std::string data_dir) {
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  // always store output in the last memory partition with 0 offset
  memory->set_partition(-1);
  memory->set_offset(0);
  bool transpose = param.matrix_param().opcode() == "conv2d" ||
                   param.matrix_param().opcode() == "conv2d_mx" ||
                   param.has_pooling_param();
  load_tensor(output_tensor, data_dir, transpose, false, false, true);
}

bool DataLoader::is_double_precision(const codegen::Tensor& tensor) {
  // FIXME: replace with proper check
  return false;
}

float* DataLoader::read_tensor_from_file(const std::string& filename, int size,
                                         bool random_data) {
  float* value_ptr = new float[size];

  if (!random_data) {
    std::ifstream input_stream(filename, std::ios::binary);
    if (!input_stream.good()) {
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    }
    input_stream.read(reinterpret_cast<char*>(value_ptr), size * sizeof(float));
    if (!input_stream) {
      throw std::runtime_error(
          "Failed to read the expected amount of data from the file");
    }
  } else {
    static std::default_random_engine engine;
    static std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (int i = 0; i < size; i++) {
      value_ptr[i] = distribution(engine);
    }
  }

  return value_ptr;
}
