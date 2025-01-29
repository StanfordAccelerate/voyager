#include "test/common/DataLoader.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

DataLoader::DataLoader(ArrayMemory* memory_interface, bool is_dut)
    : memory_interface(memory_interface), is_dut(is_dut) {}

void DataLoader::load_tensor(const codegen::Tensor& tensor,
                             std::string data_dir, bool transpose,
                             bool replication) {
  auto repeated_field = tensor.shape();
  std::vector<size_t> shape(repeated_field.begin(), repeated_field.end());

  int size = 1;
  for (int dim : shape) {
    size *= dim;
  }

  // if size is 1, then it is a scalar, so it should not be
  // written to memory
  if (size == 1) {
    return;
  }

  std::string filename = data_dir + "/" + tensor.node() + ".bin";
  auto array_ptr = read_tensor_from_file(filename, size);
  auto array = xt::adapt(array_ptr, size, xt::no_ownership(), shape);

  // Accelerator expect the data to be layed out in a different order
  if (transpose) {
    if (shape.size() == 4) {
      array = xt::transpose(array, {2, 3, 1, 0});
    } else if (shape.size() == 2) {
      array = xt::transpose(array, {1, 0});
    }
  }

  auto memory = tensor.memory();
  int partition = memory.partition();
  long long offset = memory.offset();

  std::cerr << "Loading tensor file " << filename << std::endl;
  std::cerr << "Datatype: " << tensor.dtype() << std::endl;
  std::cerr << "Offset: " << offset << std::endl;
  std::cerr << "transpose: " << transpose << std::endl;

  // number of elements packed into a single word for replication
  const int packing_factor = IC_DIMENSION / 4 * 3;
  if (replication) {
    std::cerr << "packing factor: " << packing_factor << std::endl;
  }

  int address = 0;
  for (auto it = array.begin(); it != array.end(); ++it) {
    if (tensor.dtype() == "int8") {
      memory_interface->write_data_to_memory<DataTypes::int8>(offset, partition,
                                                              address, *it);
    } else if (tensor.dtype() == "bfloat16") {
      memory_interface->write_data_to_memory<DataTypes::bfloat16>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "int24") {
      memory_interface->write_data_to_memory<DataTypes::int24>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "int32") {
      memory_interface->write_data_to_memory<DataTypes::int32>(
          offset, partition, address, *it);
    } else if (tensor.dtype() == "e8m0") {
      memory_interface->write_data_to_memory<DataTypes::e8m0>(offset, partition,
                                                              address, *it);
    } else if (tensor.dtype() == "fp8_e5m3") {
      memory_interface->write_data_to_memory<DataTypes::fp8_e5m3>(
          offset, partition, address, *it);
    } else {
      // if unspecified, we will assume it's INPUT_DATATYPE
      memory_interface->write_data_to_memory<INPUT_DATATYPE>(offset, partition,
                                                             address, *it);
    }

    address++;
    if (replication && address % IC_DIMENSION == packing_factor) {
      address += IC_DIMENSION - packing_factor;
    }
  }

  delete[] array_ptr;
}

void DataLoader::load_inputs(const codegen::Operator param,
                             std::string data_dir, bool random_data) {
  // convolution layer inputs/outputs need to be permuted. If the matrix
  // operation is a convolution, the following fused vector operations will
  // need to be permuted as well. This logic should be refined in the future.
  std::string output_node = "";
  if (param.has_matrix_op()) {
    const auto& matrix_op = param.matrix_op();
    const auto& input = matrix_op.has_mx_input() ? matrix_op.mx_input().input()
                                                 : matrix_op.input();

    bool is_conv2d =
        matrix_op.opcode() == "conv2d" || matrix_op.opcode() == "conv2d_mx";

    bool replication = input.shape(1) == 3 && is_dut;

    load_tensor(input, data_dir, is_conv2d, replication);

    if (matrix_op.has_mx_input()) {
      load_tensor(matrix_op.mx_input().scale(), data_dir, is_conv2d);
    }

    if (matrix_op.opcode() == "matmul" || matrix_op.opcode() == "matmul_mx") {
      const auto& weight = matrix_op.has_mx_weight()
                               ? matrix_op.mx_weight().input()
                               : matrix_op.weight();
      load_tensor(weight, data_dir);

      if (matrix_op.has_mx_weight()) {
        load_tensor(matrix_op.mx_weight().scale(), data_dir);
      }
    }
    output_node = matrix_op.name();
  } else if (param.has_pooling_op()) {
    const auto& pooling_op = param.pooling_op();
    load_tensor(pooling_op.input(), data_dir, true);
  } else if (param.has_reduce_op()) {
    const auto& reduce_op = param.reduce_op();
    load_tensor(reduce_op.input(), data_dir);
  } else if (param.has_reshape_op()) {
    const auto& reshape_op = param.reshape_op();
    load_tensor(reshape_op.input(), data_dir);
  } else if (param.has_slicing_op()) {
    const auto& slicing_op = param.slicing_op();
    load_tensor(slicing_op.input(), data_dir);
  } else if (param.vector_ops_size() > 0) {
    const auto vector_op = param.vector_ops(0);
    load_tensor(vector_op.input(), data_dir);
  }

  for (const auto& vector_op : param.vector_ops()) {
    if (vector_op.has_other()) {
      // Load the other tensor if it is not the output of last operation and
      // it is a constant tensor. Might fail if input or other tensor is a nop.
      const auto input = vector_op.input();
      const auto other = vector_op.other();
      const auto tensor_to_load = other.node() == output_node ? input : other;
      if (tensor_to_load.node().find("constant") == std::string::npos) {
        load_tensor(tensor_to_load, data_dir);
      }
    }
    output_node = vector_op.name();
  }
}

void DataLoader::load_weights(const codegen::Operator param,
                              std::string data_dir, bool random_data) {
  if (param.has_matrix_op() && param.matrix_op().opcode() != "matmul" &&
      param.matrix_op().opcode() != "matmul_mx") {
    const auto matrix_op = param.matrix_op();
    const auto input = matrix_op.has_mx_input() ? matrix_op.mx_input().input()
                                                : matrix_op.input();
    const auto weight = matrix_op.has_mx_weight()
                            ? matrix_op.mx_weight().input()
                            : matrix_op.weight();

    // Transpose linear weights except for matrix vector multiply
    int dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      dim *= input.shape(i);
    }
    bool transpose = dim > 1;
    load_tensor(weight, data_dir, transpose);

    if (matrix_op.has_mx_weight()) {
      load_tensor(matrix_op.mx_weight().scale(), data_dir, transpose);
    }

    if (matrix_op.has_bias()) {
      // bias is hardcoded to double precision right now
      load_tensor(matrix_op.bias(), data_dir);
    }
  }

  if (param.has_nop()) {
    for (const auto& input : param.nop().inputs()) {
      if (input.node().find("constant") != std::string::npos ||
          input.node().find("arg") != std::string::npos) {
        load_tensor(input, data_dir);
      }
    }
  }

  for (const auto& vector_op : param.vector_ops()) {
    if (vector_op.has_other()) {
      // Check both input and other tensors to see if they are parameters.
      const auto input = vector_op.input();
      if (input.node().find("constant") != std::string::npos ||
          input.node().find("arg") != std::string::npos) {
        load_tensor(input, data_dir);
      }
      const auto other = vector_op.other();
      if (other.node().find("constant") != std::string::npos ||
          other.node().find("arg") != std::string::npos) {
        load_tensor(other, data_dir);
      }
    }
  }
}

void DataLoader::load_outputs(const codegen::Operator param,
                              std::string data_dir) {
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  // always store output in the last memory partition with 0 offset
  memory->set_partition(-1);
  memory->set_offset(0);
  bool transpose = param.matrix_op().opcode() == "conv2d" ||
                   param.matrix_op().opcode() == "conv2d_mx" ||
                   param.has_pooling_op();
  load_tensor(output_tensor, data_dir, transpose);
}

float* DataLoader::read_tensor_from_file(const std::string& filename,
                                         int size) {
  float* value_ptr = new float[size];
  std::ifstream input_stream(filename, std::ios::binary);
  if (!input_stream.good()) {
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  }
  input_stream.read(reinterpret_cast<char*>(value_ptr), size * sizeof(float));
  if (!input_stream) {
    throw std::runtime_error(
        "Failed to read the expected amount of data from the file");
  }
  return value_ptr;
}
