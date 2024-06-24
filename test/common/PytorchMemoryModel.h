#ifndef PYTORCH_MEMORY_MODEL_H
#define PYTORCH_MEMORY_MODEL_H

#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

// Function to check if the tensor requires double precision
// Can be updated in the future.
inline bool is_double_precision(const codegen::Tensor& tensor) {
  // FIXME: turn off to directly compare with the old implementation
  return false;
  return tensor.dtype().find("8") == std::string::npos;
}

inline float* read_tensor_from_file(const std::string& filename, int size,
                                    bool random_data) {
  float* value_ptr = new float[size];

  if (!random_data) {
    std::ifstream input_stream(filename, std::ios::binary);
    if (!input_stream.good())
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    input_stream.read(reinterpret_cast<char*>(value_ptr), size * sizeof(float));
    if (!input_stream)
      throw std::runtime_error(
          "Failed to read the expected amount of data from the file");
  } else {
    static std::default_random_engine engine;
    static std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (int i = 0; i < size; i++) {
      value_ptr[i] = distribution(engine);
    }
  }

  return value_ptr;
}

class PyTorchMemoryModel {
 public:
  PyTorchMemoryModel(bool);

  void load_inputs(const codegen::AcceleratorParam param, std::string data_dir,
                   bool random_data = false);
  void load_weights(const codegen::AcceleratorParam param, std::string data_dir,
                    bool random_data = false);
  void load_outputs(const codegen::AcceleratorParam param,
                    std::string data_dir);

 protected:
  void load_tensor(const codegen::Tensor& tensor, std::string data_dir,
                   bool is_conv2d = false, bool random_data = false);
  virtual void write_to_memory(const int address, const float value,
                               const int parttion, bool double_precision) = 0;

  // special addressing is sometimes needed for DUT memory (ex. replication)
  bool isDut;
};

inline PyTorchMemoryModel::PyTorchMemoryModel(bool isDut) : isDut(isDut) {}

inline void PyTorchMemoryModel::load_tensor(const codegen::Tensor& tensor,
                                            std::string data_dir,
                                            bool is_conv2d, bool random_data) {
  auto repeated_field = tensor.shape();
  std::vector<size_t> shape(repeated_field.begin(), repeated_field.end());

  int size = 1;
  for (int dim : shape) {
    size *= dim;
  }

  std::string filename = data_dir + "/" + tensor.node() + ".bin";
  auto array_ptr = read_tensor_from_file(filename, size, random_data);
  xt::xarray<float> array =
      xt::adapt(array_ptr, size, xt::no_ownership(), shape);

  // Accelerator expect the data to be layed out in a different order
  bool is_weight = tensor.node().find("param_constant") != std::string::npos;
  if (is_conv2d && shape.size() == 4) {
    array = xt::transpose(array, {2, 3, 1, 0});
  } else if (is_weight && shape.size() == 2) {
    array = xt::transpose(array, {1, 0});
  }

  auto memory = tensor.memory();
  int partition = memory.partition();
  int offset = memory.offset();
  bool double_precision = is_double_precision(tensor);
  int address_multiplier = double_precision ? 2 : 1;

  std::cerr << "Loading tensor file " << filename << std::endl;
  std::cerr << "array size: " << size << std::endl;
  std::cerr << "memory partition: " << memory.partition() << std::endl;
  std::cerr << "memory offset: " << memory.offset() << std::endl;
  std::cerr << "dtype: " << tensor.dtype() << std::endl;
  std::cerr << "double precision: " << double_precision << std::endl;

  int address = 0;
  for (auto it = array.begin(); it != array.end(); ++it) {
    write_to_memory(offset + address_multiplier * address, *it, partition,
                    true);
    address++;
  }

  delete[] array_ptr;
}

inline void PyTorchMemoryModel::load_inputs(
    const codegen::AcceleratorParam param, std::string data_dir,
    bool random_data) {
  if (param.has_matrix_param()) {
    const codegen::MatrixParam& matrix_param = param.matrix_param();
    load_tensor(matrix_param.input(), data_dir, true);
  }

  if (param.has_pooling_param()) {
    const codegen::PoolingParam& pooling_param = param.pooling_param();
    load_tensor(pooling_param.input(), data_dir);
  }

  if (param.has_reduce_param()) {
    const codegen::ReduceParam& reduce_param = param.reduce_param();
    load_tensor(reduce_param.input(), data_dir);
  }

  if (param.has_shape_param()) {
    const codegen::ShapeParam& shape_param = param.shape_param();
    load_tensor(shape_param.input(), data_dir);
  }

  for (auto& vector_param : param.vector_params()) {
    if (vector_param.has_other()) {
      auto other_tensor = vector_param.other();
      if (other_tensor.node().find("param_constant") == std::string::npos) {
        load_tensor(other_tensor, data_dir);
      }
    }
  }
}

inline void PyTorchMemoryModel::load_weights(
    const codegen::AcceleratorParam param, std::string data_dir,
    bool random_data) {
  if (param.has_matrix_param()) {
    const codegen::MatrixParam& matrix_param = param.matrix_param();
    load_tensor(matrix_param.weight(), data_dir, true);

    if (matrix_param.has_bias()) {
      load_tensor(matrix_param.bias(), data_dir, true);
    }
  }

  for (auto& vector_param : param.vector_params()) {
    if (vector_param.has_other()) {
      auto other_tensor = vector_param.other();
      if (other_tensor.node().find("param_constant") != std::string::npos) {
        load_tensor(other_tensor, data_dir);
      }
    }
  }
}

inline void PyTorchMemoryModel::load_outputs(
    const codegen::AcceleratorParam param, std::string data_dir) {
  bool is_conv2d =
      param.has_matrix_param() && param.matrix_param().opcode() == "conv2d";
  // HACK: hardcode to alwasy store output in the last partition
  codegen::Tensor output_tensor;
  output_tensor.CopyFrom(param.output());
  auto memory = output_tensor.mutable_memory();
  memory->set_partition(-1);
  memory->set_offset(0);
  load_tensor(output_tensor, data_dir, is_conv2d);
}

#endif  // PYTORCH_MEMORY_MODEL_H
