#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/PytorchMemoryModel.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"
#include "test/common/operations/MatrixOps.h"
#include "test/common/operations/Pooling.h"
#include "test/common/operations/ReduceOps.h"
#include "test/common/operations/VectorOps.h"
#include "test/compiler/proto/param.pb.h"

template <typename INPUT_T, typename ACCUMULATE_T>
ACCUMULATE_T *get_input(const codegen::Tensor &tensor, const INPUT_T *arg) {
  int input_size = get_size(tensor);
  bool double_precision = is_double_precision(tensor);

  ACCUMULATE_T *input_tensor = new ACCUMULATE_T[input_size];
  for (int i = 0; i < input_size; i++) {
    input_tensor[i] = read_tensor(arg, i, double_precision);
  }
  return input_tensor;
}

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T>
void run_pytorch_op(const codegen::AcceleratorParam param,
                    std::vector<INPUT_T *> args) {
  int arg_index = 0;
  ACCUMULATE_T *output_tensor;

  if (param.has_reduce_param()) {
    const auto &reduce_param = param.reduce_param();
    if (reduce_param.opcode() == "softmax") {
      const auto &input = reduce_param.input();
      const auto input_tensor =
          get_input<INPUT_T, ACCUMULATE_T>(input, args[arg_index++]);
      const auto input_shape = get_shape(input);
      output_tensor = softmax(input_tensor, input_shape);
      delete[] input_tensor;
    } else {
      std::cerr << "Unsupported reduce instruction: " << reduce_param.opcode()
                << std::endl;
      exit(1);
    }
  }

  if (param.has_pooling_param()) {
    const auto input = param.pooling_param().input();
    const auto input_tensor =
        get_input<INPUT_T, ACCUMULATE_T>(input, args[arg_index++]);
    output_tensor = pooling<INPUT_T, ACCUMULATE_T>(input_tensor, param);
    delete[] input_tensor;
  }

  if (param.has_matrix_param()) {
    const auto &matrix_param = param.matrix_param();
    const auto &inputs = matrix_param.input();
    int dim = 1;
    for (int i = 0; i < inputs.shape_size() - 1; i++) {
      dim *= inputs.shape(i);
    }

    if (dim == 1) {
      output_tensor =
          matrix_vector_multiply<INPUT_T, ACCUMULATE_T, INTERMEDIATE_T>(
              args[0], args[1], args[2], matrix_param);
    } else {
      output_tensor = gemm<INPUT_T, ACCUMULATE_T, INTERMEDIATE_T>(
          args[0], args[1], args[2], param);
    }
    arg_index = 3;
  } else if (param.vector_params_size() > 0) {
    // fetch the input of the first vector instruction
    const auto &vector_param = param.vector_params(0);
    output_tensor = get_input<INPUT_T, ACCUMULATE_T>(vector_param.input(),
                                                     args[arg_index++]);
  }

  for (const auto &vector_param : param.vector_params()) {
    if (activations.find(vector_param.opcode()) != activations.end()) {
      // TODO: Implement different activation functions
      int input_size = get_size(vector_param.input());
      for (int i = 0; i < input_size; i++) {
        relu(output_tensor[i]);
      }
    } else if (arithmetics.find(vector_param.opcode()) != arithmetics.end()) {
      ACCUMULATE_T *input_tensor = output_tensor;
      const auto input_shape = get_shape(vector_param.input());

      const auto &other = vector_param.other();
      ACCUMULATE_T *other_tensor =
          get_input<INPUT_T, ACCUMULATE_T>(other, args[arg_index++]);
      const auto other_shape = get_shape(other);

      output_tensor =
          perform_elwise_operation(input_tensor, input_shape, other_tensor,
                                   other_shape, vector_param.opcode());

      delete[] input_tensor;
      delete[] other_tensor;
    } else {
      std::cerr << "Unsupported vector instruction: " << vector_param.opcode()
                << std::endl;
      exit(1);
    }
  }

  int output_size = get_size(param.output());
  bool double_precision = is_double_precision(param.output());
  for (int i = 0; i < output_size; i++) {
    save_tensor(args.back(), i, output_tensor[i], double_precision);
  }

  delete[] output_tensor;
}

inline void run_pytorch_model(const codegen::AcceleratorParam &param,
                              std::vector<float *> args) {
  run_pytorch_op<float, float, float>(param, args);
}

inline void run_pytorch_model(const codegen::AcceleratorParam &param,
                              std::vector<INPUT_DATATYPE *> args) {
  run_pytorch_op<INPUT_DATATYPE, INTERMEDIATE_DTYPE, ACCUM_DATATYPE>(param,
                                                                     args);
}

#ifndef NO_UNIVERSAL
inline void run_pytorch_model(const codegen::AcceleratorParam &param,
                              std::vector<UniversalPosit *> args) {
  run_pytorch_op<UniversalPosit, UniversalPositAccum, UniversalPositAccum>(
      param, args);
}
#endif
