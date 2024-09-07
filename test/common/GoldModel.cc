#include "GoldModel.h"

#include <vector>

#include "test/common/VerificationTypes.h"
#include "test/common/operations/MatrixOps.h"
#include "test/common/operations/Pooling.h"
#include "test/common/operations/ReduceOps.h"
#include "test/common/operations/ReshapeOps.h"
#include "test/common/operations/VectorOps.h"

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T,
          typename VECTOR_T>
void run_operation(const codegen::AcceleratorParam param,
                   std::vector<std::any> args) {
  int arg_index = 0;
  std::any output_tensor;

  if (param.has_reduce_param()) {
    const auto &reduce_param = param.reduce_param();
    if (reduce_param.opcode() == "softmax") {
      const auto &input = reduce_param.input();
      const auto input_shape = get_shape(input);
      output_tensor = softmax<VECTOR_T>(args[arg_index++], input_shape);
    } else {
      std::cerr << "Unsupported reduce instruction: " << reduce_param.opcode()
                << std::endl;
      exit(1);
    }
  }

  if (param.has_pooling_param()) {
    const auto input = param.pooling_param().input();
    output_tensor = pooling<VECTOR_T>(args[arg_index++], param);
  }

  if (param.has_reshape_param()) {
    const auto &reshape_param = param.reshape_param();
    const auto &input = reshape_param.input();
    output_tensor = permute<INPUT_T>(args[arg_index++], reshape_param);
  }

  if (param.has_matrix_param()) {
    const auto &matrix_param = param.matrix_param();

    // Permute input tensor
    const auto &input = matrix_param.input();
    std::any input_tensor = args[0];
    if (input.has_permutation()) {
      input_tensor = permute<INPUT_T>(input_tensor, input.permutation());
    }

    // Permute weight tensor
    const auto &weight = matrix_param.weight();
    std::any weight_tensor = args[1];
    if (weight.has_permutation()) {
      weight_tensor = permute<INPUT_T>(weight_tensor, weight.permutation());
    }

    int dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      dim *= input.shape(i);
    }

    if (dim == 1) {
      output_tensor =
          matrix_vector_multiply<INPUT_T, ACCUMULATE_T, INTERMEDIATE_T>(
              input_tensor, weight_tensor, args[2], matrix_param);
    } else {
      output_tensor = gemm<INPUT_T, ACCUMULATE_T, INTERMEDIATE_T>(
          input_tensor, weight_tensor, args[2], param);
    }
    arg_index = 3;
  } else if (param.vector_params_size() > 0) {
    // fetch the input of the first vector instruction
    const auto &vector_param = param.vector_params(0);
    output_tensor = std::any_cast<INPUT_T *>(args[arg_index++]);
  }

  for (const auto &vector_param : param.vector_params()) {
    if (activations.find(vector_param.opcode()) != activations.end()) {
      ACCUMULATE_T *tensor = std::any_cast<ACCUMULATE_T *>(output_tensor);
      // TODO: Implement different activation functions
      int input_size = get_size(vector_param.input());
      for (int i = 0; i < input_size; i++) {
        relu(tensor[i]);
      }
    } else if (arithmetics.find(vector_param.opcode()) != arithmetics.end()) {
      ACCUMULATE_T *input_tensor = std::any_cast<ACCUMULATE_T *>(output_tensor);
      const auto input_shape = get_shape(vector_param.input());

      const auto &other = vector_param.other();
      ACCUMULATE_T *other_tensor =
          std::any_cast<ACCUMULATE_T *>(args[arg_index]);
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
  if (param.output().has_permutation()) {
    output_tensor =
        permute<INPUT_T>(std::any_cast<ACCUMULATE_T *>(output_tensor),
                         param.output().permutation());
  }

  bool double_precision = is_double_precision(param.output());
  // save the output tensor
  char *output_bytes = std::any_cast<char *>(args.back());
  INPUT_T *output_tensor_casted = std::any_cast<INPUT_T *>(output_tensor);
  for (int i = 0; i < output_size; i++) {
    ac_int<INPUT_T::width> bits =
        static_cast<INPUT_T>(output_tensor_casted[i]).bits_rep();
    for (int j = 0; j < INPUT_T::width / 8; j++) {
      output_bytes[i * INPUT_T::width / 8 + j] = bits.template slc<8>(j * 8);
    }

    // delete[] output_tensor_casted;
  }
}

// void run_gold_model(const codegen::AcceleratorParam &param,
//                     std::vector<float *> args) {
//   run_operation<float, float, float, float>(param, args);
// }

void run_gold_model(const codegen::AcceleratorParam &param,
                    std::vector<std::any> args) {
  run_operation<INPUT_DATATYPE, INTERMEDIATE_DTYPE, ACCUM_DATATYPE,
                VECTOR_DATATYPE>(param, args);
}
