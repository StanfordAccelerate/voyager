#include <fstream>
#include <sstream>
#include <string>

#include "test/common/VerificationTypes.h"

std::string array_to_string(int array[], int size) {
  std::string returnstring = "{";
  for (int i = 0; i < size; i++) {
    returnstring += std::to_string(array[i]);
    if (i != size - 1) returnstring += ",";
  }
  returnstring += "}";
  return returnstring;
}

std::string formatOperation(SimplifiedParams params, std::string operation) {
  std::stringstream ss;
  ss << "const SimplifiedParams " << operation << "_params = {\n"
     << params.INPUT_OFFSET << ", // INPUT_OFFSET\n"
     << params.WEIGHT_OFFSET << ", // WEIGHT_OFFSET\n"
     << params.OUTPUT_OFFSET << ", // OUTPUT_OFFSET\n"
     << params.WEIGHT_TRANSPOSE << ", // WEIGHT_TRANSPOSE\n"

     << "{" << array_to_string(params.loops[0], 6) << ", "
     << array_to_string(params.loops[1], 6) << "}, // LOOPS\n"
     << array_to_string(params.inputXLoopIndex, 2) << ", // INPUTX\n"
     << array_to_string(params.inputYLoopIndex, 2) << ", // INPUTY\n"
     << array_to_string(params.reductionLoopIndex, 2) << ", // REDUCTION\n"
     << array_to_string(params.weightLoopIndex, 2) << ", // WEIGHT\n"
     << params.fxIndex << ", // fxIndex\n"
     << params.fyIndex << ", // fyIndex\n"
     << array_to_string(params.weightReuseIndex, 2) << ", // weightReuseIndex\n"
     << params.STRIDE << ", // stride\n"
     << (params.REPLICATION ? "true" : "false") << ", // replication\n"

     << (params.RELU ? "true" : "false") << ", // ReLU\n"
     << (params.BIAS ? "true" : "false") << ", // bias\n"
     << params.BIAS_OFFSET << ", // BIAS_OFFSET\n"
     << (params.RESIDUAL ? "true" : "false") << ", // residual\n"
     << params.RESIDUAL_OFFSET << ", // RESIDUAL_OFFSET\n"

     << (params.MAXPOOL ? "true" : "false") << ", // MAXPOOL\n"
     << (params.AVGPOOL ? "true" : "false") << ", // AVGPOOL\n"

     << (params.WEIGHT ? "true" : "false") << ", // WEIGHT\n"
     << "false, // STORE_IN_ACC\n"
     << "false, // ACC_FROM_ACC\n"

     << (params.SOFTMAX ? "true" : "false") << ", // SOFTMAX\n"
     << (params.ATTENTION_MASK ? "true" : "false") << ", // ATTENTION_MASK\n"
     << (params.ATTENTION_SCALING ? "true" : "false")
     << ", // ATTENTION_SCALING\n"
     << (params.FC ? "true" : "false") << ", // FC\n"
     << (params.NO_NORM ? "true" : "false") << ", // NO_NORM\n"

     << (params.SOFTMAX_GRAD ? "true" : "false") << ", // SOFTMAX_GRAD\n"
     << (params.FC_GRAD ? "true" : "false") << ", // FC_GRAD\n"
     << (params.NO_NORM_GRAD ? "true" : "false") << ", // NO_NORM_GRAD\n"
     << (params.RELU_GRAD ? "true" : "false") << ", // RELU_GRAD\n"
     << (params.BIAS_GRAD ? "true" : "false") << ", // BIAS_GRAD\n"
     << (params.CROSS_ENTROPY_GRAD ? "true" : "false")
     << ", // CROSS_ENTROPY_GRAD\n"
     << (params.MSE_GRAD ? "true" : "false") << ", // MSE_GRAD\n"
     << (params.BCE_WITH_LOGITS_GRAD ? "true" : "false")
     << ", // BCE_WITH_LOGITS_GRAD\n"

     << (params.INPUT_TRANSPOSE ? "true" : "false") << ", // INPUT_TRANSPOSE\n"
     << (params.CONCAT_INPUT ? "true" : "false") << ", // CONCAT_INPUT\n"
     << (params.CONCAT_WEIGHT ? "true" : "false") << ", // CONCAT_WEIGHT\n"
     << (params.SPLIT_OUTPUT ? "true" : "false") << ", // SPLIT_OUTPUT\n"

     << (params.GRAD_CLIPPING ? "true" : "false") << ", // GRAD_CLIPPING\n"

     << (params.WEIGHT_SPLITTING ? "true" : "false")
     << ", // WEIGHT_SPLITTING\n"
     << params.WEIGHT_GRADIENT_OFFSET << ", // WEIGHT_GRADIENT_OFFSET\n"
     << params.BIAS_GRADIENT_OFFSET << ", // BIAS_GRADIENT_OFFSET\n"
     << params.learningRate << ", // learningRate\n"

     << (params.ACC_T_INPUT ? "true" : "false") << ", // ACC_T_INPUT\n"
     << (params.ACC_T_WEIGHT ? "true" : "false") << ", // ACC_T_WEIGHT\n"
     << (params.ACC_T_OUTPUT ? "true" : "false") << ", // ACC_T_OUTPUT\n"
     << "};\n";

  return ss.str();
}
