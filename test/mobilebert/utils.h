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

void formatBoolField(std::string name, bool value, std::stringstream &ss) {
  ss << "." << name << " = " << (value ? "true" : "false") << ",\n";
}

std::string formatOperation2(SimplifiedParams params, std::string operation) {
  std::stringstream ss;
  ss << "const SimplifiedParams " << operation << "_params = {\n"
     << ".INPUT_OFFSET = " << params.INPUT_OFFSET << ",\n"
     << ".WEIGHT_OFFSET = " << params.WEIGHT_OFFSET << ",\n"
     << ".OUTPUT_OFFSET = " << params.OUTPUT_OFFSET << ",\n";

  formatBoolField("WEIGHT_TRANSPOSE", params.WEIGHT_TRANSPOSE, ss);

  ss << ".loops = {" << array_to_string(params.loops[0], 6) << ", "
     << array_to_string(params.loops[1], 6) << "},\n"
     << ".inputXLoopIndex = " << array_to_string(params.inputXLoopIndex, 2)
     << ",\n"
     << ".inputYLoopIndex = " << array_to_string(params.inputYLoopIndex, 2)
     << ",\n"
     << ".reductionLoopIndex = "
     << array_to_string(params.reductionLoopIndex, 2) << ",\n"
     << ".weightLoopIndex = " << array_to_string(params.weightLoopIndex, 2)
     << ",\n"
     << ".fxIndex = " << params.fxIndex << ",\n"
     << ".fyIndex = " << params.fyIndex << ",\n"
     << ".weightReuseIndex = " << array_to_string(params.weightReuseIndex, 2)
     << ",\n"
     << ".STRIDE = " << params.STRIDE << ",\n";

  formatBoolField("REPLICATION", params.REPLICATION, ss);

  formatBoolField("RELU", params.RELU, ss);
  formatBoolField("BIAS", params.BIAS, ss);
  ss << ".BIAS_OFFSET = " << params.BIAS_OFFSET << ",\n";
  formatBoolField("RESIDUAL", params.RESIDUAL, ss);
  ss << ".RESIDUAL_OFFSET = " << params.RESIDUAL_OFFSET << ",\n";

  formatBoolField("MAXPOOL", params.MAXPOOL, ss);
  formatBoolField("AVGPOOL", params.AVGPOOL, ss);
  formatBoolField("WEIGHT", params.WEIGHT, ss);

  formatBoolField("STORE_IN_ACC", params.STORE_IN_ACC, ss);
  formatBoolField("ACC_FROM_ACC", params.ACC_FROM_ACC, ss);

  formatBoolField("SOFTMAX", params.SOFTMAX, ss);
  formatBoolField("ATTENTION_MASK", params.ATTENTION_MASK, ss);
  formatBoolField("ATTENTION_SCALING", params.ATTENTION_SCALING, ss);
  formatBoolField("FC", params.FC, ss);
  formatBoolField("NO_NORM", params.NO_NORM, ss);

  formatBoolField("SOFTMAX_GRAD", params.SOFTMAX_GRAD, ss);
  formatBoolField("FC_GRAD", params.FC_GRAD, ss);
  formatBoolField("NO_NORM_GRAD", params.NO_NORM_GRAD, ss);
  formatBoolField("RELU_GRAD", params.RELU_GRAD, ss);
  formatBoolField("BIAS_GRAD", params.BIAS_GRAD, ss);
  formatBoolField("CROSS_ENTROPY_GRAD", params.CROSS_ENTROPY_GRAD, ss);
  formatBoolField("MSE_GRAD", params.MSE_GRAD, ss);
  formatBoolField("BCE_WITH_LOGITS_GRAD", params.BCE_WITH_LOGITS_GRAD, ss);

  formatBoolField("INPUT_TRANSPOSE", params.INPUT_TRANSPOSE, ss);
  formatBoolField("CONCAT_INPUT", params.CONCAT_INPUT, ss);
  formatBoolField("CONCAT_WEIGHT", params.CONCAT_WEIGHT, ss);
  formatBoolField("SPLIT_OUTPUT", params.SPLIT_OUTPUT, ss);

  formatBoolField("GRAD_CLIPPING", params.GRAD_CLIPPING, ss);
  ss << "};" << std::endl;

  return ss.str();
}
