#include <sys/wait.h>
#include <unistd.h>

#include <fstream>
#include <locale>
#include <sstream>
#include <string>

#include "test/mobilebert/params.h"

string array_to_string(int array[], int size) {
  string returnstring = "{";
  for (int i = 0; i < size; i++) {
    returnstring += std::to_string(array[i]);
    if (i != size - 1) returnstring += ",";
  }
  returnstring += "}";
  return returnstring;
}

extern "C" int sc_main(int argc, char* argv[]) {
  ofstream myfile;
  myfile.open("verification_params.h");
  myfile << "#ifndef MOBILEBERT_PARAMS\n"
         << "#define MOBILEBERT_PARAMS\n\n";
  for (auto op : inferenceOperations) {
    SimplifiedParams params = inferenceParams.at(op.second);
    Files files = inferenceTestFiles.at(op.first);
    MemoryOffsets memOffsets = inferenceMemOffsets.at(op.first);

    params.INPUT_OFFSET = memOffsets.INPUT_OFFSET;
    params.WEIGHT_OFFSET =
        params.WEIGHT ? memOffsets.WEIGHT_OFFSET : memOffsets.WEIGHT_OFFSET;
    params.OUTPUT_OFFSET = memOffsets.OUTPUT_OFFSET;
    params.BIAS_OFFSET = memOffsets.BIAS_OFFSET;
    params.RESIDUAL_OFFSET = memOffsets.RESIDUAL_OFFSET;

    myfile << "const SimplifiedParams " << op.first << "_params = {\n"
           << "    " << params.INPUT_OFFSET << ", // INPUT_OFFSET\n"
           << "    " << params.WEIGHT_OFFSET << ", // WEIGHT_OFFSET\n"
           << "    " << params.OUTPUT_OFFSET << ", // OUTPUT_OFFSET\n"
           << "    " << params.TRANSPOSE << ", // TRANSPOSE\n"
           << "    "
           << "{" << array_to_string(params.loops[0], 6) << ", "
           << array_to_string(params.loops[1], 6) << "}, // LOOPS\n"
           << "    " << array_to_string(params.inputXLoopIndex, 2)
           << ", // INPUTX\n"
           << "    " << array_to_string(params.inputYLoopIndex, 2)
           << ", // INPUTY\n"
           << "    " << array_to_string(params.reductionLoopIndex, 2)
           << ", // REDUCTION\n"
           << "    " << array_to_string(params.weightLoopIndex, 2)
           << ", // WEIGHT\n"
           << "    " << params.fxIndex << ", // fxIndex\n"
           << "    " << params.fyIndex << ", // fyIndex\n"
           << "    " << array_to_string(params.weightReuseIndex, 2)
           << ", // weightReuseIndex\n"
           << "    " << params.STRIDE << ", // stride\n"
           << "    " << (params.REPLICATION ? "true" : "false")
           << ", // replication\n"
           << "    " << (params.RELU ? "true" : "false") << ", // ReLU\n"
           << "    " << (params.BIAS ? "true" : "false") << ", // bias\n"
           << "    " << params.BIAS_OFFSET << ", // BIAS_OFFSET\n"
           << "    " << (params.RESIDUAL ? "true" : "false")
           << ", // residual\n"
           << "    " << params.RESIDUAL_OFFSET << ", // RESIDUAL_OFFSET\n"
           << "    " << (params.MAXPOOL ? "true" : "false") << ", // maxpool\n "
           << "    " << (params.AVGPOOL ? "true" : "false") << ", // avgpool\n "
           << "    " << (params.SOFTMAX ? "true" : "false") << ", // softmax\n "
           << "    " << (params.FC ? "true" : "false") << ", // FC\n"
           << "    " << (params.NO_NORM ? "true" : "false") << ", // nonorm\n "
           << "    " << (params.WEIGHT ? "true" : "false") << ", // weight\n"
           << "    " << (params.ATTENTION_SCALING ? "true" : "false")
           << ", // ATTENTION_SCALING\n"
           << "    false, // STORE_IN_ACC\n"
           << "    false, // ACC_FROM_ACC\n"
           << "    " << (params.INPUT_TRANSPOSE ? "true" : "false")
           << ", // INPUT_TRANSPOSE\n"
           << "    " << (params.SPLIT_HEAD ? "true" : "false")
           << ", // SPLIT_HEAD\n"
           << "    " << (params.CONCAT_HEAD ? "true" : "false")
           << ", // CONCAT_HEAD\n"
           << "};\n\n";
  }

  for (auto op : inferenceOperations) {
    SimplifiedParams params = inferenceParams.at(op.second);
    myfile << "const MemoryMap " << op.first << "_mmap = {SRAM, "
           << (params.WEIGHT ? "RRAM" : "SRAM") << ", RRAM, SRAM, SRAM};\n";
  }
  myfile.close();
  return 0;
}