#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "test/common/DataLoader.h"
#include "test/common/GoldModel.h"
// #include "test/common/Harness.h"
#include "test/common/UniversalPosit.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/resnet/params.h"
#include "test/simple/params.h"

#define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
#define RRAM_MEMORY_SIZE (12 * 1024 * 1024)

void validateMapping(SimplifiedParams params);
void run_sequence(const std::string& group,
                  const std::vector<std::string>& tests,
                  const std::unordered_set<std::string>& comparisons);
std::vector<std::string> parse_csv(const std::string& csv);

// NOTE: Binary data files are always supplied in [Y][X][C][K] ordering, except
// FC which is in [K][C]

std::vector<std::string> parse_csv(std::string& csv) {
  // Remove spaces
  std::remove(csv.begin(), csv.end(), ' ');

  std::stringstream ss(csv);
  std::string token;
  std::vector<std::string> result;

  // Tokenize
  while (std::getline(ss, token, ',')) {
    result.push_back(token);
  }

  return result;
}

void validateMapping(SimplifiedParams params) {
  int x0 = params.loops[1][params.inputXLoopIndex[1]];
  int y0 = params.loops[1][params.inputYLoopIndex[1]];
  int c0 = params.loops[1][params.reductionLoopIndex[1]];
  int k0 = params.loops[1][params.weightLoopIndex[1]];
  int fx = params.loops[1][params.fxIndex];
  int fy = params.loops[1][params.fyIndex];
  int stride = params.STRIDE;

  if (params.FC || params.SOFTMAX ||
      params.NO_NORM) {  // don't check for vector ops
    return;
  }

  // Input buffer
  int input_buffer_tile_size = (x0 * stride + fx - 1) * (y0 * stride + fy - 1);
  if (params.REPLICATION) {
    // don't check temporarily
    input_buffer_tile_size = 1;
  }
  if (input_buffer_tile_size > INPUT_BUFFER_SIZE) {
    std::cout << "[ERROR] Input buffer tile size violation." << std::endl;
    std::terminate();
  }

  // Weight buffer
  if (fx * fy * k0 > WEIGHT_BUFFER_SIZE) {
    std::cout << "[ERROR] Weight buffer tile size violation." << std::endl;
    std::terminate();
  }

  if (x0 * y0 * k0 > ACCUMULATION_BUFFER_SIZE) {
    std::cout << "[ERROR] Accumulation buffer tile size violation."
              << std::endl;
    std::terminate();
  }
}

void run_sequence(const std::string& group,
                  const std::vector<std::string>& tests,
                  const std::vector<std::string>& comparisons) {
  // Set data parameters
  bool use_data_file = true;
  std::string data_dir;
  std::map<std::string, MemoryMap>* mem_map;
  std::map<std::string, SimplifiedParams>* param_map;
  std::map<std::string, Files>* file_map;

  if (group == "resnet") {
    data_dir = resnetDataDir;
    mem_map = &resnetMemoryMap;
    param_map = &resnetParams;
    file_map = &resnetFiles;
  } else {
    data_dir = mobilebertDataDir;
    mem_map = &mobilebertMemoryMap;
    param_map = &mobilebert;
    file_map = &mobilebertFiles;
  }

  // Memory allocation
  INPUT_DATATYPE* acc_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* acc_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  INPUT_DATATYPE* hls_gold_sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* hls_gold_rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  UniversalPosit* uni_gold_sram_memory = new UniversalPosit[SRAM_MEMORY_SIZE];
  UniversalPosit* uni_gold_rram_memory = new UniversalPosit[RRAM_MEMORY_SIZE];
  float* float_gold_sram_memory = new float[SRAM_MEMORY_SIZE];
  float* float_gold_rram_memory = new float[RRAM_MEMORY_SIZE];
  uint64_t* trash = new uint64_t[RRAM_MEMORY_SIZE];

  if (acc_sram_memory == nullptr || acc_rram_memory == nullptr ||
      hls_gold_sram_memory == nullptr || hls_gold_rram_memory == nullptr ||
      uni_gold_sram_memory == nullptr || uni_gold_rram_memory == nullptr ||
      float_gold_sram_memory == nullptr || float_gold_rram_memory == nullptr ||
      trash == nullptr)
    throw std::runtime_error("Failed to allocate simulation memory");

  // Load first layer input
  load_memory(
      (*param_map)[tests[0]], data_dir, (*file_map)[tests[0]],
      (*mem_map)[tests[0]], use_data_file, acc_sram_memory, acc_rram_memory,
      hls_gold_sram_memory + (*param_map)[tests[0]].INPUT_OFFSET,
      (INPUT_DATATYPE*)trash, (INPUT_DATATYPE*)trash, (INPUT_DATATYPE*)trash,
      (INPUT_DATATYPE*)trash, (INPUT_DATATYPE*)trash,
      uni_gold_sram_memory + (*param_map)[tests[0]].INPUT_OFFSET,
      (UniversalPosit*)trash, (UniversalPosit*)trash, (UniversalPosit*)trash,
      (UniversalPosit*)trash, (UniversalPosit*)trash,
      float_gold_sram_memory + (*param_map)[tests[0]].INPUT_OFFSET,
      (float*)trash, (float*)trash, (float*)trash, (float*)trash,
      (float*)trash);

  // Load weights, biases, and comparisons
  for (const std::string& test : tests) {
    load_wb((*param_map)[test], data_dir, (*file_map)[test], (*mem_map)[test],
            use_data_file, acc_sram_memory, acc_rram_memory,
            (INPUT_DATATYPE*)trash,
            hls_gold_rram_memory + (*param_map)[test].WEIGHT_OFFSET,
            hls_gold_rram_memory + (*param_map)[test].BIAS_OFFSET,
            (INPUT_DATATYPE*)trash, (INPUT_DATATYPE*)trash,
            (INPUT_DATATYPE*)trash, (UniversalPosit*)trash,
            uni_gold_rram_memory + (*param_map)[test].WEIGHT_OFFSET,
            uni_gold_rram_memory + (*param_map)[test].BIAS_OFFSET,
            (UniversalPosit*)trash, (UniversalPosit*)trash,
            (UniversalPosit*)trash, (float*)trash,
            float_gold_rram_memory + (*param_map)[test].WEIGHT_OFFSET,
            float_gold_rram_memory + (*param_map)[test].BIAS_OFFSET,
            (float*)trash, (float*)trash, (float*)trash);
  }

  // Run tests in sequence
  int X, Y, C, K, FX, FY, STRIDE;
  for (const std::string& test : tests) {
    validateMapping((*param_map)[test]);
    X = (*param_map)[test].loops[0][resnetParams[test].inputXLoopIndex[0]] *
        (*param_map)[test].loops[1][resnetParams[test].inputXLoopIndex[1]];
    Y = (*param_map)[test].loops[0][resnetParams[test].inputYLoopIndex[0]] *
        (*param_map)[test].loops[1][resnetParams[test].inputYLoopIndex[1]];
    C = (*param_map)[test].loops[1][resnetParams[test].reductionLoopIndex[1]] *
        DIMENSION;
    K = (*param_map)[test].loops[0][resnetParams[test].weightLoopIndex[0]] *
        (*param_map)[test].loops[1][resnetParams[test].weightLoopIndex[1]] *
        DIMENSION;
    FX = (*param_map)[test].loops[1][resnetParams[test].fxIndex];
    FY = (*param_map)[test].loops[1][resnetParams[test].fyIndex];
    STRIDE = (*param_map)[test].STRIDE;

    if ((*param_map)[test].REPLICATION) {
      FX = 7;
      C = 3;
    }

    if ((*param_map)[test].MAXPOOL) {
      X = X / 2;
      Y = Y / 2;
    }

    if ((*param_map)[test].AVGPOOL) {
      X = 1;
      Y = 1;
    }

    std::cout << "Performing " + test + ":" << std::endl;
    std::cout << "(" << X << "x" << Y << "x" << C << ")"
              << " * "
              << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
              << std::endl;

    // Run everything
    if (std::find(comparisons.begin(), comparisons.end(), "accelerator") !=
        comparisons.end()) {
      run_custom_posit_gold_model(
          (*param_map)[test],
          hls_gold_sram_memory + (*param_map)[test].INPUT_OFFSET,
          hls_gold_rram_memory + (*param_map)[test].WEIGHT_OFFSET,
          hls_gold_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
          hls_gold_rram_memory + (*param_map)[test].BIAS_OFFSET,
          hls_gold_sram_memory + (*param_map)[test].RESIDUAL_OFFSET);
    }
    if (std::find(comparisons.begin(), comparisons.end(), "customposit") !=
        comparisons.end()) {
      run_custom_posit_gold_model(
          (*param_map)[test],
          hls_gold_sram_memory + (*param_map)[test].INPUT_OFFSET,
          hls_gold_rram_memory + (*param_map)[test].WEIGHT_OFFSET,
          hls_gold_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
          hls_gold_rram_memory + (*param_map)[test].BIAS_OFFSET,
          hls_gold_sram_memory + (*param_map)[test].RESIDUAL_OFFSET);
    }
    if (std::find(comparisons.begin(), comparisons.end(), "universal") !=
        comparisons.end()) {
      run_universal_posit_gold_model(
          (*param_map)[test],
          uni_gold_sram_memory + (*param_map)[test].INPUT_OFFSET,
          uni_gold_rram_memory + (*param_map)[test].WEIGHT_OFFSET,
          uni_gold_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
          uni_gold_rram_memory + (*param_map)[test].BIAS_OFFSET,
          uni_gold_sram_memory + (*param_map)[test].RESIDUAL_OFFSET);
    }
    if (std::find(comparisons.begin(), comparisons.end(), "fp32") !=
        comparisons.end()) {
      run_fp_gold_model(
          (*param_map)[test],
          float_gold_sram_memory + (*param_map)[test].INPUT_OFFSET,
          float_gold_rram_memory + (*param_map)[test].WEIGHT_OFFSET,
          float_gold_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
          float_gold_rram_memory + (*param_map)[test].BIAS_OFFSET,
          float_gold_sram_memory + (*param_map)[test].RESIDUAL_OFFSET);
    }
  }

  // Run accelerator
  std::vector<SimplifiedParams> params_list;
  for (const std::string& test : tests) {
    params_list.push_back((*param_map)[test]);
  }
  // if (comparisons.find("accelerator") != comparisons.end()) {
  //   run_op(params_list, acc_sram_memory, acc_rram_memory,
  //   (*mem_map)["conv1"]);
  // }

  // Allocate comparison
  INPUT_DATATYPE* hls_comp = new INPUT_DATATYPE[X * Y * K];
  UniversalPosit* uni_comp = new UniversalPosit[X * Y * K];
  float* fp_comp = new float[X * Y * K];

  std::string last_test = *(tests.end() - 1);
  load_datafile_outputs((*param_map)[last_test],
                        data_dir + (*file_map)[last_test].outputs_file,
                        hls_comp, uni_comp, fp_comp);

  if (hls_comp == nullptr || uni_comp == nullptr || fp_comp == nullptr)
    throw std::runtime_error(
        "Failed to allocate simulation memory in sequence");

  for (int i = 0; i < comparisons.size(); i += 2) {
    std::string diff_file = "test_outputs/" + group + '.' + *(tests.begin()) +
                            '.' + *(--tests.end()) + '.' + comparisons[i] +
                            "_vs_" + comparisons[i + 1];
    std::cout << diff_file << std::endl;

    if ((comparisons[i] == "accelerator" &&
         comparisons[i + 1] == "customposit") ||
        (comparisons[i + 1] == "accelerator" &&
         comparisons[i] == "customposit")) {
      compare_arrays(
          acc_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
          hls_gold_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
          X * Y * K, diff_file);
    } else if ((comparisons[i] == "accelerator" &&
                comparisons[i + 1] == "file") ||
               (comparisons[i + 1] == "accelerator" &&
                comparisons[i] == "file")) {
      compare_arrays(acc_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
                     hls_comp, X * Y * K, diff_file);
    } else if ((comparisons[i] == "customposit" &&
                comparisons[i + 1] == "file") ||
               (comparisons[i + 1] == "customposit" &&
                comparisons[i] == "file")) {
      compare_arrays(acc_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
                     hls_comp, X * Y * K, diff_file);
    } else if ((comparisons[i] == "universal" &&
                comparisons[i + 1] == "customposit") ||
               (comparisons[i + 1] == "universal" &&
                comparisons[i] == "customposit")) {
      compare_arrays(
          hls_gold_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
          uni_gold_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
          X * Y * K, diff_file);
    } else if ((comparisons[i] == "universal" &&
                comparisons[i + 1] == "file") ||
               (comparisons[i + 1] == "universal" &&
                comparisons[i] == "file")) {
      compare_arrays(
          uni_gold_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
          uni_comp, X * Y * K, diff_file);
    } else if ((comparisons[i] == "fp32" && comparisons[i + 1] == "file") ||
               (comparisons[i + 1] == "fp32" && comparisons[i] == "file")) {
      compare_arrays(
          float_gold_sram_memory + (*param_map)[last_test].OUTPUT_OFFSET,
          fp_comp, X * Y * K, diff_file);
    } else {
      std::cout << "Comparison not supported." << std::endl;
    }
  }
  // // if (comparisons.find("accelerator") != comparisons.end() &&
  // //     comparisons.find("pytorch") != comparisons.end()) {
  //   std::string acc_diff_file =
  //       "test_outputs/" + group + "resnet." + test +
  //       "unigold_vs_pytorch.txt";
  //   std::cout << acc_diff_file << std::endl;
  //   compare_arrays(acc_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
  //                  uni_comp, X * Y * K, unigold_diff_file);
  // // }

  // // if (comparisons.find("accelerator") != comparisons.end()) {
  //   std::string unigold_diff_file =
  //       "test_outputs/" + group + "resnet." + test +
  //       "unigold_vs_pytorch.txt";
  //   std::cout << unigold_diff_file << std::endl;
  //   compare_arrays(uni_gold_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
  //                  uni_comp, X * Y * K, unigold_diff_file);
  // // }

  // // if (comparisons.find("customposit") != comparisons.end()) {
  //   std::string hlsgold_diff_file =
  //       "test_outputs/" + group + "resnet." + test +
  //       "hlsgold_vs_pytorch.txt";
  //   std::cout << hlsgold_diff_file << std::endl;
  //   compare_arrays(hls_gold_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
  //                  hls_comp, X * Y * K, hlsgold_diff_file);
  // // }

  // // if (comparisons.find("fp32") != comparisons.end()){
  //   std::string fpgold_diff_file =
  //       "test_outputs/" + group + "resnet." + test + "fpgold_vs_pytorch.txt";
  //   std::cout << fpgold_diff_file << std::endl;
  //   compare_arrays(float_gold_sram_memory + (*param_map)[test].OUTPUT_OFFSET,
  //                  fp_comp, X * Y * K, fpgold_diff_file);
  // // }

  // std::ofstream wf("pybuild/result.txt", std::ios::out | std::ios::trunc);
  // if (!wf.good()) throw std::runtime_error("File write failed");

  // INPUT_DATATYPE* output =
  //     hls_gold_sram_memory + (*param_map)["fc"].OUTPUT_OFFSET;
  // int index = 0;
  // float max = (float)output[0];
  // for (int i = 0; i < 1000; i++) {
  //   // std::cout << (float)output[i] << " " << max<<" "<< i << std::endl;
  //   if ((float)output[i] >= max) {
  //     index = i;
  //     max = (float)output[i];
  //   }
  // }
  // std::cout << "index" << index << "u" << std::endl;
  // wf << index << '\n';
  // wf.close();
}

extern "C" int sc_main(int argc, char* argv[]) {
  SimplifiedParams params;

  const char* groupName = std::getenv("GROUP");
  const char* testNames = std::getenv("TESTS");
  const char* compNames = std::getenv("MODELS");

  if (!(testNames && groupName)) {
    std::cout << "Warning! No group/test specified! Please set the environment "
                 "variables GROUP and TEST"
              << std::endl;
    return -1;
  }

  std::string group(groupName);
  std::string tests(testNames);
  std::string comps(compNames);

  std::vector<std::string> testList = parse_csv(tests);
  std::vector<std::string> compList = parse_csv(comps);

  std::string fullName = "test_outputs/" + group + "." + testList[0] + ".";

  std::cout << "Running: " << group << ": " << testList[0];
  if (testList.size() > 1) {
    for (auto it = testList.begin() + 1; it != testList.end(); it++) {
      std::cout << " -> " << (*it);
    }
  }
  std::cout << std::endl;

  if (group != "simple" && group != "mobilebert" && group != "resnet") {
    throw std::runtime_error("Group: " + group + " not found");
  }

  // Get sequence to run
  std::vector<std::string> sequence;
  if (testList.size() > 1) {
    if (group == "resnet") {
      bool log = false;
      for (const std::string& test : resnet_order) {
        if (test == testList[0]) {
          log = true;
        }
        if (log) {
          sequence.push_back(test);
        }
        if (test == testList[1]) {
          log = false;
        }
      }
    } else {
      throw std::runtime_error("Sequences only work for resnet right now");
    }
  } else {
    sequence.push_back(testList[0]);
  }

  run_sequence(group, sequence, compList);
  return 0;
}