#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

#define NO_SYSC
// clang-format off
#include "src/PositTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/GoldModel.h"
#include "test/common/SimpleMemoryModel.h"
#include "test/common/Utils.h"
#include "test/mobilebert/MobileBERT.h"
#include "test/resnet/ResNet.h"

// turn on to use posit gold model instead of float
#define USE_POSIT

/* Test accuracy of model on a dataset using the Posit gold model.
 */
int main(int argc, char* argv[]) {
#ifdef USE_POSIT
  std::cout << "Running accuracy test with Posit gold model..." << std::endl;
#else
  std::cout << "Running accuracy test with float gold model..." << std::endl;
#endif

  // model name is first argument
  std::string modelName = std::string(argv[1]);
  // path to folder with model parameters and dataset is second argument
  std::string modelPath = std::string(argv[2]);

  // Get vector of workloads to run
  std::vector<Workload> workloads;
  if (modelName == "mobilebert") {
    MobileBERT mobilebert("mobilebert", "inference", "");
    workloads = mobilebert.getFullForwardPass();
  } else if (modelName == "resnet18") {
    ResNet resnet18("resnet18", "");
    workloads = resnet18.getAllWorkloads();
  } else {
    std::cout << "ERROR: Model name not recognized." << std::endl;
  }

  validateMemoryMapping(workloads);

// Create the memory model and load the model parameters
#ifdef USE_POSIT
  SimpleMemoryModel<INPUT_DATATYPE> memory(false);
#else
  SimpleMemoryModel<float> memory(false);
#endif

  std::cout << "Loading model parameters..." << std::endl;
  for (Workload& workload : workloads) {
    if (modelName == "mobilebert") {
      if (workload.loadWeight) {
        workload.files.weights_file.insert(0, modelPath + "/params/");
        workload.files.bias_file.insert(0, modelPath + "/params/");
        memory.loadModelParams(workload.params, workload.files,
                               workload.memoryMap, true);
      }
    } else if (modelName == "resnet18") {
      workload.files.weights_file.insert(0, modelPath + "/params/");
      workload.files.bias_file.insert(0, modelPath + "/params/");
      memory.loadModelParams(workload.params, workload.files,
                             workload.memoryMap, true);
    }
  }

  std::cout << "Running inference..." << std::endl;

  int numSamples = 0;
  int numCorrect = 0;

  int numClasses;
  if (modelName == "mobilebert") {
    numClasses = 2;
  } else if (modelName == "resnet18") {
    numClasses = 1000;
  }

  // iterate through all folders in modelPath
  // each folder consists of two inputs
  for (const auto& entry : std::filesystem::directory_iterator(modelPath)) {
    if (entry.is_directory()) {
      // get folder name
      std::string folderName = entry.path().filename().string();

      // skip params folder
      if (folderName == "params") {
        continue;
      }

      // load the model inputs for this sample
      Workload firstWorkload = workloads[0];
      if (modelName == "mobilebert") {
        firstWorkload.files.inputs_file =
            modelPath + "/" + folderName + "/embedding_output";
      } else if (modelName == "resnet18") {
        firstWorkload.files.inputs_file =
            modelPath + "/" + folderName + "/input";
      }
      memory.loadModelActivations(firstWorkload.params, firstWorkload.files,
                                  firstWorkload.memoryMap, true);

      if (modelName == "mobilebert") {
        // for loading attention mask, we make a custom workload
        MobileBERT mobilebert("mobilebert", "inference", "");
        Workload dummyAttentionMaskWorkload = mobilebert.getWorkloadsInRange(
            {"attention_self_attention_scores_0"})[0];
        dummyAttentionMaskWorkload.files.inputs_file = "";
        dummyAttentionMaskWorkload.files.weights_file = "";
        dummyAttentionMaskWorkload.files.bias_file =
            modelPath + "/" + folderName + "/attention_mask";

        dummyAttentionMaskWorkload.params.ATTENTION_MASK = true;
        dummyAttentionMaskWorkload.params.RESIDUAL = false;
        dummyAttentionMaskWorkload.params.RELU_GRAD = false;
        dummyAttentionMaskWorkload.params.SOFTMAX_GRAD = false;
        dummyAttentionMaskWorkload.params.WEIGHT_SPLITTING = false;
        dummyAttentionMaskWorkload.memoryMap.bias = SRAM;
        memory.loadModelActivations(dummyAttentionMaskWorkload.params,
                                    dummyAttentionMaskWorkload.files,
                                    dummyAttentionMaskWorkload.memoryMap, true);
      }

      std::cout << "Running inference on " << folderName << "..." << std::endl;
      // run every workload in the forward pass
      for (Workload& workload : workloads) {
#ifdef USE_POSIT
        run_custom_posit_gold_model
#else
        run_fp_gold_model
#endif
            (workload.params, memory.sram + workload.params.INPUT_OFFSET,
             (workload.memoryMap.weights ? memory.rram : memory.sram) +
                 workload.params.WEIGHT_OFFSET,
             memory.sram + workload.params.OUTPUT_OFFSET,
             (workload.memoryMap.bias ? memory.rram : memory.sram) +
                 workload.params.BIAS_OFFSET,
             memory.sram + workload.params.RESIDUAL_OFFSET,
             memory.sram + workload.params.WEIGHT_RESIDUAL_OFFSET);
      }
      numSamples++;

      // get final output
      float logits[numClasses];
      for (int i = 0; i < numClasses; i++) {
        float fp_logit;
        if (workloads.back().params.ACC_T_OUTPUT) {
#ifdef USE_POSIT

          int bits = 0;
          for (int precision = 0; precision < 2; precision++) {
            bits =
                (bits << 8) | memory
                                  .sram[workloads.back().params.OUTPUT_OFFSET +
                                        2 * i + (1 - precision)]
                                  .bits;
          }
          ACCUM_DATATYPE acc_logit;
          acc_logit.bits = bits;
          fp_logit = acc_logit;
#else
          fp_logit = memory.sram[workloads.back().params.OUTPUT_OFFSET + 2 * i];
#endif
        } else {
          fp_logit = memory.sram[workloads.back().params.OUTPUT_OFFSET + i];
        }
        logits[i] = fp_logit;
      }

      // get class with highest probability
      int maxIndex = 0;
      for (int i = 0; i < numClasses; i++) {
        if (logits[i] > logits[maxIndex]) {
          maxIndex = i;
        }
      }

      // get ground truth from folder name
      int groundTruth;
      if (modelName == "mobilebert") {
        // last character of folder name is 0 or 1
        groundTruth = folderName.back() - '0';
      } else if (modelName == "resnet18") {
        // split folder name by underscore, first part is ground truth
        groundTruth = std::stoi(folderName.substr(0, folderName.find("_")));
      }

      // check if output matches ground truth
      if (maxIndex == groundTruth) {
        numCorrect++;
      }
      std::cout << "Classification: " << maxIndex
                << " , Expected: " << groundTruth << std::endl;

      std::cout << "Accuracy: " << numCorrect << "/" << numSamples << " ("
                << float(numCorrect) / numSamples * 100 << "%)" << std::endl;
    }
  }
}
