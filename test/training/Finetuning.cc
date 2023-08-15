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

#define FP32

#ifdef FP32
#define DTYPE float
#else
#define DTYPE Posit<8, 1>
#endif

// Milestone 0- unit test every backward pass operation
void milestone0(SimpleMemoryModel<DTYPE> *memory);
// Milestone 1- backward pass only, assuming all activations are present
void milestone1(SimpleMemoryModel<DTYPE> *memory);
// Milestone 2- forward pass + backward pass with activations recomputed
void milestone2(SimpleMemoryModel<DTYPE> *memory);
// Milestone 3- use LoRA weights in both forward and backward pass
void milestone3(SimpleMemoryModel<DTYPE> *memory);
// Milestone 4- perform weight update on LoRA weights
void milestone4(SimpleMemoryModel<DTYPE> *memory);

// Runs the gold model for a given workload
void runWorkload(SimpleMemoryModel<DTYPE> *memory, const Workload &workload) {
  SimplifiedParams params = workload.params;
  MemoryMap memoryMap = workload.memoryMap;
#ifdef FP32
  run_fp_gold_model
#else
  run_custom_posit_gold_model
#endif
      (params, memory->sram + params.INPUT_OFFSET,
       (memoryMap.weights ? memory->rram : memory->sram) + params.WEIGHT_OFFSET,
       memory->sram + params.OUTPUT_OFFSET, memory->rram + params.BIAS_OFFSET,
       memory->sram + params.RESIDUAL_OFFSET,
       memory->sram + params.WEIGHT_RESIDUAL_OFFSET);
}

void adjustPathForStep(std::string &file, int stepNumber) {
  if (file == "") {
    return;
  }
  file.replace(file.find("step_51_"), std::string("step_51_").length(),
               "step_" + std::to_string(stepNumber) + "/");
}

void checkOutputs(SimpleMemoryModel<DTYPE> *memory, const Workload &workload) {
  SimplifiedParams params = workload.params;
  MemoryMap memoryMaps = workload.memoryMap;

  int X, Y, C, K, FX, FY, STRIDE;
  X = params.loops[0][params.inputXLoopIndex[0]] *
      params.loops[1][params.inputXLoopIndex[1]];
  Y = params.loops[0][params.inputYLoopIndex[0]] *
      params.loops[1][params.inputYLoopIndex[1]];
  C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  K = params.loops[0][params.weightLoopIndex[0]] *
      params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  FX = params.loops[1][params.fxIndex];
  FY = params.loops[1][params.fyIndex];
  STRIDE = params.STRIDE;

  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  if (params.MAXPOOL) {
    X /= 2;
    Y /= 2;
  }

  if (params.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  size_t size = X * Y * K;

  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    size = X * Y;
  } else if (params.CROSS_ENTROPY_GRAD) {
    size = X;
  } else if (params.NO_NORM_GRAD) {
    size = C;
  } else if (params.WEIGHT_UPDATE) {
    size = X * C;
  }

  memory->loadReferenceOutput(workload.params, workload.files,
                              workload.params.ACC_T_OUTPUT);
  for (int i = 0; i < 10; i++) {
    std::cout << memory->sram[params.OUTPUT_OFFSET + i] << " "
              << memory->reference[i] << std::endl;
  }
}

// Initialize the model weights and biases
void initializeModel(SimpleMemoryModel<DTYPE> *memory,
                     const std::string &modelPath) {
  // Load weights
  MobileBERT mobilebert("mobilebert", "inference", "");
  std::vector<Workload> forwardPass = mobilebert.getFullForwardPass();
  for (Workload &workload : forwardPass) {
    if (workload.loadWeight) {
      workload.files.weights_file.insert(0, modelPath + "/params/");
      workload.files.bias_file.insert(0, modelPath + "/params/");
      memory->loadModelParams(workload.params, workload.files,
                              workload.memoryMap, true);
    }
  }
}

int main(int argc, char *argv[]) {
  SimpleMemoryModel<DTYPE> *memory = new SimpleMemoryModel<DTYPE>(false);
  initializeModel(memory, "models/mobilebert/binary_data/tiny_pretrained/");

  std::cout << "Loaded pretrained model parameters." << std::endl;

  milestone1(memory);
}

// Milestone 1- backward pass only, with loading of activations
void milestone1(SimpleMemoryModel<DTYPE> *memory) {
  MobileBERT mobilebert("mobilebert", "backward",
                        "models/mobilebert/binary_data/tiny_pretrained/");

  std::vector<Workload> backwardPass = mobilebert.getFullBackwardPass();
  bool flag = false;
  int memoryWatch = 0;
  DTYPE memoryWatchVal = 0;
  for (int stepNumber = 0; stepNumber < 1; stepNumber++) {
    for (Workload &workload : backwardPass) {
      std::cout << "Running " << workload.name << std::endl;

      // adjust all files to point to the correct step number
      adjustPathForStep(workload.files.inputs_file, stepNumber);
      adjustPathForStep(workload.files.weights_file, stepNumber);
      adjustPathForStep(workload.files.bias_file, stepNumber);
      adjustPathForStep(workload.files.outputs_file, stepNumber);
      adjustPathForStep(workload.files.residual_file, stepNumber);

      // remove file entry if it doesn't have "activations" in it
      if (workload.files.inputs_file.find("activations") == std::string::npos) {
        workload.files.inputs_file = "";
      }
      if (workload.files.weights_file.find("activations") ==
          std::string::npos) {
        workload.files.weights_file = "";
      }
      if (workload.files.bias_file.find("activations") == std::string::npos) {
        workload.files.bias_file = "";
      }
      if (workload.files.residual_file.find("activations") ==
          std::string::npos) {
        workload.files.residual_file = "";
      }

      memory->loadModelActivations(workload.params, workload.files,
                                   workload.memoryMap, true);

      runWorkload(memory, workload);
      checkOutputs(memory, workload);
    }
  }
}

// Milestone 2- forward pass with checkpoints saved, backward pass with
// activations recomputed from checkpoint
void milestone2(SimpleMemoryModel<DTYPE> *memory) {
  MobileBERT mobilebert("mobilebert", "inference",
                        "models/mobilebert/binary_data/tiny_pretrained/");
}

// Milestone 3- use LoRA weights in both forward and backward pass
void milestone3(SimpleMemoryModel<DTYPE> *memory) {
  MobileBERT mobilebert("mobilebert", "inference",
                        "models/mobilebert/binary_data/tiny_pretrained/");
}

// Milestone 4- perform weight update on LoRA weights
void milestone4(SimpleMemoryModel<DTYPE> *memory) {
  MobileBERT mobilebert("mobilebert", "inference",
                        "models/mobilebert/binary_data/tiny_pretrained/");
}