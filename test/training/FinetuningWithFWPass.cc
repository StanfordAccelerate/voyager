#include <filesystem>
#include <fstream>
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
#include "test/training/DTYPE.h"
#include "test/training/DatasetIterator.h"

// Milestone 2- forward pass with checkpoints saved, backward pass with
// activations recomputed from checkpoint
// In addition, we will set the memory offsets ourselves

#include "memory_plan.h"

#define OPERATION(layer, task) #layer, #task

void getMobileBERTParams(std::string layerName, std::string taskName,
                         SimplifiedParams &params, MemoryMap &memoryMap);

void adjustMemoryOffsets(SimplifiedParams &params, int inputOffset,
                         int weightOffset, int outputOffset, int biasOffset,
                         int residualOffset) {
  params.INPUT_OFFSET = inputOffset;
  params.WEIGHT_OFFSET = weightOffset;
  params.OUTPUT_OFFSET = outputOffset;
  params.BIAS_OFFSET = biasOffset;
  params.RESIDUAL_OFFSET = residualOffset;
}

SimpleMemoryModel<DTYPE> *memory;

void runWorkload(SimplifiedParams params, MemoryMap memoryMap) {
#ifdef FP32
  run_fp_gold_model
#else
  run_custom_posit_gold_model
#endif
      (params, memory->sram + params.INPUT_OFFSET,
       (memoryMap.weights ? memory->rram : memory->sram) + params.WEIGHT_OFFSET,
       memory->sram + params.OUTPUT_OFFSET,
       (memoryMap.bias ? memory->rram : memory->sram) + params.BIAS_OFFSET,
       memory->sram + params.RESIDUAL_OFFSET,
       memory->sram + params.WEIGHT_RESIDUAL_OFFSET);
}

void run_op(const std::string &layerName, const std::string &task,
            int inputOffset, int weightOffset, int outputOffset, int biasOffset,
            int residualOffset) {
  SimplifiedParams params;
  MemoryMap memoryMap;
  getMobileBERTParams(layerName, task, params, memoryMap);
  adjustMemoryOffsets(params, inputOffset, weightOffset, outputOffset,
                      biasOffset, residualOffset);

  runWorkload(params, memoryMap);
}

// clang-format off
#include "forward_pass.h"
#include "backward_pass.h"
// clang-format on

void load_sample(DatasetIterator &dataset) {
  std::string sampleFolder = dataset.getSample();

  MobileBERT mobilebert("mobilebert", "inference", "");
  std::vector<Workload> workloads = mobilebert.getFullForwardPass();

  // load the model inputs for this sample
  Workload firstWorkload = workloads[0];
  firstWorkload.files.inputs_file =
      sampleFolder + "/activations/mobilebert_embeddings";
  firstWorkload.params.INPUT_OFFSET = INPUT;
  memory->loadModelActivations(firstWorkload.params, firstWorkload.files,
                               firstWorkload.memoryMap, true);

  // for loading attention mask, we make a custom workload
  Workload dummyAttentionMaskWorkload =
      mobilebert.getWorkloadsInRange({"attention_self_attention_scores_0"})[0];
  dummyAttentionMaskWorkload.files.inputs_file = "";
  dummyAttentionMaskWorkload.files.weights_file = "";
  dummyAttentionMaskWorkload.files.bias_file =
      sampleFolder + "/activations/mobilebert_attention_mask";

  dummyAttentionMaskWorkload.params.ATTENTION_MASK = true;
  dummyAttentionMaskWorkload.params.RESIDUAL = false;
  dummyAttentionMaskWorkload.params.RELU_GRAD = false;
  dummyAttentionMaskWorkload.params.SOFTMAX_GRAD = false;
  dummyAttentionMaskWorkload.params.WEIGHT_SPLITTING = false;
  dummyAttentionMaskWorkload.memoryMap.bias = SRAM;
  dummyAttentionMaskWorkload.params.BIAS_OFFSET = MASK;
  memory->loadModelActivations(dummyAttentionMaskWorkload.params,
                               dummyAttentionMaskWorkload.files,
                               dummyAttentionMaskWorkload.memoryMap, true);

  // open file
  std::ifstream labelFile(sampleFolder + "/activations/mobilebert_labels",
                          std::ios::binary);
  double *tmpValuesArray = new double[1024];
  labelFile.read((char *)tmpValuesArray, 1024 * sizeof(double));
  for (int i = 0; i < 2; i++) {
    memory->sram[LABEL + i] = tmpValuesArray[i];
    std::cout << memory->sram[LABEL + i] << std::endl;
  }
  for (int i = 2; i < 16; i++) {
    memory->sram[LABEL + i] = 0;
  }
  // load correct label

  std::cout << "Loaded sample from " << sampleFolder << std::endl;
}

void initialize_model(const std::string &modelPath) {
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
  std::cout << "Loaded pretrained model parameters." << std::endl;
}

int main(int argc, char **argv) {
  memory = new SimpleMemoryModel<DTYPE>(false);
  initialize_model("models/mobilebert/binary_data/tiny_pretrained/");
  DatasetIterator dataset("models/mobilebert/binary_data/tiny_pretrained/");
  for (int i = 0; i < 100; i++) {
    load_sample(dataset);

    full_forward_pass();
    full_backward_pass();
  }
}
