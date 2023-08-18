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
#include "test/training/DTYPE.h"
#include "test/training/DatasetIterator.h"

// Milestone 2- forward pass with checkpoints saved, backward pass with
// activations recomputed from checkpoint
// In addition, we will set the memory offsets ourselves

#define NUM_ENCODER_LAYERS 21
#define NUM_HEADS 4
#define NUM_FFN 2

#define ATTENTION_HEAD_SIZE (128 * 32)
#define INTRA_BOTTLENECK_SIZE (128 * 128)
#define INTRA_BOTTLENECK_BIAS_SIZE (128 * 2)
#define INTERMEDIATE_SIZE (128 * 512)
#define INTERMEDIATE_BIAS_SIZE (512 * 2)
#define ENCODER_WEIGHT_SIZE                             \
  (8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE + \
   3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE)

// memory map
#define LABEL (0)
#define LABEL_SIZE 16
#define INPUT (LABEL + LABEL_SIZE)
#define INPUT_SIZE (128 * 512)
#define MASK (INPUT + INPUT_SIZE)
#define MASK_SIZE (128 * 2)
#define CHECKPOINT (MASK + MASK_SIZE)
#define CHECKPOINT_SIZE (128 * 512 * 4)  // 4 checkpoints
#define ENCODER_SCRATCH (CHECKPOINT + CHECKPOINT_SIZE)

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

void run_layer(const std::string &layerName, int inputOffset, int weightOffset,
               int outputOffset, int biasOffset, int residualOffset) {
  SimplifiedParams params;
  MemoryMap memoryMap;
  getMobileBERTParams(layerName, "inference", params, memoryMap);
  adjustMemoryOffsets(params, inputOffset, weightOffset, outputOffset,
                      biasOffset, residualOffset);

  runWorkload(params, memoryMap);
  // std::cout << layerName << std::endl;
  // for (int i = 0; i < 10; i++) {
  //   std::cout << memory->sram[outputOffset + i] << std::endl;
  // }
  // std::cout << "----------------" << std::endl;
}

void encoder_forward_pass(int encoderLayer) {
  int activationBase = ENCODER_SCRATCH;
  int weightBase = encoderLayer * ENCODER_WEIGHT_SIZE;

  // Handle checkpointing
  // outputs of encoder layers 4, 9, 14, 19 are checkpointed
  int encoderLayerOutput;
  if (encoderLayer % 5 == 4) {  // layers 4, 9, 14, 19
    encoderLayerOutput = CHECKPOINT + (encoderLayer / 5) * INPUT_SIZE;
  } else {
    encoderLayerOutput = activationBase;
  }
  int encoderLayerInput;
  if (encoderLayer == 0) {
    encoderLayerInput = INPUT;
  } else if (encoderLayer % 5 == 0) {  // layers 5, 10, 15, 20
    encoderLayerInput = CHECKPOINT + (encoderLayer / 5 - 1) * INPUT_SIZE;
  } else {
    encoderLayerInput = activationBase;
  }

  // bottleneck_input_dense
  run_layer("bottleneck_attention_dense", encoderLayerInput,
            weightBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE,
            weightBase + 2 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
            0);
  // bottleneck_input_LayerNorm
  run_layer(
      "bottleneck_attention_LayerNorm", activationBase + INTERMEDIATE_SIZE,
      weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
      weightBase + 2 * INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_BIAS_SIZE, 0);

  // query projection
  run_layer("attention_self_query_layer",
            activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE,
            weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
                6 * INTRA_BOTTLENECK_BIAS_SIZE,
            0);
  // key projection
  run_layer("attention_self_key_layer",
            activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
                7 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
            weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                7 * INTRA_BOTTLENECK_BIAS_SIZE,
            0);
  // value projection
  run_layer("attention_self_value_layer", encoderLayerInput,
            weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                8 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                8 * INTRA_BOTTLENECK_BIAS_SIZE,
            0);

  for (int head = 0; head < NUM_HEADS; head++) {
    // scores
    run_layer("attention_self_attention_scores_0",
              activationBase + INTERMEDIATE_SIZE + head * ATTENTION_HEAD_SIZE,
              activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                  head * ATTENTION_HEAD_SIZE,
              activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
              MASK, 0);
    // probs
    run_layer("attention_self_attention_probs_0",
              activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 0,
              activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE, 0,
              0);
    // context
    run_layer("attention_self_context_layer_0",
              activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
                  head * ATTENTION_HEAD_SIZE,
              activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
                  head * ATTENTION_HEAD_SIZE,
              0, 0);
  }

  // bottleneck_attention_dense
  run_layer("bottleneck_input_dense", encoderLayerInput, weightBase,
            activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + INTERMEDIATE_SIZE, 0);
  // bottleneck_attention_LayerNorm
  run_layer("bottleneck_input_LayerNorm",
            activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE,
            weightBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_BIAS_SIZE, 0);

  // attention_output_dense
  run_layer("attention_output_dense",
            activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
            weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                9 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                9 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE);
  // attention_output_LayerNorm
  run_layer("attention_output_LayerNorm",
            activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                10 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE,
            weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                11 * INTRA_BOTTLENECK_BIAS_SIZE,
            0);

  for (int ffn = 0; ffn < NUM_FFN; ffn++) {
    // ffn_intermediate_dense
    run_layer("ffn_0_intermediate_dense", activationBase + INTERMEDIATE_SIZE,
              weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                  12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
              weightBase + 4 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                  12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              0);
    // ffn_output_dense
    run_layer("ffn_0_output_dense",
              activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
              weightBase + 4 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + +INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE);
    // ffn_output_LayerNorm
    run_layer("ffn_0_output_LayerNorm",
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 13 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE,
              weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 14 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              0);
  }

  // output_bottleneck_dense
  run_layer("output_bottleneck_dense", activationBase + INTERMEDIATE_SIZE,
            weightBase + 7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
            weightBase + 8 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            encoderLayerInput);

  // output_bottleneck_LayerNorm
  run_layer("output_bottleneck_LayerNorm",
            activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
            weightBase + 8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            encoderLayerOutput,
            weightBase + 8 * INTERMEDIATE_SIZE + 4 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            0);
}

void forward_pass(int startingEncoder, int endingEncoder) {
  for (int encoderLayer = startingEncoder; encoderLayer < endingEncoder;
       encoderLayer++) {
    encoder_forward_pass(encoderLayer);
  }
}

void full_forward_pass() {
  forward_pass(0, NUM_ENCODER_LAYERS);

  // classifier
  run_layer("classifier", ENCODER_SCRATCH,
            (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
                8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            ENCODER_SCRATCH + INTERMEDIATE_SIZE,
            (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
                8 * INTERMEDIATE_SIZE + 21 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            0);

  std::cout << "Classifier Output: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << memory->sram[ENCODER_SCRATCH + INTERMEDIATE_SIZE + i]
              << std::endl;
  }
}

void encoder_backward_pass(int encoderLayer) {
  for (int head = 0; head < NUM_HEADS; head++) {
    // multi-headed attention
  }
}

void full_backward_pass() {
  // classifier

  for (int encoderLayer = NUM_ENCODER_LAYERS - 1; encoderLayer >= 0;
       encoderLayer--) {
    encoder_backward_pass(encoderLayer);
  }
}

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
  load_sample(dataset);

  full_forward_pass();
}
