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
#define LORA_WQ_B_SIZE (128 * 16 * 2)     // 16b precision
#define LORA_WQ_A_SIZE (16 * 128 * 2)     // 16b precision
#define LORA_WV_B_SIZE (512 * 16 * 2)     // 16b precision
#define LORA_WV_A_SIZE (16 * 128 * 2)     // 16b precision
#define CLASSIFIER_W_SIZE (512 * 16 * 2)  // 16b precision

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
#define ENCODER_SCRATCH_SIZE (128 * 512 + 128 * 128 * 7)
#define BACKPROP_SCRATCH (ENCODER_SCRATCH + ENCODER_SCRATCH_SIZE)
#define BACKPROP_SCRATCH_SIZE (128 * 512 + 128 * 128 * 7)
#define LORA_W (BACKPROP_SCRATCH + BACKPROP_SCRATCH_SIZE)
#define LORA_W_SIZE                                                      \
  ((LORA_WQ_B_SIZE + LORA_WQ_A_SIZE + LORA_WV_B_SIZE + LORA_WV_A_SIZE) * \
   NUM_ENCODER_LAYERS)
#define CLASSIFIER_W (LORA_W + LORA_W_SIZE)
#define LORA_G (CLASSIFIER_W + CLASSIFIER_W_SIZE)
#define LORA_G_SIZE (LORA_W_SIZE)
#define CLASSIFIER_G (LORA_G + LORA_G_SIZE)

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

void run_layer(const std::string &layerName, const std::string &task,
               int inputOffset, int weightOffset, int outputOffset,
               int biasOffset, int residualOffset) {
  SimplifiedParams params;
  MemoryMap memoryMap;
  getMobileBERTParams(layerName, task, params, memoryMap);
  adjustMemoryOffsets(params, inputOffset, weightOffset, outputOffset,
                      biasOffset, residualOffset);

  runWorkload(params, memoryMap);
}

// define enum with 7 values
enum ForwardPassVariant {
  COMPLETE_FORWARD_PASS,
  FORWARD_PASS_FFN_1_INTERMEDIATE,  // from start to FFN 1 Intermediate
  FORWARD_PASS_FFN_0_INTERMEDIATE,  // only FFN 0 Intermediate
  FORWARD_PASS_MHA_0,               // from start to attention_probs 0
  FORWARD_PASS_MHA_1,               // only scores 1, probs 1
  FORWARD_PASS_MHA_2,               // only scores 2, probs 2
  FORWARD_PASS_MHA_3,               // only scores 3, probs 3
};

void encoder_forward_pass(int encoderLayer,
                          ForwardPassVariant variant = COMPLETE_FORWARD_PASS) {
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

  if (variant == COMPLETE_FORWARD_PASS ||
      variant == FORWARD_PASS_FFN_1_INTERMEDIATE ||
      variant == FORWARD_PASS_MHA_0) {
    // bottleneck_input_dense
    run_layer(
        "bottleneck_attention_dense", "inference", encoderLayerInput,
        weightBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
        activationBase + INTERMEDIATE_SIZE,
        weightBase + 2 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE, 0);
    // bottleneck_input_LayerNorm
    run_layer(
        "bottleneck_attention_LayerNorm", "inference",
        activationBase + INTERMEDIATE_SIZE,
        weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
        weightBase + 2 * INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_BIAS_SIZE, 0);

    // query projection
    run_layer(
        "attention_self_query_layer", "inference",
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
        weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
        activationBase + INTERMEDIATE_SIZE,
        weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
            6 * INTRA_BOTTLENECK_BIAS_SIZE,
        0);
    // key projection
    run_layer("attention_self_key_layer", "inference",
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
                  7 * INTRA_BOTTLENECK_BIAS_SIZE,
              activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
              weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                  7 * INTRA_BOTTLENECK_BIAS_SIZE,
              0);
    // value projection
    run_layer("attention_self_value_layer", "inference", encoderLayerInput,
              weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                  8 * INTRA_BOTTLENECK_BIAS_SIZE,
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                  8 * INTRA_BOTTLENECK_BIAS_SIZE,
              0);
  }

  int startingHead = 0;
  if (variant == FORWARD_PASS_MHA_1) {
    startingHead = 1;
  } else if (variant == FORWARD_PASS_MHA_2) {
    startingHead = 2;
  } else if (variant == FORWARD_PASS_MHA_3) {
    startingHead = 3;
  }

  if (variant == COMPLETE_FORWARD_PASS ||
      variant == FORWARD_PASS_FFN_1_INTERMEDIATE ||
      variant == FORWARD_PASS_MHA_0 || variant == FORWARD_PASS_MHA_1 ||
      variant == FORWARD_PASS_MHA_2 || variant == FORWARD_PASS_MHA_3) {
    for (int head = startingHead; head < NUM_HEADS; head++) {
      // scores
      run_layer("attention_self_attention_scores_0", "inference",
                activationBase + INTERMEDIATE_SIZE + head * ATTENTION_HEAD_SIZE,
                activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                    head * ATTENTION_HEAD_SIZE,
                activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
                MASK, 0);
      // probs
      run_layer(
          "attention_self_attention_probs_0", "inference",
          activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 0,
          activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE, 0, 0);

      if (variant == FORWARD_PASS_MHA_0 || variant == FORWARD_PASS_MHA_1 ||
          variant == FORWARD_PASS_MHA_2 || variant == FORWARD_PASS_MHA_3) {
        return;
      }

      // context
      run_layer("attention_self_context_layer_0", "inference",
                activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
                activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
                    head * ATTENTION_HEAD_SIZE,
                activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
                    head * ATTENTION_HEAD_SIZE,
                0, 0);
    }

    // bottleneck_attention_dense
    run_layer("bottleneck_input_dense", "inference", encoderLayerInput,
              weightBase,
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + INTERMEDIATE_SIZE, 0);
    // bottleneck_attention_LayerNorm
    run_layer("bottleneck_input_LayerNorm", "inference",
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
              activationBase + INTERMEDIATE_SIZE,
              weightBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_BIAS_SIZE,
              0);

    // attention_output_dense
    run_layer("attention_output_dense", "inference",
              activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
              weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                  9 * INTRA_BOTTLENECK_BIAS_SIZE,
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                  9 * INTRA_BOTTLENECK_BIAS_SIZE,
              activationBase + INTERMEDIATE_SIZE);
    // attention_output_LayerNorm
    run_layer("attention_output_LayerNorm", "inference",
              activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
              weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                  10 * INTRA_BOTTLENECK_BIAS_SIZE,
              activationBase + INTERMEDIATE_SIZE,
              weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                  11 * INTRA_BOTTLENECK_BIAS_SIZE,
              0);
  }

  for (int ffn = 0; ffn < NUM_FFN; ffn++) {
    // ffn_intermediate_dense
    run_layer("ffn_0_intermediate_dense", "inference",
              activationBase + INTERMEDIATE_SIZE + ffn * INTRA_BOTTLENECK_SIZE,
              weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                  12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
              weightBase + 4 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                  12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              0);

    if (ffn == 0 && variant == FORWARD_PASS_FFN_0_INTERMEDIATE) {
      return;
    }
    if (ffn == 1 && variant == FORWARD_PASS_FFN_1_INTERMEDIATE) {
      return;
    }
    // ffn_output_dense
    run_layer("ffn_0_output_dense", "inference",
              activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
              weightBase + 4 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + +INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
              weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE + ffn * INTRA_BOTTLENECK_SIZE);

    // ffn_output_LayerNorm
    run_layer("ffn_0_output_LayerNorm", "inference",
              activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
              weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 13 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              activationBase + INTERMEDIATE_SIZE +
                  (NUM_FFN - 1 - ffn) * INTRA_BOTTLENECK_SIZE,
              weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                  3 * INTRA_BOTTLENECK_SIZE + 14 * INTRA_BOTTLENECK_BIAS_SIZE +
                  ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                         3 * INTRA_BOTTLENECK_BIAS_SIZE),
              0);
  }

  // output_bottleneck_dense
  run_layer("output_bottleneck_dense", "inference",
            activationBase + INTERMEDIATE_SIZE,
            weightBase + 7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
            weightBase + 8 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            encoderLayerInput);

  // output_bottleneck_LayerNorm
  run_layer("output_bottleneck_LayerNorm", "inference",
            activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
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
  run_layer("classifier", "inference", ENCODER_SCRATCH,
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

void forward_pass_from_checkpoint(int endEncoderLayer,
                                  ForwardPassVariant variant) {
  // the other variants run from a partial scratch space, so no need to start at
  // the checkpoint
  if (variant == FORWARD_PASS_FFN_1_INTERMEDIATE ||
      variant == FORWARD_PASS_MHA_0) {
    // start from checkpoint and go to endEncoderLayer
    // checkpointed layer is 5, 10, 15, 20
    int closestCheckpoint = endEncoderLayer / 5 * 5;
    for (int encoderLayer = closestCheckpoint; encoderLayer < endEncoderLayer;
         encoderLayer++) {
      encoder_forward_pass(encoderLayer);
    }
  }
  encoder_forward_pass(endEncoderLayer, variant);
}

void encoder_backward_pass(int encoderLayer) {
  int activationBase = ENCODER_SCRATCH;
  int activationGradientBase = BACKPROP_SCRATCH;
  int weightBase = encoderLayer * ENCODER_WEIGHT_SIZE;

  // These namings are weird, they actually refer to the layer before

  run_layer(
      "output_bottleneck_dense", "backward",
      activationGradientBase + (encoderLayer == 20 ? INTERMEDIATE_SIZE : 0),
      weightBase + 8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
          3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase, 0, 0);

  run_layer("output_LayerNorm", "backward", activationGradientBase,
            weightBase + 7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationGradientBase + INTERMEDIATE_SIZE, 0, 0);

  for (int ffn = NUM_FFN - 1; ffn >= 0; ffn--) {
    run_layer(
        "output_dense", "backward", activationGradientBase + INTERMEDIATE_SIZE,
        weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
            3 * INTRA_BOTTLENECK_SIZE + 13 * INTRA_BOTTLENECK_BIAS_SIZE +
            ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                   3 * INTRA_BOTTLENECK_BIAS_SIZE),
        activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 0,
        0);

    forward_pass_from_checkpoint(
        encoderLayer, ffn == NUM_FFN - 1 ? FORWARD_PASS_FFN_1_INTERMEDIATE
                                         : FORWARD_PASS_FFN_0_INTERMEDIATE);

    run_layer(
        "intermediate_dense", "backward",
        activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
        weightBase + 4 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
            3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE +
            ffn * (2 * INTERMEDIATE_SIZE + +INTERMEDIATE_BIAS_SIZE +
                   3 * INTRA_BOTTLENECK_BIAS_SIZE),
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        0, activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE);

    run_layer(
        "ffn_0_output_LayerNorm", "backward",
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
            12 * INTRA_BOTTLENECK_BIAS_SIZE +
            ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                   3 * INTRA_BOTTLENECK_BIAS_SIZE),
        activationGradientBase + INTERMEDIATE_SIZE, 0,
        activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE);
  }

  run_layer("attention_output_dense", "backward",
            activationGradientBase + INTERMEDIATE_SIZE,
            weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
                10 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            0, 0);

  run_layer(
      "attention_self_context_layer", "backward",
      activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
      weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
          9 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE, 0,
      0);

  run_layer("bottleneck_input_dense", "backward",
            activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
            activationGradientBase + INTERMEDIATE_SIZE, 0, 0);

  for (int head = 0; head < NUM_HEADS; head++) {
    forward_pass_from_checkpoint(encoderLayer, head == 0   ? FORWARD_PASS_MHA_0
                                               : head == 1 ? FORWARD_PASS_MHA_1
                                               : head == 2
                                                   ? FORWARD_PASS_MHA_2
                                                   : FORWARD_PASS_MHA_3);
    run_layer("attention_self_value_layer_0", "backward",
              activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
              activationGradientBase + INTERMEDIATE_SIZE +
                  2 * INTRA_BOTTLENECK_SIZE + head * ATTENTION_HEAD_SIZE,
              activationGradientBase + INTERMEDIATE_SIZE +
                  INTRA_BOTTLENECK_SIZE + head * ATTENTION_HEAD_SIZE,
              0, 0);

    run_layer(
        "attention_self_attention_probs_0", "backward",
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
        0, 0);

    run_layer(
        "attention_self_attention_scores_0", "backward",
        activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
        0,
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        0, activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE);

    run_layer(
        "attention_self_query_layer_0", "backward",
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        0, 0);

    run_layer(
        "attention_self_key_layer_0", "backward",
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        activationBase + INTERMEDIATE_SIZE + head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        0, 0);
  }

  run_layer(
      "query_to_bottleneck_attention_LayerNorm", "backward",
      activationGradientBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
      weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE, 0,
      0);

  run_layer(
      "key_to_bottleneck_attention_LayerNorm", "backward",
      activationGradientBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE,
      weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
          7 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE, 0,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE);

  run_layer(
      "bottleneck_attention_dense", "backward",
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
      weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 0,
      0);

  run_layer("shared_attention_input_to_hidden_states", "backward",
            activationGradientBase + INTERMEDIATE_SIZE, weightBase,
            activationGradientBase, 0, activationGradientBase);
  run_layer("value_to_hidden_states", "backward",
            activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
            weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                8 * INTRA_BOTTLENECK_BIAS_SIZE,
            activationGradientBase, 0, activationGradientBase);
  run_layer(
      "bottlenecked_hidden_states", "backward",
      activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
      weightBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase, 0, activationGradientBase);
}

void full_backward_pass() {
  // cross-entropy gradient
  run_layer("classifier", "backward", ENCODER_SCRATCH + INTERMEDIATE_SIZE, 0,
            BACKPROP_SCRATCH, 0, 0);

  // classifier gradient
  run_layer("output_bottleneck_LayerNorm", "backward", BACKPROP_SCRATCH,
            (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
                8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
                3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
            BACKPROP_SCRATCH + INTERMEDIATE_SIZE, 0, 0);

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
