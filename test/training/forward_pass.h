#pragma once

#include "memory_plan.h"

#ifdef SOC
typedef enum
#else
// define enum with 7 values
enum ForwardPassVariant
#endif
{ COMPLETE_FORWARD_PASS,
  FORWARD_PASS_FFN_1_INTERMEDIATE,  // from start to FFN 1 Intermediate
  FORWARD_PASS_FFN_0_INTERMEDIATE,  // only FFN 0 Intermediate
  FORWARD_PASS_MHA_0,               // from start to attention_probs 0
  FORWARD_PASS_MHA_1,               // only scores 1, probs 1
  FORWARD_PASS_MHA_2,               // only scores 2, probs 2
  FORWARD_PASS_MHA_3,               // only scores 3, probs 3
#ifdef SOC
} ForwardPassVariant;
#else
};
#endif

void encoder_forward_pass(int encoderLayer, ForwardPassVariant variant) {
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
    run_op(OPERATION(bottleneck_attention_dense, inference), encoderLayerInput,
           weightBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);
    // bottleneck_input_LayerNorm
    run_op(OPERATION(bottleneck_attention_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    // query projection
    run_op(OPERATION(attention_self_query_layer, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               6 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);
    // key projection
    run_op(OPERATION(attention_self_key_layer, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               7 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               7 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);
    // value projection
    run_op(OPERATION(attention_self_value_layer, inference), encoderLayerInput,
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
      run_op(OPERATION(attention_self_attention_scores_0, inference),
             activationBase + INTERMEDIATE_SIZE + head * ATTENTION_HEAD_SIZE,
             activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
                 head * ATTENTION_HEAD_SIZE,
             activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
             MASK, 0);
      // probs
      run_op(OPERATION(attention_self_attention_probs_0, inference),
             activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 0,
             activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE, 0,
             0);

      if (variant == FORWARD_PASS_MHA_0 || variant == FORWARD_PASS_MHA_1 ||
          variant == FORWARD_PASS_MHA_2 || variant == FORWARD_PASS_MHA_3) {
        return;
      }

      // context
      run_op(OPERATION(attention_self_context_layer_0, inference),
             activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
             activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
                 head * ATTENTION_HEAD_SIZE,
             activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
                 head * ATTENTION_HEAD_SIZE,
             0, 0);
    }

    // bottleneck_attention_dense
    run_op(OPERATION(bottleneck_input_dense, inference), encoderLayerInput,
           weightBase,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + INTERMEDIATE_SIZE, 0);
    // bottleneck_attention_LayerNorm
    run_op(OPERATION(bottleneck_input_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_BIAS_SIZE, 0);

    // attention_output_dense
    run_op(OPERATION(attention_output_dense, inference),
           activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               9 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
               9 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE);
    // attention_output_LayerNorm
    run_op(OPERATION(attention_output_LayerNorm, inference),
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
    run_op(OPERATION(ffn_0_intermediate_dense, inference),
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
    run_op(OPERATION(ffn_0_output_dense, inference),
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
    run_op(OPERATION(ffn_0_output_LayerNorm, inference),
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
  run_op(OPERATION(output_bottleneck_dense, inference),
         activationBase + INTERMEDIATE_SIZE,
         weightBase + 7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         weightBase + 8 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         encoderLayerInput);

  // output_bottleneck_LayerNorm
  run_op(OPERATION(output_bottleneck_LayerNorm, inference),
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
    encoder_forward_pass(encoderLayer, COMPLETE_FORWARD_PASS);
  }
}

void full_forward_pass() {
  forward_pass(0, NUM_ENCODER_LAYERS);

  // classifier
  run_op(OPERATION(classifier, inference), ENCODER_SCRATCH,
         (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
             8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         ENCODER_SCRATCH + INTERMEDIATE_SIZE,
         (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
             8 * INTERMEDIATE_SIZE + 21 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         0);

#ifndef SOC
  std::cout << "Classifier Output: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << memory->sram[ENCODER_SCRATCH + INTERMEDIATE_SIZE + i]
              << std::endl;
  }
#endif
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
      encoder_forward_pass(encoderLayer, COMPLETE_FORWARD_PASS);
    }
  }
  encoder_forward_pass(endEncoderLayer, variant);
}