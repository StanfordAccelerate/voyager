#pragma once

#include "memory_plan.h"

// #define LORA

#ifdef SOC
typedef enum
#else
// define enum with 7 values
enum ForwardPassVariant
#endif
{ COMPLETE_FORWARD_PASS,
  FORWARD_PASS_FFN_1_INTERMEDIATE,    // from start to FFN 1 Intermediate
  FORWARD_PASS_FFN_0_INTERMEDIATE,    // only FFN 0 Intermediate
  FORWARD_PASS_MHA_0,                 // from start to attention_probs 0
  FORWARD_PASS_MHA_1,                 // only scores 1, probs 1
  FORWARD_PASS_MHA_2,                 // only scores 2, probs 2
  FORWARD_PASS_MHA_3,                 // only scores 3, probs 3
  FORWARD_PASS_BOTTLENECK_ATTENTION,  // from start to bottleneck attention
                                      // layernorm
#ifdef SOC
} ForwardPassVariant;
#else
};
#endif

void checkLayerOutput(int memoryOffset, int size, std::string file) {
  std::string fullFileName =
      "models/mobilebert/binary_data/tiny_pretrained/step_0/activations/" +
      file;
  double *expected = readFileAsDouble(fullFileName, size, true);
  int mismatchCount = 0;
  for (int i = 0; i < size; i++) {
    if (fabs(expected[i] - memory->sram[memoryOffset + i]) > 0.001) {
      std::cout << "Mismatch at " << i << std::endl;
      std::cout << expected[i] << " vs " << memory->sram[memoryOffset + i]
                << std::endl;
      mismatchCount++;
    }
  }
  if (mismatchCount > 0) {
    std::cout << mismatchCount << " mismatches" << std::endl;
    std::cout << "file: " << file << std::endl;
    std::abort();
  }
}

void encoder_forward_pass(int encoderLayer, ForwardPassVariant variant) {
  int activationBase = ENCODER_SCRATCH;
  int weightBase = encoderLayer * ENCODER_WEIGHT_SIZE;
  int loraWeightBase = LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE;

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

  // if (encoderLayer == 20) {
  //   std::cout << "Input:" << std::endl;
  //   for (int i = 0; i < 10; i++) {
  //     std::cout << memory->sram[encoderLayerInput + i] << std::endl;
  //   }
  //   std::cout << std::endl;
  //   for (int i = 63 * 512; i < 63 * 512 + 10; i++) {
  //     std::cout << memory->sram[encoderLayerInput + i] << std::endl;
  //   }

  //   std::cout << std::endl;
  //   for (int i = 64 * 512; i < 64 * 512 + 10; i++) {
  //     std::cout << memory->sram[encoderLayerInput + i] << std::endl;
  //   }
  //   std::abort();
  // }

  if (variant == COMPLETE_FORWARD_PASS ||
      variant == FORWARD_PASS_FFN_1_INTERMEDIATE ||
      variant == FORWARD_PASS_MHA_0 ||
      variant == FORWARD_PASS_BOTTLENECK_ATTENTION) {
    // bottleneck_input_dense
    run_op(OPERATION(bottleneck_attention_dense, inference), encoderLayerInput,
           weightBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    checkLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                     "mobilebert_encoder_layer_" +
                         std::to_string(encoderLayer) +
                         "_bottleneck_attention_dense");

    // bottleneck_input_LayerNorm
    run_op(OPERATION(bottleneck_attention_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    checkLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_bottleneck_attention_LayerNorm");

    // std::cerr << "bottleneck attention LayerNorm" << std::endl;
    // int offsets = activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE;
    // for (int i = 0; i < 128; i++) {
    //   for (int j = 0; j < 128; j++) {
    //     std::cerr << memory->sram[offsets + i * 128 + j] << "\t";
    //   }
    //   std::cerr << std::endl;
    // }
    // std::cerr << std::endl << std::endl;

    if (variant == FORWARD_PASS_BOTTLENECK_ATTENTION) {
      return;
    }

#ifdef LORA

    // std::cout << "query weight combination" << std::endl;
    // LoRA query weight (A * B) + W
    run_op(OPERATION(attention_self_query_weight, inference), loraWeightBase,
           loraWeightBase + LORA_WQ_A_SIZE,
           activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE, 0,
           weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE);

    // double *combined_query_weight =
    //     readFileAsDouble("combined_query_weight", 128 * 128, true);
    // int count = 0;
    // for (int i = 0; i < 128 * 128; i++) {
    //   if (fabs(combined_query_weight[i] -
    //            memory->sram[activationBase + INTERMEDIATE_SIZE +
    //                         2 * INTRA_BOTTLENECK_SIZE + i]) > 0.001) {
    //     // std::cout << "Mismatch at " << i << std::endl;
    //     count++;
    //     std::cout << combined_query_weight[i] << "/"
    //               << memory->sram[activationBase + INTERMEDIATE_SIZE +
    //                               2 * INTRA_BOTTLENECK_SIZE + i]
    //               << " ";
    //     // std::abort();
    //   }
    // }
    // std::cout << std::endl << count << " mismatches" << std::endl;
    // std::abort();

    // query projection
    run_op(OPERATION(attention_self_query_layer, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               6 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);
#else
    // query projection
    run_op(OPERATION(attention_self_query_layer, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               6 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    checkLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                     "mobilebert_encoder_layer_" +
                         std::to_string(encoderLayer) +
                         "_attention_self_query_layer");
#endif
    // key projection
    run_op(OPERATION(attention_self_key_layer, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               7 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               7 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    checkLayerOutput(
        activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_attention_self_key_layer");

#ifdef LORA

    // double *value_lora_A_weight =
    //     readFileAsDouble("value_lora_A_weight", 128 * 128, true);
    // for (int i = 0; i < 128 * 128; i++) {
    //   if (fabs(value_lora_A_weight[i] -
    //            memory->sram[loraWeightBase + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE
    //            +
    //                         i * 2]) > 0.001) {
    //     std::cout << "Mismatch at " << i << std::endl;
    //     std::cout << value_lora_A_weight[i] << " vs "
    //               << memory->sram[loraWeightBase + LORA_WQ_A_SIZE +
    //                               LORA_WQ_B_SIZE + i * 2]
    //               << std::endl;
    //     std::abort();
    //   }
    // }
    // std::abort();

    // LoRA value weight (A * B) + W
    run_op(OPERATION(attention_self_value_weight, inference),
           loraWeightBase + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE,
           loraWeightBase + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE,
           activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 0,
           weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE);

    // double *combined_value_weight =
    //     readFileAsDouble("combined_value_weight", 512 * 128, true);
    // for (int i = 0; i < 512 * 128; i++) {
    //   if (fabs(combined_value_weight[i] -
    //            memory->sram[activationBase + INTERMEDIATE_SIZE +
    //                         3 * INTRA_BOTTLENECK_SIZE + i]) > 0.001) {
    //     std::cout << "Mismatch at " << i << std::endl;
    //     std::cout << combined_value_weight[i] << " vs "
    //               << memory->sram[activationBase + INTERMEDIATE_SIZE +
    //                               3 * INTRA_BOTTLENECK_SIZE + i]
    //               << std::endl;
    //     std::abort();
    //   }
    // }
    // std::abort();
    // std::cout << "Value weights:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //   std::cout << memory->sram[activationBase + INTERMEDIATE_SIZE +
    //                             3 * INTRA_BOTTLENECK_SIZE + i]
    //             << std::endl;
    // }
    // std::abort();

    // std::cout << "Value inputs:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //   std::cout << memory->sram[encoderLayerInput + i] << std::endl;
    // }

    // std::cout << "Value bias:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //   std::cout << memory->rram[weightBase + 3 * INTERMEDIATE_SIZE +
    //                             2 * INTRA_BOTTLENECK_SIZE +
    //                             8 * INTRA_BOTTLENECK_BIAS_SIZE + i]
    //             << std::endl;
    // }

    // std::cout << "*******************" << std::endl;

    // value projection
    run_op(OPERATION(attention_self_value_layer, inference), encoderLayerInput,
           activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    // std::cout << "Value weights:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //   std::cout << memory->sram[activationBase + INTERMEDIATE_SIZE +
    //                             3 * INTRA_BOTTLENECK_SIZE + i]
    //             << std::endl;
    // }

    // std::cout << "Value output:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //   std::cout << memory->sram[activationBase + INTERMEDIATE_SIZE +
    //                             INTRA_BOTTLENECK_SIZE + i]
    //             << std::endl;
    // }
    // std::abort();
#else
    // value projection
    run_op(OPERATION(attention_self_value_layer, inference), encoderLayerInput,
           weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    checkLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_attention_self_value_layer");
#endif
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
             activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE, 0,
             0);

      if (variant == FORWARD_PASS_MHA_0 || variant == FORWARD_PASS_MHA_1 ||
          variant == FORWARD_PASS_MHA_2 || variant == FORWARD_PASS_MHA_3) {
        return;
      }

      // context
      run_op(OPERATION(attention_self_context_layer_0, inference),
             activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
             activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
                 head * ATTENTION_HEAD_SIZE,
             activationBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
                 head * ATTENTION_HEAD_SIZE,
             0, 0);

      checkLayerOutput(
          activationBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
              head * ATTENTION_HEAD_SIZE,
          128 * 32,
          "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
              "_attention_self_context_layer_" + std::to_string(head));
    }

    // bottleneck_attention_dense
    run_op(OPERATION(bottleneck_input_dense, inference), encoderLayerInput,
           weightBase,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + INTERMEDIATE_SIZE, 0);

    checkLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_bottleneck_input_dense");

    // bottleneck_attention_LayerNorm
    run_op(OPERATION(bottleneck_input_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_BIAS_SIZE, 0);

    checkLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                     "mobilebert_encoder_layer_" +
                         std::to_string(encoderLayer) +
                         "_bottleneck_input_LayerNorm");

    // attention_output_dense
    run_op(OPERATION(attention_output_dense, inference),
           activationBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               9 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
               9 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE);

    checkLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_attention_output_dense");

    // attention_output_LayerNorm
    run_op(OPERATION(attention_output_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
               10 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
               11 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    checkLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                     "mobilebert_encoder_layer_" +
                         std::to_string(encoderLayer) +
                         "_attention_output_LayerNorm");
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

  checkLayerOutput(
      activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 128 * 512,
      "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
          "_output_bottleneck_dense");

  // output_bottleneck_LayerNorm
  run_op(OPERATION(output_bottleneck_LayerNorm, inference),
         activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         weightBase + 8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         encoderLayerOutput,
         weightBase + 8 * INTERMEDIATE_SIZE + 4 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         0);

  checkLayerOutput(encoderLayerOutput, 128 * 512,
                   "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
                       "_output_bottleneck_LayerNorm");
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
  // the other variants run from a partial scratch space, so no need to start
  // at the checkpoint
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