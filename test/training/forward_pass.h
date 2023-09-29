#pragma once

#include "memory_plan.h"

#define LORA

// #define CHECK_OUTPUT

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

#ifndef SOC
void checkFWDLayerOutput(int memoryOffset, int size, std::string file) {
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
#endif

void debug_print(char *str) {
#ifdef SOC
  gdb_print(str);
#else
  std::cout << str << std::endl;
#endif
}

void print_memory(int address, bool SRAM) {
#ifdef SOC
  if (SRAM) {
    gdb_print_memory(SRAM_BASE + address, 16, P8);
  } else {
    gdb_print_memory(RRAM_BASE + address, 16, P8);
  }

  gdb_sw_breakpoint();
#else
  if (SRAM) {
    for (int i = 0; i < 16; i++) {
      std::cout << memory->sram[address + i] << " ";
    }
  } else {
    for (int i = 0; i < 16; i++) {
      std::cout << memory->rram[address + i] << " ";
    }
  }
  std::cout << std::endl;
#endif
}

void dump_memory(int address, int size, char *file) {
#ifdef SOC
  gdb_dump_memory(address, size, file);
#else
  std::ofstream myfile(std::string(file));
  for (int i = 0; i < size; i++) {
    myfile << memory->sram[address + i].bits << std::endl;
  }
  myfile.close();
#endif
}

void encoder_forward_pass(int encoderLayer, ForwardPassVariant variant) {
  int activationBase = ENCODER_SCRATCH;
  int weightBase = encoderLayer * ENCODER_WEIGHT_SIZE;
  int loraWeightBase = LORA_W + encoderLayer * LORA_W_PER_ENC_SIZE;

  // Handle checkpointing
  int encoderLayerOutput;
  if (encoderLayer % CHECKPOINT_INTERVAL == CHECKPOINT_INTERVAL - 1) {
    encoderLayerOutput =
        CHECKPOINT + (encoderLayer / CHECKPOINT_INTERVAL) * INPUT_SIZE;
  } else {
    encoderLayerOutput = activationBase;
  }
  int encoderLayerInput;
  if (encoderLayer == 0) {
    encoderLayerInput = INPUT;
  } else if (encoderLayer % CHECKPOINT_INTERVAL == 0) {
    encoderLayerInput =
        CHECKPOINT + (encoderLayer / CHECKPOINT_INTERVAL - 1) * INPUT_SIZE;
  } else {
    encoderLayerInput = activationBase;
  }

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

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                        "mobilebert_encoder_layer_" +
                            std::to_string(encoderLayer) +
                            "_bottleneck_attention_dense");
#endif

    // bottleneck_input_LayerNorm
    run_op(OPERATION(bottleneck_attention_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_bottleneck_attention_LayerNorm");
#endif

    if (variant == FORWARD_PASS_BOTTLENECK_ATTENTION) {
      return;
    }

#ifdef LORA

    // gdb_sw_breakpoint();

    // LoRA query weight (A * B) + W
    run_op(OPERATION(attention_self_query_weight, inference), loraWeightBase,
           loraWeightBase + LORA_WQ_A_SIZE,
           activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE, 0,
           weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE);

    // debug_print("lora_A");
    // print_memory(loraWeightBase, true);

    // debug_print("lora_B");
    // print_memory(loraWeightBase + LORA_WQ_A_SIZE, true);

    // debug_print("lora_query_weight");
    // print_memory(activationBase + INTERMEDIATE_SIZE + 2 *
    // INTRA_BOTTLENECK_SIZE,
    //              true);

    // query_projection
    run_op(OPERATION(attention_self_query_layer, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               6 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    // debug_print("query_projection");
    // print_memory(activationBase + INTERMEDIATE_SIZE, true);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                        "mobilebert_encoder_layer_" +
                            std::to_string(encoderLayer) +
                            "_attention_self_query_layer");
#endif
#else
    // query projection
    run_op(OPERATION(attention_self_query_layer, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               6 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                        "mobilebert_encoder_layer_" +
                            std::to_string(encoderLayer) +
                            "_attention_self_query_layer");
#endif
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

    // debug_print("key_projection");
    // print_memory(activationBase + INTERMEDIATE_SIZE + 2 *
    // INTRA_BOTTLENECK_SIZE,
    //              true);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(
        activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_attention_self_key_layer");
#endif

#ifdef LORA

    // LoRA value weight (A * B) + W
    run_op(OPERATION(attention_self_value_weight, inference),
           loraWeightBase + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE,
           loraWeightBase + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE,
           activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 0,
           weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE);

    // debug_print("lora_A");
    // print_memory(loraWeightBase + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE, true);

    // debug_print("lora_B");
    // print_memory(
    //     loraWeightBase + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE,
    //     true);

    // debug_print("lora_value_weight tile 0");
    // print_memory(activationBase + INTERMEDIATE_SIZE + 3 *
    // INTRA_BOTTLENECK_SIZE,
    //              true);

    // debug_print("lora_value_weight tile 1");
    // print_memory(activationBase + INTERMEDIATE_SIZE +
    //                  3 * INTRA_BOTTLENECK_SIZE + 128 * 16,
    //              true);

    // debug_print("lora_value_weight tile 2");
    // print_memory(activationBase + INTERMEDIATE_SIZE +
    //                  3 * INTRA_BOTTLENECK_SIZE + 128 * 2 * 16,
    //              true);

    // debug_print("lora_value_weight tile 3");
    // print_memory(activationBase + INTERMEDIATE_SIZE +
    //                  3 * INTRA_BOTTLENECK_SIZE + 128 * 3 * 16,
    //              true);

    // value projection
    run_op(OPERATION(attention_self_value_layer, inference), encoderLayerInput,
           activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

    // debug_print("value_projection");
    // print_memory(activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
    //              true);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_attention_self_value_layer");
#endif

#else
    // value projection
    run_op(OPERATION(attention_self_value_layer, inference), encoderLayerInput,
           weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               8 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_attention_self_value_layer");
#endif
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

#ifdef CHECK_OUTPUT
      checkFWDLayerOutput(
          activationBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
              head * ATTENTION_HEAD_SIZE,
          128 * 32,
          "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
              "_attention_self_context_layer_" + std::to_string(head));
#endif
    }

    // bottleneck_attention_dense
    run_op(OPERATION(bottleneck_input_dense, inference), encoderLayerInput,
           weightBase,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + INTERMEDIATE_SIZE, 0);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_bottleneck_input_dense");
#endif

    // bottleneck_attention_LayerNorm
    run_op(OPERATION(bottleneck_input_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_BIAS_SIZE, 0);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                        "mobilebert_encoder_layer_" +
                            std::to_string(encoderLayer) +
                            "_bottleneck_input_LayerNorm");
#endif

    // attention_output_dense
    run_op(OPERATION(attention_output_dense, inference),
           activationBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
               9 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
               9 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 128 * 128,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_attention_output_dense");
#endif

    // attention_output_LayerNorm
    run_op(OPERATION(attention_output_LayerNorm, inference),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
               10 * INTRA_BOTTLENECK_BIAS_SIZE,
           activationBase + INTERMEDIATE_SIZE,
           weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
               11 * INTRA_BOTTLENECK_BIAS_SIZE,
           0);

#ifdef CHECK_OUTPUT
    checkFWDLayerOutput(activationBase + INTERMEDIATE_SIZE, 128 * 128,
                        "mobilebert_encoder_layer_" +
                            std::to_string(encoderLayer) +
                            "_attention_output_LayerNorm");
#endif
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

#ifdef CHECK_OUTPUT
  checkFWDLayerOutput(
      activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE, 128 * 512,
      "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
          "_output_bottleneck_dense");
#endif

  // output_bottleneck_LayerNorm
  run_op(OPERATION(output_bottleneck_LayerNorm, inference),
         activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         weightBase + 8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         encoderLayerOutput,
         weightBase + 8 * INTERMEDIATE_SIZE + 4 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         0);

  // debug_print("encoder_layer_output");
  // print_memory(encoderLayerOutput, true);

#ifdef CHECK_OUTPUT
  checkFWDLayerOutput(encoderLayerOutput, 128 * 512,
                      "mobilebert_encoder_layer_" +
                          std::to_string(encoderLayer) +
                          "_output_bottleneck_LayerNorm");
#endif
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
  run_op(OPERATION(classifier, inference), ENCODER_SCRATCH, CLASSIFIER_W,
         ENCODER_SCRATCH + INTERMEDIATE_SIZE, CLASSIFIER_B, 0);

#ifndef SOC
  std::cout << "Classifier Output: " << std::endl;
  for (int i = 0; i < 2; i++) {
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
    int closestCheckpoint =
        endEncoderLayer / CHECKPOINT_INTERVAL * CHECKPOINT_INTERVAL;
    for (int encoderLayer = closestCheckpoint; encoderLayer < endEncoderLayer;
         encoderLayer++) {
      encoder_forward_pass(encoderLayer, COMPLETE_FORWARD_PASS);
    }
  }
  encoder_forward_pass(endEncoderLayer, variant);
}