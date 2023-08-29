#pragma once
#include "memory_plan.h"

// #define CHECK_OUTPUT

#ifndef SOC
void checkBWDLayerOutput(int memoryOffset, int size, std::string file,
                         bool weightGradient = false) {
  std::string fullFileName =
      "models/mobilebert/binary_data/tiny_pretrained/step_0/" +
      std::string(weightGradient ? "weight_gradients/"
                                 : "activation_gradients/") +
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

void encoder_backward_pass(int encoderLayer, int step) {
  int activationBase = ENCODER_SCRATCH;
  int activationGradientBase = BACKPROP_SCRATCH;
  int weightBase = encoderLayer * ENCODER_WEIGHT_SIZE;
  int loraWeightOffset = encoderLayer * LORA_W_PER_ENC_SIZE;

  // These namings are weird, they actually refer to the layer before

  run_op(OPERATION(output_bottleneck_dense, backward),
         activationGradientBase + (encoderLayer == 20 ? INTERMEDIATE_SIZE : 0),
         weightBase + 8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase, 0, 0);

#ifdef CHECK_OUTPUT
  checkBWDLayerOutput(activationGradientBase, INTERMEDIATE_SIZE,
                      "mobilebert_encoder_layer_" +
                          std::to_string(encoderLayer) +
                          "_output_bottleneck_dense");
#endif

  run_op(OPERATION(output_LayerNorm, backward), activationGradientBase,
         weightBase + 7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE, 0, 0);

#ifdef CHECK_OUTPUT
  checkBWDLayerOutput(activationGradientBase + INTERMEDIATE_SIZE,
                      INTRA_BOTTLENECK_SIZE,
                      "mobilebert_encoder_layer_" +
                          std::to_string(encoderLayer) + "_output_LayerNorm");
#endif

  for (int ffn = NUM_FFN - 1; ffn >= 0; ffn--) {
    run_op(OPERATION(output_dense, backward),
           activationGradientBase + INTERMEDIATE_SIZE,
           weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
               3 * INTRA_BOTTLENECK_SIZE + 13 * INTRA_BOTTLENECK_BIAS_SIZE +
               ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                      3 * INTRA_BOTTLENECK_BIAS_SIZE),
           activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           0, 0);

    forward_pass_from_checkpoint(
        encoderLayer, ffn == NUM_FFN - 1 ? FORWARD_PASS_FFN_1_INTERMEDIATE
                                         : FORWARD_PASS_FFN_0_INTERMEDIATE);

    run_op(
        OPERATION(intermediate_dense, backward),
        activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
        weightBase + 4 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
            3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE +
            ffn * (2 * INTERMEDIATE_SIZE + +INTERMEDIATE_BIAS_SIZE +
                   3 * INTRA_BOTTLENECK_BIAS_SIZE),
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        0, activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE);

    run_op(
        OPERATION(ffn_0_output_LayerNorm, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
            12 * INTRA_BOTTLENECK_BIAS_SIZE +
            ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                   3 * INTRA_BOTTLENECK_BIAS_SIZE),
        activationGradientBase + INTERMEDIATE_SIZE, 0,
        activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE);
  }

  run_op(OPERATION(attention_output_dense, backward),
         activationGradientBase + INTERMEDIATE_SIZE,
         weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             10 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 0,
         0);

#ifdef CHECK_OUTPUT
  checkBWDLayerOutput(
      activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
      INTRA_BOTTLENECK_SIZE,
      "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
          "_attention_output_dense");
#endif

  run_op(OPERATION(attention_self_context_layer, backward),
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             9 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         0, 0);

#ifdef CHECK_OUTPUT
  checkBWDLayerOutput(
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
      INTRA_BOTTLENECK_SIZE,
      "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
          "_attention_self_context_layer");
#endif

  run_op(OPERATION(bottleneck_input_dense, backward),
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE, 0, 0);

  if (encoderLayer > 0)
#ifdef CHECK_OUTPUT
    checkBWDLayerOutput(
        activationGradientBase + INTERMEDIATE_SIZE, INTRA_BOTTLENECK_SIZE,
        "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
            "_bottleneck_input_dense");
#endif

  for (int head = 0; head < NUM_HEADS; head++) {
    forward_pass_from_checkpoint(encoderLayer, head == 0   ? FORWARD_PASS_MHA_0
                                               : head == 1 ? FORWARD_PASS_MHA_1
                                               : head == 2
                                                   ? FORWARD_PASS_MHA_2
                                                   : FORWARD_PASS_MHA_3);
    run_op(OPERATION(attention_self_value_layer_0, backward),
           activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
           activationGradientBase + INTERMEDIATE_SIZE +
               2 * INTRA_BOTTLENECK_SIZE + head * ATTENTION_HEAD_SIZE,
           activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               head * ATTENTION_HEAD_SIZE,
           0, 0);

    run_op(
        OPERATION(attention_self_attention_probs_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
        0, 0);

    run_op(
        OPERATION(attention_self_attention_scores_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
        0,
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        0, activationBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE);

    run_op(
        OPERATION(attention_self_query_layer_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        0, 0);

    run_op(
        OPERATION(attention_self_key_layer_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        activationBase + INTERMEDIATE_SIZE + head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        0, 0);
  }

  /* ========================== query lora ========================== */

  // recompute bottleneck attention LayerNorm
  forward_pass_from_checkpoint(encoderLayer, FORWARD_PASS_BOTTLENECK_ATTENTION);

  // query weight gradient
  run_op(OPERATION(attention_self_query_weight, gradient),
         activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  // quantize lora_A weight
  run_op(OPERATION(attention_self_query_lora_A_quant, gradient),
         LORA_W + loraWeightOffset, 0,
         activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  // query lora_B gradient
  run_op(OPERATION(attention_self_query_lora_B_weight, gradient),
          activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
          activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
          LORA_G + loraWeightOffset + LORA_WQ_A_SIZE,
          0,
          LORA_G + loraWeightOffset + LORA_WQ_A_SIZE);

  // std::cerr << "query lora_B gradient" << std::endl;
  // output_loc = LORA_G + loraWeightOffset + LORA_WQ_A_SIZE;
  // for (int i = 0; i < 128; i++) {
  //   for (int j = 0; j < 16; j++) {
  //     int offset = i * 16 + j;
  //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
  //   }
  //   std::cerr << std::endl;
  // }
  // std::cerr << std::endl;

  // quantize lora_B weight
  run_op(OPERATION(attention_self_query_lora_B_quant, gradient),
         LORA_W + loraWeightOffset + LORA_WQ_A_SIZE, 0,
         activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  // query lora_A gradient
  run_op(OPERATION(attention_self_query_lora_A_weight, gradient),
          activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
          activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
          LORA_G + loraWeightOffset,
          0,
          LORA_G + loraWeightOffset);

  // std::cerr << "query lora_A gradient" << std::endl;
  // output_loc = LORA_G + loraWeightOffset;
  // for (int i = 0; i < 128; i++) {
  //   for (int j = 0; j < 16; j++) {
  //     int offset = i * 16 + j;
  //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
  //   }
  //   std::cerr << std::endl;
  // }
  // std::cerr << std::endl << std::endl;

  if (step % GRADIENT_ACCUMULATION_STEPS == 0) {
    // query lora_A weight update
    run_op(OPERATION(attention_self_query_lora_A_weight, weight),
            LORA_G + loraWeightOffset,
            LORA_W + loraWeightOffset, 0, 0, 0, step);

    // query lora_B weight update
    run_op(OPERATION(attention_self_query_lora_B_weight, weight),
            LORA_G + loraWeightOffset + LORA_WQ_A_SIZE,
            LORA_W + loraWeightOffset + LORA_WQ_A_SIZE, 0, 0, 0, step);

    // std::cerr << "query lora_A weight" << std::endl;
    // output_loc = LORA_W + loraWeightOffset;
    // for (int i = 0; i < 128; i++) {
    //   for (int j = 0; j < 16; j++) {
    //     int offset = i * 16 + j;
    //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
    //   }
    //   std::cerr << std::endl;
    // }
    // std::cerr << std::endl << std::endl;

    // std::cerr << "query lora_B weight" << std::endl;
    // output_loc = LORA_W + loraWeightOffset + LORA_WQ_A_SIZE;
    // for (int i = 0; i < 128; i++) {
    //   for (int j = 0; j < 16; j++) {
    //     int offset = i * 16 + j;
    //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
    //   }
    //   std::cerr << std::endl;
    // }
    // std::cerr << std::endl << std::endl;
  }

  /* ========================== query lora ========================== */

  run_op(OPERATION(query_to_bottleneck_attention_LayerNorm, backward),
         activationGradientBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
         weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  run_op(
      OPERATION(key_to_bottleneck_attention_LayerNorm, backward),
      activationGradientBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE,
      weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
          7 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE, 0,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE);

  run_op(OPERATION(bottleneck_attention_dense, backward),
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         0, 0);

#ifdef CHECK_OUTPUT
  checkBWDLayerOutput(
      activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
      INTRA_BOTTLENECK_SIZE,
      "mobilebert_encoder_layer_" + std::to_string(encoderLayer) +
          "_bottleneck_attention_dense");
#endif

  run_op(OPERATION(bottlenecked_hidden_states, backward),
         activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         weightBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase, 0, activationGradientBase);
  run_op(OPERATION(shared_attention_input_to_hidden_states, backward),
         activationGradientBase + INTERMEDIATE_SIZE, weightBase,
         activationGradientBase, 0, activationGradientBase);
  run_op(OPERATION(value_to_hidden_states, backward),
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             8 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase, 0, activationGradientBase);

#ifdef CHECK_OUTPUT
  checkBWDLayerOutput(activationGradientBase, INTERMEDIATE_SIZE,
                      "mobilebert_encoder_layer_" +
                          std::to_string(encoderLayer) + "_hidden_states");
#endif

  /* ========================== value lora gradient ==========================
   */

  int encoderLayerInput;
  if (encoderLayer == 0) {
    encoderLayerInput = INPUT;
  } else if (encoderLayer % 5 == 0) {  // layers 5, 10, 15, 20
    encoderLayerInput = CHECKPOINT + (encoderLayer / 5 - 1) * INPUT_SIZE;
  } else {
    encoderLayerInput = activationBase;
  }

  // value weight gradient
  run_op(OPERATION(attention_self_value_weight, gradient), encoderLayerInput,
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  // quantize value lora_A weight
  run_op(OPERATION(attention_self_value_lora_A_quant, gradient),
         LORA_W + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE, 0,
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  // value lora_B gradient
  run_op(OPERATION(attention_self_value_lora_B_weight, gradient),
          activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
          activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
          LORA_G + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE,
          0,
          LORA_G + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE);

  // std::cerr << "value lora_B gradient" << std::endl;
  // output_loc = LORA_G + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE;
  // for (int i = 0; i < 128; i++) {
  //   for (int j = 0; j < 16; j++) {
  //     int offset = i * 16 + j;
  //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
  //   }
  //   std::cerr << std::endl;
  // }
  // std::cerr << std::endl;

  // quantize lora_B weight
  run_op(OPERATION(attention_self_value_lora_B_quant, gradient),
         LORA_W + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE +
             LORA_WV_A_SIZE,
         0,
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  // value lora_A gradient
  run_op(OPERATION(attention_self_value_lora_A_weight, gradient),
          activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
          activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
          LORA_G + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE,
          0,
          LORA_G + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE);

  // std::cerr << "value lora_A gradient" << std::endl;
  // output_loc = LORA_G + loraWeightOffset;
  // for (int i = 0; i < 128; i++) {
  //   for (int j = 0; j < 16; j++) {
  //     int offset = i * 16 + j;
  //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
  //   }
  //   std::cerr << std::endl;
  // }
  // std::cerr << std::endl << std::endl;

  if (step % GRADIENT_ACCUMULATION_STEPS == 0) {
    // value lora_A weight update
    run_op(OPERATION(attention_self_value_lora_A_weight, weight),
            LORA_G + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE,
            LORA_W + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE, 0, 0, 0, step);

    // value lora_B weight update
    run_op(OPERATION(attention_self_value_lora_B_weight, weight),
            LORA_G + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE,
            LORA_W + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE, 0, 0, 0, step);

    // std::cerr << "value lora_A weight" << std::endl;
    // output_loc = LORA_W + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE;
    // for (int i = 0; i < 512; i++) {
    //   for (int j = 0; j < 16; j++) {
    //     int offset = i * 16 + j;
    //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
    //   }
    //   std::cerr << std::endl;
    // }
    // std::cerr << std::endl << std::endl;

    // std::cerr << "value lora_B weight" << std::endl;
    // output_loc = LORA_W + loraWeightOffset + LORA_WQ_A_SIZE + LORA_WQ_B_SIZE + LORA_WV_A_SIZE;
    // for (int i = 0; i < 128; i++) {
    //   for (int j = 0; j < 16; j++) {
    //     int offset = i * 16 + j;
    //     std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
    //   }
    //   std::cerr << std::endl;
    // }
    // std::cerr << std::endl << std::endl;
  }

  /* ========================== value lora gradient ========================== */
}

void full_backward_pass(int step = 0) {
  // cross-entropy gradient
  run_op(OPERATION(classifier, backward), ENCODER_SCRATCH + INTERMEDIATE_SIZE,
         LABEL, BACKPROP_SCRATCH, 0, 0);

  // for (int j = 0; j < 16; j++) {
  //   std::cerr << memory->sram[BACKPROP_SCRATCH + j] << '\t';
  // }
  // std::cerr << std::endl << std::endl;

  // classifier weight gradient
  run_op(OPERATION(classifier_weight, gradient), BACKPROP_SCRATCH,
         ENCODER_SCRATCH, CLASSIFIER_G, 0, 0);

  // std::cerr << "classifier weight gradient" << std::endl;
  // for (int i = 0; i < 16; i++) {
  //   for (int j = 0; j < 512; j++) {
  //     int offset = i * 512 + j;
  //     std::cerr << memory->sram[CLASSIFIER_G + 2 * offset] << '\t';
  //   }
  //   std::cerr << std::endl;
  // }

  // std::cerr << "classifier orig weight" << std::endl;
  // for (int i = 0; i < 16; i++) {
  //   for (int j = 0; j < 512; j++) {
  //     int offset = i * 512 + j;
  //     std::cerr << memory->sram[CLASSIFIER_W + 2 * offset] << '\t';
  //   }
  //   std::cerr << std::endl;
  // }

  // classifier weight update
  if (step % GRADIENT_ACCUMULATION_STEPS == 0) {
    run_op(OPERATION(classifier_weight, weight),
          CLASSIFIER_G, CLASSIFIER_W, 0, 0, 0, step);
  }

  // std::cerr << "classifier weight update" << std::endl;
  // for (int i = 0; i < 16; i++) {
  //   for (int j = 0; j < 512; j++) {
  //     int offset = i * 512 + j;
  //     std::cerr << memory->sram[CLASSIFIER_W + 2 * offset] << '\t';
  //   }
  //   std::cerr << std::endl;
  // }

  // classifier bias
  run_op(OPERATION(classifier_bias, gradient), BACKPROP_SCRATCH, 0,
         CLASSIFIER_G + CLASSIFIER_W_SIZE, 0, 0);

  // classifier weight update
  if (step % GRADIENT_ACCUMULATION_STEPS == 0) {
    run_op(OPERATION(classifier_bias, weight),
          CLASSIFIER_G + CLASSIFIER_W_SIZE,
          CLASSIFIER_W + CLASSIFIER_W_SIZE,
          0, 0, 0, step);
  }

  // std::cerr << "classifier bias update" << std::endl;
  // for (int i = 0; i < 16; i++) {
  //   std::cerr << memory->sram[CLASSIFIER_W + CLASSIFIER_W_SIZE + 2 * i] << '\t';
  // }
  // std::cerr << std::endl;

  // classifier gradient
  run_op(OPERATION(output_bottleneck_LayerNorm, backward), BACKPROP_SCRATCH,
         (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
             8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         BACKPROP_SCRATCH + INTERMEDIATE_SIZE, 0, 0);

  for (int encoderLayer = NUM_ENCODER_LAYERS - 1; encoderLayer >= 0;
       encoderLayer--) {
    encoder_backward_pass(encoderLayer, step);
  }
}