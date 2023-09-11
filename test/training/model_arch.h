#pragma once

#define NUM_ENCODER_LAYERS 21
#define NUM_HEADS 4
#define NUM_FFN 2

#define NUM_CHECKPOINTS 2
#define CHECKPOINT_INTERVAL (NUM_ENCODER_LAYERS / NUM_CHECKPOINTS)

#define START_LR 2e-2
#define GRADIENT_ACCUMULATION_STEPS 16
#define MAX_TRAINING_STEPS (6900 * GRADIENT_ACCUMULATION_STEPS)

#define ATTENTION_HEAD_SIZE (128 * 32)
#define INTRA_BOTTLENECK_SIZE (128 * 128)
#define INTRA_BOTTLENECK_BIAS_SIZE (128 * 2)
#define INTERMEDIATE_SIZE (128 * 512)
#define INTERMEDIATE_BIAS_SIZE (512 * 2)
#define ENCODER_WEIGHT_SIZE                             \
  (8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE + \
   3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE)
#define LORA_WQ_B_SIZE (128 * 16 * 2)     // 16b precision
#define LORA_WQ_A_SIZE (128 * 16 * 2)     // 16b precision
#define LORA_WV_B_SIZE (128 * 16 * 2)     // 16b precision
#define LORA_WV_A_SIZE (512 * 16 * 2)     // 16b precision
#define CLASSIFIER_W_SIZE (512 * 16 * 2)  // 16b precision
#define CLASSIFIER_B_SIZE (16 * 2)        // 16b precision
