#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

using namespace std;

map<string, SimplifiedParams> inferenceParams{
    // (128 x 512) * (512 x 128)
    {"inputBottleneck",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         131072,                                      // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{4, 1, 2, 1, 1, 1}, {32, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {5, 5},                                      // weightReuseIndex
         1,                                           // stride
         false,                                       // replication
         false,                                       // RELU
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // NO-NORM
         true,                                        // weight
         false,                                       // ATTENTION_SCALING
         false,                                       // STORE_IN_ACC
         false,                                       // ACC_FROM_ACC
         false,                                       // INPUT_TRANSPOSE
         false,                                       // SPLIT_HEAD
         false,                                       // CONCAT_HEAD
     }},

    // elementwise product and addition for matrix of size:
    // (128 x 128) * 128
    {"bottleneckLayerNorm",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         10 * 1024,                                   // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {2, 0},                                      // REDUCTION
         {3, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         1,                                           // stride
         false,                                       // replication
         false,                                       // RELU
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         true,                                        // NO_NORM
         true,                                        // weight
         false,                                       // ATTENTION_SCALING
         false,                                       // STORE_IN_ACC
         false,                                       // ACC_FROM_ACC
         false,                                       // INPUT_TRANSPOSE
         false,                                       // SPLIT_HEAD
         false,                                       // CONCAT_HEAD
     }},

    // (128 x 128) x (128 x 128)
    {"qkProjection",
     {
         131072,                                     // INPUT_OFFSET
         65536,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
         true,                                       // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         true,                                       // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

    {"vProjection",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         131072,                                      // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{4, 1, 2, 1, 1, 1}, {32, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {5, 5},                                      // weightReuseIndex
         1,                                           // stride
         false,                                       // replication
         false,                                       // RELU
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // NO-NORM
         true,                                        // weight
         false,                                       // ATTENTION_SCALING
         false,                                       // STORE_IN_ACC
         false,                                       // ACC_FROM_ACC
         false,                                       // INPUT_TRANSPOSE
         true,                                        // SPLIT_HEAD
         false,                                       // CONCAT_HEAD
     }},

    // Q x K
    // (128 x 32) * (32 x 128)
    {"qkAttention",
     {
         0,                                          // INPUT_OFFSET
         69632,                                      // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         true,                                       // TRANSPOSE
         {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
         false,                                      // weight
         true,                                       // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

    // Attention probability
    // (128 x 128)
    {"softmax",
     {
         0,                                             // INPUT_OFFSET
         0,                                             // WEIGHT_OFFSET
         128 * 32,                                      // OUTPUT_OFFSET
         false,                                         // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
         {0, 5},                                        // INPUTX
         {1, 4},                                        // INPUTY
         {3, 0},                                        // REDUCTION
         {2, 1},                                        // WEIGHT
         3,                                             // fxIndex
         2,                                             // fyIndex
         {5, 5},                                        // weightReuseIndex
         1,                                             // stride
         false,                                         // replication
         false,                                         // RELU
         false,                                         // bias
         30 * 1024,                                     // BIAS_OFFSET
         false,                                         // residual
         40 * 1024,                                     // RESIDUAL_OFFSET
         false,                                         // maxpool
         false,                                         // avgpool
         true,                                          // SOFTMAX
         false,                                         // FC
         false,                                         // NO-NORM
         false,                                         // weight
         false,                                         // ATTENTION_SCALING
         false,                                         // STORE_IN_ACC
         false,                                         // ACC_FROM_ACC
         false,                                         // INPUT_TRANSPOSE
         false,                                         // SPLIT_HEAD
         false,                                         // CONCAT_HEAD
     }},

    // QK x V
    // (128 x 128) * (128 x 32)
    {"vAttention",
     {
         131072,                                     // INPUT_OFFSET
         73728,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
         false,                                      // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

    // Self-attention output
    // (128 x 128) x (128 x 128)
    {"outputAttention",
     {
         0,                                          // INPUT_OFFSET
         77824,                                      // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         true,                                       // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
         true,                                       // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         true,                                       // CONCAT_HEAD
     }},

    // FFN 1
    // (128 x 128) * (128 x 512)
    {"ffn1",
     {
         131072,                                     // INPUT_OFFSET
         94208,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         true,                                       // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
         true,                                       // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

    // FFN 2
    // (128 x 512) x (512 x 128)
    {"ffn2",
     {
         0,                                           // INPUT_OFFSET
         159744,                                      // WEIGHT_OFFSET
         131072,                                      // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {32, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {5, 5},                                      // weightReuseIndex
         1,                                           // stride
         false,                                       // replication
         false,                                       // RELU
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         true,                                        // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // NO-NORM
         true,                                        // weight
         false,                                       // ATTENTION_SCALING
         false,                                       // STORE_IN_ACC
         false,                                       // ACC_FROM_ACC
         false,                                       // INPUT_TRANSPOSE
         false,                                       // SPLIT_HEAD
         false,                                       // CONCAT_HEAD
     }},

    // output bottleneck
    // (128 x 128) x (128 x 512)
    {"outputBottleneck",
     {
         131072,                                     // INPUT_OFFSET
         225280,                                     // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         true,                                       // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
         true,                                       // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

    // elementwise product and addition for matrix of size:
    // 128 x 512
    {"outputLayerNorm",
     {
         0,                                             // INPUT_OFFSET
         0,                                             // WEIGHT_OFFSET
         10 * 1024,                                     // OUTPUT_OFFSET
         false,                                         // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 128}},  // LOOPS
         {0, 5},                                        // INPUTX
         {1, 4},                                        // INPUTY
         {2, 0},                                        // REDUCTION
         {3, 1},                                        // WEIGHT
         3,                                             // fxIndex
         2,                                             // fyIndex
         {4, 5},                                        // weightReuseIndex
         1,                                             // stride
         false,                                         // replication
         false,                                         // RELU
         true,                                          // bias
         30 * 1024,                                     // BIAS_OFFSET
         false,                                         // residual
         40 * 1024,                                     // RESIDUAL_OFFSET
         false,                                         // maxpool
         false,                                         // avgpool
         false,                                         // SOFTMAX
         false,                                         // FC
         true,                                          // NO_NORM
         true,                                          // weight
         false,                                         // ATTENTION_SCALING
         false,                                         // STORE_IN_ACC
         false,                                         // ACC_FROM_ACC
         false,                                         // INPUT_TRANSPOSE
         false,                                         // SPLIT_HEAD
         false,                                         // CONCAT_HEAD
     }},

    // (1 x 512) * (512 x 2)
    {"classifier",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         10 * 1024,                                  // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 1, 1}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         true,                                       // FC
         false,                                      // NO_NORM
         true,                                       // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},
};

vector<pair<string, string>> inferenceOperations{
    {"bottleneck_input_dense", "inputBottleneck"},
    {"bottleneck_input_LayerNorm", "bottleneckLayerNorm"},
    {"bottleneck_attention_dense", "inputBottleneck"},
    {"bottleneck_attention_LayerNorm", "bottleneckLayerNorm"},
    {"attention_self_query", "qkProjection"},
    {"attention_self_key", "qkProjection"},
    {"attention_self_attention_scores_0", "qkAttention"},
    {"attention_self_attention_scores_1", "qkAttention"},
    {"attention_self_attention_scores_2", "qkAttention"},
    {"attention_self_attention_scores_3", "qkAttention"},
    {"attention_self_attention_probs_0", "softmax"},
    {"attention_self_attention_probs_1", "softmax"},
    {"attention_self_attention_probs_2", "softmax"},
    {"attention_self_attention_probs_3", "softmax"},
    {"attention_self_value", "vProjection"},
    {"attention_self_context_layer_0", "vAttention"},
    {"attention_self_context_layer_1", "vAttention"},
    {"attention_self_context_layer_2", "vAttention"},
    {"attention_self_context_layer_3", "vAttention"},
    {"attention_output_dense", "outputAttention"},
    {"attention_output_LayerNorm", "bottleneckLayerNorm"},
    {"ffn_0_intermediate_dense", "ffn1"},
    {"ffn_0_output_dense", "ffn2"},
    {"ffn_0_output_LayerNorm", "bottleneckLayerNorm"},
    {"ffn_1_intermediate_dense", "ffn1"},
    {"ffn_1_output_dense", "ffn2"},
    {"ffn_1_output_LayerNorm", "bottleneckLayerNorm"},
    {"ffn_2_intermediate_dense", "ffn1"},
    {"ffn_2_output_dense", "ffn2"},
    {"ffn_2_output_LayerNorm", "bottleneckLayerNorm"},
    {"intermediate_dense", "ffn1"},
    {"output_dense", "ffn2"},
    {"output_LayerNorm", "bottleneckLayerNorm"},
    {"output_bottleneck_dense", "outputBottleneck"},
    {"output_bottleneck_LayerNorm", "outputLayerNorm"},
    {"classifier", "classifier"},
};

map<string, MemoryOffsets> inferenceMemOffsets{
    {"bottleneck_input_dense",
     {
         0,
         0,
         4 * HIDDEN_SIZE,
         WEIGHT_INTERMEDIATE_SIZE,
     }},
    {"bottleneck_input_LayerNorm",
     {
         4 * HIDDEN_SIZE,
         WEIGHT_INTERMEDIATE_SIZE + BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
         WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_dense",
     {
         0,
         WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_LayerNorm",
     {
         4 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_HIDDEN_SIZE,
         6 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_query",
     {
         6 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 6 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             6 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_key",
     {
         6 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
         7 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_0",
     {
         4 * HIDDEN_SIZE,
         7 * HIDDEN_SIZE,
         8 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_1",
     {
         4 * HIDDEN_SIZE + HEAD_SIZE,
         7 * HIDDEN_SIZE + HEAD_SIZE,
         9 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_2",
     {
         4 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         7 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         10 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_3",
     {
         4 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         7 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         11 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_0",
     {
         8 * HIDDEN_SIZE,
         -1,
         12 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_1",
     {
         9 * HIDDEN_SIZE,
         -1,
         13 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_2",
     {
         10 * HIDDEN_SIZE,
         -1,
         14 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_3",
     {
         11 * HIDDEN_SIZE,
         -1,
         15 * HIDDEN_SIZE,
     }},
    {"attention_self_value",
     {
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_context_layer_0",
     {
         12 * HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         6 * HIDDEN_SIZE,
     }},
    {"attention_self_context_layer_1",
     {
         13 * HIDDEN_SIZE,
         4 * HIDDEN_SIZE + 1 * HEAD_SIZE,
         6 * HIDDEN_SIZE + 1 * HEAD_SIZE,
     }},
    {"attention_self_context_layer_2",
     {
         14 * HIDDEN_SIZE,
         4 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         6 * HIDDEN_SIZE + 2 * HEAD_SIZE,
     }},
    {"attention_self_context_layer_3",
     {
         15 * HIDDEN_SIZE,
         4 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         6 * HIDDEN_SIZE + 3 * HEAD_SIZE,
     }},
    {"attention_output_dense",
     {
         6 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
     }},
    {"attention_output_LayerNorm",
     {
         4 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             10 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             11 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_intermediate_dense",
     {
         5 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
         6 * HIDDEN_SIZE,
         4 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_output_dense",
     {
         6 * HIDDEN_SIZE,
         4 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
     }},
    {"ffn_0_output_LayerNorm",
     {
         4 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 13 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 14 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_intermediate_dense",
     {
         5 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         6 * HIDDEN_SIZE,
         6 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_output_dense",
     {
         6 * HIDDEN_SIZE,
         6 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
     }},
    {"ffn_1_output_LayerNorm",
     {
         4 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 16 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 17 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_intermediate_dense",
     {
         5 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         6 * HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_output_dense",
     {
         6 * HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
     }},
    {"ffn_2_output_LayerNorm",
     {
         4 * HIDDEN_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 19 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 20 * BIAS_HIDDEN_SIZE,
     }},
    {"intermediate_dense",
     {
         5 * HIDDEN_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
         6 * HIDDEN_SIZE,
         10 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
     }},
    {"output_dense",
     {
         6 * HIDDEN_SIZE,
         10 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
         4 * HIDDEN_SIZE,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
     }},
    {"output_LayerNorm",
     {
         4 * HIDDEN_SIZE,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 22 * BIAS_HIDDEN_SIZE,
         5 * HIDDEN_SIZE,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 23 * BIAS_HIDDEN_SIZE,
     }},
    {"output_bottleneck_dense",
     {
         5 * HIDDEN_SIZE,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
         6 * HIDDEN_SIZE,
         12 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
         0,
     }},
    {"output_bottleneck_LayerNorm",
     {
         6 * HIDDEN_SIZE,
         12 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
         0,
         12 * WEIGHT_INTERMEDIATE_SIZE + 6 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"classifier",
     {
         0,
         0,
         4 * HIDDEN_SIZE,
         16 * BIAS_INTERMEDIATE_SIZE,
     }},
};

std::map<std::string, Files> inferenceTestFiles{
    // Input Bottleneck
    {"bottleneck_input_dense",
     {
         "hidden_states",
         "bottleneck_input_dense_weight",
         "bottleneck_input_dense_bias",
         "bottleneck_input_dense",
     }},
    {"bottleneck_input_LayerNorm",
     {
         "bottleneck_input_dense",
         "bottleneck_input_LayerNorm_weight",
         "bottleneck_input_LayerNorm_bias",
         "bottleneck_input_LayerNorm",
     }},
    {"bottleneck_attention_dense",
     {
         "hidden_states",
         "bottleneck_attention_dense_weight",
         "bottleneck_attention_dense_bias",
         "bottleneck_attention_dense",
     }},
    {"bottleneck_attention_LayerNorm",
     {
         "bottleneck_attention_dense",
         "bottleneck_attention_LayerNorm_weight",
         "bottleneck_attention_LayerNorm_bias",
         "bottleneck_attention_LayerNorm",
     }},

    // QKV Projection
    {"attention_self_query",
     {
         "bottleneck_attention_LayerNorm",
         "attention_self_query_weight",
         "attention_self_query_bias",
         "attention_self_query_layer",
     }},
    {"attention_self_key",
     {
         "bottleneck_attention_LayerNorm",
         "attention_self_key_weight",
         "attention_self_key_bias",
         "attention_self_key_layer",
     }},
    {"attention_self_value",
     {
         "hidden_states",
         "attention_self_value_weight",
         "attention_self_value_bias",
         "attention_self_value_layer",
     }},

    // Multi-Head Self Attention
    {"attention_self_attention_scores_0",
     {
         "attention_self_query_layer_0",
         "attention_self_key_layer_0",
         "",
         "attention_self_attention_scores_scaled_0",
     }},
    {"attention_self_attention_scores_1",
     {
         "attention_self_query_layer_1",
         "attention_self_key_layer_1",
         "",
         "attention_self_attention_scores_scaled_1",
     }},
    {"attention_self_attention_scores_2",
     {
         "attention_self_query_layer_2",
         "attention_self_key_layer_2",
         "",
         "attention_self_attention_scores_scaled_2",
     }},
    {"attention_self_attention_scores_3",
     {
         "attention_self_query_layer_3",
         "attention_self_key_layer_3",
         "",
         "attention_self_attention_scores_scaled_3",
     }},

    {"attention_self_attention_probs_0",
     {
         "attention_self_attention_scores_scaled_0",
         "",
         "",
         "attention_self_attention_probs_0",
     }},
    {"attention_self_attention_probs_1",
     {
         "attention_self_attention_scores_scaled_1",
         "",
         "",
         "attention_self_attention_probs_1",
     }},
    {"attention_self_attention_probs_2",
     {
         "attention_self_attention_scores_scaled_2",
         "",
         "",
         "attention_self_attention_probs_2",
     }},
    {"attention_self_attention_probs_3",
     {
         "attention_self_attention_scores_scaled_3",
         "",
         "",
         "attention_self_attention_probs_3",
     }},

    {"attention_self_context_layer_0",
     {
         "attention_self_attention_probs_0",
         "attention_self_value_layer_0",
         "",
         "attention_self_context_layer_0",
     }},
    {"attention_self_context_layer_1",
     {
         "attention_self_attention_probs_1",
         "attention_self_value_layer_1",
         "",
         "attention_self_context_layer_1",
     }},
    {"attention_self_context_layer_2",
     {
         "attention_self_attention_probs_2",
         "attention_self_value_layer_2",
         "",
         "attention_self_context_layer_2",
     }},
    {"attention_self_context_layer_3",
     {
         "attention_self_attention_probs_3",
         "attention_self_value_layer_3",
         "",
         "attention_self_context_layer_3",
     }},

    {"attention_output_dense",
     {
         "attention_self_context_layer_unskewed",
         "attention_output_dense_weight",
         "attention_output_dense_bias",
         "attention_output_residual",
         "bottleneck_input_LayerNorm",
     }},
    {"attention_output_LayerNorm",
     {
         "attention_output_residual",
         "attention_output_LayerNorm_weight",
         "attention_output_LayerNorm_bias",
         "attention_output_LayerNorm",
     }},

    // FFN Layers
    {"ffn_0_intermediate_dense",
     {
         "attention_output_LayerNorm",
         "ffn_0_intermediate_dense_weight",
         "ffn_0_intermediate_dense_bias",
         "ffn_0_intermediate_intermediate_act_fn",
     }},
    {"ffn_0_output_dense",
     {
         "ffn_0_intermediate_intermediate_act_fn",
         "ffn_0_output_dense_weight",
         "ffn_0_output_dense_bias",
         "ffn_0_output_residual",
         "attention_output_LayerNorm",
     }},
    {"ffn_0_output_LayerNorm",
     {
         "ffn_0_output_residual",
         "ffn_0_output_LayerNorm_weight",
         "ffn_0_output_LayerNorm_bias",
         "ffn_0_output_LayerNorm",
     }},

    {"ffn_1_intermediate_dense",
     {
         "ffn_0_output_LayerNorm",
         "ffn_1_intermediate_dense_weight",
         "ffn_1_intermediate_dense_bias",
         "ffn_1_intermediate_intermediate_act_fn",
     }},
    {"ffn_1_output_dense",
     {
         "ffn_1_intermediate_intermediate_act_fn",
         "ffn_1_output_dense_weight",
         "ffn_1_output_dense_bias",
         "ffn_1_output_residual",
         "ffn_0_output_LayerNorm",
     }},
    {"ffn_1_output_LayerNorm",
     {
         "ffn_1_output_residual",
         "ffn_1_output_LayerNorm_weight",
         "ffn_1_output_LayerNorm_bias",
         "ffn_1_output_LayerNorm",
     }},

    {"ffn_2_intermediate_dense",
     {
         "ffn_1_output_LayerNorm",
         "ffn_2_intermediate_dense_weight",
         "ffn_2_intermediate_dense_bias",
         "ffn_2_intermediate_intermediate_act_fn",
     }},
    {"ffn_2_output_dense",
     {
         "ffn_2_intermediate_intermediate_act_fn",
         "ffn_2_output_dense_weight",
         "ffn_2_output_dense_bias",
         "ffn_2_output_residual",
         "ffn_1_output_LayerNorm",
     }},
    {"ffn_2_output_LayerNorm",
     {
         "ffn_2_output_residual",
         "ffn_2_output_LayerNorm_weight",
         "ffn_2_output_LayerNorm_bias",
         "ffn_2_output_LayerNorm",
     }},

    {"intermediate_dense",
     {
         "ffn_2_output_LayerNorm",
         "intermediate_dense_weight",
         "intermediate_dense_bias",
         "intermediate_intermediate_act_fn",
     }},
    {"output_dense",
     {
         "intermediate_intermediate_act_fn",
         "output_dense_weight",
         "output_dense_bias",
         "output_residual",
         "ffn_2_output_LayerNorm",
     }},
    {"output_LayerNorm",
     {
         "output_residual",
         "output_LayerNorm_weight",
         "output_LayerNorm_bias",
         "output_LayerNorm",
     }},

    {"output_bottleneck_dense",
     {
         "output_LayerNorm",
         "output_bottleneck_dense_weight",
         "output_bottleneck_dense_bias",
         "output_bottleneck_residual",
         "hidden_states",
     }},
    {"output_bottleneck_LayerNorm",
     {
         "output_bottleneck_residual",
         "output_bottleneck_LayerNorm_weight",
         "output_bottleneck_LayerNorm_bias",
         "output_bottleneck_LayerNorm",
     }},

    {"classifier",
     {
         "mobilebert_encoder_layer_23_output_bottleneck_LayerNorm",
         "classifier_weight",
         "classifier_bias",
         "mobilebert_classifier",
     }},
};
