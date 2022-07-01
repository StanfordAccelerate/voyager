#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

std::vector<std::string> backpropOrder{
    "output_bottleneck_LayerNorm",
    "output_bottleneck_dense",
    "output_LayerNorm",
    "output_dense",
    "intermediate_dense",
    "ffn_2_output_LayerNorm",
    "ffn_2_output_dense",
    "ffn_2_intermediate_dense",
    "ffn_1_output_LayerNorm",
    "ffn_1_output_dense",
    "ffn_1_intermediate_dense",
    "ffn_0_output_LayerNorm",
    "ffn_0_output_dense",
    "ffn_0_intermediate_dense",
    "attention_output_LayerNorm",
    "attention_output_dense",
    "bottleneck_input_dense",
    "attention_self_context_layer",
    "attention_self_value_layer_3",
    "attention_self_value_layer_2",
    "attention_self_value_layer_1",
    "attention_self_value_layer_0",
    "attention_self_attention_probs_0",
    "attention_self_attention_probs_1",
    "attention_self_attention_probs_2",
    "attention_self_attention_probs_3",
    "attention_self_attention_scores_0",
    "attention_self_attention_scores_1",
    "attention_self_attention_scores_2",
    "attention_self_attention_scores_3",
    "attention_self_query_layer_3",
    "attention_self_query_layer_2",
    "attention_self_query_layer_1",
    "attention_self_query_layer_0",
    "attention_self_key_layer_3",
    "attention_self_key_layer_2",
    "attention_self_key_layer_1",
    "attention_self_key_layer_0",
    "bottleneck_attention_LayerNorm_0",
    "bottleneck_attention_LayerNorm_1",
    "bottleneck_attention_dense",
    "hidden_states_0",
    "hidden_states_1",
    "hidden_states_2",
};

std::map<std::string, std::string> backpropParamsMapping{
    {"output_bottleneck_LayerNorm", "classifier"},
    {"output_bottleneck_dense", "outputLayerNorm"},
    {"output_LayerNorm", "outputBottleneck"},
    {"output_dense", "bottleneckLayerNorm"},
    {"intermediate_dense", "ffn2"},
    {"ffn_2_output_LayerNorm", "ffn1"},
    {"ffn_2_output_dense", "bottleneckLayerNorm"},
    {"ffn_2_intermediate_dense", "ffn2"},
    {"ffn_1_output_LayerNorm", "ffn1"},
    {"ffn_1_output_dense", "bottleneckLayerNorm"},
    {"ffn_1_intermediate_dense", "ffn2"},
    {"ffn_0_output_LayerNorm", "ffn1"},
    {"ffn_0_output_dense", "bottleneckLayerNorm"},
    {"ffn_0_intermediate_dense", "ffn2"},
    {"attention_output_LayerNorm", "ffn1"},
    {"attention_output_dense", "bottleneckLayerNorm"},
    {"bottleneck_input_dense", "bottleneckLayerNorm"},
    {"attention_self_context_layer", "outputAttention"},
    {"attention_self_value_layer_0", "vAttention"},
    {"attention_self_value_layer_1", "vAttention"},
    {"attention_self_value_layer_2", "vAttention"},
    {"attention_self_value_layer_3", "vAttention"},
    {"attention_self_attention_probs_0", "attentionProbs"},
    {"attention_self_attention_probs_1", "attentionProbs"},
    {"attention_self_attention_probs_2", "attentionProbs"},
    {"attention_self_attention_probs_3", "attentionProbs"},
    {"attention_self_attention_scores_0", "softmax"},
    {"attention_self_attention_scores_1", "softmax"},
    {"attention_self_attention_scores_2", "softmax"},
    {"attention_self_attention_scores_3", "softmax"},
    {"attention_self_query_layer_0", "qAttention"},
    {"attention_self_query_layer_1", "qAttention"},
    {"attention_self_query_layer_2", "qAttention"},
    {"attention_self_query_layer_3", "qAttention"},
    {"attention_self_key_layer_0", "kAttention"},
    {"attention_self_key_layer_1", "kAttention"},
    {"attention_self_key_layer_2", "kAttention"},
    {"attention_self_key_layer_3", "kAttention"},
    {"bottleneck_attention_LayerNorm_0", "qProjection"},
    {"bottleneck_attention_LayerNorm_1", "kProjection"},
    {"bottleneck_attention_dense", "bottleneckLayerNorm"},
    {"hidden_states_0", "inputBottleneck"},
    {"hidden_states_1", "vProjection"},
    {"hidden_states_2", "inputBottleneck"},
};

std::map<std::string, SimplifiedParams> backpropParams{
    // (128 x 128) * (128 x 512)
    {"inputBottleneck",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         131072,                                      // OUTPUT_OFFSET
         true,                                        // TRANSPOSE
         {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
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
         false,                                       // bias
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

    {"vProjection",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         131072,                                      // OUTPUT_OFFSET
         true,                                        // TRANSPOSE
         {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
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
         false,                                       // bias
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
         true,                                        // CONCAT_HEAD
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
         false,                                       // bias
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
    {"qProjection",
     {
         131072,                                     // INPUT_OFFSET
         65536,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         true,                                       // TRANSPOSE
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
         false,                                      // bias
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
         true,                                       // CONCAT_HEAD
     }},

    // (128 x 128) x (128 x 128)
    {"kProjection",
     {
         131072,                                     // INPUT_OFFSET
         65536,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         true,                                       // TRANSPOSE
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
         false,                                      // bias
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

    // Q x K
    // (128 x 128) * (128 x 32)
    {"qAttention",
     {
         0,                                          // INPUT_OFFSET
         69632,                                      // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
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

    // Q x K
    // (128 x 128) * (128 x 32)
    {"kAttention",
     {
         0,                                          // INPUT_OFFSET
         69632,                                      // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
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
         true,                                       // INPUT_TRANSPOSE
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
         false,                                         // SOFTMAX
         false,                                         // FC
         false,                                         // NO-NORM
         false,                                         // weight
         false,                                         // ATTENTION_SCALING
         false,                                         // STORE_IN_ACC
         false,                                         // ACC_FROM_ACC
         false,                                         // INPUT_TRANSPOSE
         false,                                         // SPLIT_HEAD
         false,                                         // CONCAT_HEAD
         true,                                          // SOFTMAX_GRAD
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
         true,                                       // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

    // QK x V
    // (128 x 32) * (32 x 128)
    {"attentionProbs",
     {
         131072,                                     // INPUT_OFFSET
         73728,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         true,                                       // TRANSPOSE
         {{4, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 32}},  // LOOPS
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
         true,                                       // TRANSPOSE
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
         false,                                      // bias
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

    // FFN 1
    // (128 x 512) * (512 x 128)
    {"ffn1",
     {
         131072,                                      // INPUT_OFFSET
         94208,                                       // WEIGHT_OFFSET
         0,                                           // OUTPUT_OFFSET
         true,                                        // TRANSPOSE
         {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
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
         false,                                       // bias
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

    // FFN 2
    // (128 x 128) x (128 x 512)
    {"ffn2",
     {
         0,                                          // INPUT_OFFSET
         159744,                                     // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         true,                                       // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
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
         true,                                       // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
         false,                                      // SOFTMAX_GRAD
         false,                                      // NO_NORM_GRAD
         true,                                       // RELU_GRAD
     }},

    // output bottleneck
    // (128 x 512) x (512 x 128)
    {"outputBottleneck",
     {
         131072,                                      // INPUT_OFFSET
         225280,                                      // WEIGHT_OFFSET
         0,                                           // OUTPUT_OFFSET
         true,                                        // TRANSPOSE
         {{4, 1, 8, 1, 1, 1}, {32, 1, 1, 1, 1, 32}},  // LOOPS
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
         false,                                       // bias
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
    // (128 x 512) * 512
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
         false,                                         // bias
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

    // FIXME: This should be a FC or matmul?
    // (1 x 16) * (16 x 512)
    {"classifier",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         10 * 1024,                                  // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 32, 1, 1, 1, 1}},  // LOOPS
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

std::map<std::string, MemoryOffsets> backpropMemOffsets{
    {"output_bottleneck_LayerNorm",
     {
         0,
         12 * WEIGHT_INTERMEDIATE_SIZE + 7 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
         12 * WEIGHT_INTERMEDIATE_SIZE + 23 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"output_bottleneck_dense",
     {
         INTERMEDIATE_SIZE,
         12 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
         0,
         12 * WEIGHT_INTERMEDIATE_SIZE + 6 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"output_LayerNorm",
     {
         0,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
         12 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"output_dense",
     {
         INTERMEDIATE_SIZE,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 22 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 23 * BIAS_HIDDEN_SIZE,
     }},
    {"intermediate_dense",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         10 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
         4 * INTERMEDIATE_SIZE + 24 * HIDDEN_SIZE,
     }},
    {"ffn_2_output_LayerNorm",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
         10 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
     }},
    {"ffn_2_output_dense",
     {
         INTERMEDIATE_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 19 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 20 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_intermediate_dense",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         3 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
     }},
    {"ffn_1_output_LayerNorm",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
     }},
    {"ffn_1_output_dense",
     {
         INTERMEDIATE_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 16 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 17 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_intermediate_dense",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         6 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         2 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
     }},
    {"ffn_0_output_LayerNorm",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
         6 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
     }},
    {"ffn_0_output_dense",
     {
         INTERMEDIATE_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 13 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 14 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_intermediate_dense",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         4 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
     }},
    {"attention_output_LayerNorm",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
         4 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
     }},
    {"attention_output_dense",
     {
         INTERMEDIATE_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             10 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             11 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_input_dense",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         WEIGHT_INTERMEDIATE_SIZE + BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"attention_self_context_layer",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_value_layer_0",
     {
         INTERMEDIATE_SIZE + 11 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
     }},
    {"attention_self_value_layer_1",
     {
         INTERMEDIATE_SIZE + 12 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + HEAD_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE + HEAD_SIZE,
     }},
    {"attention_self_value_layer_2",
     {
         INTERMEDIATE_SIZE + 13 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE + 2 * HEAD_SIZE,
     }},
    {"attention_self_value_layer_3",
     {
         INTERMEDIATE_SIZE + 14 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE + 3 * HEAD_SIZE,
     }},
    {"attention_self_attention_probs_0",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_1",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + HEAD_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + HEAD_SIZE,
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_2",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_3",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE,
     }},

    {"attention_self_attention_scores_0",
     {
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 11 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_1",
     {
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 12 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_2",
     {
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 13 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_3",
     {
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 14 * HIDDEN_SIZE,
     }},

    {"attention_self_query_layer_0",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE,
     }},
    {"attention_self_query_layer_1",
     {
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + HEAD_SIZE,
     }},
    {"attention_self_query_layer_2",
     {
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + 2 * HEAD_SIZE,
     }},
    {"attention_self_query_layer_3",
     {
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + 3 * HEAD_SIZE,
     }},
    {"attention_self_key_layer_0",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 7 * HIDDEN_SIZE,
     }},
    {"attention_self_key_layer_1",
     {
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE + HEAD_SIZE,
         INTERMEDIATE_SIZE + 7 * HIDDEN_SIZE + HEAD_SIZE,
     }},
    {"attention_self_key_layer_2",
     {
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 7 * HIDDEN_SIZE + 2 * HEAD_SIZE,
     }},
    {"attention_self_key_layer_3",
     {
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 7 * HIDDEN_SIZE + 3 * HEAD_SIZE,
     }},

    {"bottleneck_attention_LayerNorm_0",
     {
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 6 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             6 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_LayerNorm_1",
     {
         INTERMEDIATE_SIZE + 7 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
     }},
    {"bottleneck_attention_dense",
     {
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_HIDDEN_SIZE,
     }},
    {"hidden_states_0",
     {
         INTERMEDIATE_SIZE,
         0,
         0,
         WEIGHT_INTERMEDIATE_SIZE,
         0,
     }},
    {"hidden_states_1",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
         0,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
         0,
     }},
    {"hidden_states_2",
     {
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
         0,
     }},
};

std::map<std::string, Files> backpropTestFiles{
    {"output_bottleneck_LayerNorm",
     {
         "mobilebert_classifier",
         "classifier_weight",
         "",
         "mobilebert_encoder_layer_23_output_bottleneck_LayerNorm",
     }},
    {"output_bottleneck_dense",
     {
         "output_bottleneck_LayerNorm",
         "output_bottleneck_LayerNorm_weight",
         "",
         "output_bottleneck_dense",
     }},
    {"output_LayerNorm",
     {
         "output_bottleneck_dense",
         "output_bottleneck_dense_weight",
         "",
         "output_LayerNorm",
     }},
    {"output_dense",
     {
         "output_LayerNorm",
         "output_LayerNorm_weight",
         "",
         "output_dense",
     }},
    {"intermediate_dense",
     {
         "output_dense",
         "output_dense_weight",
         "",
         "intermediate_dense",
         "intermediate_dense",
     }},

    {"ffn_2_output_LayerNorm",
     {
         "intermediate_dense",
         "intermediate_dense_weight",
         "",
         "ffn_2_output_LayerNorm",
         "output_dense",
     }},
    {"ffn_2_output_dense",
     {
         "ffn_2_output_LayerNorm",
         "ffn_2_output_LayerNorm_weight",
         "",
         "ffn_2_output_dense",
     }},
    {"ffn_2_intermediate_dense",
     {
         "ffn_2_output_dense",
         "ffn_2_output_dense_weight",
         "",
         "ffn_2_intermediate_dense",
         "ffn_2_intermediate_dense",
     }},

    {"ffn_1_output_LayerNorm",
     {
         "ffn_2_intermediate_dense",
         "ffn_2_intermediate_dense_weight",
         "",
         "ffn_1_output_LayerNorm",
         "ffn_2_output_dense",
     }},
    {"ffn_1_output_dense",
     {
         "ffn_1_output_LayerNorm",
         "ffn_1_output_LayerNorm_weight",
         "",
         "ffn_1_output_dense",
     }},
    {"ffn_1_intermediate_dense",
     {
         "ffn_1_output_dense",
         "ffn_1_output_dense_weight",
         "",
         "ffn_1_intermediate_dense",
         "ffn_1_intermediate_dense",
     }},

    {"ffn_0_output_LayerNorm",
     {
         "ffn_1_intermediate_dense",
         "ffn_1_intermediate_dense_weight",
         "",
         "ffn_0_output_LayerNorm",
         "ffn_1_output_dense",
     }},
    {"ffn_0_output_dense",
     {
         "ffn_0_output_LayerNorm",
         "ffn_0_output_LayerNorm_weight",
         "",
         "ffn_0_output_dense",
     }},
    {"ffn_0_intermediate_dense",
     {
         "ffn_0_output_dense",
         "ffn_0_output_dense_weight",
         "",
         "ffn_0_intermediate_dense",
         "ffn_0_intermediate_dense",
     }},

    {"attention_output_LayerNorm",
     {
         "ffn_0_intermediate_dense",
         "ffn_0_intermediate_dense_weight",
         "",
         "attention_output_LayerNorm",
         "ffn_0_output_dense",
     }},
    {"attention_output_dense",
     {
         "attention_output_LayerNorm",
         "attention_output_LayerNorm_weight",
         "",
         "attention_output_dense",
     }},
    {"bottleneck_input_dense",
     {
         "attention_output_dense",
         "bottleneck_input_LayerNorm_weight",
         "",
         "bottleneck_input_dense",
     }},
    {"attention_self_context_layer",
     {
         "attention_output_dense",
         "attention_output_dense_weight",
         "",
         "attention_self_context_layer",
     }},

    {"attention_self_value_layer_0",
     {
         "attention_self_attention_probs_0",
         "attention_self_context_layer_0",
         "",
         "attention_self_value_layer_0",
     }},
    {"attention_self_value_layer_1",
     {
         "attention_self_attention_probs_1",
         "attention_self_context_layer_1",
         "",
         "attention_self_value_layer_1",
     }},
    {"attention_self_value_layer_2",
     {
         "attention_self_attention_probs_2",
         "attention_self_context_layer_2",
         "",
         "attention_self_value_layer_2",
     }},
    {"attention_self_value_layer_3",
     {
         "attention_self_attention_probs_3",
         "attention_self_context_layer_3",
         "",
         "attention_self_value_layer_3",
     }},

    {"attention_self_attention_probs_0",
     {
         "attention_self_context_layer_0",
         "attention_self_value_layer_0",
         "",
         "attention_self_attention_probs_0",
     }},
    {"attention_self_attention_probs_1",
     {
         "attention_self_context_layer_1",
         "attention_self_value_layer_1",
         "",
         "attention_self_attention_probs_1",
     }},
    {"attention_self_attention_probs_2",
     {
         "attention_self_context_layer_2",
         "attention_self_value_layer_2",
         "",
         "attention_self_attention_probs_2",

     }},
    {"attention_self_attention_probs_3",
     {
         "attention_self_context_layer_3",
         "attention_self_value_layer_3",
         "",
         "attention_self_attention_probs_3",
     }},

    {"attention_self_attention_scores_0",
     {
         "attention_self_attention_probs_0",
         "",
         "",
         "attention_self_attention_scores_0",
         "attention_self_attention_probs_0",
     }},
    {"attention_self_attention_scores_1",
     {
         "attention_self_attention_probs_1",
         "",
         "",
         "attention_self_attention_scores_1",
         "attention_self_attention_probs_1",
     }},
    {"attention_self_attention_scores_2",
     {
         "attention_self_attention_probs_2",
         "",
         "",
         "attention_self_attention_scores_2",
         "attention_self_attention_probs_2",
     }},
    {"attention_self_attention_scores_3",
     {
         "attention_self_attention_probs_3",
         "",
         "",
         "attention_self_attention_scores_3",
         "attention_self_attention_probs_3",
     }},

    {"attention_self_query_layer_0",
     {
         "attention_self_attention_scores_0",
         "attention_self_key_layer_0",
         "",
         "attention_self_query_layer_0",
     }},
    {"attention_self_query_layer_1",
     {
         "attention_self_attention_scores_1",
         "attention_self_key_layer_1",
         "",
         "attention_self_query_layer_1",
     }},
    {"attention_self_query_layer_2",
     {
         "attention_self_attention_scores_2",
         "attention_self_key_layer_2",
         "",
         "attention_self_query_layer_2",
     }},
    {"attention_self_query_layer_3",
     {
         "attention_self_attention_scores_3",
         "attention_self_key_layer_3",
         "",
         "attention_self_query_layer_3",
     }},

    {"attention_self_key_layer_0",
     {
         "attention_self_attention_scores_0",
         "attention_self_query_layer_0",
         "",
         "attention_self_key_layer_0",
     }},
    {"attention_self_key_layer_1",
     {
         "attention_self_attention_scores_1",
         "attention_self_query_layer_1",
         "",
         "attention_self_key_layer_1",
     }},
    {"attention_self_key_layer_2",
     {
         "attention_self_attention_scores_2",
         "attention_self_query_layer_2",
         "",
         "attention_self_key_layer_2",
     }},
    {"attention_self_key_layer_3",
     {
         "attention_self_attention_scores_3",
         "attention_self_query_layer_3",
         "",
         "attention_self_key_layer_3",
     }},

    {"bottleneck_attention_LayerNorm_0",
     {
         "attention_self_query",
         "attention_self_query_weight",
         "",
         "bottleneck_attention_LayerNorm",
     }},
    {"bottleneck_attention_LayerNorm_1",
     {
         "attention_self_key",
         "attention_self_key_weight",
         "",
         "bottleneck_attention_LayerNorm",
     }},
    {"bottleneck_attention_dense",
     {
         "bottleneck_attention_LayerNorm",
         "bottleneck_attention_LayerNorm_weight",
         "",
         "bottleneck_attention_dense",
     }},

    {"hidden_states_0",
     {
         "bottleneck_input_dense",
         "bottleneck_input_dense_weight",
         "",
         "hidden_states",
     }},
    {"hidden_states_1",
     {
         "attention_self_value",
         "attention_self_value_weight",
         "",
         "hidden_states",
     }},
    {"hidden_states_2",
     {
         "bottleneck_attention_dense",
         "bottleneck_attention_dense_weight",
         "",
         "hidden_states",
     }},
};
