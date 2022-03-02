#pragma once

#include <array>
#include <map>
#include <string>

std::map<std::string, SimplifiedParams> mobilebert{
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
     }},

    // elementwise product and addition for matrix of size:
    // (128 x 128) * 128
    {"bottleneckLayerNorm",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         10 * 1024,                                   // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {8, 1, 1, 1, 128, 1}},  // LOOPS
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
     }},

    // Q x K
    // (128 x 32) * (32 x 128)
    {"qkAttention",
     {
         0,                                          // INPUT_OFFSET
         69632,                                      // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
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
     }},

    // elementwise product and addition for matrix of size:
    // 128 x 512
    {"outputLayerNorm",
     {
         0,                                            // INPUT_OFFSET
         0,                                            // WEIGHT_OFFSET
         10 * 1024,                                    // OUTPUT_OFFSET
         false,                                        // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 128, 1}},  // LOOPS
         {0, 5},                                       // INPUTX
         {1, 4},                                       // INPUTY
         {2, 0},                                       // REDUCTION
         {3, 1},                                       // WEIGHT
         3,                                            // fxIndex
         2,                                            // fyIndex
         {4, 5},                                       // weightReuseIndex
         1,                                            // stride
         false,                                        // replication
         false,                                        // RELU
         true,                                         // bias
         30 * 1024,                                    // BIAS_OFFSET
         false,                                        // residual
         40 * 1024,                                    // RESIDUAL_OFFSET
         false,                                        // maxpool
         false,                                        // avgpool
         false,                                        // SOFTMAX
         false,                                        // FC
         true,                                         // NO_NORM
     }},

    // (128 x 512) * (512 x 2)
    {"classifier",
     {
         0,                                            // INPUT_OFFSET
         0,                                            // WEIGHT_OFFSET
         10 * 1024,                                    // OUTPUT_OFFSET
         false,                                        // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 128, 1}},  // LOOPS
         {0, 5},                                       // INPUTX
         {1, 4},                                       // INPUTY
         {2, 0},                                       // REDUCTION
         {3, 1},                                       // WEIGHT
         3,                                            // fxIndex
         2,                                            // fyIndex
         {4, 5},                                       // weightReuseIndex
         1,                                            // stride
         false,                                        // replication
         false,                                        // RELU
         true,                                         // bias
         30 * 1024,                                    // BIAS_OFFSET
         false,                                        // residual
         40 * 1024,                                    // RESIDUAL_OFFSET
         false,                                        // maxpool
         false,                                        // avgpool
         false,                                        // SOFTMAX
         false,                                        // FC
         false,                                        // NO_NORM
     }},
};

std::array<std::string, 35> mobilebertOrder{"bottleneck_input_dense",
                                            "bottleneck_input_LayerNorm",
                                            "bottleneck_attention_dense",
                                            "bottleneck_attention_LayerNorm",
                                            "attention_self_query",
                                            "attention_self_key",
                                            "attention_self_attention_scores_0",
                                            "attention_self_attention_scores_1",
                                            "attention_self_attention_scores_2",
                                            "attention_self_attention_scores_3",
                                            "attention_self_attention_probs_0",
                                            "attention_self_attention_probs_1",
                                            "attention_self_attention_probs_2",
                                            "attention_self_attention_probs_3",
                                            "attention_self_value",
                                            "attention_self_context_layer_0",
                                            "attention_self_context_layer_1",
                                            "attention_self_context_layer_2",
                                            "attention_self_context_layer_3",
                                            "attention_output_dense",
                                            "attention_output_LayerNorm",
                                            "ffn_0_intermediate_dense",
                                            "ffn_0_output_dense",
                                            "ffn_0_output_LayerNorm",
                                            "ffn_1_intermediate_dense",
                                            "ffn_1_output_dense",
                                            "ffn_1_output_LayerNorm",
                                            "ffn_2_intermediate_dense",
                                            "ffn_2_output_dense",
                                            "ffn_2_output_LayerNorm",
                                            "intermediate_dense",
                                            "output_dense",
                                            "output_LayerNorm",
                                            "output_bottleneck_dense",
                                            "output_bottleneck_LayerNorm"};

std::map<std::string, std::string> mobilebertOperations{
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
    {"attention_self_value", "inputBottleneck"},
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
    {"classifier", "classifier"}};

const int BLOCK_SIZE = 128 * 128;
std::map<std::string, Offsets> mobilebertOffsets{
    {"bottleneck_input_dense", {0, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE}},
    {"bottleneck_input_LayerNorm",
     {4 * BLOCK_SIZE, 0, 5 * BLOCK_SIZE, BLOCK_SIZE}},
    {"bottleneck_attention_dense", {0, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE}},
    {"bottleneck_attention_LayerNorm",
     {4 * BLOCK_SIZE, 0, 6 * BLOCK_SIZE, BLOCK_SIZE}},
    {"attention_self_query", {6 * BLOCK_SIZE, 0, 4 * BLOCK_SIZE, BLOCK_SIZE}},
    {"attention_self_key", {6 * BLOCK_SIZE, 0, 7 * BLOCK_SIZE, BLOCK_SIZE}},
    {"attention_self_attention_scores_0",
     {4 * BLOCK_SIZE, 7 * BLOCK_SIZE, 8 * BLOCK_SIZE}},
    {"attention_self_attention_scores_1",
     {4 * BLOCK_SIZE + 1 * 128 * 32, 7 * BLOCK_SIZE + 1 * 32 * 128,
      9 * BLOCK_SIZE}},
    {"attention_self_attention_scores_2",
     {4 * BLOCK_SIZE + 2 * 128 * 32, 7 * BLOCK_SIZE + 2 * 32 * 128,
      10 * BLOCK_SIZE}},
    {"attention_self_attention_scores_3",
     {4 * BLOCK_SIZE + 3 * 128 * 32, 7 * BLOCK_SIZE + 3 * 32 * 128,
      11 * BLOCK_SIZE}},
    {"attention_self_attention_probs_0", {8 * BLOCK_SIZE, 0, 12 * BLOCK_SIZE}},
    {"attention_self_attention_probs_1", {9 * BLOCK_SIZE, 0, 13 * BLOCK_SIZE}},
    {"attention_self_attention_probs_2", {10 * BLOCK_SIZE, 0, 14 * BLOCK_SIZE}},
    {"attention_self_attention_probs_3", {11 * BLOCK_SIZE, 0, 15 * BLOCK_SIZE}},
    {"attention_self_value", {0, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE}},
    {"attention_self_context_layer_0",
     {12 * BLOCK_SIZE, 4 * BLOCK_SIZE, 6 * BLOCK_SIZE}},
    {"attention_self_context_layer_1",
     {13 * BLOCK_SIZE, 4 * BLOCK_SIZE + 1 * 128 * 32,
      6 * BLOCK_SIZE + 1 * 128 * 32}},
    {"attention_self_context_layer_2",
     {14 * BLOCK_SIZE, 4 * BLOCK_SIZE + 2 * 128 * 32,
      6 * BLOCK_SIZE + 2 * 128 * 32}},
    {"attention_self_context_layer_3",
     {15 * BLOCK_SIZE, 4 * BLOCK_SIZE + 3 * 128 * 32,
      6 * BLOCK_SIZE + 3 * 128 * 32}},
    {"attention_output_dense",
     {6 * BLOCK_SIZE, 0, 4 * BLOCK_SIZE, BLOCK_SIZE, 5 * BLOCK_SIZE}},
    {"attention_output_LayerNorm",
     {4 * BLOCK_SIZE, 0, 5 * BLOCK_SIZE, BLOCK_SIZE}},
    {"ffn_0_intermediate_dense",
     {5 * BLOCK_SIZE, 0, 6 * BLOCK_SIZE, 4 * BLOCK_SIZE}},
    {"ffn_0_output_dense",
     {6 * BLOCK_SIZE, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE, 5 * BLOCK_SIZE}},
    {"ffn_0_output_LayerNorm", {4 * BLOCK_SIZE, 0, 5 * BLOCK_SIZE, BLOCK_SIZE}},
    {"ffn_1_intermediate_dense",
     {5 * BLOCK_SIZE, 0, 6 * BLOCK_SIZE, 4 * BLOCK_SIZE}},
    {"ffn_1_output_dense",
     {6 * BLOCK_SIZE, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE, 5 * BLOCK_SIZE}},
    {"ffn_1_output_LayerNorm", {4 * BLOCK_SIZE, 0, 5 * BLOCK_SIZE, BLOCK_SIZE}},
    {"ffn_2_intermediate_dense",
     {5 * BLOCK_SIZE, 0, 6 * BLOCK_SIZE, 4 * BLOCK_SIZE}},
    {"ffn_2_output_dense",
     {6 * BLOCK_SIZE, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE, 5 * BLOCK_SIZE}},
    {"ffn_2_output_LayerNorm", {4 * BLOCK_SIZE, 0, 5 * BLOCK_SIZE, BLOCK_SIZE}},
    {"intermediate_dense", {5 * BLOCK_SIZE, 0, 6 * BLOCK_SIZE, 4 * BLOCK_SIZE}},
    {"output_dense",
     {6 * BLOCK_SIZE, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE, 5 * BLOCK_SIZE}},
    {"output_LayerNorm", {4 * BLOCK_SIZE, 0, 5 * BLOCK_SIZE, BLOCK_SIZE}},
    {"output_bottleneck_dense",
     {5 * BLOCK_SIZE, 0, 6 * BLOCK_SIZE, 4 * BLOCK_SIZE, 0}},
    {"output_bottleneck_LayerNorm", {6 * BLOCK_SIZE, 0, 0, 4 * BLOCK_SIZE}},
    {"classifier", {0, 0, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE}}};

std::string mobilebertDataDir = "data/mobilebert/";

std::map<std::string, Files> mobilebertFiles{
    // Input Bottleneck
    {"bottleneck_input_dense",
     {"hidden_states", "bottleneck_input_dense_weight",
      "bottleneck_input_dense_bias", "bottleneck_input_dense"}},
    {"bottleneck_input_LayerNorm",
     {"bottleneck_input_dense", "bottleneck_input_LayerNorm_weight",
      "bottleneck_input_LayerNorm_bias", "bottleneck_input_LayerNorm"}},
    {"bottleneck_attention_dense",
     {"hidden_states", "bottleneck_attention_dense_weight",
      "bottleneck_attention_dense_bias", "bottleneck_attention_dense"}},
    {"bottleneck_attention_LayerNorm",
     {"bottleneck_attention_dense", "bottleneck_attention_LayerNorm_weight",
      "bottleneck_attention_LayerNorm_bias", "bottleneck_attention_LayerNorm"}},

    // QKV Projection
    {"attention_self_query",
     {"bottleneck_attention_LayerNorm", "attention_self_query_weight",
      "attention_self_query_bias", "attention_self_query"}},
    {"attention_self_key",
     {"bottleneck_attention_LayerNorm", "attention_self_key_weight",
      "attention_self_key_bias", "attention_self_key"}},
    {"attention_self_value",
     {"hidden_states", "attention_self_value_weight",
      "attention_self_value_bias", "attention_self_value"}},

    // TODO: permute QKV into 4 sub-tensors

    // Multi-Head Self Attention
    // TODO: add division
    {"attention_self_attention_scores_0",
     {"attention_self_query_layer_0", "attention_self_key_layer_0", "",
      "attention_self_attention_scores_0"}},
    {"attention_self_attention_scores_1",
     {"attention_self_query_layer_1", "attention_self_key_layer_1", "",
      "attention_self_attention_scores_1"}},
    {"attention_self_attention_scores_2",
     {"attention_self_query_layer_2", "attention_self_key_layer_2", "",
      "attention_self_attention_scores_2"}},
    {"attention_self_attention_scores_3",
     {"attention_self_query_layer_3", "attention_self_key_layer_3", "",
      "attention_self_attention_scores_3"}},

    {"attention_self_attention_probs_0",
     {"attention_self_attention_scores_div_0",
      "attention_self_attention_scores_0", "",
      "attention_self_attention_probs_0"}},
    {"attention_self_attention_probs_1",
     {"attention_self_attention_scores_div_1",
      "attention_self_attention_scores_1", "",
      "attention_self_attention_probs_1"}},
    {"attention_self_attention_probs_2",
     {"attention_self_attention_scores_div_2",
      "attention_self_attention_scores_2", "",
      "attention_self_attention_probs_2"}},
    {"attention_self_attention_probs_3",
     {"attention_self_attention_scores_div_3",
      "attention_self_attention_scores_3", "",
      "attention_self_attention_probs_3"}},

    {"attention_self_context_layer_0",
     {"attention_self_attention_probs_0", "attention_self_value_layer_0", "",
      "attention_self_context_layer_0"}},
    {"attention_self_context_layer_1",
     {"attention_self_attention_probs_1", "attention_self_value_layer_1", "",
      "attention_self_context_layer_1"}},
    {"attention_self_context_layer_2",
     {"attention_self_attention_probs_2", "attention_self_value_layer_2", "",
      "attention_self_context_layer_2"}},
    {"attention_self_context_layer_3",
     {"attention_self_attention_probs_3", "attention_self_value_layer_3", "",
      "attention_self_context_layer_3"}},

    // TODO: concat attention_self_context_layer into a single tensor

    {"attention_output_dense",
     {"attention_self_context_layer", "attention_output_dense_weight",
      "attention_output_dense_bias", "attention_output_residual",
      "bottleneck_input_LayerNorm"}},
    {"attention_output_LayerNorm",
     {"attention_output_residual", "attention_output_LayerNorm_weight",
      "attention_output_LayerNorm_bias", "attention_output_LayerNorm"}},

    // FFN Layers
    {"ffn_0_intermediate_dense",
     {"attention_output_LayerNorm", "ffn_0_intermediate_dense_weight",
      "ffn_0_intermediate_dense_bias", "ffn_0_intermediate_dense"}},
    {"ffn_0_output_dense",
     {"ffn_0_intermediate_dense", "ffn_0_output_dense_weight",
      "ffn_0_output_dense_bias", "ffn_0_output_residual",
      "attention_output_LayerNorm"}},
    {"ffn_0_output_LayerNorm",
     {"ffn_0_output_residual", "ffn_0_output_LayerNorm_weight",
      "ffn_0_output_LayerNorm_bias", "ffn_0_output_LayerNorm"}},
    {"ffn_1_intermediate_dense",
     {"ffn_0_output_LayerNorm", "ffn_1_intermediate_dense_weight",
      "ffn_1_intermediate_dense_bias", "ffn_1_intermediate_dense"}},
    {"ffn_1_output_dense",
     {"ffn_1_intermediate_dense", "ffn_1_output_dense_weight",
      "ffn_1_output_dense_bias", "ffn_1_output_residual",
      "ffn_0_output_LayerNorm"}},
    {"ffn_1_output_LayerNorm",
     {"ffn_1_output_residual", "ffn_1_output_LayerNorm_weight",
      "ffn_1_output_LayerNorm_bias", "ffn_1_output_LayerNorm"}},
    {"ffn_2_intermediate_dense",
     {"ffn_1_output_LayerNorm", "ffn_2_intermediate_dense_weight",
      "ffn_2_intermediate_dense_bias", "ffn_2_intermediate_dense"}},
    {"ffn_2_output_dense",
     {"ffn_2_intermediate_dense", "ffn_2_output_dense_weight",
      "ffn_2_output_dense_bias", "ffn_2_output_residual",
      "ffn_1_output_LayerNorm"}},
    {"ffn_2_output_LayerNorm",
     {"ffn_2_output_residual", "ffn_2_output_LayerNorm_weight",
      "ffn_2_output_LayerNorm_bias", "ffn_2_output_LayerNorm"}},
    {"intermediate_dense",
     {"ffn_2_output_LayerNorm", "intermediate_dense_weight",
      "intermediate_dense_bias", "intermediate_dense"}},
    {"output_dense",
     {"intermediate_dense", "output_dense_weight", "output_dense_bias",
      "output_residual", "ffn_2_output_LayerNorm"}},
    {"output_LayerNorm",
     {"output_residual", "output_LayerNorm_weight", "output_LayerNorm_bias",
      "output_LayerNorm"}},
    {"output_bottleneck_dense",
     {"output_LayerNorm", "output_bottleneck_dense_weight",
      "output_bottleneck_dense_bias", "output_bottleneck_residual",
      "hidden_states"}},
    {"output_bottleneck_LayerNorm",
     {"output_bottleneck_residual", "output_bottleneck_LayerNorm_weight",
      "output_bottleneck_LayerNorm_bias", "output_bottleneck_LayerNorm"}},
    {"classifier", {}}};

std::map<std::string, MemoryMap> mobilebertMemoryMap{
    {"bottleneck_input_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"bottleneck_input_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"bottleneck_attention_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"bottleneck_attention_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_query", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_key", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_value", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_scores_0", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_scores_1", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_scores_2", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_scores_3", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_probs_0", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_probs_1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_probs_2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_probs_3", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_context_layer_0", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_self_context_layer_1", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_self_context_layer_2", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_self_context_layer_3", {SRAM, SRAM, RRAM, SRAM, SRAM}},
    {"attention_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_output_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_0_intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_0_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_0_output_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_1_intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_1_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_1_output_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_2_intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_2_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_2_output_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"output_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"output_bottleneck_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"output_bottleneck_LayerNorm", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"classifier", {SRAM, RRAM, RRAM, SRAM, SRAM}}};
