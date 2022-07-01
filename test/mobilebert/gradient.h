#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

std::map<std::string, std::string> gradientParamsMapping{
    {"classifier", "classifier"},
    {"output_bottleneck_LayerNorm", "outputLayerNorm"},
    {"output_bottleneck_dense", "outputBottleneck"},
    {"output_LayerNorm", "bottleneckLayerNorm"},
    {"output_dense", "ffn2"},
    {"intermediate_dense", "outputBottleneck"},
    {"ffn_2_output_LayerNorm", "bottleneckLayerNorm"},
    {"ffn_2_output_dense", "ffn2"},
    {"ffn_2_intermediate_dense", "outputBottleneck"},
    {"ffn_1_output_LayerNorm", "bottleneckLayerNorm"},
    {"ffn_1_output_dense", "ffn2"},
    {"ffn_1_intermediate_dense", "outputBottleneck"},
    {"ffn_0_output_LayerNorm", "bottleneckLayerNorm"},
    {"ffn_0_output_dense", "ffn2"},
    {"ffn_0_intermediate_dense", "outputBottleneck"},
    {"attention_output_LayerNorm", "bottleneckLayerNorm"},
    {"attention_output_dense", "outputAttention"},
    {"attention_self_value_layer_0", "vProjection"},
    {"attention_self_query_layer_0", "qkProjection"},
    {"attention_self_key_layer_0", "qkProjection"},
    {"bottleneck_attention_LayerNorm_1", "bottleneckLayerNorm"},
    {"bottleneck_attention_dense", "inputBottleneck"},
    {"bottleneck_input_LayerNorm", "bottleneckLayerNorm"},
    {"bottleneck_input_dense", "inputBottleneck"},
    {"hidden_states_2", "outputLayerNorm"},
};

std::map<std::string, SimplifiedParams> gradientParams{
    // FIXME: This should be a FC or matmul?
    // (16 x 1) * (1 x 512)
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
         false,                                      // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         false,                                      // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

    // elementwise multiplication of matrices of size:
    // (128 x 512) * (128 x 512)
    {"outputLayerNorm",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         10 * 1024,                                   // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{4, 1, 1, 8, 1, 1}, {32, 1, 1, 1, 1, 32}},  // LOOPS
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
         false,                                       // NO_NORM
         false,                                       // weight
         false,                                       // ATTENTION_SCALING
         false,                                       // STORE_IN_ACC
         false,                                       // ACC_FROM_ACC
         false,                                       // INPUT_TRANSPOSE
         false,                                       // SPLIT_HEAD
         false,                                       // CONCAT_HEAD,
         false,                                       // SOFTMAX_GRAD
         true,                                        // NO_NORM_GRAD
     }},

    // output bottleneck
    // (512 x 128) x (128 x 128)
    {"outputBottleneck",
     {
         131072,                                      // INPUT_OFFSET
         225280,                                      // WEIGHT_OFFSET
         0,                                           // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{16, 1, 8, 1, 1, 1}, {8, 1, 1, 1, 1, 32}},  // LOOPS
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
         false,                                       // weight
         false,                                       // ATTENTION_SCALING
         false,                                       // STORE_IN_ACC
         false,                                       // ACC_FROM_ACC
         true,                                        // INPUT_TRANSPOSE
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
         false,                                      // TRANSPOSE
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
         false,                                      // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         true,                                       // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},

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
         false,                                      // SOFTMAX_GRAD;
         false,                                      // NO_NORM_GRAD;
         false,                                      // RELU_GRAD;
         true,                                       // WEIGHT_PERMUTE;
     }},

    // (128 x 128) * (128 x 512)
    {"vProjection",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
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
         false,                                      // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         true,                                       // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         true,                                       // CONCAT_HEAD
     }},

    // TODO: Add unit test (no head concatenation)
    // Self-attention output
    // (128 x 128) x (128 x 128)
    {"qkProjection",
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
         true,                                       // CONCAT_HEAD
     }},

    // elementwise multiplication of matrices of size:
    // (128 x 128) * (128 x 128)
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
         false,                                       // NO_NORM
         false,                                       // weight
         false,                                       // ATTENTION_SCALING
         false,                                       // STORE_IN_ACC
         false,                                       // ACC_FROM_ACC
         false,                                       // INPUT_TRANSPOSE
         false,                                       // SPLIT_HEAD
         false,                                       // CONCAT_HEAD
         false,                                       // SOFTMAX_GRAD
         true,                                        // NO_NORM_GRAD
     }},

    // (128 x 128) * (128 x 512)
    {"inputBottleneck",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
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
         false,                                      // weight
         false,                                      // ATTENTION_SCALING
         false,                                      // STORE_IN_ACC
         false,                                      // ACC_FROM_ACC
         true,                                       // INPUT_TRANSPOSE
         false,                                      // SPLIT_HEAD
         false,                                      // CONCAT_HEAD
     }},
};

std::map<std::string, MemoryOffsets> gradientMemOffsets{
    {"classifier",
     {
         0,
         7 * INTERMEDIATE_SIZE + 26 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"output_bottleneck_LayerNorm",
     {
         0,
         5 * INTERMEDIATE_SIZE + 26 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"output_bottleneck_dense",
     {
         0,
         5 * INTERMEDIATE_SIZE + 25 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},

    {"output_LayerNorm",
     {
         0,
         5 * INTERMEDIATE_SIZE + 24 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"output_dense",
     {
         0,
         4 * INTERMEDIATE_SIZE + 24 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"intermediate_dense",
     {
         0,
         4 * INTERMEDIATE_SIZE + 23 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_2_output_LayerNorm",
     {
         0,
         4 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_2_output_dense",
     {
         0,
         3 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_2_intermediate_dense",
     {
         0,
         3 * INTERMEDIATE_SIZE + 21 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_1_output_LayerNorm",
     {
         0,
         3 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_1_output_dense",
     {
         0,
         2 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_1_intermediate_dense",
     {
         0,
         2 * INTERMEDIATE_SIZE + 19 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_0_output_LayerNorm",
     {
         0,
         2 * INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_0_output_dense",
     {
         0,
         INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"ffn_0_intermediate_dense",
     {
         0,
         INTERMEDIATE_SIZE + 17 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},

    {"attention_output_LayerNorm",
     {
         0,
         INTERMEDIATE_SIZE + 16 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"attention_output_dense",
     {
         0,
         INTERMEDIATE_SIZE + 15 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},

    {"attention_self_value_layer_0",
     {
         0,
         0,
         INTERMEDIATE_SIZE,
     }},
    {"attention_self_key_layer_0",
     {
         0,
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"attention_self_query_layer_0",
     {
         0,
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"bottleneck_attention_LayerNorm_1",
     {
         0,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"bottleneck_attention_dense",
     {
         0,
         0,
         INTERMEDIATE_SIZE,
     }},
    {"bottleneck_input_LayerNorm",
     {
         0,
         INTERMEDIATE_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"bottleneck_input_dense",
     {
         0,
         0,
         INTERMEDIATE_SIZE,
     }},
    {"hidden_states_2",
     {
         0,
         5 * INTERMEDIATE_SIZE + 26 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE,
     }},
};

std::map<std::string, Files> gradientTestFiles{
    {"classifier",
     {
         "mobilebert_classifier",
         "mobilebert_encoder_layer_23_output_bottleneck_LayerNorm",
         "",
         "classifier_weight",
     }},
    {"output_bottleneck_LayerNorm",
     {
         "output_bottleneck_LayerNorm",
         "output_bottleneck_residual",
         "",
         "output_bottleneck_LayerNorm_weight",
     }},
    {"output_bottleneck_dense",
     {
         "output_bottleneck_dense",
         "output_LayerNorm",
         "",
         "output_bottleneck_dense_weight",
     }},

    {"output_LayerNorm",
     {
         "output_LayerNorm",
         "output_residual",
         "",
         "output_LayerNorm_weight",
     }},
    {"output_dense",
     {
         "output_dense",
         "intermediate_intermediate_act_fn",
         "",
         "output_dense_weight",
     }},
    {"intermediate_dense",
     {
         "intermediate_dense",
         "ffn_2_output_LayerNorm",
         "",
         "intermediate_dense_weight",
     }},

    {"ffn_2_output_LayerNorm",
     {
         "ffn_2_output_LayerNorm",
         "ffn_2_output_residual",
         "",
         "ffn_2_output_LayerNorm_weight",
     }},
    {"ffn_2_output_dense",
     {
         "ffn_2_output_dense",
         "ffn_2_intermediate_intermediate_act_fn",
         "",
         "ffn_2_output_dense_weight",
     }},
    {"ffn_2_intermediate_dense",
     {
         "ffn_2_intermediate_dense",
         "ffn_1_output_LayerNorm",
         "",
         "ffn_2_intermediate_dense_weight",
     }},

    {"ffn_1_output_LayerNorm",
     {
         "ffn_1_output_LayerNorm",
         "ffn_1_output_residual",
         "",
         "ffn_1_output_LayerNorm_weight",
     }},
    {"ffn_1_output_dense",
     {
         "ffn_1_output_dense",
         "ffn_1_intermediate_intermediate_act_fn",
         "",
         "ffn_1_output_dense_weight",
     }},
    {"ffn_1_intermediate_dense",
     {
         "ffn_1_intermediate_dense",
         "ffn_0_output_LayerNorm",
         "",
         "ffn_1_intermediate_dense_weight",
     }},

    {"ffn_0_output_LayerNorm",
     {
         "ffn_0_output_LayerNorm",
         "ffn_0_output_residual",
         "",
         "ffn_0_output_LayerNorm_weight",
     }},
    {"ffn_0_output_dense",
     {
         "ffn_0_output_dense",
         "ffn_0_intermediate_intermediate_act_fn",
         "",
         "ffn_0_output_dense_weight",
     }},
    {"ffn_0_intermediate_dense",
     {
         "ffn_0_intermediate_dense",
         "attention_output_LayerNorm",
         "",
         "ffn_0_intermediate_dense_weight",
     }},

    {"attention_output_LayerNorm",
     {
         "attention_output_LayerNorm",
         "attention_output_residual",
         "",
         "attention_output_LayerNorm_weight",
     }},
    {"attention_output_dense",
     {
         "attention_output_dense",
         "attention_self_context_layer",
         "",
         "attention_output_dense_weight",
     }},

    {"attention_self_value_layer_0",
     {
         "attention_self_value",
         "hidden_states",
         "",
         "attention_self_value_weight",
     }},
    {"attention_self_query_layer_0",
     {
         "attention_self_query",
         "bottleneck_attention_LayerNorm",
         "",
         "attention_self_query_weight",
     }},
    {"attention_self_key_layer_0",
     {
         "attention_self_key",
         "bottleneck_attention_LayerNorm",
         "",
         "attention_self_key_weight",
     }},

    {"bottleneck_attention_LayerNorm_1",
     {
         "bottleneck_attention_LayerNorm",
         "bottleneck_attention_dense",
         "",
         "bottleneck_attention_LayerNorm_weight",
     }},
    {"bottleneck_attention_dense",
     {
         "bottleneck_attention_dense",
         "hidden_states",
         "",
         "bottleneck_attention_dense_weight",
     }},

    {"bottleneck_input_LayerNorm",
     {
         "bottleneck_input_LayerNorm",
         "bottleneck_input_dense",
         "",
         "bottleneck_input_LayerNorm_weight",
     }},
    {"bottleneck_input_dense",
     {
         "bottleneck_input_dense",
         "hidden_states",
         "",
         "bottleneck_input_dense_weight",
     }},
    {"hidden_states_2",
     {
         "output_bottleneck_LayerNorm",
         "output_bottleneck_residual",
         "",
         "output_bottleneck_LayerNorm_weight",
     }},
};
