#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

std::vector<std::string> inferenceOrder{
    "bottleneck_input_dense",
    "bottleneck_input_LayerNorm",
    "bottleneck_attention_dense",
    "bottleneck_attention_LayerNorm",
    "attention_self_query_layer",
    "attention_self_key_layer",
    "attention_self_attention_scores_0",
    "attention_self_attention_scores_1",
    "attention_self_attention_scores_2",
    "attention_self_attention_scores_3",
    "attention_self_attention_probs_0",
    "attention_self_attention_probs_1",
    "attention_self_attention_probs_2",
    "attention_self_attention_probs_3",
    "attention_self_value_layer",
    "attention_self_context_layer_0",
    "attention_self_context_layer_1",
    "attention_self_context_layer_2",
    "attention_self_context_layer_3",
    "attention_output_dense",
    "attention_output_LayerNorm",
    "ffn_0_intermediate_dense",
    "ffn_0_output_dense",
    "ffn_0_output_LayerNorm",
    "intermediate_dense",
    "output_dense",
    "output_LayerNorm",
    "output_bottleneck_dense",
    "output_bottleneck_LayerNorm",
    "classifier",
};

std::map<std::string, std::string> inferenceParamsMapping{
    {"bottleneck_input_dense", "inputBottleneck"},
    {"bottleneck_input_LayerNorm", "bottleneckLayerNorm"},
    {"bottleneck_attention_dense", "inputBottleneck"},
    {"bottleneck_attention_LayerNorm", "bottleneckLayerNorm"},
    {"attention_self_query_layer", "qkProjection"},
    {"attention_self_key_layer", "qkProjection"},
    {"attention_self_value_layer", "vProjection"},
    {"attention_self_attention_scores_0", "qkAttention"},
    {"attention_self_attention_scores_1", "qkAttention"},
    {"attention_self_attention_scores_2", "qkAttention"},
    {"attention_self_attention_scores_3", "qkAttention"},
    {"attention_self_attention_probs_0", "softmax"},
    {"attention_self_attention_probs_1", "softmax"},
    {"attention_self_attention_probs_2", "softmax"},
    {"attention_self_attention_probs_3", "softmax"},
    {"attention_self_context_layer_0", "context"},
    {"attention_self_context_layer_1", "context"},
    {"attention_self_context_layer_2", "context"},
    {"attention_self_context_layer_3", "context"},
    {"attention_output_dense", "outputAttention"},
    {"attention_output_LayerNorm", "bottleneckLayerNorm"},
    {"ffn_0_intermediate_dense", "ffn1"},
    {"ffn_0_output_dense", "ffn2"},
    {"ffn_0_output_LayerNorm", "bottleneckLayerNorm"},
    {"intermediate_dense", "ffn1"},
    {"output_dense", "ffn2"},
    {"output_LayerNorm", "bottleneckLayerNorm"},
    {"output_bottleneck_dense", "outputBottleneck"},
    {"output_bottleneck_LayerNorm", "outputLayerNorm"},
    {"classifier", "classifier"},
};

std::map<std::string, SimplifiedParams> inferenceParams{
    // (128 x 512) * (512 x 128)
    {"inputBottleneck",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .WEIGHT = true,
     }},

    // (128 x 128) * 128
    {"bottleneckLayerNorm",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .WEIGHT = true,
         .NO_NORM = true,
     }},

    // (128 x 128) x (128 x 128)
    {"qkProjection",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .WEIGHT = true,
         .SPLIT_OUTPUT = true,
     }},

    // (128 x 512) x (512 x 128)
    {"vProjection",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .WEIGHT = true,
         .SPLIT_OUTPUT = true,
     }},

    // (128 x 32) x (32 x 128)
    {"qkAttention",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{4, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .ATTENTION_SCALING = true,
     }},

    // (128 x 128)
    {"softmax",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .SOFTMAX = true,
     }},

    // (128 x 128)
    {"softmax_no_mask",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .SOFTMAX = true,
     }},

    // (128 x 128) x (128 x 32)
    {"context",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
     }},

    // (128 x 128) x (128 x 128)
    {"outputAttention",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .RESIDUAL = true,
         .WEIGHT = true,
         .CONCAT_INPUT = true,
     }},

    // (128 x 128) x (128 x 512)
    {"ffn1",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .RELU = true,
         .BIAS = true,
         .WEIGHT = true,
     }},

    // (128 x 512) x (512 x 128)
    {"ffn2",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .RESIDUAL = true,
         .WEIGHT = true,
     }},

    // (128 x 128) x (128 x 512)
    {"outputBottleneck",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .RESIDUAL = true,
         .WEIGHT = true,
     }},

    // (128 x 512) * 512
    {"outputLayerNorm",
     {
         .loops = {{4, 1, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .WEIGHT = true,
         .NO_NORM = true,
     }},

    // (1 x 512) x (512 x 16)
    {"classifier",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 1, 1}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS = true,
         .WEIGHT = true,
         .FC = true,
     }},
};

std::map<std::string, MemoryOffsets> inferenceMemOffsets{
    {"bottleneck_input_dense",
     {
         0,
         0,
         INTERMEDIATE_SIZE,
         WEIGHT_INTERMEDIATE_SIZE,
     }},
    {"bottleneck_input_LayerNorm",
     {
         INTERMEDIATE_SIZE,
         WEIGHT_INTERMEDIATE_SIZE + BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
         WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_dense",
     {
         0,
         WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_LayerNorm",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_query_layer",
     {
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 6 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             6 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_key_layer",
     {
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_0",
     {
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 7 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_1",
     {
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE + HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + HEAD_SIZE,
         INTERMEDIATE_SIZE + 8 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_2",
     {
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 9 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_scores_3",
     {
         INTERMEDIATE_SIZE + 5 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 10 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_0",
     {
         INTERMEDIATE_SIZE + 7 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 11 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_1",
     {
         INTERMEDIATE_SIZE + 8 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 12 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_2",
     {
         INTERMEDIATE_SIZE + 9 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 13 * HIDDEN_SIZE,
     }},
    {"attention_self_attention_probs_3",
     {
         INTERMEDIATE_SIZE + 10 * HIDDEN_SIZE,
         0,
         INTERMEDIATE_SIZE + 14 * HIDDEN_SIZE,
     }},
    {"attention_self_value_layer",
     {
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_context_layer_0",
     {
         INTERMEDIATE_SIZE + 11 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 15 * HIDDEN_SIZE,
     }},
    {"attention_self_context_layer_1",
     {
         INTERMEDIATE_SIZE + 12 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 1 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 15 * HIDDEN_SIZE + 1 * HEAD_SIZE,
     }},
    {"attention_self_context_layer_2",
     {
         INTERMEDIATE_SIZE + 13 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 2 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 15 * HIDDEN_SIZE + 2 * HEAD_SIZE,
     }},
    {"attention_self_context_layer_3",
     {
         INTERMEDIATE_SIZE + 14 * HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 2 * HIDDEN_SIZE + 3 * HEAD_SIZE,
         INTERMEDIATE_SIZE + 15 * HIDDEN_SIZE + 3 * HEAD_SIZE,
     }},
    {"attention_output_dense",
     {
         INTERMEDIATE_SIZE + 15 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 16 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 3 * HIDDEN_SIZE,
     }},
    {"attention_output_LayerNorm",
     {
         INTERMEDIATE_SIZE + 16 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             10 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 17 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             11 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_intermediate_dense",
     {
         INTERMEDIATE_SIZE + 17 * HIDDEN_SIZE,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         4 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_output_dense",
     {
         INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         4 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
         2 * INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
         INTERMEDIATE_SIZE + 17 * HIDDEN_SIZE,
     }},
    {"ffn_0_output_LayerNorm",
     {
         2 * INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 13 * BIAS_HIDDEN_SIZE,
         2 * INTERMEDIATE_SIZE + 19 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 14 * BIAS_HIDDEN_SIZE,
     }},
    {"intermediate_dense",
     {
         2 * INTERMEDIATE_SIZE + 19 * HIDDEN_SIZE,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         2 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         6 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
     }},
    {"output_dense",
     {
         2 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         6 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         3 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
         2 * INTERMEDIATE_SIZE + 19 * HIDDEN_SIZE,
     }},
    {"output_LayerNorm",
     {
         3 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 16 * BIAS_HIDDEN_SIZE,
         3 * INTERMEDIATE_SIZE + 21 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 17 * BIAS_HIDDEN_SIZE,
     }},
    {"output_bottleneck_dense",
     {
         3 * INTERMEDIATE_SIZE + 21 * HIDDEN_SIZE,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         3 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         0,
     }},
    {"output_bottleneck_LayerNorm",
     {
         3 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         4 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
     }},
    {"classifier",
     {
         4 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
         5 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         8 * WEIGHT_INTERMEDIATE_SIZE + 21 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
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
    {"attention_self_query_layer",
     {
         "bottleneck_attention_LayerNorm",
         "attention_self_query_weight",
         "attention_self_query_bias",
         "attention_self_query_layer",
     }},
    {"attention_self_key_layer",
     {
         "bottleneck_attention_LayerNorm",
         "attention_self_key_weight",
         "attention_self_key_bias",
         "attention_self_key_layer",
     }},
    {"attention_self_value_layer",
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
         "mobilebert_attention_mask",
         "attention_self_attention_scores_0",
     }},
    {"attention_self_attention_scores_1",
     {
         "attention_self_query_layer_1",
         "attention_self_key_layer_1",
         "mobilebert_attention_mask",
         "attention_self_attention_scores_1",
     }},
    {"attention_self_attention_scores_2",
     {
         "attention_self_query_layer_2",
         "attention_self_key_layer_2",
         "mobilebert_attention_mask",
         "attention_self_attention_scores_2",
     }},
    {"attention_self_attention_scores_3",
     {
         "attention_self_query_layer_3",
         "attention_self_key_layer_3",
         "mobilebert_attention_mask",
         "attention_self_attention_scores_3",
     }},

    {"attention_self_attention_probs_0",
     {
         "attention_self_attention_scores_0",
         "",
         "",
         "attention_self_attention_probs_0",
     }},
    {"attention_self_attention_probs_1",
     {
         "attention_self_attention_scores_1",
         "",
         "",
         "attention_self_attention_probs_1",
     }},
    {"attention_self_attention_probs_2",
     {
         "attention_self_attention_scores_2",
         "",
         "",
         "attention_self_attention_probs_2",
     }},
    {"attention_self_attention_probs_3",
     {
         "attention_self_attention_scores_3",
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
         "attention_self_context_layer",
         "attention_output_dense_weight",
         "attention_output_dense_bias",
         "attention_output_dense",
         "bottleneck_input_LayerNorm",
     }},
    {"attention_output_LayerNorm",
     {
         "attention_output_dense",
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
         "ffn_0_output_dense",
         "attention_output_LayerNorm",
     }},
    {"ffn_0_output_LayerNorm",
     {
         "ffn_0_output_dense",
         "ffn_0_output_LayerNorm_weight",
         "ffn_0_output_LayerNorm_bias",
         "ffn_0_output_LayerNorm",
     }},

    {"intermediate_dense",
     {
         "ffn_0_output_LayerNorm",
         "intermediate_dense_weight",
         "intermediate_dense_bias",
         "intermediate_intermediate_act_fn",
     }},
    {"output_dense",
     {
         "intermediate_intermediate_act_fn",
         "output_dense_weight",
         "output_dense_bias",
         "output_dense",
         "ffn_0_output_LayerNorm",
     }},
    {"output_LayerNorm",
     {
         "output_dense",
         "output_LayerNorm_weight",
         "output_LayerNorm_bias",
         "output_LayerNorm",
     }},

    {"output_bottleneck_dense",
     {
         "output_LayerNorm",
         "output_bottleneck_dense_weight",
         "output_bottleneck_dense_bias",
         "output_bottleneck_dense",
         "hidden_states",
     }},
    {"output_bottleneck_LayerNorm",
     {
         "output_bottleneck_dense",
         "output_bottleneck_LayerNorm_weight",
         "output_bottleneck_LayerNorm_bias",
         "output_bottleneck_LayerNorm",
     }},

    {"classifier",
     {
         "mobilebert_encoder_layer_23_output_bottleneck_LayerNorm",
         "classifier_weight",
         "classifier_bias",
         "classifier",
     }},
};
