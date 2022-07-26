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
    "attention_self_query_layer_0",
    "attention_self_query_layer_1",
    "attention_self_query_layer_2",
    "attention_self_query_layer_3",
    "attention_self_key_layer_0",
    "attention_self_key_layer_1",
    "attention_self_key_layer_2",
    "attention_self_key_layer_3",
    "bottleneck_attention_LayerNorm_0",
    "bottleneck_attention_LayerNorm_1",
    "bottleneck_attention_dense",
    "hidden_states_0",
    "hidden_states_1",
    "hidden_states_2",
};

std::map<std::string, std::string> backpropParamsMapping{
    {"classifier", "crossEntropyLoss"},
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
    // (1 x 16)
    {"crossEntropyLoss",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .CROSS_ENTROPY_GRAD = true,
     }},

    // (1 x 16) x (16 x 512)
    {"classifier",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {1, 32, 1, 1, 1, 1}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .WEIGHT = true,
     }},

    // (128 x 512) * 512
    {"outputLayerNorm",
     {
         .loops = {{8, 1, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .WEIGHT = true,
         .NO_NORM = true,
     }},

    // (128 x 512) x (512 x 128)
    {"outputBottleneck",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .WEIGHT = true,
     }},

    // (128 x 128) x (128 x 512)
    {"ffn2",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .WEIGHT = true,
         .RELU_GRAD = true,
     }},

    // (128 x 512) x (512 x 128)
    {"ffn1",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .RESIDUAL = true,
         .WEIGHT = true,
     }},

    // (128 x 128) x (128 x 128)
    {"outputAttention",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .WEIGHT = true,
         .SPLIT_OUTPUT = true,
     }},

    // (128 x 32) * (32 x 128)
    {"attentionProbs",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
     }},

    // (128 x 128) x (128 x 32)
    {"vAttention",
     {
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
     }},

    // (128 x 128) x (128 x 512)
    {"vProjection",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {5, 5},
         .STRIDE = 1,
         .RESIDUAL = true,
         .WEIGHT = true,
         .CONCAT_INPUT = true,
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
         .SOFTMAX_GRAD = true,
     }},

    // (128 x 128) x (128 x 32)
    {"qAttention",
     {
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {5, 5},
         .STRIDE = 1,
     }},

    // (128 x 128) x (128 x 128)
    {"qProjection",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {5, 5},
         .STRIDE = 1,
         .WEIGHT = true,
         .CONCAT_INPUT = true,
     }},

    // (128 x 128) * (128 x 32)
    {"kAttention",
     {
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {5, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
     }},

    // (128 x 128) x (128 x 128)
    {"kProjection",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {5, 5},
         .STRIDE = 1,
         .RESIDUAL = true,
         .WEIGHT = true,
         .CONCAT_INPUT = true,
     }},

    // (128 x 128) * 128
    {"bottleneckLayerNorm",
     {
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {2, 0},
         .weightLoopIndex = {3, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .WEIGHT = true,
         .NO_NORM = true,
     }},

    // (128 x 128) x (128 x 512)
    {"inputBottleneck",
     {
         .WEIGHT_TRANSPOSE = true,
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {5, 5},
         .STRIDE = 1,
         .RESIDUAL = true,
         .WEIGHT = true,
     }},
};

std::map<std::string, MemoryOffsets> backpropMemOffsets{
    {"classifier",
     {
         0,
         0,
         0,
         0,
     }},
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
    {"classifier",
     {
         "mobilebert_logits",
         "mobilebert_labels",
         "",
         "mobilebert_logits",
     }},
    {"output_bottleneck_LayerNorm",
     {
         "mobilebert_logits",
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
         "attention_self_query_layer",
         "attention_self_query_weight",
         "",
         "bottleneck_attention_LayerNorm",
     }},
    {"bottleneck_attention_LayerNorm_1",
     {
         "attention_self_key_layer",
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
         "attention_self_mixed_value_layer",
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
