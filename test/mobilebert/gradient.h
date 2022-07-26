#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

std::map<std::string, std::string> gradientParamsMapping{
    {"classifier_weight", "classifier_weight"},
    {"output_bottleneck_LayerNorm_weight", "outputLayerNorm"},
    {"output_bottleneck_LayerNorm_bias", "hiddenReduction"},
    {"output_bottleneck_dense_weight", "outputBottleneck"},
    {"output_bottleneck_dense_bias", "hiddenReduction"},
    {"output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"output_LayerNorm_bias", "bottleneckReduction"},
    {"output_dense_weight", "ffn2"},
    {"output_dense_bias", "bottleneckReduction"},
    {"intermediate_dense_weight", "outputBottleneck"},
    {"intermediate_dense_bias", "hiddenReduction"},
    {"ffn_2_output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"ffn_2_output_LayerNorm_bias", "bottleneckReduction"},
    {"ffn_2_output_dense_weight", "ffn2"},
    {"ffn_2_output_dense_bias", "bottleneckReduction"},
    {"ffn_2_intermediate_dense_weight", "outputBottleneck"},
    {"ffn_2_intermediate_dense_bias", "hiddenReduction"},
    {"ffn_1_output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"ffn_1_output_LayerNorm_bias", "bottleneckReduction"},
    {"ffn_1_output_dense_weight", "ffn2"},
    {"ffn_1_output_dense_bias", "bottleneckReduction"},
    {"ffn_1_intermediate_dense_weight", "outputBottleneck"},
    {"ffn_1_intermediate_dense_bias", "hiddenReduction"},
    {"ffn_0_output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"ffn_0_output_LayerNorm_bias", "bottleneckReduction"},
    {"ffn_0_output_dense_weight", "ffn2"},
    {"ffn_0_output_dense_bias", "bottleneckReduction"},
    {"ffn_0_intermediate_dense_weight", "outputBottleneck"},
    {"ffn_0_intermediate_dense_bias", "hiddenReduction"},
    {"attention_output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"attention_output_LayerNorm_bias", "bottleneckReduction"},
    {"attention_output_dense_weight", "outputAttention"},
    {"attention_output_dense_bias", "bottleneckReduction"},
    {"attention_self_value_weight", "vProjection"},
    {"attention_self_value_bias", "qkvBias"},
    {"attention_self_query_weight", "qkProjection"},
    {"attention_self_query_bias", "qkvBias"},
    {"attention_self_key_weight", "qkProjection"},
    {"attention_self_key_bias", "qkvBias"},
    {"bottleneck_attention_LayerNorm_weight", "bottleneckLayerNorm"},
    {"bottleneck_attention_LayerNorm_bias", "bottleneckReduction"},
    {"bottleneck_attention_dense_weight", "inputBottleneck"},
    {"bottleneck_attention_dense_bias", "bottleneckReduction"},
    {"bottleneck_input_LayerNorm_weight", "bottleneckLayerNorm"},
    {"bottleneck_input_LayerNorm_bias", "bottleneckReduction"},
    {"bottleneck_input_dense_weight", "inputBottleneck"},
    {"bottleneck_input_dense_bias", "bottleneckReduction"},
};

std::map<std::string, SimplifiedParams> gradientParams{
    // (16 x 1) x (1 x 512)
    {"classifier_weight",
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
         .GRAD_CLIPPING = true,
     }},

    // (128 x 512) * (128 x 512)
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
         .NO_NORM_GRAD = true,
         .GRAD_CLIPPING = true,
     }},

    // (128 x 128) x (128 x 512)
    {"outputBottleneck",
     {
         .loops = {{8, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
         .GRAD_CLIPPING = true,
     }},

    // (512 x 128) x (128 x 128)
    {"ffn2",
     {
         .loops = {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
         .GRAD_CLIPPING = true,
     }},

    // (128 x 128) x (128 x 128)
    {"outputAttention",
     {
         .loops = {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
         .CONCAT_INPUT = true,
         .GRAD_CLIPPING = true,
     }},

    // (512 x 128) x (128 x 128)
    {"vProjection",
     {
         .loops = {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
         .CONCAT_WEIGHT = true,
         .GRAD_CLIPPING = true,
     }},

    // (128 x 128) x (128 x 128)
    {"qkProjection",
     {
         .loops = {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
         .CONCAT_WEIGHT = true,
         .GRAD_CLIPPING = true,
     }},

    // (128 x 128) * (128 x 128)
    {"bottleneckLayerNorm",
     {
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .NO_NORM_GRAD = true,
         .GRAD_CLIPPING = true,
     }},

    // (512 x 128) x (128 x 128)
    {"inputBottleneck",
     {
         .loops = {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .INPUT_TRANSPOSE = true,
         .GRAD_CLIPPING = true,
     }},

    // (128 x 128)
    {"bottleneckReduction",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS_GRAD = true,
         .GRAD_CLIPPING = true,
     }},

    // (128 x 128)
    {"qkvBias",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS_GRAD = true,
         .CONCAT_WEIGHT = true,
         .GRAD_CLIPPING = true,
     }},

    // (128 x 512)
    {"hiddenReduction",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 1}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .BIAS_GRAD = true,
         .GRAD_CLIPPING = true,
     }},
};

std::map<std::string, MemoryOffsets> gradientMemOffsets{
    {"classifier_weight",
     {
         0,
         6 * INTERMEDIATE_SIZE + 26 * HIDDEN_SIZE,
         12 * WEIGHT_INTERMEDIATE_SIZE + 7 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"classifier_bias",
     {
         0,  // unused
         0,
         12 * WEIGHT_INTERMEDIATE_SIZE + 23 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"output_bottleneck_LayerNorm_weight",
     {
         5 * INTERMEDIATE_SIZE + 26 * HIDDEN_SIZE,
         0,
         12 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"output_bottleneck_LayerNorm_bias",
     {
         0,  // unused
         0,
         12 * WEIGHT_INTERMEDIATE_SIZE + 6 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"output_bottleneck_dense_weight",
     {
         5 * INTERMEDIATE_SIZE + 25 * HIDDEN_SIZE,
         0,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},
    {"output_bottleneck_dense_bias",
     {
         0,  // unused
         0,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 24 * BIAS_HIDDEN_SIZE,
     }},

    {"output_LayerNorm_weight",
     {
         5 * INTERMEDIATE_SIZE + 24 * HIDDEN_SIZE,
         0,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 22 * BIAS_HIDDEN_SIZE,
     }},
    {"output_LayerNorm_bias",
     {
         0,  // unused
         0,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 23 * BIAS_HIDDEN_SIZE,
     }},
    {"output_dense_weight",
     {
         4 * INTERMEDIATE_SIZE + 24 * HIDDEN_SIZE,
         0,
         10 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
     }},
    {"output_dense_bias",
     {
         0,  // unused
         0,
         11 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
     }},
    {"intermediate_dense_weight",
     {
         4 * INTERMEDIATE_SIZE + 23 * HIDDEN_SIZE,
         0,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
     }},
    {"intermediate_dense_bias",
     {
         0,
         0,
         10 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 21 * BIAS_HIDDEN_SIZE,
     }},

    {"ffn_2_output_LayerNorm_weight",
     {
         4 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         0,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 19 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_output_LayerNorm_bias",
     {
         0,
         0,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 20 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_output_dense_weight",
     {
         3 * INTERMEDIATE_SIZE + 22 * HIDDEN_SIZE,
         0,
         8 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_output_dense_bias",
     {
         0,
         0,
         9 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_intermediate_dense_weight",
     {
         3 * INTERMEDIATE_SIZE + 21 * HIDDEN_SIZE,
         0,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_2_intermediate_dense_bias",
     {
         0,
         0,
         8 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 18 * BIAS_HIDDEN_SIZE,
     }},

    {"ffn_1_output_LayerNorm_weight",
     {
         3 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         0,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 16 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_output_LayerNorm_bias",
     {
         0,
         0,
         7 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 17 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_output_dense_weight",
     {
         2 * INTERMEDIATE_SIZE + 20 * HIDDEN_SIZE,
         0,
         6 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_output_dense_bias",
     {
         0,  // unused
         0,
         6 * WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_intermediate_dense_weight",
     {
         0,
         0,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_1_intermediate_dense_bias",
     {
         0,  // unused
         0,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 15 * BIAS_HIDDEN_SIZE,
     }},

    {"ffn_0_output_LayerNorm_weight",
     {
         2 * INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         0,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 13 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_output_LayerNorm_bias",
     {
         0,  // unused
         0,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 14 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_output_dense_weight",
     {
         INTERMEDIATE_SIZE + 18 * HIDDEN_SIZE,
         0,
         4 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_output_dense_bias",
     {
         0,  // unused
         0,
         5 * WEIGHT_INTERMEDIATE_SIZE + BIAS_INTERMEDIATE_SIZE +
             3 * WEIGHT_HIDDEN_SIZE + 12 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_intermediate_dense_weight",
     {
         INTERMEDIATE_SIZE + 17 * HIDDEN_SIZE,
         0,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
     }},
    {"ffn_0_intermediate_dense_bias",
     {
         0,  // unused
         0,
         4 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             12 * BIAS_HIDDEN_SIZE,
     }},

    {"attention_output_LayerNorm_weight",
     {
         INTERMEDIATE_SIZE + 16 * HIDDEN_SIZE,
         0,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             10 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_output_LayerNorm_bias",
     {
         0,  // unused
         0,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             11 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_output_dense_weight",
     {
         INTERMEDIATE_SIZE + 15 * HIDDEN_SIZE,
         0,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_output_dense_bias",
     {
         0,  // unused
         0,
         3 * WEIGHT_INTERMEDIATE_SIZE + 3 * WEIGHT_HIDDEN_SIZE +
             9 * BIAS_HIDDEN_SIZE,
     }},

    {"attention_self_value_weight",
     {
         0,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_value_bias",
     {
         0,
         0,
         3 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             8 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_key_weight",
     {
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_key_bias",
     {
         0,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 2 * WEIGHT_HIDDEN_SIZE +
             7 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_query_weight",
     {
         INTERMEDIATE_SIZE + 4 * HIDDEN_SIZE,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 6 * BIAS_HIDDEN_SIZE,
     }},
    {"attention_self_query_bias",
     {
         0,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + WEIGHT_HIDDEN_SIZE +
             6 * BIAS_HIDDEN_SIZE,
     }},

    {"bottleneck_attention_LayerNorm_weight",
     {
         INTERMEDIATE_SIZE + HIDDEN_SIZE,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 4 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_LayerNorm_bias",
     {

         0,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 5 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_dense_weight",
     {
         0,
         0,
         WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_attention_dense_bias",
     {
         0,
         0,
         2 * WEIGHT_INTERMEDIATE_SIZE + 3 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_input_LayerNorm_weight",
     {
         INTERMEDIATE_SIZE,
         0,
         WEIGHT_INTERMEDIATE_SIZE + BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_input_LayerNorm_bias",
     {
         INTERMEDIATE_SIZE,
         0,
         WEIGHT_INTERMEDIATE_SIZE + 2 * BIAS_HIDDEN_SIZE,
     }},
    {"bottleneck_input_dense_weight",
     {
         0,
         0,
         0,
     }},
    {"bottleneck_input_dense_bias",
     {
         0,
         0,
         WEIGHT_INTERMEDIATE_SIZE,
     }},
};

std::map<std::string, Files> gradientTestFiles{
    {"classifier_weight",
     {
         "mobilebert_logits",
         "mobilebert_encoder_layer_23_output_bottleneck_LayerNorm",
         "",
         "classifier_weight",
     }},
    {"classifier_bias",
     {
         "",
         "mobilebert_logits",
         "",
         "classifier_bias",
     }},
    {"output_bottleneck_LayerNorm_weight",
     {
         "output_bottleneck_residual",
         "output_bottleneck_LayerNorm",
         "",
         "output_bottleneck_LayerNorm_weight",
     }},
    {"output_bottleneck_LayerNorm_bias",
     {
         "",
         "output_bottleneck_LayerNorm",
         "",
         "output_bottleneck_LayerNorm_bias",
     }},
    {"output_bottleneck_dense_weight",
     {
         "output_LayerNorm",
         "output_bottleneck_dense",
         "",
         "output_bottleneck_dense_weight",
     }},
    {"output_bottleneck_dense_bias",
     {
         "",
         "output_bottleneck_dense",
         "",
         "output_bottleneck_dense_bias",
     }},

    {"output_LayerNorm_weight",
     {
         "output_residual",
         "output_LayerNorm",
         "",
         "output_LayerNorm_weight",
     }},
    {"output_LayerNorm_bias",
     {
         "",
         "output_LayerNorm",
         "",
         "output_LayerNorm_bias",
     }},
    {"output_dense_weight",
     {
         "intermediate_intermediate_act_fn",
         "output_dense",
         "",
         "output_dense_weight",
     }},
    {"output_dense_bias",
     {
         "",
         "output_dense",
         "",
         "output_dense_bias",
     }},
    {"intermediate_dense_weight",
     {
         "ffn_2_output_LayerNorm",
         "intermediate_dense",
         "",
         "intermediate_dense_weight",
     }},
    {"intermediate_dense_bias",
     {
         "",
         "intermediate_dense",
         "",
         "intermediate_dense_bias",
     }},

    {"ffn_2_output_LayerNorm_weight",
     {
         "ffn_2_output_residual",
         "ffn_2_output_LayerNorm",
         "",
         "ffn_2_output_LayerNorm_weight",
     }},
    {"ffn_2_output_LayerNorm_bias",
     {
         "",
         "ffn_2_output_LayerNorm",
         "",
         "ffn_2_output_LayerNorm_bias",
     }},
    {"ffn_2_output_dense_weight",
     {
         "ffn_2_intermediate_intermediate_act_fn",
         "ffn_2_output_dense",
         "",
         "ffn_2_output_dense_weight",
     }},
    {"ffn_2_output_dense_bias",
     {
         "ffn_2_intermediate_intermediate_act_fn",
         "ffn_2_output_dense",
         "",
         "ffn_2_output_dense_bias",
     }},
    {"ffn_2_intermediate_dense_weight",
     {
         "ffn_1_output_LayerNorm",
         "ffn_2_intermediate_dense",
         "",
         "ffn_2_intermediate_dense_weight",
     }},
    {"ffn_2_intermediate_dense_bias",
     {
         "",
         "ffn_2_intermediate_dense",
         "",
         "ffn_2_intermediate_dense_bias",
     }},

    {"ffn_1_output_LayerNorm_weight",
     {
         "ffn_1_output_residual",
         "ffn_1_output_LayerNorm",
         "",
         "ffn_1_output_LayerNorm_weight",
     }},
    {"ffn_1_output_LayerNorm_bias",
     {
         "",
         "ffn_1_output_LayerNorm",
         "",
         "ffn_1_output_LayerNorm_bias",
     }},
    {"ffn_1_output_dense_weight",
     {
         "ffn_1_intermediate_intermediate_act_fn",
         "ffn_1_output_dense",
         "",
         "ffn_1_output_dense_weight",
     }},
    {"ffn_1_output_dense_bias",
     {
         "ffn_1_intermediate_intermediate_act_fn",
         "ffn_1_output_dense",
         "",
         "ffn_1_output_dense_bias",
     }},
    {"ffn_1_intermediate_dense_weight",
     {
         "ffn_0_output_LayerNorm",
         "ffn_1_intermediate_dense",
         "",
         "ffn_1_intermediate_dense_weight",
     }},
    {"ffn_1_intermediate_dense_bias",
     {
         "",
         "ffn_1_intermediate_dense",
         "",
         "ffn_1_intermediate_dense_bias",
     }},

    {"ffn_0_output_LayerNorm_weight",
     {
         "ffn_0_output_residual",
         "ffn_0_output_LayerNorm",
         "",
         "ffn_0_output_LayerNorm_weight",
     }},
    {"ffn_0_output_LayerNorm_bias",
     {
         "",
         "ffn_0_output_LayerNorm",
         "",
         "ffn_0_output_LayerNorm_bias",
     }},
    {"ffn_0_output_dense_weight",
     {
         "ffn_0_intermediate_intermediate_act_fn",
         "ffn_0_output_dense",
         "",
         "ffn_0_output_dense_weight",
     }},
    {"ffn_0_output_dense_bias",
     {
         "",
         "ffn_0_output_dense",
         "",
         "ffn_0_output_dense_bias",
     }},
    {"ffn_0_intermediate_dense_weight",
     {
         "attention_output_LayerNorm",
         "ffn_0_intermediate_dense",
         "",
         "ffn_0_intermediate_dense_weight",
     }},
    {"ffn_0_intermediate_dense_bias",
     {
         "",
         "ffn_0_intermediate_dense",
         "",
         "ffn_0_intermediate_dense_bias",
     }},

    {"attention_output_LayerNorm_weight",
     {
         "attention_output_residual",
         "attention_output_LayerNorm",
         "",
         "attention_output_LayerNorm_weight",
     }},
    {"attention_output_LayerNorm_bias",
     {
         "",
         "attention_output_LayerNorm",
         "",
         "attention_output_LayerNorm_bias",
     }},
    {"attention_output_dense_weight",
     {
         "attention_self_context_layer",
         "attention_output_dense",
         "",
         "attention_output_dense_weight",
     }},
    {"attention_output_dense_bias",
     {
         "",
         "attention_output_dense",
         "",
         "attention_output_dense_bias",
     }},

    {"attention_self_value_weight",
     {
         "hidden_states",
         "attention_self_value_layer",
         "",
         "attention_self_value_weight",
     }},
    {"attention_self_value_bias",
     {
         "",
         "attention_self_value_layer",
         "",
         "attention_self_value_bias",
     }},
    {"attention_self_query_weight",
     {
         "bottleneck_attention_LayerNorm",
         "attention_self_query_layer",
         "",
         "attention_self_query_weight",
     }},
    {"attention_self_query_bias",
     {
         "",
         "attention_self_query_layer",
         "",
         "attention_self_query_bias",
     }},
    {"attention_self_key_weight",
     {
         "bottleneck_attention_LayerNorm",
         "attention_self_key_layer",
         "",
         "attention_self_key_weight",
     }},
    {"attention_self_key_bias",
     {
         "",
         "attention_self_key_layer",
         "",
         "attention_self_key_bias",
     }},

    {"bottleneck_attention_LayerNorm_weight",
     {
         "bottleneck_attention_dense",
         "bottleneck_attention_LayerNorm",
         "",
         "bottleneck_attention_LayerNorm_weight",
     }},
    {"bottleneck_attention_LayerNorm_bias",
     {
         "",
         "bottleneck_attention_LayerNorm",
         "",
         "bottleneck_attention_LayerNorm_bias",
     }},
    {"bottleneck_attention_dense_weight",
     {
         "hidden_states",
         "bottleneck_attention_dense",
         "",
         "bottleneck_attention_dense_weight",
     }},
    {"bottleneck_attention_dense_bias",
     {
         "",
         "bottleneck_attention_dense",
         "",
         "bottleneck_attention_dense_bias",
     }},

    {"bottleneck_input_LayerNorm_weight",
     {
         "bottleneck_input_dense",
         "bottleneck_input_LayerNorm",
         "",
         "bottleneck_input_LayerNorm_weight",
     }},
    {"bottleneck_input_LayerNorm_bias",
     {
         "",
         "bottleneck_input_LayerNorm",
         "",
         "bottleneck_input_LayerNorm_bias",
     }},
    {"bottleneck_input_dense_weight",
     {
         "hidden_states",
         "bottleneck_input_dense",
         "",
         "bottleneck_input_dense_weight",
     }},
    {"bottleneck_input_dense_bias",
     {
         "",
         "bottleneck_input_dense",
         "",
         "bottleneck_input_dense_bias",
     }},
};
