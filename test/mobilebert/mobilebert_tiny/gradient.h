#pragma once

#include <map>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

std::map<std::string, std::string> gradientParamsMapping{
    {"classifier_weight", "classifier_weight"},
    {"classifier_bias", "classifier_bias"},
    {"output_bottleneck_LayerNorm_weight", "outputLayerNorm"},
    {"output_bottleneck_LayerNorm_bias", "interBottleneckBias"},
    {"output_bottleneck_dense_weight", "outputBottleneck"},
    {"output_bottleneck_dense_bias", "interBottleneckBias"},
    {"output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"output_LayerNorm_bias", "intraBottleneckBias"},
    {"output_dense_weight", "ffn2"},
    {"output_dense_bias", "intraBottleneckBias"},
    {"intermediate_dense_weight", "outputBottleneck"},
    {"intermediate_dense_bias", "interBottleneckBias"},
    {"ffn_0_output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"ffn_0_output_LayerNorm_bias", "intraBottleneckBias"},
    {"ffn_0_output_dense_weight", "ffn2"},
    {"ffn_0_output_dense_bias", "intraBottleneckBias"},
    {"ffn_0_intermediate_dense_weight", "outputBottleneck"},
    {"ffn_0_intermediate_dense_bias", "interBottleneckBias"},
    {"attention_output_LayerNorm_weight", "bottleneckLayerNorm"},
    {"attention_output_LayerNorm_bias", "intraBottleneckBias"},
    {"attention_output_dense_weight", "outputAttention"},
    {"attention_output_dense_bias", "intraBottleneckBias"},
    {"attention_self_value_weight", "qkvProjection"},
    {"attention_self_value_bias", "qkvBias"},
    {"attention_self_query_weight", "qkvProjection"},
    {"attention_self_query_bias", "qkvBias"},
    {"attention_self_key_weight", "qkvProjection"},
    {"attention_self_key_bias", "qkvBias"},
    {"bottleneck_input_LayerNorm_weight", "bottleneckLayerNorm"},
    {"bottleneck_input_LayerNorm_bias", "intraBottleneckBias"},
    {"bottleneck_input_dense_weight", "inputBottleneck"},
    {"bottleneck_input_dense_bias", "intraBottleneckBias"},
};

std::map<std::string, SimplifiedParams> gradientParams{
    // (16 x 1) x (1 x 512)
    {"classifier_weight",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {1, 32, 1, 1, 1, 16}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .FC_GRAD = true,
         .GRAD_CLIPPING = true,
     }},

    // (16 x 1)
    {"classifier_bias",
     {
         .loops = {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}},
         .inputXLoopIndex = {0, 5},
         .inputYLoopIndex = {1, 4},
         .reductionLoopIndex = {3, 0},
         .weightLoopIndex = {2, 1},
         .fxIndex = 3,
         .fyIndex = 2,
         .weightReuseIndex = {4, 5},
         .STRIDE = 1,
         .GRAD_CLIPPING_UNIT_TEST = true,
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
         .loops = {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},
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

    // (128 x 128) x (128 x 128)
    {"qkvProjection",
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
    {"intraBottleneckBias",
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
         .ACC_T_OUTPUT = true,
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
         .ACC_T_OUTPUT = true,
     }},

    // (128 x 512)
    {"interBottleneckBias",
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
         .ACC_T_OUTPUT = true,
     }},
};

std::map<std::string, MemoryOffsets> gradientMemOffsets{
    {"classifier_weight",
     {
         0,
         4 * INTERMEDIATE_SIZE + 22 * INTRA_BOTTLENECK_SIZE,
         8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"classifier_bias",
     {
         0,
         0,
         8 * INTERMEDIATE_SIZE + 21 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_bottleneck_LayerNorm_weight",
     {
         3 * INTERMEDIATE_SIZE + 22 * INTRA_BOTTLENECK_SIZE,
         0,
         8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_bottleneck_LayerNorm_bias",
     {
         0,
         0,
         8 * INTERMEDIATE_SIZE + 4 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_bottleneck_dense_weight",
     {
         3 * INTERMEDIATE_SIZE + 21 * INTRA_BOTTLENECK_SIZE,
         0,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_bottleneck_dense_bias",
     {
         0,
         0,
         8 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},

    {"output_LayerNorm_weight",
     {
         3 * INTERMEDIATE_SIZE + 20 * INTRA_BOTTLENECK_SIZE,
         0,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 16 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_LayerNorm_bias",
     {
         0,
         0,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 17 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_dense_weight",
     {
         2 * INTERMEDIATE_SIZE + 20 * INTRA_BOTTLENECK_SIZE,
         0,
         6 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_dense_bias",
     {
         0,
         0,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"intermediate_dense_weight",
     {
         2 * INTERMEDIATE_SIZE + 19 * INTRA_BOTTLENECK_SIZE,
         0,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"intermediate_dense_bias",
     {
         0,
         0,
         6 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},

    {"ffn_0_output_LayerNorm_weight",
     {
         2 * INTERMEDIATE_SIZE + 18 * INTRA_BOTTLENECK_SIZE,
         0,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 13 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"ffn_0_output_LayerNorm_bias",
     {
         0,
         0,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 14 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"ffn_0_output_dense_weight",
     {
         INTERMEDIATE_SIZE + 18 * INTRA_BOTTLENECK_SIZE,
         0,
         4 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"ffn_0_output_dense_bias",
     {
         0,
         0,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"ffn_0_intermediate_dense_weight",
     {
         INTERMEDIATE_SIZE + 17 * INTRA_BOTTLENECK_SIZE,
         0,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             12 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"ffn_0_intermediate_dense_bias",
     {
         0,
         0,
         4 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             12 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},

    {"attention_output_LayerNorm_weight",
     {
         INTERMEDIATE_SIZE + 16 * INTRA_BOTTLENECK_SIZE,
         0,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             10 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_output_LayerNorm_bias",
     {
         0,
         0,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             11 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_output_dense_weight",
     {
         INTERMEDIATE_SIZE + 15 * INTRA_BOTTLENECK_SIZE,
         0,
         3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             9 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_output_dense_bias",
     {
         0,
         0,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             9 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},

    {"attention_self_value_weight",
     {
         0,
         0,
         2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             8 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_value_bias",
     {
         0,
         0,
         3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             8 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_key_weight",
     {
         INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
         0,
         2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
             7 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_key_bias",
     {
         0,
         0,
         2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             7 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_query_weight",
     {
         INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
         0,
         2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_query_bias",
     {
         0,
         0,
         2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
             6 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},

    {"bottleneck_input_LayerNorm_weight",
     {
         INTERMEDIATE_SIZE,
         0,
         INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"bottleneck_input_LayerNorm_bias",
     {
         0,
         0,
         INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_BIAS_SIZE,
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
         INTERMEDIATE_SIZE,
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
         "mobilebert_logits",
         "",
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
         "ffn_0_output_LayerNorm",
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
         "bottleneck_input_LayerNorm",
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
         "bottleneck_input_LayerNorm",
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
         "bottleneck_input_LayerNorm",
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
