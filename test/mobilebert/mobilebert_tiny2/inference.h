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
    "attention_self_value_layer",
    "attention_self_attention_scores_0",
    "attention_self_attention_probs_0",
    "attention_self_attention_scores_1",
    "attention_self_attention_probs_1",
    "attention_self_attention_scores_2",
    "attention_self_attention_probs_2",
    "attention_self_attention_scores_3",
    "attention_self_attention_probs_3",
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

// (128 x 512) * (512 x 128)
const SimplifiedParams input_bottleneck = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
};

// (128 x 128) * 128
const SimplifiedParams input_bottleneck_LayerNorm = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = true,
};

// (128 x 16) x (16 x 128) + (128 x 128)
const SimplifiedParams merged_query_weight = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {1, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = false,
    .BIAS_OFFSET = 0,
    .RESIDUAL = true,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = false,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = false,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = false,
    .CONCAT_WEIGHT = false,
    .SPLIT_OUTPUT = false,
    .MERGE_LORA_WEIGHT = true,
};

// (512 x 16) x (16 x 128) + (512 x 128)
const SimplifiedParams value_weight = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{8, 1, 1, 1, 1, 1}, {1, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = false,
    .BIAS_OFFSET = 0,
    .RESIDUAL = true,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = false,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = false,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = false,
    .CONCAT_WEIGHT = false,
    .SPLIT_OUTPUT = false,
    .MERGE_LORA_WEIGHT = true,
};

const SimplifiedParams query_key_projection = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = false,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = false,
    .CONCAT_WEIGHT = false,
    .SPLIT_OUTPUT = true,
};

// (128 x 512) x (512 x 128)
const SimplifiedParams value_projection = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = false,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = false,
    .CONCAT_WEIGHT = false,
    .SPLIT_OUTPUT = true,
};

// (128 x 32) x (32 x 128)
const SimplifiedParams attention_score = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = true,
    .loops = {{2, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = false,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = true,
    .FC = false,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = false,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = false,
    .CONCAT_WEIGHT = false,
    .SPLIT_OUTPUT = false,
    .GRAD_CLIPPING = false,
    .GRAD_CLIPPING_UNIT_TEST = false,
    .WEIGHT_SPLITTING = false,
    .WEIGHT_RESIDUAL_OFFSET = 0,
    .learningRate = 0.0,
    .ACC_T_INPUT = false,
    .ACC_T_WEIGHT = false,
    .ACC_T_OUTPUT = true,
};

// (128 x 128)
const SimplifiedParams softmax = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = false,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = false,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = true,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = true,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = false,
    .CONCAT_WEIGHT = false,
    .SPLIT_OUTPUT = false,
    .GRAD_CLIPPING = false,
    .GRAD_CLIPPING_UNIT_TEST = false,
    .WEIGHT_SPLITTING = false,
    .WEIGHT_RESIDUAL_OFFSET = 0,
    .learningRate = 0.0,
    .ACC_T_INPUT = true,
};

// (128 x 128) x (128 x 32)
const SimplifiedParams context_layer = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
};

// (128 x 128) x (128 x 128)
const SimplifiedParams attention_output_dense = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = true,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = false,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = true,
};

// (128 x 128) x (128 x 512)
const SimplifiedParams intermediate_dense = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 2, 1, 1, 1}, {8, 16, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = true,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
};

// (128 x 512) x (512 x 128)
const SimplifiedParams output_dense = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = true,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
};

// (128 x 128) x (128 x 512)
const SimplifiedParams output_bottleneck_dense = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 2, 1, 1, 1}, {8, 16, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = true,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
};

// (128 x 512) * 512
const SimplifiedParams output_bottleneck_LayerNorm = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{2, 1, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 64}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = false,
    .NO_NORM = true,
};

// (1 x 512) x (512 x 16)
const SimplifiedParams classifier = {
    .INPUT_OFFSET = 0,
    .WEIGHT_OFFSET = 0,
    .OUTPUT_OFFSET = 0,
    .WEIGHT_TRANSPOSE = false,
    .loops = {{1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 1, 1}},
    .inputXLoopIndex = {0, 5},
    .inputYLoopIndex = {1, 4},
    .reductionLoopIndex = {3, 0},
    .weightLoopIndex = {2, 1},
    .fxIndex = 3,
    .fyIndex = 2,
    .weightReuseIndex = {4, 5},
    .STRIDE = 1,
    .REPLICATION = false,
    .RELU = false,
    .BIAS = true,
    .BIAS_OFFSET = 0,
    .RESIDUAL = false,
    .RESIDUAL_OFFSET = 0,
    .MAXPOOL = false,
    .AVGPOOL = false,
    .WEIGHT = true,
    .STORE_IN_ACC = false,
    .ACC_FROM_ACC = false,
    .SOFTMAX = false,
    .ATTENTION_SCALING = false,
    .FC = true,
    .NO_NORM = false,
    .SOFTMAX_GRAD = false,
    .FC_GRAD = false,
    .NO_NORM_GRAD = false,
    .RELU_GRAD = false,
    .BIAS_GRAD = false,
    .CROSS_ENTROPY_GRAD = false,
    .MSE_GRAD = false,
    .BCE_WITH_LOGITS_GRAD = false,
    .INPUT_TRANSPOSE = false,
    .CONCAT_INPUT = false,
    .CONCAT_WEIGHT = false,
    .SPLIT_OUTPUT = false,
    .GRAD_CLIPPING = false,
    .GRAD_CLIPPING_UNIT_TEST = false,
    .WEIGHT_SPLITTING = false,
    .WEIGHT_RESIDUAL_OFFSET = 0,
    .learningRate = 0.0,
    .ACC_T_INPUT = false,
    .ACC_T_WEIGHT = false,
    .ACC_T_OUTPUT = false,
};

std::map<std::string, SimplifiedParams> inferenceParams;
inferenceParams["bottleneck_input_dense"] = input_bottleneck;
inferenceParams["bottleneck_input_LayerNorm"] = input_bottleneck_LayerNorm;
inferenceParams["bottleneck_attention_dense"] = input_bottleneck;
inferenceParams["bottleneck_attention_LayerNorm"] = input_bottleneck_LayerNorm;
inferenceParams["attention_self_query_weight"] = merged_query_weight;
inferenceParams["attention_self_query_layer"] = query_key_projection;
inferenceParams["attention_self_key_layer"] = query_key_projection;
inferenceParams["attention_self_value_weight"] = value_weight;
inferenceParams["attention_self_value_layer"] = value_projection;
inferenceParams["attention_self_attention_scores_0"] = attention_score;
inferenceParams["attention_self_attention_scores_1"] = attention_score;
inferenceParams["attention_self_attention_scores_2"] = attention_score;
inferenceParams["attention_self_attention_scores_3"] = attention_score;
inferenceParams["attention_self_attention_probs_0"] = softmax;
inferenceParams["attention_self_attention_probs_1"] = softmax;
inferenceParams["attention_self_attention_probs_2"] = softmax;
inferenceParams["attention_self_attention_probs_3"] = softmax;
inferenceParams["attention_self_context_layer_0"] = context_layer;
inferenceParams["attention_self_context_layer_1"] = context_layer;
inferenceParams["attention_self_context_layer_2"] = context_layer;
inferenceParams["attention_self_context_layer_3"] = context_layer;
inferenceParams["attention_output_dense"] = attention_output_dense;
inferenceParams["attention_output_LayerNorm"] = input_bottleneck_LayerNorm;
inferenceParams["ffn_0_intermediate_dense"] = intermediate_dense;
inferenceParams["ffn_0_output_dense"] = output_dense;
inferenceParams["ffn_0_output_LayerNorm"] = input_bottleneck_LayerNorm;
inferenceParams["intermediate_dense"] = intermediate_dense;
inferenceParams["output_dense"] = output_dense;
inferenceParams["output_LayerNorm"] = input_bottleneck_LayerNorm;
inferenceParams["output_bottleneck_dense"] = output_bottleneck_dense;
inferenceParams["output_bottleneck_LayerNorm"] = output_bottleneck_LayerNorm;
inferenceParams["classifier"] = classifier;

const std::map<std::string, MemoryOffsets> inferenceMemOffsets{
    {"bottleneck_input_dense",
     {
         0,
         0,
         INTERMEDIATE_SIZE,
         INTERMEDIATE_SIZE,
     }},
    {"bottleneck_input_LayerNorm",
     {
         INTERMEDIATE_SIZE,
         INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"bottleneck_attention_dense",
     {
         0,
         INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         2 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"bottleneck_attention_LayerNorm",
     {
         INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
         2 * INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    // TODO: query lora weights
    {"attention_self_query_weight",
     {
         -1,
         -1,
         -1,
         0,
         -1,
     }},
    // TODO: value lora weights
    {"attention_self_value_weight",
     {
         -1,
         -1,
         -1,
         0,
         -1,
     }},
    {"attention_self_query_layer",
     {
         INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
         2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
         2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
             6 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_key_layer",
     {
         INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
         2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
             7 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE,
         2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             7 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_value_layer",
     {
         0,
         2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             8 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             8 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"attention_self_attention_scores_0",
     {
         INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 7 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_attention_probs_0",
     {
         INTERMEDIATE_SIZE + 7 * INTRA_BOTTLENECK_SIZE,
         0,
         INTERMEDIATE_SIZE + 7 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_attention_scores_1",
     {
         INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE + ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE + ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 8 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_attention_probs_1",
     {
         INTERMEDIATE_SIZE + 8 * INTRA_BOTTLENECK_SIZE,
         0,
         INTERMEDIATE_SIZE + 8 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_attention_scores_2",
     {
         INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
             2 * ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
             2 * ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 9 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_attention_probs_2",
     {
         INTERMEDIATE_SIZE + 9 * INTRA_BOTTLENECK_SIZE,
         0,
         INTERMEDIATE_SIZE + 9 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_attention_scores_3",
     {
         INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
             3 * ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
             3 * ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 10 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_attention_probs_3",
     {
         INTERMEDIATE_SIZE + 10 * INTRA_BOTTLENECK_SIZE,
         0,
         INTERMEDIATE_SIZE + 10 * INTRA_BOTTLENECK_SIZE,
     }},

    {"attention_self_context_layer_0",
     {
         INTERMEDIATE_SIZE + 7 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 11 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_self_context_layer_1",
     {
         INTERMEDIATE_SIZE + 8 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             1 * ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 11 * INTRA_BOTTLENECK_SIZE +
             1 * ATTENTION_HEAD_SIZE,
     }},
    {"attention_self_context_layer_2",
     {
         INTERMEDIATE_SIZE + 9 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             2 * ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 11 * INTRA_BOTTLENECK_SIZE +
             2 * ATTENTION_HEAD_SIZE,
     }},
    {"attention_self_context_layer_3",
     {
         INTERMEDIATE_SIZE + 10 * INTRA_BOTTLENECK_SIZE,
         INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             3 * ATTENTION_HEAD_SIZE,
         INTERMEDIATE_SIZE + 11 * INTRA_BOTTLENECK_SIZE +
             3 * ATTENTION_HEAD_SIZE,
     }},

    {"attention_output_dense",
     {
         INTERMEDIATE_SIZE + 11 * INTRA_BOTTLENECK_SIZE,
         3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             9 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 12 * INTRA_BOTTLENECK_SIZE,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             9 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
     }},
    {"attention_output_LayerNorm",
     {
         INTERMEDIATE_SIZE + 12 * INTRA_BOTTLENECK_SIZE,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             10 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 13 * INTRA_BOTTLENECK_SIZE,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             11 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"ffn_0_intermediate_dense",
     {
         INTERMEDIATE_SIZE + 13 * INTRA_BOTTLENECK_SIZE,
         3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             12 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 14 * INTRA_BOTTLENECK_SIZE,
         4 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             12 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"ffn_0_output_dense",
     {
         INTERMEDIATE_SIZE + 14 * INTRA_BOTTLENECK_SIZE,
         4 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE,
         2 * INTERMEDIATE_SIZE + 14 * INTRA_BOTTLENECK_SIZE,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE,
         INTERMEDIATE_SIZE + 13 * INTRA_BOTTLENECK_SIZE,
     }},
    {"ffn_0_output_LayerNorm",
     {
         2 * INTERMEDIATE_SIZE + 14 * INTRA_BOTTLENECK_SIZE,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 13 * INTRA_BOTTLENECK_BIAS_SIZE,
         2 * INTERMEDIATE_SIZE + 15 * INTRA_BOTTLENECK_SIZE,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 14 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"intermediate_dense",
     {
         2 * INTERMEDIATE_SIZE + 15 * INTRA_BOTTLENECK_SIZE,
         5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
         2 * INTERMEDIATE_SIZE + 16 * INTRA_BOTTLENECK_SIZE,
         6 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_dense",
     {
         2 * INTERMEDIATE_SIZE + 16 * INTRA_BOTTLENECK_SIZE,
         6 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
         3 * INTERMEDIATE_SIZE + 16 * INTRA_BOTTLENECK_SIZE,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 15 * INTRA_BOTTLENECK_BIAS_SIZE,
         2 * INTERMEDIATE_SIZE + 15 * INTRA_BOTTLENECK_SIZE,
     }},
    {"output_LayerNorm",
     {
         3 * INTERMEDIATE_SIZE + 16 * INTRA_BOTTLENECK_SIZE,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 16 * INTRA_BOTTLENECK_BIAS_SIZE,
         3 * INTERMEDIATE_SIZE + 17 * INTRA_BOTTLENECK_SIZE,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 17 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"output_bottleneck_dense",
     {
         3 * INTERMEDIATE_SIZE + 17 * INTRA_BOTTLENECK_SIZE,
         7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         3 * INTERMEDIATE_SIZE + 18 * INTRA_BOTTLENECK_SIZE,
         8 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         0,
     }},
    {"output_bottleneck_LayerNorm",
     {
         3 * INTERMEDIATE_SIZE + 18 * INTRA_BOTTLENECK_SIZE,
         8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         0,
         8 * INTERMEDIATE_SIZE + 4 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
    {"classifier",
     {
         0,
         8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         5 * INTERMEDIATE_SIZE + 18 * INTRA_BOTTLENECK_SIZE,
         8 * INTERMEDIATE_SIZE + 21 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
     }},
};

const std::map<std::string, Files> inferenceTestFiles{
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
    // TODO: query lora weights
    {"attention_self_query_weight", {"", "", "", ""}},
    // TODO: value lora weights
    {"attention_self_value_weight", {"", "", "", ""}},
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
         "mobilebert_encoder_layer_20_output_bottleneck_LayerNorm",
         "classifier_weight",
         "classifier_bias",
         "classifier",
     }},
};
