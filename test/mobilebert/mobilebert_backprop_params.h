#ifndef MOBILEBERT_BACKPROP_PARAMS
#define MOBILEBERT_BACKPROP_PARAMS

#include "test/common/VerificationTypes.h"

const SimplifiedParams output_bottleneck_dense_params = {
    29175968,                                     // INPUT_OFFSET
    581120,                                       // WEIGHT_OFFSET
    29110432,                                     // OUTPUT_OFFSET
    0,                                            // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                       // INPUTX
    {1, 4},                                       // INPUTY
    {3, 0},                                       // REDUCTION
    {2, 1},                                       // WEIGHT
    3,                                            // fxIndex
    2,                                            // fyIndex
    {4, 5},                                       // weightReuseIndex
    1,                                            // stride
    false,                                        // replication
    false,                                        // ReLU
    false,                                        // bias
    0,                                            // BIAS_OFFSET
    false,                                        // residual
    29110432,                                     // RESIDUAL_OFFSET
    false,                                        // MAXPOOL
    false,                                        // AVGPOOL
    true,                                         // WEIGHT
    false,                                        // STORE_IN_ACC
    false,                                        // ACC_FROM_ACC
    false,                                        // SOFTMAX
    false,                                        // ATTENTION_MASK
    false,                                        // ATTENTION_SCALING
    false,                                        // FC
    true,                                         // NO_NORM
    false,                                        // SOFTMAX_GRAD
    false,                                        // FC_GRAD
    false,                                        // NO_NORM_GRAD
    false,                                        // RELU_GRAD
    false,                                        // BIAS_GRAD
    false,                                        // CROSS_ENTROPY_GRAD
    false,                                        // MSE_GRAD
    false,                                        // BCE_WITH_LOGITS_GRAD
    false,                                        // INPUT_TRANSPOSE
    false,                                        // CONCAT_INPUT
    false,                                        // CONCAT_WEIGHT
    false,                                        // SPLIT_OUTPUT
    false,                                        // GRAD_CLIPPING
    false,                                        // WEIGHT_SPLITTING
    0,                                            // WEIGHT_GRADIENT_OFFSET
    0,                                            // BIAS_GRADIENT_OFFSET
    0,                                            // learningRate
    false,                                        // ACC_T_INPUT
    false,                                        // ACC_T_WEIGHT
    false,                                        // ACC_T_OUTPUT
};

const SimplifiedParams output_bottleneck_dense_weight_params = {
    630912,                                     // INPUT_OFFSET
    29110432,                                   // WEIGHT_OFFSET
    15612560,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15612560,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams output_bottleneck_dense_bias_params = {
    90240,                                      // INPUT_OFFSET
    29110432,                                   // WEIGHT_OFFSET
    15678096,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15678096,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    true,                                       // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    true,                                       // ACC_T_OUTPUT
};

const SimplifiedParams output_LayerNorm_params = {
    29110432,                                    // INPUT_OFFSET
    514560,                                      // WEIGHT_OFFSET
    29175968,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    false,                                       // residual
    29110432,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams output_LayerNorm_weight_params = {
    614528,                                     // INPUT_OFFSET
    29175968,                                   // WEIGHT_OFFSET
    15612048,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15612048,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    true,                                       // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams output_LayerNorm_bias_params = {
    90240,                                     // INPUT_OFFSET
    29175968,                                  // WEIGHT_OFFSET
    15612304,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15612304,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams output_dense_params = {
    29175968,                                   // INPUT_OFFSET
    514048,                                     // WEIGHT_OFFSET
    29192352,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {2, 0},                                     // REDUCTION
    {3, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    true,                                       // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams output_dense_weight_params = {
    548992,                                      // INPUT_OFFSET
    29192352,                                    // WEIGHT_OFFSET
    15546256,                                    // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    15546256,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    false,                                       // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    true,                                        // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    true,                                        // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams output_dense_bias_params = {
    90240,                                     // INPUT_OFFSET
    29192352,                                  // WEIGHT_OFFSET
    15611792,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15611792,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams intermediate_dense_params = {
    29192352,                                    // INPUT_OFFSET
    448256,                                      // WEIGHT_OFFSET
    29208736,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    false,                                       // residual
    548992,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    true,                                        // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams intermediate_dense_weight_params = {
    532608,                                     // INPUT_OFFSET
    29208736,                                   // WEIGHT_OFFSET
    15479696,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15479696,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams intermediate_dense_bias_params = {
    90240,                                      // INPUT_OFFSET
    29208736,                                   // WEIGHT_OFFSET
    15545232,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15545232,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    true,                                       // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    true,                                       // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_LayerNorm_params = {
    29208736,                                    // INPUT_OFFSET
    381696,                                      // WEIGHT_OFFSET
    29175968,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    29192352,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_LayerNorm_weight_params = {
    516224,                                     // INPUT_OFFSET
    29175968,                                   // WEIGHT_OFFSET
    15479184,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15479184,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    true,                                       // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_LayerNorm_bias_params = {
    90240,                                     // INPUT_OFFSET
    29175968,                                  // WEIGHT_OFFSET
    15479440,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15479440,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_dense_params = {
    29175968,                                   // INPUT_OFFSET
    381184,                                     // WEIGHT_OFFSET
    29192352,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {2, 0},                                     // REDUCTION
    {3, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    true,                                       // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_dense_weight_params = {
    450688,                                      // INPUT_OFFSET
    29192352,                                    // WEIGHT_OFFSET
    15413392,                                    // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    15413392,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    false,                                       // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    true,                                        // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    true,                                        // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_dense_bias_params = {
    90240,                                     // INPUT_OFFSET
    29192352,                                  // WEIGHT_OFFSET
    15478928,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15478928,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_intermediate_dense_params = {
    29192352,                                    // INPUT_OFFSET
    315392,                                      // WEIGHT_OFFSET
    29208736,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    false,                                       // residual
    450688,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    true,                                        // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_intermediate_dense_weight_params = {
    434304,                                     // INPUT_OFFSET
    29208736,                                   // WEIGHT_OFFSET
    15346832,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15346832,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_intermediate_dense_bias_params = {
    90240,                                      // INPUT_OFFSET
    29208736,                                   // WEIGHT_OFFSET
    15412368,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15412368,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    true,                                       // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    true,                                       // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_LayerNorm_params = {
    29208736,                                    // INPUT_OFFSET
    248832,                                      // WEIGHT_OFFSET
    29175968,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    29192352,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_LayerNorm_weight_params = {
    417920,                                     // INPUT_OFFSET
    29175968,                                   // WEIGHT_OFFSET
    15346320,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15346320,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    true,                                       // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_LayerNorm_bias_params = {
    90240,                                     // INPUT_OFFSET
    29175968,                                  // WEIGHT_OFFSET
    15346576,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15346576,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_dense_params = {
    29175968,                                   // INPUT_OFFSET
    248320,                                     // WEIGHT_OFFSET
    29192352,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {2, 0},                                     // REDUCTION
    {3, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    true,                                       // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_dense_weight_params = {
    401536,                                     // INPUT_OFFSET
    29192352,                                   // WEIGHT_OFFSET
    15329680,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15329680,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    true,                                       // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_dense_bias_params = {
    90240,                                     // INPUT_OFFSET
    29192352,                                  // WEIGHT_OFFSET
    15346064,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15346064,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_input_LayerNorm_weight_params = {
    155776,                                     // INPUT_OFFSET
    29192352,                                   // WEIGHT_OFFSET
    15163792,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15163792,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    true,                                       // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_input_LayerNorm_bias_params = {
    90240,                                     // INPUT_OFFSET
    29192352,                                  // WEIGHT_OFFSET
    15164048,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15164048,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_input_dense_params = {
    29192352,                                   // INPUT_OFFSET
    65792,                                      // WEIGHT_OFFSET
    29175968,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {2, 0},                                     // REDUCTION
    {3, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    true,                                       // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_input_dense_weight_params = {
    90240,                                       // INPUT_OFFSET
    29175968,                                    // WEIGHT_OFFSET
    15098000,                                    // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    15098000,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    false,                                       // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    true,                                        // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    true,                                        // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_input_dense_bias_params = {
    90240,                                     // INPUT_OFFSET
    29175968,                                  // WEIGHT_OFFSET
    15163536,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15163536,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_context_layer_params = {
    29192352,                                   // INPUT_OFFSET
    231680,                                     // WEIGHT_OFFSET
    29208736,                                   // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    true,                                       // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_value_layer_3_params = {
    385152,                                     // INPUT_OFFSET
    29221024,                                   // WEIGHT_OFFSET
    29204640,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_value_layer_2_params = {
    368768,                                     // INPUT_OFFSET
    29216928,                                   // WEIGHT_OFFSET
    29200544,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_value_layer_1_params = {
    352384,                                     // INPUT_OFFSET
    29212832,                                   // WEIGHT_OFFSET
    29196448,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_value_layer_0_params = {
    336000,                                     // INPUT_OFFSET
    29208736,                                   // WEIGHT_OFFSET
    29192352,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_value_weight_params = {
    90240,                                       // INPUT_OFFSET
    29192352,                                    // WEIGHT_OFFSET
    15263888,                                    // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    15263888,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    false,                                       // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    true,                                        // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    true,                                        // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    true,                                        // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_value_bias_params = {
    90240,                                     // INPUT_OFFSET
    29192352,                                  // WEIGHT_OFFSET
    15329424,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15329424,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    true,                                      // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_0_params = {
    29208736,                                   // INPUT_OFFSET
    188544,                                     // WEIGHT_OFFSET
    29225120,                                   // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_1_params = {
    29212832,                                   // INPUT_OFFSET
    192640,                                     // WEIGHT_OFFSET
    29241504,                                   // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_2_params = {
    29216928,                                   // INPUT_OFFSET
    196736,                                     // WEIGHT_OFFSET
    29257888,                                   // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_3_params = {
    29221024,                                   // INPUT_OFFSET
    200832,                                     // WEIGHT_OFFSET
    29274272,                                   // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_0_params = {
    29225120,                                      // INPUT_OFFSET
    90240,                                         // WEIGHT_OFFSET
    29208736,                                      // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    336000,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    false,                                         // SOFTMAX
    false,                                         // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    true,                                          // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    0,                                             // WEIGHT_GRADIENT_OFFSET
    0,                                             // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_1_params = {
    29241504,                                      // INPUT_OFFSET
    90240,                                         // WEIGHT_OFFSET
    29225120,                                      // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    352384,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    false,                                         // SOFTMAX
    false,                                         // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    true,                                          // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    0,                                             // WEIGHT_GRADIENT_OFFSET
    0,                                             // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_2_params = {
    29257888,                                      // INPUT_OFFSET
    90240,                                         // WEIGHT_OFFSET
    29241504,                                      // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    368768,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    false,                                         // SOFTMAX
    false,                                         // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    true,                                          // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    0,                                             // WEIGHT_GRADIENT_OFFSET
    0,                                             // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_3_params = {
    29274272,                                      // INPUT_OFFSET
    90240,                                         // WEIGHT_OFFSET
    29257888,                                      // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    385152,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    false,                                         // SOFTMAX
    false,                                         // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    true,                                          // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    0,                                             // WEIGHT_GRADIENT_OFFSET
    0,                                             // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_query_layer_3_params = {
    29257888,                                   // INPUT_OFFSET
    266368,                                     // WEIGHT_OFFSET
    29286560,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_query_layer_2_params = {
    29241504,                                   // INPUT_OFFSET
    262272,                                     // WEIGHT_OFFSET
    29282464,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_query_layer_1_params = {
    29225120,                                   // INPUT_OFFSET
    258176,                                     // WEIGHT_OFFSET
    29278368,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_query_layer_0_params = {
    29208736,                                   // INPUT_OFFSET
    254080,                                     // WEIGHT_OFFSET
    29274272,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_query_weight_params = {
    221312,                                     // INPUT_OFFSET
    29274272,                                   // WEIGHT_OFFSET
    15230608,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15230608,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    true,                                       // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_query_bias_params = {
    90240,                                     // INPUT_OFFSET
    29274272,                                  // WEIGHT_OFFSET
    15246992,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15246992,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    true,                                      // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_key_layer_3_params = {
    29257888,                                   // INPUT_OFFSET
    249984,                                     // WEIGHT_OFFSET
    29302944,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_key_layer_2_params = {
    29241504,                                   // INPUT_OFFSET
    245888,                                     // WEIGHT_OFFSET
    29298848,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_key_layer_1_params = {
    29225120,                                   // INPUT_OFFSET
    241792,                                     // WEIGHT_OFFSET
    29294752,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_key_layer_0_params = {
    29208736,                                   // INPUT_OFFSET
    237696,                                     // WEIGHT_OFFSET
    29290656,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_key_weight_params = {
    221312,                                     // INPUT_OFFSET
    29290656,                                   // WEIGHT_OFFSET
    15247248,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15247248,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    true,                                       // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    true,                                       // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_key_bias_params = {
    90240,                                     // INPUT_OFFSET
    29290656,                                  // WEIGHT_OFFSET
    15263632,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15263632,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    true,                                      // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_LayerNorm_q_params = {
    29274272,                                   // INPUT_OFFSET
    132608,                                     // WEIGHT_OFFSET
    29208736,                                   // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    true,                                       // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_LayerNorm_k_params = {
    29290656,                                   // INPUT_OFFSET
    149248,                                     // WEIGHT_OFFSET
    29225120,                                   // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {5, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    29208736,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    true,                                       // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_LayerNorm_weight_params = {
    172160,                                     // INPUT_OFFSET
    29225120,                                   // WEIGHT_OFFSET
    15230096,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    true,                                       // residual
    15230096,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    true,                                       // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    true,                                       // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_LayerNorm_bias_params = {
    90240,                                     // INPUT_OFFSET
    29225120,                                  // WEIGHT_OFFSET
    15230352,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15230352,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_dense_params = {
    29225120,                                   // INPUT_OFFSET
    132096,                                     // WEIGHT_OFFSET
    29208736,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {2, 0},                                     // REDUCTION
    {3, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    29110432,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    true,                                       // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    0,                                          // WEIGHT_GRADIENT_OFFSET
    0,                                          // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_dense_weight_params = {
    90240,                                       // INPUT_OFFSET
    29208736,                                    // WEIGHT_OFFSET
    15164304,                                    // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{16, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    15164304,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    false,                                       // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    true,                                        // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    true,                                        // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_dense_bias_params = {
    90240,                                     // INPUT_OFFSET
    29208736,                                  // WEIGHT_OFFSET
    15229840,                                  // OUTPUT_OFFSET
    0,                                         // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    1,                                         // stride
    false,                                     // replication
    false,                                     // ReLU
    false,                                     // bias
    0,                                         // BIAS_OFFSET
    true,                                      // residual
    15229840,                                  // RESIDUAL_OFFSET
    false,                                     // MAXPOOL
    false,                                     // AVGPOOL
    false,                                     // WEIGHT
    false,                                     // STORE_IN_ACC
    false,                                     // ACC_FROM_ACC
    false,                                     // SOFTMAX
    false,                                     // ATTENTION_MASK
    false,                                     // ATTENTION_SCALING
    false,                                     // FC
    false,                                     // NO_NORM
    false,                                     // SOFTMAX_GRAD
    false,                                     // FC_GRAD
    false,                                     // NO_NORM_GRAD
    false,                                     // RELU_GRAD
    true,                                      // BIAS_GRAD
    false,                                     // CROSS_ENTROPY_GRAD
    false,                                     // MSE_GRAD
    false,                                     // BCE_WITH_LOGITS_GRAD
    false,                                     // INPUT_TRANSPOSE
    false,                                     // CONCAT_INPUT
    false,                                     // CONCAT_WEIGHT
    false,                                     // SPLIT_OUTPUT
    true,                                      // GRAD_CLIPPING
    false,                                     // WEIGHT_SPLITTING
    0,                                         // WEIGHT_GRADIENT_OFFSET
    0,                                         // BIAS_GRADIENT_OFFSET
    0,                                         // learningRate
    false,                                     // ACC_T_INPUT
    false,                                     // ACC_T_WEIGHT
    true,                                      // ACC_T_OUTPUT
};

const SimplifiedParams shared_attention_input_to_hidden_states_params = {
    29175968,                                    // INPUT_OFFSET
    0,                                           // WEIGHT_OFFSET
    29110432,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {5, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    29110432,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams value_to_hidden_states_params = {
    29192352,                                    // INPUT_OFFSET
    165888,                                      // WEIGHT_OFFSET
    29110432,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {5, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    29110432,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    true,                                        // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams bottlenecked_hidden_states_params = {
    29208736,                                    // INPUT_OFFSET
    66304,                                       // WEIGHT_OFFSET
    29175968,                                    // OUTPUT_OFFSET
    1,                                           // WEIGHT_TRANSPOSE
    {{8, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {5, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    false,                                       // bias
    0,                                           // BIAS_OFFSET
    true,                                        // residual
    29110432,                                    // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    0,                                           // WEIGHT_GRADIENT_OFFSET
    0,                                           // BIAS_GRADIENT_OFFSET
    0,                                           // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

#endif
