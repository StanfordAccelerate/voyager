#ifndef MOBILEBERT_PARAMS
#define MOBILEBERT_PARAMS

const SimplifiedParams bottleneck_input_dense_params = {
    180352,                                      // INPUT_OFFSET
    0,                                           // WEIGHT_OFFSET
    245888,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                        // bias
    65536,                                       // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
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
    15188112,                                    // WEIGHT_GRADIENT_OFFSET
    15253648,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams bottleneck_input_LayerNorm_params = {
    245888,                                      // INPUT_OFFSET
    65792,                                       // WEIGHT_OFFSET
    295040,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
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
    true,                                        // bias
    66048,                                       // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
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
    15253904,                                    // WEIGHT_GRADIENT_OFFSET
    15254160,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams bottleneck_attention_dense_params = {
    180352,                                      // INPUT_OFFSET
    66304,                                       // WEIGHT_OFFSET
    262272,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                        // bias
    131840,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
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
    15254416,                                    // WEIGHT_GRADIENT_OFFSET
    15319952,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams bottleneck_attention_LayerNorm_params = {
    262272,                                      // INPUT_OFFSET
    132096,                                      // WEIGHT_OFFSET
    311424,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
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
    true,                                        // bias
    132352,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
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
    15320208,                                    // WEIGHT_GRADIENT_OFFSET
    15320464,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_query_layer_params = {
    311424,                                     // INPUT_OFFSET
    132608,                                     // WEIGHT_OFFSET
    327808,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                       // bias
    148992,                                     // BIAS_OFFSET
    false,                                      // residual
    180352,                                     // RESIDUAL_OFFSET
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
    15320720,                                   // WEIGHT_GRADIENT_OFFSET
    15337104,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_key_layer_params = {
    311424,                                     // INPUT_OFFSET
    149248,                                     // WEIGHT_OFFSET
    344192,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                       // bias
    165632,                                     // BIAS_OFFSET
    false,                                      // residual
    180352,                                     // RESIDUAL_OFFSET
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
    15337360,                                   // WEIGHT_GRADIENT_OFFSET
    15353744,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_scores_0_params = {
    327808,                                     // INPUT_OFFSET
    344192,                                     // WEIGHT_OFFSET
    360576,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
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
    15532304,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_scores_1_params = {
    331904,                                     // INPUT_OFFSET
    348288,                                     // WEIGHT_OFFSET
    376960,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
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
    15536400,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_scores_2_params = {
    336000,                                     // INPUT_OFFSET
    352384,                                     // WEIGHT_OFFSET
    393344,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
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
    15540496,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_scores_3_params = {
    340096,                                     // INPUT_OFFSET
    356480,                                     // WEIGHT_OFFSET
    409728,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
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
    15544592,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_probs_0_params = {
    360576,                                        // INPUT_OFFSET
    180224,                                        // WEIGHT_OFFSET
    426112,                                        // OUTPUT_OFFSET
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
    180352,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
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
    15368464,                                      // WEIGHT_GRADIENT_OFFSET
    15188112,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_probs_1_params = {
    376960,                                        // INPUT_OFFSET
    180224,                                        // WEIGHT_OFFSET
    442496,                                        // OUTPUT_OFFSET
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
    180352,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
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
    15368464,                                      // WEIGHT_GRADIENT_OFFSET
    15188112,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_probs_2_params = {
    393344,                                        // INPUT_OFFSET
    180224,                                        // WEIGHT_OFFSET
    458880,                                        // OUTPUT_OFFSET
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
    180352,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
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
    15368464,                                      // WEIGHT_GRADIENT_OFFSET
    15188112,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_attention_probs_3_params = {
    409728,                                        // INPUT_OFFSET
    180224,                                        // WEIGHT_OFFSET
    475264,                                        // OUTPUT_OFFSET
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
    180352,                                        // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
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
    15368464,                                      // WEIGHT_GRADIENT_OFFSET
    15188112,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_value_layer_params = {
    180352,                                      // INPUT_OFFSET
    165888,                                      // WEIGHT_OFFSET
    278656,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                        // bias
    231424,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
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
    true,                                        // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15354000,                                    // WEIGHT_GRADIENT_OFFSET
    15419536,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_context_layer_0_params = {
    426112,                                     // INPUT_OFFSET
    278656,                                     // WEIGHT_OFFSET
    491648,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
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
    15466768,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_context_layer_1_params = {
    442496,                                     // INPUT_OFFSET
    282752,                                     // WEIGHT_OFFSET
    495744,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
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
    15470864,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_context_layer_2_params = {
    458880,                                     // INPUT_OFFSET
    286848,                                     // WEIGHT_OFFSET
    499840,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
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
    15474960,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_self_context_layer_3_params = {
    475264,                                     // INPUT_OFFSET
    290944,                                     // WEIGHT_OFFSET
    503936,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
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
    180352,                                     // RESIDUAL_OFFSET
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
    15479056,                                   // WEIGHT_GRADIENT_OFFSET
    15188112,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_output_dense_params = {
    491648,                                     // INPUT_OFFSET
    231680,                                     // WEIGHT_OFFSET
    508032,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                       // bias
    248064,                                     // BIAS_OFFSET
    true,                                       // residual
    295040,                                     // RESIDUAL_OFFSET
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
    15419792,                                   // WEIGHT_GRADIENT_OFFSET
    15436176,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
const SimplifiedParams attention_output_LayerNorm_params = {
    508032,                                      // INPUT_OFFSET
    248320,                                      // WEIGHT_OFFSET
    524416,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
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
    true,                                        // bias
    248576,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
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
    15436432,                                    // WEIGHT_GRADIENT_OFFSET
    15436688,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams ffn_0_intermediate_dense_params = {
    524416,                                      // INPUT_OFFSET
    248832,                                      // WEIGHT_OFFSET
    540800,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // ReLU
    true,                                        // bias
    314368,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
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
    15436944,                                    // WEIGHT_GRADIENT_OFFSET
    15502480,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams ffn_0_output_dense_params = {
    540800,                                      // INPUT_OFFSET
    315392,                                      // WEIGHT_OFFSET
    606336,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                        // bias
    380928,                                      // BIAS_OFFSET
    true,                                        // residual
    524416,                                      // RESIDUAL_OFFSET
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
    15503504,                                    // WEIGHT_GRADIENT_OFFSET
    15569040,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams ffn_0_output_LayerNorm_params = {
    606336,                                      // INPUT_OFFSET
    381184,                                      // WEIGHT_OFFSET
    622720,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
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
    true,                                        // bias
    381440,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
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
    15569296,                                    // WEIGHT_GRADIENT_OFFSET
    15569552,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams intermediate_dense_params = {
    622720,                                      // INPUT_OFFSET
    381696,                                      // WEIGHT_OFFSET
    639104,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // ReLU
    true,                                        // bias
    447232,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
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
    15569808,                                    // WEIGHT_GRADIENT_OFFSET
    15635344,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams output_dense_params = {
    639104,                                      // INPUT_OFFSET
    448256,                                      // WEIGHT_OFFSET
    704640,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
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
    true,                                        // bias
    513792,                                      // BIAS_OFFSET
    true,                                        // residual
    622720,                                      // RESIDUAL_OFFSET
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
    15636368,                                    // WEIGHT_GRADIENT_OFFSET
    15701904,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams output_LayerNorm_params = {
    704640,                                      // INPUT_OFFSET
    514048,                                      // WEIGHT_OFFSET
    721024,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
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
    true,                                        // bias
    514304,                                      // BIAS_OFFSET
    false,                                       // residual
    180352,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
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
    15702160,                                    // WEIGHT_GRADIENT_OFFSET
    15702416,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams output_bottleneck_dense_params = {
    721024,                                      // INPUT_OFFSET
    514560,                                      // WEIGHT_OFFSET
    737408,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
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
    true,                                        // bias
    580096,                                      // BIAS_OFFSET
    true,                                        // residual
    180352,                                      // RESIDUAL_OFFSET
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
    15702672,                                    // WEIGHT_GRADIENT_OFFSET
    15768208,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};
const SimplifiedParams output_bottleneck_LayerNorm_params = {
    737408,                                       // INPUT_OFFSET
    581120,                                       // WEIGHT_OFFSET
    802944,                                       // OUTPUT_OFFSET
    0,                                            // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 32}},  // LOOPS
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
    true,                                         // bias
    582144,                                       // BIAS_OFFSET
    false,                                        // residual
    180352,                                       // RESIDUAL_OFFSET
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
    15769232,                                     // WEIGHT_GRADIENT_OFFSET
    15770256,                                     // BIAS_GRADIENT_OFFSET
    -0.02,                                        // learningRate
    false,                                        // ACC_T_INPUT
    false,                                        // ACC_T_WEIGHT
    false,                                        // ACC_T_OUTPUT
};
const SimplifiedParams classifier_params = {
    15122560,                                   // INPUT_OFFSET
    13996032,                                   // WEIGHT_OFFSET
    15188096,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 1, 1}},  // LOOPS
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
    true,                                       // bias
    14012416,                                   // BIAS_OFFSET
    false,                                      // residual
    14499968,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    true,                                       // FC
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
    29184144,                                   // WEIGHT_GRADIENT_OFFSET
    29200528,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};
#endif
