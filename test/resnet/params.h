#pragma once

#include <array>
#include <map>
#include <string>

#include "test/common/VerificationTypes.h"

const SimplifiedParams conv1_params = {
    45056,                                       // INPUT_OFFSET
    0,                                           // WEIGHT_OFFSET
    847872,                                      // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{7, 7, 2, 1, 1, 1}, {1, 2, 7, 2, 16, 16}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    2,                                           // stride
    true,                                        // replication
    true,                                        // RELU
    true,                                        // bias
    9408,                                        // BIAS_OFFSET
    false,                                       // residual
    45056,                                       // RESIDUAL_OFFSET
    true,                                        // maxpool
    false,                                       // avgpool
    false,                                       // softmax
    false,                                       // FC
    false,                                       // no-norm
};

const SimplifiedParams layer1_0_conv1_params = {
    847872,                                      // INPUT_OFFSET
    9472,                                        // WEIGHT_OFFSET
    45056,                                       // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // RELU
    true,                                        // bias
    46336,                                       // BIAS_OFFSET
    false,                                       // residual
    45056,                                       // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};

const SimplifiedParams layer1_0_conv2_params = {
    45056,                                       // INPUT_OFFSET
    46400,                                       // WEIGHT_OFFSET
    1249280,                                     // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // RELU
    true,                                        // bias
    83264,                                       // BIAS_OFFSET
    true,                                        // residual
    847872,                                      // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};

const SimplifiedParams layer1_1_conv1_params = {
    1249280,                                     // INPUT_OFFSET
    83328,                                       // WEIGHT_OFFSET
    446464,                                      // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // RELU
    true,                                        // bias
    120192,                                      // BIAS_OFFSET
    false,                                       // residual
    45056,                                       // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};

const SimplifiedParams layer1_1_conv2_params = {
    446464,                                      // INPUT_OFFSET
    120256,                                      // WEIGHT_OFFSET
    1650688,                                     // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // RELU
    true,                                        // bias
    157120,                                      // BIAS_OFFSET
    true,                                        // residual
    1249280,                                     // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};

const SimplifiedParams layer2_0_downsample_params = {
    1650688,                                     // INPUT_OFFSET
    157184,                                      // WEIGHT_OFFSET
    847872,                                      // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {4, 2, 1, 1, 14, 14}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    2,                                           // stride
    false,                                       // replication
    false,                                       // RELU
    true,                                        // bias
    165376,                                      // BIAS_OFFSET
    false,                                       // residual
    45056,                                       // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};

const SimplifiedParams layer2_0_conv1_params = {
    1650688,                                   // INPUT_OFFSET
    165504,                                    // WEIGHT_OFFSET
    446464,                                    // OUTPUT_OFFSET
    false,                                     // TRANSPOSE
    {{4, 4, 4, 1, 1, 1}, {4, 2, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    2,                                         // stride
    false,                                     // replication
    true,                                      // RELU
    true,                                      // bias
    239232,                                    // BIAS_OFFSET
    false,                                     // residual
    45056,                                     // RESIDUAL_OFFSET
    false,                                     // maxpool
    false,                                     // avgpool
    false,                                     // SOFTMAX
    false,                                     // FC
    false,                                     // no-norm
};

const SimplifiedParams layer2_0_conv2_params = {
    446464,                                      // INPUT_OFFSET
    239360,                                      // WEIGHT_OFFSET
    1249280,                                     // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // RELU
    true,                                        // bias
    386816,                                      // BIAS_OFFSET
    true,                                        // residual
    847872,                                      // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};
const SimplifiedParams layer2_1_conv1_params = {
    1249280,                                     // INPUT_OFFSET
    386944,                                      // WEIGHT_OFFSET
    45056,                                       // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // RELU
    true,                                        // bias
    534400,                                      // BIAS_OFFSET
    false,                                       // residual
    45056,                                       // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};
const SimplifiedParams layer2_1_conv2_params = {
    45056,                                       // INPUT_OFFSET
    534528,                                      // WEIGHT_OFFSET
    1650688,                                     // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // RELU
    true,                                        // bias
    681984,                                      // BIAS_OFFSET
    true,                                        // residual
    1249280,                                     // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};

const SimplifiedParams layer3_0_downsample_params = {
    1650688,                                   // INPUT_OFFSET
    682112,                                    // WEIGHT_OFFSET
    847872,                                    // OUTPUT_OFFSET
    false,                                     // TRANSPOSE
    {{2, 2, 2, 1, 1, 1}, {8, 8, 1, 1, 7, 7}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    2,                                         // stride
    false,                                     // replication
    false,                                     // RELU
    true,                                      // bias
    714880,                                    // BIAS_OFFSET
    false,                                     // residual
    45056,                                     // RESIDUAL_OFFSET
    false,                                     // maxpool
    false,                                     // avgpool
    false,                                     // SOFTMAX
    false,                                     // FC
    false,                                     // no-norm
};

const SimplifiedParams layer3_0_conv1_params = {
    1650688,                                   // INPUT_OFFSET
    715136,                                    // WEIGHT_OFFSET
    45056,                                     // OUTPUT_OFFSET
    false,                                     // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {8, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                    // INPUTX
    {1, 4},                                    // INPUTY
    {3, 0},                                    // REDUCTION
    {2, 1},                                    // WEIGHT
    3,                                         // fxIndex
    2,                                         // fyIndex
    {4, 5},                                    // weightReuseIndex
    2,                                         // stride
    false,                                     // replication
    true,                                      // RELU
    true,                                      // bias
    1010048,                                   // BIAS_OFFSET
    false,                                     // residual
    45056,                                     // RESIDUAL_OFFSET
    false,                                     // maxpool
    false,                                     // avgpool
    false,                                     // SOFTMAX
    false,                                     // FC
    false,                                     // no-norm
};

const SimplifiedParams layer3_0_conv2_params = {
    45056,                                      // INPUT_OFFSET
    1010304,                                    // WEIGHT_OFFSET
    1249280,                                    // OUTPUT_OFFSET
    false,                                      // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {16, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    true,                                       // RELU
    true,                                       // bias
    1600128,                                    // BIAS_OFFSET
    true,                                       // residual
    847872,                                     // RESIDUAL_OFFSET
    false,                                      // maxpool
    false,                                      // avgpool
    false,                                      // SOFTMAX
    false,                                      // FC
    false,                                      // no-norm
};
const SimplifiedParams layer3_1_conv1_params = {
    1249280,                                    // INPUT_OFFSET
    4194304,                                    // WEIGHT_OFFSET
    446464,                                     // OUTPUT_OFFSET
    false,                                      // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {16, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    true,                                       // RELU
    true,                                       // bias
    4784128,                                    // BIAS_OFFSET
    false,                                      // residual
    45056,                                      // RESIDUAL_OFFSET
    false,                                      // maxpool
    false,                                      // avgpool
    false,                                      // SOFTMAX
    false,                                      // FC
    false,                                      // no-norm
};
const SimplifiedParams layer3_1_conv2_params = {
    446464,                                     // INPUT_OFFSET
    4784384,                                    // WEIGHT_OFFSET
    1650688,                                    // OUTPUT_OFFSET
    false,                                      // TRANSPOSE
    {{2, 2, 4, 1, 1, 1}, {16, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    true,                                       // RELU
    true,                                       // bias
    5374208,                                    // BIAS_OFFSET
    true,                                       // residual
    1249280,                                    // RESIDUAL_OFFSET
    false,                                      // maxpool
    false,                                      // avgpool
    false,                                      // SOFTMAX
    false,                                      // FC
    false,                                      // no-norm
};
const SimplifiedParams layer4_0_downsample_params = {
    1650688,                                     // INPUT_OFFSET
    1600384,                                     // WEIGHT_OFFSET
    847872,                                      // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{1, 1, 2, 1, 1, 1}, {16, 16, 1, 1, 7, 7}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    2,                                           // stride
    false,                                       // replication
    false,                                       // RELU
    true,                                        // bias
    1731456,                                     // BIAS_OFFSET
    false,                                       // residual
    45056,                                       // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    false,                                       // FC
    false,                                       // no-norm
};
const SimplifiedParams layer4_0_conv1_params = {
    1650688,                                    // INPUT_OFFSET
    8388608,                                    // WEIGHT_OFFSET
    446464,                                     // OUTPUT_OFFSET
    false,                                      // TRANSPOSE
    {{1, 1, 8, 1, 1, 1}, {16, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    2,                                          // stride
    false,                                      // replication
    true,                                       // RELU
    true,                                       // bias
    9568256,                                    // BIAS_OFFSET
    false,                                      // residual
    45056,                                      // RESIDUAL_OFFSET
    false,                                      // maxpool
    false,                                      // avgpool
    false,                                      // SOFTMAX
    false,                                      // FC
    false,                                      // no-norm
};
const SimplifiedParams layer4_0_conv2_params = {
    446464,                                     // INPUT_OFFSET
    1731968,                                    // WEIGHT_OFFSET
    1249280,                                    // OUTPUT_OFFSET
    false,                                      // TRANSPOSE
    {{1, 1, 8, 1, 1, 1}, {32, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    true,                                       // RELU
    true,                                       // bias
    4091264,                                    // BIAS_OFFSET
    true,                                       // residual
    847872,                                     // RESIDUAL_OFFSET
    false,                                      // maxpool
    false,                                      // avgpool
    false,                                      // SOFTMAX
    false,                                      // FC
    false,                                      // no-norm
};
const SimplifiedParams layer4_1_conv1_params = {
    1249280,                                    // INPUT_OFFSET
    5374464,                                    // WEIGHT_OFFSET
    45056,                                      // OUTPUT_OFFSET
    false,                                      // TRANSPOSE
    {{1, 1, 8, 1, 1, 1}, {32, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    true,                                       // RELU
    true,                                       // bias
    7733760,                                    // BIAS_OFFSET
    false,                                      // residual
    45056,                                      // RESIDUAL_OFFSET
    false,                                      // maxpool
    false,                                      // avgpool
    false,                                      // SOFTMAX
    false,                                      // FC
    false,                                      // no-norm
};
const SimplifiedParams layer4_1_conv2_params = {
    45056,                                      // INPUT_OFFSET
    9437184,                                    // WEIGHT_OFFSET
    446464,                                     // OUTPUT_OFFSET
    false,                                      // TRANSPOSE
    {{1, 1, 8, 1, 1, 1}, {32, 4, 3, 3, 7, 7}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    true,                                       // RELU
    true,                                       // bias
    11796480,                                   // BIAS_OFFSET
    true,                                       // residual
    1249280,                                    // RESIDUAL_OFFSET
    false,                                      // maxpool
    true,                                       // avgpool
    false,                                      // SOFTMAX
    false,                                      // FC
    false,                                      // no-norm
};

// map to vector processor instead
const SimplifiedParams fc_params = {
    446464,                                      // INPUT_OFFSET
    7734272,                                     // WEIGHT_OFFSET
    1650688,                                     // OUTPUT_OFFSET
    false,                                       // TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {32, 63, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // RELU
    true,                                        // bias
    8246272,                                     // BIAS_OFFSET
    false,                                       // residual
    45056,                                       // RESIDUAL_OFFSET
    false,                                       // maxpool
    false,                                       // avgpool
    false,                                       // SOFTMAX
    true,                                        // FC
    false,                                       // no-norm
};
const SimplifiedParams softmax_params = {
    1650688,                                      // INPUT_OFFSET
    0,                                            // WEIGHT_OFFSET
    1651688,                                      // OUTPUT_OFFSET
    false,                                        // TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1008}},  // LOOPS
    {0, 5},                                       // INPUTX
    {1, 4},                                       // INPUTY
    {3, 0},                                       // REDUCTION
    {2, 1},                                       // WEIGHT
    3,                                            // fxIndex
    2,                                            // fyIndex
    {4, 5},                                       // weightReuseIndex
    1,                                            // stride
    false,                                        // replication
    false,                                        // RELU
    false,                                        // bias
    0,                                            // BIAS_OFFSET
    false,                                        // residual
    45056,                                        // RESIDUAL_OFFSET
    false,                                        // maxpool
    false,                                        // avgpool
    true,                                         // SOFTMAX
    false,                                        // FC
    false,                                        // no-norm
};

std::map<std::string, SimplifiedParams> resnetParams{
    {"conv1", conv1_params},
    {"layer1_0_conv1", layer1_0_conv1_params},
    {"layer1_0_conv2", layer1_0_conv2_params},
    {"layer1_1_conv1", layer1_1_conv1_params},
    {"layer1_1_conv2", layer1_1_conv2_params},
    {"layer2_0_downsample", layer2_0_downsample_params},
    {"layer2_0_conv1", layer2_0_conv1_params},
    {"layer2_0_conv2", layer2_0_conv2_params},
    {"layer2_1_conv1", layer2_1_conv1_params},
    {"layer2_1_conv2", layer2_1_conv2_params},
    {"layer3_0_downsample", layer3_0_downsample_params},
    {"layer3_0_conv1", layer3_0_conv1_params},
    {"layer3_0_conv2", layer3_0_conv2_params},
    {"layer3_1_conv1", layer3_1_conv1_params},
    {"layer3_1_conv2", layer3_1_conv2_params},
    {"layer4_0_downsample", layer4_0_downsample_params},
    {"layer4_0_conv1", layer4_0_conv1_params},
    {"layer4_0_conv2", layer4_0_conv2_params},
    {"layer4_1_conv1", layer4_1_conv1_params},
    {"layer4_1_conv2", layer4_1_conv2_params},
    {"fc", fc_params},
    {"softmax", softmax_params}};

std::array<std::string, 22> resnet_order{"conv1",
                                         "layer1_0_conv1",
                                         "layer1_0_conv2",
                                         "layer1_1_conv1",
                                         "layer1_1_conv2",
                                         "layer2_0_downsample",
                                         "layer2_0_conv1",
                                         "layer2_0_conv2",
                                         "layer2_1_conv1",
                                         "layer2_1_conv2",
                                         "layer3_0_downsample",
                                         "layer3_0_conv1",
                                         "layer3_0_conv2",
                                         "layer3_1_conv1",
                                         "layer3_1_conv2",
                                         "layer4_0_downsample",
                                         "layer4_0_conv1",
                                         "layer4_0_conv2",
                                         "layer4_1_conv1",
                                         "layer4_1_conv2",
                                         "fc",
                                         "softmax"};

std::string resnetDataDir = "/sim2/shared/MINOTAUR/nn_data/resnet/";

std::map<std::string, Files> resnetFiles{
    {"conv1", {"conv1_input", "conv1_weight", "conv1_bias", "conv1_comp"}},
    {"layer1_0_conv1",
     {"layer1_0_conv1_input", "layer1_0_conv1_weight", "layer1_0_conv1_bias",
      "layer1_0_conv1_comp"}},
    {"layer1_0_conv2",
     {"layer1_0_conv2_input", "layer1_0_conv2_weight", "layer1_0_conv2_bias",
      "layer1_0_conv2_comp", "conv1_comp"}},
    {"layer1_1_conv1",
     {"layer1_1_conv1_input", "layer1_1_conv1_weight", "layer1_1_conv1_bias",
      "layer1_1_conv1_comp"}},
    {"layer1_1_conv2",
     {"layer1_1_conv2_input", "layer1_1_conv2_weight", "layer1_1_conv2_bias",
      "layer1_1_conv2_comp", "layer1_0_conv2_comp"}},
    {"layer2_0_downsample",
     {"layer2_0_downsample_input", "layer2_0_downsample_weight",
      "layer2_0_downsample_bias", "layer2_0_downsample_comp"}},
    {"layer2_0_conv1",
     {"layer2_0_conv1_input", "layer2_0_conv1_weight", "layer2_0_conv1_bias",
      "layer2_0_conv1_comp"}},
    {"layer2_0_conv2",
     {"layer2_0_conv2_input", "layer2_0_conv2_weight", "layer2_0_conv2_bias",
      "layer2_0_conv2_comp", "layer2_0_downsample_comp"}},
    {"layer2_1_conv1",
     {"layer2_1_conv1_input", "layer2_1_conv1_weight", "layer2_1_conv1_bias",
      "layer2_1_conv1_comp"}},
    {"layer2_1_conv2",
     {"layer2_1_conv2_input", "layer2_1_conv2_weight", "layer2_1_conv2_bias",
      "layer2_1_conv2_comp", "layer2_0_conv2_comp"}},
    {"layer3_0_downsample",
     {"layer3_0_downsample_input", "layer3_0_downsample_weight",
      "layer3_0_downsample_bias", "layer3_0_downsample_comp"}},
    {"layer3_0_conv1",
     {"layer3_0_conv1_input", "layer3_0_conv1_weight", "layer3_0_conv1_bias",
      "layer3_0_conv1_comp"}},
    {"layer3_0_conv2",
     {"layer3_0_conv2_input", "layer3_0_conv2_weight", "layer3_0_conv2_bias",
      "layer3_0_conv2_comp", "layer3_0_downsample_comp"}},
    {"layer3_1_conv1",
     {"layer3_1_conv1_input", "layer3_1_conv1_weight", "layer3_1_conv1_bias",
      "layer3_1_conv1_comp"}},
    {"layer3_1_conv2",
     {"layer3_1_conv2_input", "layer3_1_conv2_weight", "layer3_1_conv2_bias",
      "layer3_1_conv2_comp", "layer3_0_conv2_comp"}},
    {"layer4_0_downsample",
     {"layer4_0_downsample_input", "layer4_0_downsample_weight",
      "layer4_0_downsample_bias", "layer4_0_downsample_comp"}},
    {"layer4_0_conv1",
     {"layer4_0_conv1_input", "layer4_0_conv1_weight", "layer4_0_conv1_bias",
      "layer4_0_conv1_comp"}},
    {"layer4_0_conv2",
     {"layer4_0_conv2_input", "layer4_0_conv2_weight", "layer4_0_conv2_bias",
      "layer4_0_conv2_comp", "layer4_0_downsample_comp"}},
    {"layer4_1_conv1",
     {"layer4_1_conv1_input", "layer4_1_conv1_weight", "layer4_1_conv1_bias",
      "layer4_1_conv1_comp"}},
    {"layer4_1_conv2",
     {"layer4_1_conv2_input", "layer4_1_conv2_weight", "layer4_1_conv2_bias",
      "layer4_1_conv2_comp", "layer4_0_conv2_comp"}},
    {"fc", {"fc_input", "fc_weight", "fc_bias", "fc_comp"}}};

std::map<std::string, MemoryMap> resnetMemoryMap{
    {"conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer1_0_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer1_0_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer1_1_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer1_1_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer2_0_downsample", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer2_0_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer2_0_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer2_1_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer2_1_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer3_0_downsample", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer3_0_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer3_0_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer3_1_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer3_1_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer4_0_downsample", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer4_0_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer4_0_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer4_1_conv1", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"layer4_1_conv2", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"fc", {SRAM, RRAM, RRAM, SRAM, SRAM}}};
