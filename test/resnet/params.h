#pragma once

#include <array>
#include <map>
#include <string>

#include "test/common/VerificationTypes.h"

std::map<std::string, SimplifiedParams> resnetParams{
    {"conv1",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         802816,                                      // OUTPUT_OFFSET
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
         11678912,                                    // BIAS_OFFSET
         false,                                       // residual
         0 * 1024,                                    // RESIDUAL_OFFSET
         true,                                        // maxpool
         false,                                       // avgpool
         false,                                       // softmax
         false,                                       // FC
         false,                                       // no-norm
     }},

    {"layer1_0_conv1",
     {
         802816,                                      // INPUT_OFFSET
         9408,                                        // WEIGHT_OFFSET
         0,                                           // OUTPUT_OFFSET
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
         11678976,                                    // BIAS_OFFSET
         false,                                       // residual
         0 * 1024,                                    // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},

    {"layer1_0_conv2",
     {
         0,                                           // INPUT_OFFSET
         46272,                                       // WEIGHT_OFFSET
         1204224,                                     // OUTPUT_OFFSET
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
         11679040,                                    // BIAS_OFFSET
         true,                                        // residual
         802816,                                      // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},

    {"layer1_1_conv1",
     {
         1204224,                                     // INPUT_OFFSET
         83136,                                       // WEIGHT_OFFSET
         401408,                                      // OUTPUT_OFFSET
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
         11679104,                                    // BIAS_OFFSET
         false,                                       // residual
         802816,                                      // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},

    {"layer1_1_conv2",
     {
         401408,                                      // INPUT_OFFSET
         120000,                                      // WEIGHT_OFFSET
         0,                                           // OUTPUT_OFFSET
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
         11679168,                                    // BIAS_OFFSET
         true,                                        // residual
         1204224,                                     // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},

    {"layer2_0_downsample",
     {
         0,                                           // INPUT_OFFSET
         156864,                                      // WEIGHT_OFFSET
         802816,                                      // OUTPUT_OFFSET
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
         11679232,                                    // BIAS_OFFSET
         false,                                       // residual
         1204224,                                     // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},

    {"layer2_0_conv1",
     {
         802816,                                    // INPUT_OFFSET
         165056,                                    // WEIGHT_OFFSET
         401408,                                    // OUTPUT_OFFSET
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
         11679360,                                  // BIAS_OFFSET
         false,                                     // residual
         0 * 1024,                                  // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // no-norm
     }},

    {"layer2_0_conv2",
     {
         401408,                                      // INPUT_OFFSET
         238784,                                      // WEIGHT_OFFSET
         1204224,                                     // OUTPUT_OFFSET
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
         11679488,                                    // BIAS_OFFSET
         true,                                        // residual
         802816,                                      // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},
    {"layer2_1_conv1",
     {
         1204224,                                     // INPUT_OFFSET
         386240,                                      // WEIGHT_OFFSET
         0,                                           // OUTPUT_OFFSET
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
         11679616,                                    // BIAS_OFFSET
         false,                                       // residual
         802816,                                      // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},
    {"layer2_1_conv2",
     {
         0,                                           // INPUT_OFFSET
         533696,                                      // WEIGHT_OFFSET
         401408,                                      // OUTPUT_OFFSET
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
         11679744,                                    // BIAS_OFFSET
         true,                                        // residual
         1204224,                                     // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},

    {"layer3_0_downsample",
     {
         401408,                                    // INPUT_OFFSET
         681152,                                    // WEIGHT_OFFSET
         802816,                                    // OUTPUT_OFFSET
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
         11679872,                                  // BIAS_OFFSET
         false,                                     // residual
         1204224,                                   // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // no-norm
     }},

    {"layer3_0_conv1",
     {
         802816,                                    // INPUT_OFFSET
         713920,                                    // WEIGHT_OFFSET
         0,                                         // OUTPUT_OFFSET
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
         11680128,                                  // BIAS_OFFSET
         false,                                     // residual
         0 * 1024,                                  // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // no-norm
     }},

    {"layer3_0_conv2",
     {
         0,                                          // INPUT_OFFSET
         1008832,                                    // WEIGHT_OFFSET
         1204224,                                    // OUTPUT_OFFSET
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
         11680384,                                   // BIAS_OFFSET
         true,                                       // residual
         802816,                                     // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // no-norm
     }},
    {"layer3_1_conv1",
     {
         1204224,                                    // INPUT_OFFSET
         1598656,                                    // WEIGHT_OFFSET
         401408,                                     // OUTPUT_OFFSET
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
         11680640,                                   // BIAS_OFFSET
         false,                                      // residual
         802816,                                     // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // no-norm
     }},
    {"layer3_1_conv2",
     {
         401408,                                     // INPUT_OFFSET
         2188480,                                    // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
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
         11680896,                                   // BIAS_OFFSET
         true,                                       // residual
         1204224,                                    // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // no-norm
     }},
    {"layer4_0_downsample",
     {
         0,                                           // INPUT_OFFSET
         2778304,                                     // WEIGHT_OFFSET
         802816,                                      // OUTPUT_OFFSET
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
         11681152,                                    // BIAS_OFFSET
         false,                                       // residual
         1204224,                                     // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // no-norm
     }},
    {"layer4_0_conv1",
     {
         802816,                                     // INPUT_OFFSET
         2909376,                                    // WEIGHT_OFFSET
         401408,                                     // OUTPUT_OFFSET
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
         11681664,                                   // BIAS_OFFSET
         false,                                      // residual
         0 * 1024,                                   // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // no-norm
     }},
    {"layer4_0_conv2",
     {
         401408,                                     // INPUT_OFFSET
         4089024,                                    // WEIGHT_OFFSET
         1204224,                                    // OUTPUT_OFFSET
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
         11682176,                                   // BIAS_OFFSET
         true,                                       // residual
         802816,                                     // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // no-norm
     }},
    {"layer4_1_conv1",
     {
         1204224,                                    // INPUT_OFFSET
         6448320,                                    // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
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
         11682688,                                   // BIAS_OFFSET
         false,                                      // residual
         802816,                                     // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // no-norm
     }},
    {"layer4_1_conv2",
     {
         0,                                          // INPUT_OFFSET
         8807616,                                    // WEIGHT_OFFSET
         401408,                                     // OUTPUT_OFFSET
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
         11683200,                                   // BIAS_OFFSET
         true,                                       // residual
         1204224,                                    // RESIDUAL_OFFSET
         false,                                      // maxpool
         true,                                       // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // no-norm
     }},

    // map to vector processor instead
    {"fc",
     {
         401408,                                      // INPUT_OFFSET
         11166912,                                    // WEIGHT_OFFSET
         1605632,                                     // OUTPUT_OFFSET
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
         11683712,                                    // BIAS_OFFSET
         false,                                       // residual
         1204224,                                     // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         true,                                        // FC
         false,                                       // no-norm
     }},
    {"softmax",
     {
         1605632,                                      // INPUT_OFFSET
         11166912,                                     // WEIGHT_OFFSET
         0,                                            // OUTPUT_OFFSET
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
         11683712,                                     // BIAS_OFFSET
         false,                                        // residual
         1204224,                                      // RESIDUAL_OFFSET
         false,                                        // maxpool
         false,                                        // avgpool
         true,                                         // SOFTMAX
         false,                                        // FC
         false,                                        // no-norm
     }}};

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

std::string resnetDataDir = "data/resnet/";

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
