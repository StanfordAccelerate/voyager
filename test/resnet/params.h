#pragma once

#include <map>
#include <string>

std::map<std::string, Params> resnet{
    {"conv1",
     {
         0,                                           // INPUT_OFFSET
         0,                                 // WEIGHT_OFFSET
         802816,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{7, 7, 2, 1, 1, 1}, {1, 2, 7, 2, 16, 16}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         2,                                           // stride
         true,                                        // replication
         false,                                       // maxpool
         true,                                        // bias
         11166912,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_0_conv1",
     {
         802816,                                           // INPUT_OFFSET
         9408,                                 // WEIGHT_OFFSET
         0,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11166976,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_0_conv2",
     {
         0,                                           // INPUT_OFFSET
         46272,                                 // WEIGHT_OFFSET
         1204224,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167040,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_1_conv1",
     {
         1204224,                                           // INPUT_OFFSET
         83136,                                 // WEIGHT_OFFSET
         401408,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167104,                                   // BIAS_OFFSET
         true,                                       // residual
         802816,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_1_conv2",
     {
         401408,                                           // INPUT_OFFSET
         120000,                                 // WEIGHT_OFFSET
         0,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167168,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer2_0_downsample",
     {
         0,                                           // INPUT_OFFSET
         156864,                                 // WEIGHT_OFFSET
         802816,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{2, 2, 4, 1, 1, 1}, {4, 1, 1, 1, 14, 14}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         2,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167232,                                   // BIAS_OFFSET
         true,                                       // residual
         1204224,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer2_0_conv1",
     {
         802816,                                         // INPUT_OFFSET
         165056,                               // WEIGHT_OFFSET
         401408,                           // OUTPUT_OFFSET
         false,                                     // SOFTMAX
         1,                                         // SCALE
         false,                                     // TRANSPOSE
         0,                                         // VECTOR_OFFSET
         false,                                     // VEC_OP
         false,                                     // VEC_SUB
         false,                                     // VEC_SQUARE
         false,                                     // VEC_REDUCE
         true,                                      // CONST_SCALE
         0,                                         // VEC_SCALE_OFFSET
         0,                                         // VEC_SUB_OFFSET
         false,                                     // RELU
         {{4, 4, 4, 1, 1, 1}, {4, 2, 3, 3, 7, 7}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         false,                                     // matmul
         2,                                         // stride
         false,                                     // replication
         false,                                     // maxpool
         true,                                      // bias
         11167360,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false                                      // avgpool
     }},

    {"layer2_0_conv2",
     {
         401408,                                           // INPUT_OFFSET
         238784,                                 // WEIGHT_OFFSET
         1204224,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167488,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer2_1_conv1",
     {
         1204224,                                           // INPUT_OFFSET
         386240,                                 // WEIGHT_OFFSET
         0,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167616,                                   // BIAS_OFFSET
         true,                                       // residual
         802816,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer2_1_conv2",
     {
         0,                                           // INPUT_OFFSET
         533696,                                 // WEIGHT_OFFSET
         401408,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167744,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer3_0_downsample",
     {
         401408,                                           // INPUT_OFFSET
         681152,                                 // WEIGHT_OFFSET
         802816,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {8, 8, 1, 1, 14, 14}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         2,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11167872,                                   // BIAS_OFFSET
         true,                                       // residual
         1204224,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer3_0_conv_1",
     {
         802816,                                           // INPUT_OFFSET
         713920,                                 // WEIGHT_OFFSET
         0,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {8, 8, 3, 3, 14, 14}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         2,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11168128,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer3_0_conv_2",
     {
         0,                                            // INPUT_OFFSET
         1008832,                                  // WEIGHT_OFFSET
         1204224,                              // OUTPUT_OFFSET
         false,                                        // SOFTMAX
         1,                                            // SCALE
         false,                                        // TRANSPOSE
         0,                                            // VECTOR_OFFSET
         false,                                        // VEC_OP
         false,                                        // VEC_SUB
         false,                                        // VEC_SQUARE
         false,                                        // VEC_REDUCE
         true,                                         // CONST_SCALE
         0,                                            // VEC_SCALE_OFFSET
         0,                                            // VEC_SUB_OFFSET
         false,                                        // RELU
         {{1, 1, 2, 1, 1, 1}, {16, 8, 3, 3, 14, 14}},  // LOOPS
         {0, 5},                                       // INPUTX
         {1, 4},                                       // INPUTY
         {3, 0},                                       // REDUCTION
         {2, 1},                                       // WEIGHT
         3,                                            // fxIndex
         2,                                            // fyIndex
         {4, 5},                                       // weightReuseIndex
         false,                                        // matmul
         1,                                            // stride
         false,                                        // replication
         false,                                        // maxpool
         true,                                         // bias
         11168384,                                    // BIAS_OFFSET
         false,                                        // residual
         40 * 1024,                                    // RESIDUAL_OFFSET
         false                                         // avgpool
     }},
    {"layer3_1_conv_1",
     {
         1204224,                                            // INPUT_OFFSET
         1598656,                                  // WEIGHT_OFFSET
         401408,                              // OUTPUT_OFFSET
         false,                                        // SOFTMAX
         1,                                            // SCALE
         false,                                        // TRANSPOSE
         0,                                            // VECTOR_OFFSET
         false,                                        // VEC_OP
         false,                                        // VEC_SUB
         false,                                        // VEC_SQUARE
         false,                                        // VEC_REDUCE
         true,                                         // CONST_SCALE
         0,                                            // VEC_SCALE_OFFSET
         0,                                            // VEC_SUB_OFFSET
         false,                                        // RELU
         {{1, 1, 2, 1, 1, 1}, {16, 8, 3, 3, 14, 14}},  // LOOPS
         {0, 5},                                       // INPUTX
         {1, 4},                                       // INPUTY
         {3, 0},                                       // REDUCTION
         {2, 1},                                       // WEIGHT
         3,                                            // fxIndex
         2,                                            // fyIndex
         {4, 5},                                       // weightReuseIndex
         false,                                        // matmul
         1,                                            // stride
         false,                                        // replication
         false,                                        // maxpool
         true,                                         // bias
         11168640,                                    // BIAS_OFFSET
         true,                                        // residual
         802816,                                    // RESIDUAL_OFFSET
         false                                         // avgpool
     }},
    {"layer3_1_conv_2",
     {
         401408,                                            // INPUT_OFFSET
         2188480,                                  // WEIGHT_OFFSET
         0,                              // OUTPUT_OFFSET
         false,                                        // SOFTMAX
         1,                                            // SCALE
         false,                                        // TRANSPOSE
         0,                                            // VECTOR_OFFSET
         false,                                        // VEC_OP
         false,                                        // VEC_SUB
         false,                                        // VEC_SQUARE
         false,                                        // VEC_REDUCE
         true,                                         // CONST_SCALE
         0,                                            // VEC_SCALE_OFFSET
         0,                                            // VEC_SUB_OFFSET
         false,                                        // RELU
         {{1, 1, 2, 1, 1, 1}, {16, 8, 3, 3, 14, 14}},  // LOOPS
         {0, 5},                                       // INPUTX
         {1, 4},                                       // INPUTY
         {3, 0},                                       // REDUCTION
         {2, 1},                                       // WEIGHT
         3,                                            // fxIndex
         2,                                            // fyIndex
         {4, 5},                                       // weightReuseIndex
         false,                                        // matmul
         1,                                            // stride
         false,                                        // replication
         false,                                        // maxpool
         true,                                         // bias
         11168896,                                    // BIAS_OFFSET
         false,                                        // residual
         40 * 1024,                                    // RESIDUAL_OFFSET
         false                                         // avgpool
     }},
    {"layer4_0_downsample",
     {
         0,                                           // INPUT_OFFSET
         2778304,                                 // WEIGHT_OFFSET
         802816,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {16, 16, 1, 1, 7, 7}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         2,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11169152,                                   // BIAS_OFFSET
         true,                                       // residual
         1204224,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_0_conv_1",
     {
         802816,                                           // INPUT_OFFSET
         2909376,                                 // WEIGHT_OFFSET
         401408,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {16, 16, 3, 3, 7, 7}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         2,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11169664,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_0_conv_2",
     {
         401408,                                           // INPUT_OFFSET
         4089024,                                 // WEIGHT_OFFSET
         1204224,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {32, 16, 3, 3, 7, 7}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11170176,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_1_conv_1",
     {
         1204224,                                           // INPUT_OFFSET
         6448320,                                 // WEIGHT_OFFSET
         0,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {32, 16, 3, 3, 7, 7}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11170688,                                   // BIAS_OFFSET
         true,                                       // residual
         802816,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_1_conv_2",
     {
         0,                                           // INPUT_OFFSET
         8807616,                                 // WEIGHT_OFFSET
         401408,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {32, 16, 3, 3, 7, 7}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         11171200,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         true                                         // avgpool
     }},

    // map to vector processor instead
    {"fc",
     {
         401408,                                           // INPUT_OFFSET
         11166912,                                 // WEIGHT_OFFSET
         1605632,                             // OUTPUT_OFFSET
         false,                                       // SOFTMAX
         1,                                           // SCALE
         false,                                       // TRANSPOSE
         0,                                           // VECTOR_OFFSET
         false,                                       // VEC_OP
         false,                                       // VEC_SUB
         false,                                       // VEC_SQUARE
         false,                                       // VEC_REDUCE
         true,                                        // CONST_SCALE
         0,                                           // VEC_SCALE_OFFSET
         0,                                           // VEC_SUB_OFFSET
         false,                                       // RELU
         {{1, 1, 2, 1, 1, 1}, {32, 16, 3, 3, 1, 1}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         false,                                       // bias
         11171712,                                   // BIAS_OFFSET
         true,                                       // residual
         1204224,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }}};
