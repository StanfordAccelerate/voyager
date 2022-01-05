#pragma once

#include <map>
#include <string>

std::map<std::string, Params> resnet{
    {"conv1",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_0_conv1",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_0_conv2",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_1_conv1",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer1_1_conv2",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer2_0_downsample",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer2_0_conv1",
     {
         0,                                         // INPUT_OFFSET
         1024 * 1024,                               // WEIGHT_OFFSET
         2 * 1024 * 1024,                           // OUTPUT_OFFSET
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
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false                                      // avgpool
     }},

    {"layer2_0_conv2",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer2_1_conv1",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer2_1_conv2",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer3_0_downsample",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer3_0_conv_1",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},

    {"layer3_0_conv_2",
     {
         0,                                            // INPUT_OFFSET
         1024 * 1024,                                  // WEIGHT_OFFSET
         2 * 1024 * 1024,                              // OUTPUT_OFFSET
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
         30 * 1024,                                    // BIAS_OFFSET
         false,                                        // residual
         40 * 1024,                                    // RESIDUAL_OFFSET
         false                                         // avgpool
     }},
    {"layer3_1_conv_1",
     {
         0,                                            // INPUT_OFFSET
         1024 * 1024,                                  // WEIGHT_OFFSET
         2 * 1024 * 1024,                              // OUTPUT_OFFSET
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
         30 * 1024,                                    // BIAS_OFFSET
         false,                                        // residual
         40 * 1024,                                    // RESIDUAL_OFFSET
         false                                         // avgpool
     }},
    {"layer3_1_conv_2",
     {
         0,                                            // INPUT_OFFSET
         1024 * 1024,                                  // WEIGHT_OFFSET
         2 * 1024 * 1024,                              // OUTPUT_OFFSET
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
         30 * 1024,                                    // BIAS_OFFSET
         false,                                        // residual
         40 * 1024,                                    // RESIDUAL_OFFSET
         false                                         // avgpool
     }},
    {"layer4_0_downsample",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_0_conv_1",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_0_conv_2",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_1_conv_1",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }},
    {"layer4_1_conv_2",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         true                                         // avgpool
     }},

    // map to vector processor instead
    {"fc",
     {
         0,                                           // INPUT_OFFSET
         1024 * 1024,                                 // WEIGHT_OFFSET
         2 * 1024 * 1024,                             // OUTPUT_OFFSET
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
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false                                        // avgpool
     }}};
