#pragma once

#include <map>
#include <string>

std::map<std::string, SimplifiedParams> simple{
    {"simple",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         10 * 1024,                                  // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {2, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO_NORM
     }},
    {"transpose",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         10 * 1024,                                  // OUTPUT_OFFSET
         true,                                       // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {2, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO_NORM
     }},
    {"conv",
     {
         0,                                         // INPUT_OFFSET
         0,                                         // WEIGHT_OFFSET
         10 * 1024,                                 // OUTPUT_OFFSET
         false,                                     // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 3, 3, 8, 8}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {2, 0},                                    // REDUCTION
         {3, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         1,                                         // stride
         false,                                     // replication
         false,                                     // RELU
         false,                                     // bias
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // NO_NORM
     }},
    {"relu",
     {
         0,                                         // INPUT_OFFSET
         0,                                         // WEIGHT_OFFSET
         10 * 1024,                                 // OUTPUT_OFFSET
         false,                                     // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 3, 3, 8, 8}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {2, 0},                                    // REDUCTION
         {3, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         1,                                         // stride
         false,                                     // replication
         true,                                      // RELU
         false,                                     // bias
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // NO_NORM
     }},
    {"conv_with_replication",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         10 * 1024,                                  // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 7, 2, 8, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {2, 0},                                     // REDUCTION
         {3, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {4, 5},                                     // weightReuseIndex
         2,                                          // stride
         true,                                       // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO_NORM
     }},
    {"maxpool",
     {
         0,                                         // INPUT_OFFSET
         0,                                         // WEIGHT_OFFSET
         10 * 1024,                                 // OUTPUT_OFFSET
         false,                                     // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 3, 3, 8, 8}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {2, 0},                                    // REDUCTION
         {3, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         1,                                         // stride
         false,                                     // replication
         false,                                     // RELU
         false,                                     // bias
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         true,                                      // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // NO_NORM
     }},
    {"bias",
     {
         0,                                         // INPUT_OFFSET
         0,                                         // WEIGHT_OFFSET
         10 * 1024,                                 // OUTPUT_OFFSET
         false,                                     // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 3, 3, 8, 8}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {2, 0},                                    // REDUCTION
         {3, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         1,                                         // stride
         false,                                     // replication
         false,                                     // RELU
         true,                                      // bias
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // NO_NORM
     }},
    {"bias_and_residual",
     {
         0,                                         // INPUT_OFFSET
         0,                                         // WEIGHT_OFFSET
         10 * 1024,                                 // OUTPUT_OFFSET
         false,                                     // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 3, 3, 8, 8}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {2, 0},                                    // REDUCTION
         {3, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         1,                                         // stride
         false,                                     // replication
         false,                                     // RELU
         true,                                      // bias
         30 * 1024,                                 // BIAS_OFFSET
         true,                                      // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // NO_NORM
     }},
    {"avgpool",
     {
         0,                                         // INPUT_OFFSET
         0,                                         // WEIGHT_OFFSET
         10 * 1024,                                 // OUTPUT_OFFSET
         false,                                     // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 3, 3, 8, 8}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {2, 0},                                    // REDUCTION
         {3, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         1,                                         // stride
         false,                                     // replication
         false,                                     // RELU
         false,                                     // bias
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false,                                     // maxpool
         true,                                      // avgpool
         false,                                     // SOFTMAX
         false,                                     // FC
         false,                                     // NO_NORM
     }},
    {"no_norm",  // elementwise product and addition for matrix of size:
                 // 64 x 64
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         10 * 1024,                                  // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {4, 1, 1, 1, 64, 1}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {2, 0},                                     // REDUCTION
         {3, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {4, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         true,                                       // NO_NORM
     }},
    {"fc",
     {
         0,                                         // INPUT_OFFSET
         0,                                         // WEIGHT_OFFSET
         10 * 1024,                                 // OUTPUT_OFFSET
         false,                                     // TRANSPOSE
         {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         1,                                         // stride
         false,                                     // replication
         false,                                     // RELU
         false,                                     // bias
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false,                                     // maxpool
         false,                                     // avgpool
         false,                                     // SOFTMAX
         true,                                      // FC
         false,                                     // no-norm
     }}};