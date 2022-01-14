#pragma once

#include <map>
#include <string>

std::map<std::string, Params> simple{
    {"simple",
     {
         0,                                          // INPUT_OFFSET
         1024 * 1024,                                // WEIGHT_OFFSET
         2 * 1024 * 1024,                            // OUTPUT_OFFSET
         false,                                      // SOFTMAX
         1,                                          // SCALE
         false,                                      // TRANSPOSE
         0,                                          // VECTOR_OFFSET
         false,                                      // VEC_OP
         false,                                      // VEC_SUB
         false,                                      // VEC_SQUARE
         false,                                      // VEC_REDUCE
         true,                                       // CONST_SCALE
         0,                                          // VEC_SCALE_OFFSET
         0,                                          // VEC_SUB_OFFSET
         false,                                      // RELU
         {{1, 1, 1, 1, 1, 1}, {2, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         false,                                      // matmul
         1,                                          // stride
         false,                                      // replication
         false,                                      // maxpool
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false                                       // avgpool
     }},
    {"transpose",
     {
         0,                                          // INPUT_OFFSET
         1024 * 1024,                                // WEIGHT_OFFSET
         2 * 1024 * 1024,                            // OUTPUT_OFFSET
         false,                                      // SOFTMAX
         1,                                          // SCALE
         true,                                       // TRANSPOSE
         0,                                          // VECTOR_OFFSET
         false,                                      // VEC_OP
         false,                                      // VEC_SUB
         false,                                      // VEC_SQUARE
         false,                                      // VEC_REDUCE
         true,                                       // CONST_SCALE
         0,                                          // VEC_SCALE_OFFSET
         0,                                          // VEC_SUB_OFFSET
         false,                                      // RELU
         {{1, 1, 1, 1, 1, 1}, {2, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         false,                                      // matmul
         1,                                          // stride
         false,                                      // replication
         false,                                      // maxpool
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false                                       // avgpool
     }},
    {"conv",
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
         {{1, 1, 1, 1, 1, 1}, {1, 1, 3, 3, 8, 8}},  // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {2, 0},                                    // REDUCTION
         {3, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {4, 5},                                    // weightReuseIndex
         false,                                     // matmul
         1,                                         // stride
         false,                                     // replication
         false,                                     // maxpool
         false,                                     // bias
         30 * 1024,                                 // BIAS_OFFSET
         false,                                     // residual
         40 * 1024,                                 // RESIDUAL_OFFSET
         false                                      // avgpool
     }},

    {"conv_with_replication",
     {
         0,                                          // INPUT_OFFSET
         1024 * 1024,                                // WEIGHT_OFFSET
         2 * 1024 * 1024,                            // OUTPUT_OFFSET
         false,                                      // SOFTMAX
         1,                                          // SCALE
         false,                                      // TRANSPOSE
         0,                                          // VECTOR_OFFSET
         false,                                      // VEC_OP
         false,                                      // VEC_SUB
         false,                                      // VEC_SQUARE
         false,                                      // VEC_REDUCE
         true,                                       // CONST_SCALE
         0,                                          // VEC_SCALE_OFFSET
         0,                                          // VEC_SUB_OFFSET
         false,                                      // RELU
         {{1, 1, 1, 1, 1, 1}, {1, 1, 7, 2, 8, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {2, 0},                                     // REDUCTION
         {3, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {4, 5},                                     // weightReuseIndex
         false,                                      // matmul
         2,                                          // stride
         true,                                       // replication
         false,                                      // maxpool
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false                                       // avgpool
     }},

    {"maxpool",
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
         {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 16, 16}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {2, 0},                                      // REDUCTION
         {3, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         true                                         // maxpool
     }},

    {"bias",
     {
         0,                                           // INPUT_OFFSET
         10 * 1024,                                   // WEIGHT_OFFSET
         20 * 1024,                                   // OUTPUT_OFFSET
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
         {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 16, 16}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {2, 0},                                      // REDUCTION
         {3, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         true,                                        // maxpool
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
     }},

    {"bias_and_residual",
     {
         0,                                           // INPUT_OFFSET
         10 * 1024,                                   // WEIGHT_OFFSET
         20 * 1024,                                   // OUTPUT_OFFSET
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
         {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 16, 16}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {2, 0},                                      // REDUCTION
         {3, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {4, 5},                                      // weightReuseIndex
         false,                                       // matmul
         1,                                           // stride
         false,                                       // replication
         false,                                       // maxpool
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         true,                                        // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
     }},

    {"avgpool",
     {
         0,                                           // INPUT_OFFSET
         10 * 1024,                                   // WEIGHT_OFFSET
         20 * 1024,                                   // OUTPUT_OFFSET
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
         {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 16, 16}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {2, 0},                                      // REDUCTION
         {3, 1},                                      // WEIGHT
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
         true                                         // avgpool
     }},
};