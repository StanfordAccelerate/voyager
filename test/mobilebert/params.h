#pragma once

#include <map>
#include <string>
#include <array>

std::map<std::string, Params> mobilebert{
    // (128 x 512) * (512 x 128)
    // (1x128x512) * (1x1x512x128)
    {"inputBottleneck",
     {
         0,                                          // INPUT_OFFSET
         0,                                          // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
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
         {{4, 1, 2, 1, 1, 1}, {32, 4, 1, 1, 1, 32}}, // LOOPS
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

    // (128 x 128) x (128 x 32)
    {"qkvProjection",
     {
         131072,                                    // INPUT_OFFSET
         65536,                                     // WEIGHT_OFFSET
         0,                                         // OUTPUT_OFFSET
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
         {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}}, // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {5, 5},                                    // weightReuseIndex
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

    // attention- Q*KT
    // (128 x 32) * (32 x 128)
    {"qkAttention",
     {
         0,                                         // INPUT_OFFSET
         69632,                                     // WEIGHT_OFFSET
         131072,                                    // OUTPUT_OFFSET
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
         {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}}, // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {5, 5},                                    // weightReuseIndex
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

    // attention- *v
    // (128 x 128) * (128 x 32)
    {"vAttention",
     {
         131072,                                    // INPUT_OFFSET
         73728,                                     // WEIGHT_OFFSET
         0,                                         // OUTPUT_OFFSET
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
         {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}}, // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {5, 5},                                    // weightReuseIndex
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

    // wo projection
    // (128 x 128) x (128 x 128)
    {"wProjection",
     {
         0,                                         // INPUT_OFFSET
         77824,                                     // WEIGHT_OFFSET
         131072,                                    // OUTPUT_OFFSET
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
         {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}}, // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {5, 5},                                    // weightReuseIndex
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

    // FFN 1
    // (128 x 128) * (128 x 512)
    {"ffn1",
     {
         131072,                                    // INPUT_OFFSET
         94208,                                     // WEIGHT_OFFSET
         0,                                         // OUTPUT_OFFSET
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
         {{4, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 32}}, // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {5, 5},                                    // weightReuseIndex
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

    // FFN 2
    // (128 x 512) x (512 x 128)
    {"ffn2",
     {
         0,                                          // INPUT_OFFSET
         159744,                                     // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
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
         {{4, 1, 4, 1, 1, 1}, {32, 2, 1, 1, 1, 32}}, // LOOPS
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

    // output bottleneck
    // (128 x 128) x (128 x 512)
    {"outputBottleneck",
     {
         131072,                                    // INPUT_OFFSET
         225280,                                    // WEIGHT_OFFSET
         0,                                         // OUTPUT_OFFSET
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
         {{4, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 32}}, // LOOPS
         {0, 5},                                    // INPUTX
         {1, 4},                                    // INPUTY
         {3, 0},                                    // REDUCTION
         {2, 1},                                    // WEIGHT
         3,                                         // fxIndex
         2,                                         // fyIndex
         {5, 5},                                    // weightReuseIndex
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
};

std::array<std::string, 8> mobilebert_order{
    "inputBottleneck",
    "qkvProjection",
    "qkAttention",
    "vAttention",
    "wProjection",
    "ffn1",
    "ffn2",
    "outputBottleneck"};