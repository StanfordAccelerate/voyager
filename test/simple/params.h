#pragma once

const Params simple = {
    0,                        // INPUT_OFFSET
    1024 * 1024,              // WEIGHT_OFFSET
    2 * 1024 * 1024,          // OUTPUT_OFFSET
    false,                    // SOFTMAX
    1,                        // SCALE
    false,                    // TRANSPOSE
    0,                        // VECTOR_OFFSET
    false,                    // VEC_OP
    false,                    // VEC_SUB
    false,                    // VEC_SQUARE
    false,                    // VEC_REDUCE
    true,                     // CONST_SCALE
    0,                        // VEC_SCALE_OFFSET
    0,                        // VEC_SUB_OFFSET
    false,                    // RELU
    {{1, 1, 1}, {2, 2, 32}},  // LOOPS
    {1, 2},                   // INPUT
    {2, 0},                   // REDUCTION
    {0, 1}                    // WEIGHT
};
