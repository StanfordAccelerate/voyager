#pragma once

#include <ac_int.h>

#include "ArchitectureParams.h"

/***************************************/
/* Activation Functions without Kwargs */
/***************************************/

// ==================== FIXED ====================
// Sometimes, it's useful to be able to pass the activations through some fixed
// polynomial, that isn't piecewise. These maxes and clamp parameters can be
// used for this. The actual polynomial still needs to be configured.
const VECTOR_DATATYPE FIXED_MAXES[NUM_MAXES] = {-4.0, -2.0, -1.0,
                                                0.0,  1.0,  2.0};
const ac_int<1, false> FIXED_CLAMP_MIN = 0;
const ac_int<1, false> FIXED_CLAMP_MAX = 0;

// ==================== EXP ====================
const VECTOR_DATATYPE EXP_MAXES[NUM_MAXES] = {-4.0, -3.2, -2.4,
                                              -1.6, -0.8, 0.0};
const VECTOR_DATATYPE EXP_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},                       // [-inf, -4.0)
    {0.30320454, 0.12538815, 0.0135461},   // [-4.0, -3.2)
    {0.46802676, 0.22840204, 0.02964202},  // [-3.2, -2.4)
    {0.67544863, 0.40125359, 0.06565276},  // [-2.4, -1.6)
    {0.88488941, 0.66305457, 0.14746556},  // [-1.6, -0.8)
    {0.99576284, 0.94023815, 0.3207053},   // [-0.8, 0.0)
    {0.99576284, 0.94023815, 0.3207053}};  // [0.0, inf)
const ac_int<1, false> EXP_CLAMP_MIN = 1;
const ac_int<1, false> EXP_CLAMP_MAX = 1;

// ==================== GELU ====================
const VECTOR_DATATYPE GELU_MAXES[NUM_MAXES] = {-2.5, -1.6, -0.7, 0.2, 1.1, 2.0};
const VECTOR_DATATYPE GELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},                          // [-inf, -2.5)
    {-0.50243798, -0.37154151, -0.07114065},  // [-2.5, -1.6)
    {-0.16143203, 0.05471593, 0.06206480},    // [-1.6, -0.7)
    {0.00177437, 0.52101993, 0.39513909},     // [-0.7,  0.2)
    {-0.00343439, 0.57310756, 0.26492001},    // [ 0.2,  1.1)
    {-0.37716041, 1.25260942, -0.04394447},   // [ 1.1,  2.0)
    {0.0, 1.0, 0.0}};                         // [ 2.0,  inf)
const ac_int<1, false> GELU_CLAMP_MIN = 1;
const ac_int<1, false> GELU_CLAMP_MAX = 0;

// ==================== SILU ====================
const VECTOR_DATATYPE SILU_MAXES[NUM_MAXES] = {-4.0, -2.4, -0.8, 0.8, 2.4, 4.0};
const VECTOR_DATATYPE SILU_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},                          // [-inf, -4)
    {-0.64124114, -0.23600189, -0.02318127},  // [-4.0, -2.4)
    {-0.13298061, 0.18754855, 0.06505841},    // [-2.4, -0.8)
    {-0.00799941, 0.50000153, 0.26034152},    // [-0.8, 0.8)
    {-0.13298185, 0.81245762, 0.06505646},    // [0.8, 2.4)
    {-0.64122689, 1.23599516, -0.02318052},   // [2.4, 4.0)
    {0.0, 1.0, 0.0}};                         // [4, inf)
const ac_int<1, false> SILU_CLAMP_MIN = 1;
const ac_int<1, false> SILU_CLAMP_MAX = 0;

// ==================== ELU ====================
const VECTOR_DATATYPE ELU_MAXES[NUM_MAXES] = {-4.0, -3.2, -2.4,
                                              -1.6, -0.8, 0.0};
const VECTOR_DATATYPE ELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {-1.0, 0.0, 0.0},                       // [-inf, -4.0)
    {-0.69740851, 0.12502542, 0.01349293},  // [-4.0, -3.2)
    {-0.53185065, 0.22849908, 0.02966069},  // [-3.2, -2.4)
    {-0.32461989, 0.40119139, 0.06563825},  // [-2.4, -1.6)
    {-0.11511858, 0.66306802, 0.1474747},   // [-1.6, -0.8)
    {-0.00423305, 0.94028185, 0.32073335},  // [-0.8, 0.0)
    {0.0, 1.0, 0.0}};                       // [0.0, inf)
const ac_int<1, false> ELU_CLAMP_MIN = 1;
const ac_int<1, false> ELU_CLAMP_MAX = 0;

// ==================== LOGSIGMOID ====================
const VECTOR_DATATYPE LOGSIGMOID_MAXES[NUM_MAXES] = {-4.0, -2.4, -0.8,
                                                     0.8,  2.4,  4.0};
const VECTOR_DATATYPE LOGSIGMOID_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 1.0, 0.0},                         // [-inf, -4.0)
    {-0.25110293, 0.92370584, -0.00337961},  // [-4.0, -2.4)
    {-0.65666672, 0.58573602, -0.07378999},  // [-2.4, -0.8)
    {-0.69125212, 0.49927251, -0.12782968},  // [-0.8, 0.8)
    {-0.65514635, 0.40900809, -0.07141441},  // [0.8, 2.4)
    {-0.32953055, 0.13766159, -0.01488389},  // [2.4, 4.0)
    {0.0, 0.0, 0.0}};                        // [4.0, inf)
const ac_int<1, false> LOGSIGMOID_CLAMP_MIN = 0;
const ac_int<1, false> LOGSIGMOID_CLAMP_MAX = 1;

// ==================== TANH ====================
const VECTOR_DATATYPE TANH_MAXES[NUM_MAXES] = {-2.0, -1.2, -0.4, 0.4, 1.2, 2.0};
const VECTOR_DATATYPE TANH_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {-1.0, 0.0, 0.0},                        // [-inf, -2.0)
    {-0.38358838, 0.50606058, 0.10732545},   // [-2.0, -1.2)
    {0.06727078, 1.2574925, 0.42042208},     // [-1.2, -0.4)
    {0.0000031, 0.9211541, -0.00000091},     // [-0.4, 0.4)
    {-0.06726454, 1.25749229, -0.42042365},  // [0.4, 1.2)
    {0.38359671, 0.50605688, -0.10732556},   // [1.2, 2.0)
    {1.0, 0.0, 0.0}};                        // [2.0, inf)
const ac_int<1, false> TANH_CLAMP_MIN = 1;
const ac_int<1, false> TANH_CLAMP_MAX = 1;

// ==================== TANHSHRINK ====================
const VECTOR_DATATYPE TANHSHRINK_MAXES[NUM_MAXES] = {-2.0, -1.2, -0.4,
                                                     0.4,  1.2,  2.0};
const VECTOR_DATATYPE TANHSHRINK_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {1.0, 1.0, 0.0},                          // [-inf, -2.0)
    {0.38359241, 0.49395793, -0.10731657},    // [-2.0, -1.2)
    {-0.06726710, -0.25747458, -0.42041345},  // [-1.2, -0.4)
    {-0.00000091, 0.07885639, 0.00000027},    // [-0.4, 0.4)
    {0.06726528, -0.25747452, 0.42041391},    // [0.4, 1.2)
    {-0.38359488, 0.49395907, 0.10731658},    // [1.2, 2.0)
    {-1.0, 1.0, 0.0}};                        // [2.0, inf)
const ac_int<1, false> TANHSHRINK_CLAMP_MIN = 0;
const ac_int<1, false> TANHSHRINK_CLAMP_MAX = 0;

// ==================== SOFTPLUS* ====================
const VECTOR_DATATYPE SOFTPLUS_MAXES[NUM_MAXES] = {-4.0, -1.6, 0.8,
                                                   3.2,  5.6,  8.0};
const VECTOR_DATATYPE SOFTPLUS_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},                        // [-inf, -4.0)
    {0.44634462, 0.20654296, 0.02485508},   // [-4.0, -1.6)
    {0.69456034, 0.51681261, 0.12181434},   // [-1.6, 0.8)
    {0.65387269, 0.61853172, 0.05823990},   // [0.8, 3.2)
    {0.05186809, 0.99478459, -0.00054961},  // [3.2, 5.6)
    {0.21113487, 0.93790360, 0.00452905},   // [5.6, 8.0)
    {0.0, 1.0, 0.0}};                       // [8.0, inf)
const ac_int<1, false> SOFTPLUS_CLAMP_MIN = 1;
const ac_int<1, false> SOFTPLUS_CLAMP_MAX = 0;

// ==================== MISH ====================
const VECTOR_DATATYPE MISH_MAXES[NUM_MAXES] = {-4.0, -2.4, -0.8, 0.8, 2.4, 4.0};
const VECTOR_DATATYPE MISH_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},                          // [-inf, -4.0)
    {-0.86530222, -0.38268892, -0.04681395},  // [-4.0, -2.4)
    {-0.15288379, 0.21099311, 0.07686981},    // [-2.4, -0.8)
    {-0.00460037, 0.58170166, 0.30856265},    // [-0.8, 0.8)
    {-0.20598799, 1.08517069, -0.00610550},   // [0.8, 2.4)
    {-0.30646291, 1.16889979, -0.02354906},   // [2.4, 4.0)
    {0.0, 1.0, 0.0}};                         // [4.0, inf)
const ac_int<1, false> MISH_CLAMP_MIN = 1;
const ac_int<1, false> MISH_CLAMP_MAX = 0;

// ==================== SIGMOID ====================
const VECTOR_DATATYPE SIGMOID_MAXES[NUM_MAXES] = {-4.0, -2.4, -0.8,
                                                  0.8,  2.4,  4.0};
const VECTOR_DATATYPE SIGMOID_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},                        // [-inf, -4.0)
    {0.30820363, 0.12651465, 0.01341569},   // [-4.0, -2.4)
    {0.53363376, 0.31437310, 0.05255286},   // [-2.4, -0.8)
    {0.49999994, 0.23028853, 0.00000001},   // [-0.8, 0.8)
    {0.46636611, 0.31437310, -0.05255285},  // [0.8, 2.4)
    {0.69179618, 0.12651471, -0.01341569},  // [2.4, 4.0)
    {1.0, 0.0, 0.0}};                       // [4.0, inf)
const ac_int<1, false> SIGMOID_CLAMP_MIN = 1;
const ac_int<1, false> SIGMOID_CLAMP_MAX = 1;

// ==================== SELU ====================
const VECTOR_DATATYPE SELU_MAXES[NUM_MAXES] = {-4.0, -3.2, -2.4,
                                               -1.6, -0.8, 0.0};
const VECTOR_DATATYPE SELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {-1.7581, 0.0, 0.0},                    // [-inf, -4.0)
    {-1.22810960, 0.21859610, 0.02354041},  // [-4.0, -3.2)
    {-0.93479996, 0.40191462, 0.05218392},  // [-3.2, -2.4)
    {-0.57070258, 0.70532910, 0.11539528},  // [-2.4, -1.6)
    {-0.20257564, 1.16548778, 0.25919486},  // [-1.6, -0.8)
    {-0.00737264, 1.65349528, 0.56419955},  // [-0.8, 0.0)
    {0.0, 1.050701, 0.0}};                  // [0.0, inf)
const ac_int<1, false> SELU_CLAMP_MIN = 1;
const ac_int<1, false> SELU_CLAMP_MAX = 0;

// ==================== CELU* ====================
const VECTOR_DATATYPE CELU_MAXES[NUM_MAXES] = {-4.0, -3.2, -2.4,
                                               -1.6, -0.8, 0.0};
const VECTOR_DATATYPE CELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {-1.0, 0.0, 0.0},                       // [-inf, -4.0)
    {-0.6973123, 0.12509987, 0.0135055},    // [-4.0, -3.2)
    {-0.53168639, 0.22861606, 0.0296799},   // [-3.2, -2.4)
    {-0.32462875, 0.4011641, 0.06562741},   // [-2.4, -1.6)
    {-0.1151744, 0.66298204, 0.14744552},   // [-1.6, -0.8)
    {-0.00421229, 0.94038732, 0.32082382},  // [-0.8, 0.0)
    {0.0, 1.0, 0.0}};                       // [0.0, inf)
const ac_int<1, false> CELU_CLAMP_MIN = 1;
const ac_int<1, false> CELU_CLAMP_MAX = 0;

// ==================== HARDSIGMOID ====================
const VECTOR_DATATYPE HARDSIGMOID_MAXES[NUM_MAXES] = {-3.0, -2.0, -1.0,
                                                      0.0,  2.0,  3.0};
const VECTOR_DATATYPE HARDSIGMOID_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},      {0.5, 0.166667, 0.0}, {0.5, 0.166667, 0.0},
    {0.5, 0.166667, 0.0}, {0.5, 0.166667, 0.0}, {1.0, 0.0, 0.0}};
const ac_int<1, false> HARDSIGMOID_CLAMP_MIN = 1;
const ac_int<1, false> HARDSIGMOID_CLAMP_MAX = 1;

// ==================== HARDSWISH ====================
const VECTOR_DATATYPE HARDSWISH_MAXES[NUM_MAXES] = {-3.0, -2.0, -1.0,
                                                    0.0,  2.0,  3.0};
const VECTOR_DATATYPE HARDSWISH_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 0.0, 0.0},      {0.0, 0.5, 0.166667}, {0.0, 0.5, 0.166667},
    {0.0, 0.5, 0.166667}, {0.0, 0.5, 0.166667}, {0.0, 0.5, 0.166667},
    {0.0, 1.0, 0.0}};
const ac_int<1, false> HARDSWISH_CLAMP_MIN = 1;
const ac_int<1, false> HARDSWISH_CLAMP_MAX = 0;

/************************************/
/* Activation Functions with Kwargs */
/************************************/

const std::set<std::string> unary_ops_with_kwargs = {
    "hardshrink", "hardshrink_", "hardtanh",  "hardtanh_",
    "leaky_relu", "leaky_relu_", "rrelu",     "rrelu_",
    "softshrink", "softshrink_", "threshold", "threshold_"};

// ==================== HARDSHRINK ====================
const VECTOR_DATATYPE HARDSHRINK_RANGES[NUM_RANGES][NUM_COEFFS] = {
    {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
const ac_int<1, false> HARDSHRINK_CLAMP_MIN = 0;
const ac_int<1, false> HARDSHRINK_CLAMP_MAX = 0;

// ==================== HARDTANH ====================
const ac_int<1, false> HARDTANH_CLAMP_MIN = 0;
const ac_int<1, false> HARDTANH_CLAMP_MAX = 0;

// ==================== LEAKYRELU ====================
const VECTOR_DATATYPE LEAKYRELU_MAXES[NUM_MAXES] = {0.0, 0.0, 0.0,
                                                    0.0, 0.0, 0.0};
const ac_int<1, false> LEAKYRELU_CLAMP_MIN = 0;
const ac_int<1, false> LEAKYRELU_CLAMP_MAX = 0;

// ==================== RRELU ====================
const VECTOR_DATATYPE RRELU_MAXES[NUM_MAXES] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
const ac_int<1, false> RRELU_CLAMP_MIN = 0;
const ac_int<1, false> RRELU_CLAMP_MAX = 0;

// ==================== SOFTSHRINK ====================
const ac_int<1, false> SOFTSHRINK_CLAMP_MIN = 0;
const ac_int<1, false> SOFTSHRINK_CLAMP_MAX = 0;

// ==================== THRESHOLD ====================
const ac_int<1, false> THRESHOLD_CLAMP_MIN = 0;
const ac_int<1, false> THRESHOLD_CLAMP_MAX = 0;