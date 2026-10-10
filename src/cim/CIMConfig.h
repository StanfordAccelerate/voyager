#pragma once

#include "ArchitectureParams.h"

// Macro geometry and arithmetic. Element output capacity also depends
// on the weight datatype width relative to CIM_BASE_B_WIDTH.

#ifndef CIM_MACRO_INPUT_LANES
#define CIM_MACRO_INPUT_LANES 64
#endif

#ifndef CIM_MACRO_OUTPUT_LANES
#define CIM_MACRO_OUTPUT_LANES 8
#endif

#ifndef CIM_WEIGHT_SETS
#define CIM_WEIGHT_SETS 18
#endif

#ifndef CIM_BASE_A_WIDTH
#define CIM_BASE_A_WIDTH 4
#endif

#ifndef CIM_BASE_B_WIDTH
#define CIM_BASE_B_WIDTH 4
#endif

#ifndef CIM_BASE_C_WIDTH
#define CIM_BASE_C_WIDTH 20
#endif

#ifndef CIM_MACRO_WRITE_INPUT_LANES
#define CIM_MACRO_WRITE_INPUT_LANES 1
#endif

#ifndef CIM_MAC_LATENCY
#define CIM_MAC_LATENCY 1
#endif

#ifndef CIM_MODE
#define CIM_MODE 0
#endif

#ifndef CIM_SIGNED
#define CIM_SIGNED true
#endif

#ifndef CIM_TILE_INPUT_AXIS_ELEMENTS
#define CIM_TILE_INPUT_AXIS_ELEMENTS 1
#endif

#ifndef CIM_TILE_OUTPUT_AXIS_ELEMENTS
#define CIM_TILE_OUTPUT_AXIS_ELEMENTS 4
#endif

#ifndef CIM_INPUT_AXIS_TILES
#define CIM_INPUT_AXIS_TILES 1
#endif

#ifndef CIM_OUTPUT_AXIS_TILES
#define CIM_OUTPUT_AXIS_TILES 1
#endif

#ifndef CIM_A_PORT_TILES
#define CIM_A_PORT_TILES CIM_INPUT_AXIS_TILES
#endif

#ifndef CIM_B_PORT_TILES
#define CIM_B_PORT_TILES CIM_OUTPUT_AXIS_TILES
#endif

#ifndef CIM_C_PORT_TILES
#define CIM_C_PORT_TILES CIM_OUTPUT_AXIS_TILES
#endif

// CIMProcessor consumes one output-major beat containing the reduced result.
#ifndef CIM_C_BEAT_LAYOUT
#define CIM_C_BEAT_LAYOUT 1
#endif

#ifndef CIM_ARRAY_RESULT_SLOTS
#define CIM_ARRAY_RESULT_SLOTS CIM_INPUT_AXIS_TILES
#endif

#ifndef CIM_LOCAL_ACCUM_CONTEXTS
#define CIM_LOCAL_ACCUM_CONTEXTS 4
#endif

namespace cim {
// The processor passes one intermediate accumulation result at a time.
constexpr int ACCUM_TO_WB_FIFO_DEPTH = 1;
constexpr int OUTPUT_FIFO_DEPTH = 8;
constexpr int ACCUM_METADATA_FIFO_DEPTH = 2;
}  // namespace cim
