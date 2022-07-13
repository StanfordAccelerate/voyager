#define POSIT

#ifdef POSIT
#include "src/PositTypes.h"

using P8 = Posit<8, 1>;
using P16 = Posit<16, 1>;
using P8D = Posit<8, 1>::DecomposedPosit;
using P16D = Posit<16, 1>::DecomposedPosit;
#define INPUT_DATATYPE P8
#define WEIGHT_DATATYPE P8
#define ACCUM_DATATYPE P16
#define OUTPUT_DATATYPE P8
#else
#define INPUT_DATATYPE ac_int<8, true>
#define WEIGHT_DATATYPE ac_int<8, true>
#define ACCUM_DATATYPE ac_int<8, true>
#define OUTPUT_DATATYPE ac_int<8, true>
#endif

#define DIMENSION 16
#define INPUT_BUFFER_SIZE 1024
#define WEIGHT_BUFFER_SIZE 1024
#define ACCUMULATION_BUFFER_SIZE 1024

// #define PIPE_INPUT 1
