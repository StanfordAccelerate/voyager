#pragma once

#if defined(P8_1)

using P8 = Posit<8, 1>;
using P16 = Posit<16, 1>;
using P8D = Posit<8, 1>::AccumulationDatatype;
using P16D = Posit<16, 1>::AccumulationDatatype;
#define INPUT_DATATYPE P8
#define WEIGHT_DATATYPE P8
// #define ACCUM_DATATYPE P16
#define ACCUM_DATATYPE PositFP<8, 7>
#define OUTPUT_DATATYPE P8

#elif defined(E4M3)

using F8 = StdFloat<3, 4>;
using F16 = StdFloat<7, 8>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8

#elif defined(E4M3_DW)

using F8 = StdFloat<3, 4, true>;
using F16 = StdFloat<7, 8, true>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8

#elif defined(E5M2)

using F8 = StdFloat<2, 5>;
using F16 = StdFloat<7, 8>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8

#elif defined(HYBRID_FP8)

using F8 = StdFloat<3, 4>;
using F16 = StdFloat<7, 8>;
using F9 = StdFloat<3, 5>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8
#define HYBRID_TYPE F9

#elif defined(BF16)

using F16 = StdFloat<7, 8>;
using F32 = StdFloat<23, 8>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F32
#define OUTPUT_DATATYPE F16

#elif defined(BF16_ONLY)

using F16 = StdFloat<7, 8>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F16

#else

#error "No datatype specified!"

#endif

#if !defined(DIMENSION)

#error "No dimension specified!"

#endif

#define INPUT_BUFFER_SIZE 1024
#define WEIGHT_BUFFER_SIZE 1024
#define ACCUMULATION_BUFFER_SIZE 1024

// #define PIPE_INPUT 1
