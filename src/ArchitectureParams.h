#pragma once

#if defined(P8_1)

using P8 = Posit<8, 1>;
using F16 = StdFloat<7, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE P8
#define WEIGHT_DATATYPE P8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE P8
#define VECTOR_DATATYPE ACCUM_DATATYPE

#elif defined(E4M3)

using F8 = StdFloat<3, 4, false, true, AC_RND_CONV>;
using F16 = StdFloat<7, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8
#define VECTOR_DATATYPE F16

#elif defined(E4M3_NS)

using F8 = StdFloat<3, 4, false, false, AC_RND_CONV>;
using F16 = StdFloat<7, 8, false, false, AC_RND_CONV>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8
#define VECTOR_DATATYPE F16

#elif defined(E4M3_DW)

using F8 = StdFloat<3, 4, true, true, AC_RND_CONV>;
using F16 = StdFloat<7, 8, true, true, AC_RND_CONV>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8
#define VECTOR_DATATYPE F16

#elif defined(E4M3_DW_NS)

using F8 = StdFloat<3, 4, true, false, AC_RND_CONV>;
using F16 = StdFloat<7, 8, true, false, AC_RND_CONV>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8
#define VECTOR_DATATYPE F16

#elif defined(E5M2)

using F8 = StdFloat<2, 5>;
using F16 = StdFloat<7, 8>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8
#define VECTOR_DATATYPE F16

#elif defined(HYBRID_FP8)

using F8 = StdFloat<3, 4>;
using F16 = StdFloat<7, 8>;
using F9 = StdFloat<3, 5>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F8
#define HYBRID_TYPE F9
#define VECTOR_DATATYPE F16

#elif defined(BF16)

using F16 = StdFloat<7, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F16

#elif defined(FP32)

using F32 = StdFloat<23, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE F32
#define WEIGHT_DATATYPE F32
#define ACCUM_DATATYPE F32
#define OUTPUT_DATATYPE F32
#define VECTOR_DATATYPE F32

#elif defined(INT8)

using F16 = StdFloat<7, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int24
#define OUTPUT_DATATYPE DataTypes::int8
#define VECTOR_DATATYPE F16

#elif defined(INT8_32)

using F16 = StdFloat<7, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int32
#define OUTPUT_DATATYPE DataTypes::int8
#define VECTOR_DATATYPE F16

#elif defined(MXINT8)

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int32
#define ACCUM_BUFFER_DATATYPE DataTypes::bfloat16
#define OUTPUT_DATATYPE DataTypes::int8
#define VECTOR_DATATYPE DataTypes::bfloat16
#define SCALE_DATATYPE DataTypes::fp8_e8m0

#define SUPPORT_MX true

#elif defined(MXNF4)

#define INPUT_DATATYPE DataTypes::nf4
#define WEIGHT_DATATYPE DataTypes::nf4
#define ACCUM_DATATYPE DataTypes::int32
#define ACCUM_BUFFER_DATATYPE DataTypes::bfloat16
#define OUTPUT_DATATYPE DataTypes::nf4
#define VECTOR_DATATYPE DataTypes::bfloat16
#define SCALE_DATATYPE DataTypes::fp8_e5m3

#define SUPPORT_MX true

#elif defined(CFLOAT)

#define INPUT_DATATYPE CFloat
#define WEIGHT_DATATYPE CFloat
#define ACCUM_DATATYPE CFloat
#define OUTPUT_DATATYPE CFloat
#define VECTOR_DATATYPE CFloat

#else

#error "No datatype specified!"

#endif

// default datatype for the accumulation buffer
#ifndef ACCUM_BUFFER_DATATYPE
#define ACCUM_BUFFER_DATATYPE ACCUM_DATATYPE
#endif

// default to E8M0 scale
#ifndef SCALE_DATATYPE
#define SCALE_DATATYPE DataTypes::fp8_e8m0
#endif

#ifndef SUPPORT_MX
#define SUPPORT_MX false
#endif

#if !SUPPORT_MX
#define OUTPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE
#define VECTOR_INPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE
#else
#define OUTPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE, SCALE_DATATYPE
#define VECTOR_INPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE, SCALE_DATATYPE
#endif

#ifndef IC_DIMENSION
#error "No IC dimension specified!"
#endif

#ifndef OC_DIMENSION
#error "No OC dimension specified!"
#endif

#ifndef INPUT_BUFFER_SIZE
#define INPUT_BUFFER_SIZE 1024
#endif

#ifndef WEIGHT_BUFFER_SIZE
#define WEIGHT_BUFFER_SIZE 1024
#endif

#ifndef ACCUM_BUFFER_SIZE
#define ACCUM_BUFFER_SIZE 1024
#endif
