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
using F32 = StdFloat<23, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F32
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F32

#elif defined(BF16_NS)

using F16 = StdFloat<7, 8, false, false, AC_RND_CONV>;
using F32 = StdFloat<23, 8, false, false, AC_RND_CONV>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F32
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F32

#elif defined(BF16_TRN)

using F16 = StdFloat<7, 8, false, true, AC_TRN_ZERO>;
using F32 = StdFloat<23, 8, false, true, AC_TRN_ZERO>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F32
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F32

#elif defined(BF16_NS_TRN)

using F16 = StdFloat<7, 8, false, false, AC_TRN_ZERO>;
using F32 = StdFloat<23, 8, false, false, AC_TRN_ZERO>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F32
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F32

#elif defined(BF16_ONLY)

using F16 = StdFloat<7, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F16

#elif defined(BF16_ONLY_NS)

using F16 = StdFloat<7, 8, false, false, AC_RND_CONV>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F16

#elif defined(BF16_ONLY_TRN)

using F16 = StdFloat<7, 8, false, true, AC_TRN_ZERO>;

#define INPUT_DATATYPE F16
#define WEIGHT_DATATYPE F16
#define ACCUM_DATATYPE F16
#define OUTPUT_DATATYPE F16
#define VECTOR_DATATYPE F16

#elif defined(BF16_ONLY_NS_TRN)

using F16 = StdFloat<7, 8, false, false, AC_TRN_ZERO>;

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

using I8 = Int<8, true>;
using I24 = Int<24, true>;
using F16 = StdFloat<7, 8, false, true, AC_RND_CONV>;

#define INPUT_DATATYPE I8
#define WEIGHT_DATATYPE I8
#define ACCUM_DATATYPE I24
#define OUTPUT_DATATYPE I8
#define VECTOR_DATATYPE F16

#elif defined(CFLOAT)

#define INPUT_DATATYPE CFloat
#define WEIGHT_DATATYPE CFloat
#define ACCUM_DATATYPE CFloat
#define OUTPUT_DATATYPE CFloat
#define VECTOR_DATATYPE CFloat

#else

#error "No datatype specified!"

#endif

#if !defined(IC_DIMENSION)

#error "No IC dimension specified!"

#endif

#if !defined(OC_DIMENSION)

#error "No OC dimension specified!"

#endif

#define INPUT_BUFFER_SIZE 1024
#define WEIGHT_BUFFER_SIZE 1024
#define ACCUMULATION_BUFFER_SIZE 1024

// #define PIPE_INPUT 1
