#pragma once
#include "src/datatypes/DataTypes.h"

#if defined(P8_1)

#define INPUT_DATATYPE DataTypes::posit8
#define WEIGHT_DATATYPE DataTypes::posit8
#define ACCUM_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE ACCUM_DATATYPE

#define SA_INPUT_TYPE INPUT_DATATYPE::decoded
#define SA_WEIGHT_TYPE WEIGHT_DATATYPE::decoded

#elif defined(E4M3)

#define INPUT_DATATYPE DataTypes::e4m3
#define WEIGHT_DATATYPE DataTypes::e4m3
#define ACCUM_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16

#elif defined(E4M3_NS)

using F8 = StdFloat<3, 4, false, false, AC_RND_CONV>;
using F16 = StdFloat<7, 8, false, false, AC_RND_CONV>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define VECTOR_DATATYPE F16

#elif defined(E4M3_DW)

using F8 = StdFloat<3, 4, true, true, AC_RND_CONV>;
using F16 = StdFloat<7, 8, true, true, AC_RND_CONV>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define VECTOR_DATATYPE F16

#elif defined(E4M3_DW_NS)

using F8 = StdFloat<3, 4, true, false, AC_RND_CONV>;
using F16 = StdFloat<7, 8, true, false, AC_RND_CONV>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define VECTOR_DATATYPE F16

#elif defined(E5M2)

#define INPUT_DATATYPE DataTypes::e5m2
#define WEIGHT_DATATYPE DataTypes::e5m2
#define ACCUM_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16

#elif defined(HYBRID_FP8)

using F8 = StdFloat<3, 4>;
using F16 = StdFloat<7, 8>;
using F9 = StdFloat<3, 5>;

#define INPUT_DATATYPE F8
#define WEIGHT_DATATYPE F8
#define ACCUM_DATATYPE F16
#define HYBRID_TYPE F9
#define VECTOR_DATATYPE F16

#elif defined(BF16)

#define INPUT_DATATYPE DataTypes::bfloat16
#define WEIGHT_DATATYPE DataTypes::bfloat16
#define ACCUM_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16

#elif defined(FP32)

#define INPUT_DATATYPE DataTypes::float32
#define WEIGHT_DATATYPE DataTypes::float32
#define ACCUM_DATATYPE DataTypes::float32
#define VECTOR_DATATYPE DataTypes::float32

#elif defined(INT8)

using F16 = DataTypes::bfloat16;

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int24
#define VECTOR_DATATYPE F16

#elif defined(INT8_32)

using F16 = DataTypes::bfloat16;

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int32
#define VECTOR_DATATYPE F16

#elif defined(MXINT8)

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int32
#define ACCUM_BUFFER_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16
#define SCALE_DATATYPE DataTypes::fp8_e8m0

#define SUPPORT_MX true

#elif defined(MXNF4)

// #define INPUT_DATATYPE DataTypes::nf4
// #define WEIGHT_DATATYPE DataTypes::nf4
#define INPUT_DATATYPE \
  DataTypes::int1, DataTypes::int2, DataTypes::nf4, DataTypes::int6
#define WEIGHT_DATATYPE \
  DataTypes::int1, DataTypes::int2, DataTypes::nf4, DataTypes::int6
#define ACCUM_DATATYPE DataTypes::int18
#define ACCUM_BUFFER_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16
#define SCALE_DATATYPE DataTypes::fp8_e5m3

#define SA_INPUT_TYPE DataTypes::int6
#define SA_WEIGHT_TYPE DataTypes::int6

// Width of the widest data type in the list
#define MAX_INPUT_DTYPE_WIDTH 6
#define MAX_WEIGHT_DTYPE_WIDTH 6

// Number of bits used to represent the data type index
#define DTYPE_INDEX_WIDTH 2

#define IC_PORT_WIDTH (IC_DIMENSION * 4)
#define OC_PORT_WIDTH (OC_DIMENSION * 4)

#define SUPPORT_MX true

#elif defined(CFLOAT)

#define INPUT_DATATYPE CFloat
#define WEIGHT_DATATYPE CFloat
#define ACCUM_DATATYPE CFloat
#define VECTOR_DATATYPE CFloat

#else
#error "No datatype specified!"
#endif

// ================================================================
// Default Datatypes
// ================================================================

#ifndef SA_INPUT_TYPE
#define SA_INPUT_TYPE INPUT_DATATYPE
#endif

#ifndef SA_WEIGHT_TYPE
#define SA_WEIGHT_TYPE WEIGHT_DATATYPE
#endif

#ifndef ACCUM_BUFFER_DATATYPE
#define ACCUM_BUFFER_DATATYPE ACCUM_DATATYPE
#endif

#ifndef SUPPORT_MX
#define SUPPORT_MX false
#endif

#ifndef SCALE_DATATYPE
#define SCALE_DATATYPE DataTypes::fp8_e8m0
#endif

#ifndef VECTOR_INPUT_DATATYPES
#if SUPPORT_MX
#define VECTOR_INPUT_DATATYPES VECTOR_DATATYPE, SCALE_DATATYPE
#else
#define VECTOR_INPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE
#endif
#endif

#ifndef OUTPUT_DATATYPES
#if SUPPORT_MX
#define OUTPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE, SCALE_DATATYPE
#else
#define OUTPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE
#endif
#endif

// ================================================================
// Common Constants
// ================================================================

#ifndef IC_DIMENSION
#error "No IC dimension specified!"
#endif

#ifndef OC_DIMENSION
#error "No OC dimension specified!"
#endif

#define ADDRESS_WIDTH 64

// ================================================================
// Datatype Width Configuration
// ================================================================

using MatrixInputTypes = std::tuple<INPUT_DATATYPE>;
using MatrixWeightTypes = std::tuple<WEIGHT_DATATYPE>;

#ifndef DTYPE_INDEX_WIDTH
#define DTYPE_INDEX_WIDTH 1
#endif

#ifndef MAX_INPUT_DTYPE_WIDTH
#define MAX_INPUT_DTYPE_WIDTH INPUT_DATATYPE::width
#endif

#ifndef MAX_WEIGHT_DTYPE_WIDTH
#define MAX_WEIGHT_DTYPE_WIDTH WEIGHT_DATATYPE::width
#endif

// ================================================================
// Port Width Definitions
// ================================================================

#ifndef IC_PORT_WIDTH
#define IC_PORT_WIDTH (IC_DIMENSION * MAX_INPUT_DTYPE_WIDTH)
#endif

#define IC_PORT_TYPE ac_int<IC_PORT_WIDTH, false>

#ifndef OC_PORT_WIDTH
#define OC_PORT_WIDTH (OC_DIMENSION * MAX_WEIGHT_DTYPE_WIDTH)
#endif

#define OC_PORT_TYPE ac_int<OC_PORT_WIDTH, false>

// ================================================================
// Buffer Configurations
// ================================================================

#ifndef INPUT_BUFFER_SIZE
#define INPUT_BUFFER_SIZE 1024
#endif

#ifndef INPUT_BUFFER_WIDTH
#define INPUT_BUFFER_WIDTH (IC_DIMENSION * MAX_INPUT_DTYPE_WIDTH)
#endif

#ifndef WEIGHT_BUFFER_SIZE
#define WEIGHT_BUFFER_SIZE 1024
#endif

#ifndef WEIGHT_BUFFER_WIDTH
#define WEIGHT_BUFFER_WIDTH (OC_DIMENSION * MAX_WEIGHT_DTYPE_WIDTH)
#endif

#ifndef ACCUM_BUFFER_SIZE
#define ACCUM_BUFFER_SIZE 1024
#endif
