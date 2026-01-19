#pragma once

// This header depends on `DataTypes.h`, but only through macros. As a result,
// IWYU thinks that this header is not used, even though it is.
#include "datatypes/DataTypes.h"  // IWYU pragma: keep

// Constants for Approximation Unit
constexpr int NUM_MAXES = 6;
constexpr int NUM_RANGES = 7;
constexpr int NUM_COEFFS = 3;

#if defined(P8_1)

#define INPUT_DATATYPE DataTypes::posit8
#define WEIGHT_DATATYPE DataTypes::posit8
#define ACCUM_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16

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

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int24
#define VECTOR_DATATYPE DataTypes::bfloat16

#elif defined(INT8_32)

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int32
#define VECTOR_DATATYPE DataTypes::bfloat16

#elif defined(MXINT8)

#define INPUT_DATATYPE DataTypes::int8
#define WEIGHT_DATATYPE DataTypes::int8
#define ACCUM_DATATYPE DataTypes::int32
#define ACCUM_BUFFER_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16
#define SCALE_DATATYPE DataTypes::fp8_e8m0

#define SUPPORT_MX true

#elif defined(MXNF4)

#define INPUT_DATATYPE DataTypes::uint2, DataTypes::int4, DataTypes::int6
#define WEIGHT_DATATYPE DataTypes::uint2, DataTypes::int4, DataTypes::int6
#define ACCUM_DATATYPE DataTypes::int18
#define ACCUM_BUFFER_DATATYPE DataTypes::bfloat16
#define VECTOR_DATATYPE DataTypes::bfloat16
#define SCALE_DATATYPE DataTypes::fp8_e5m3
#define VU_INPUT_TYPES \
  VECTOR_DATATYPE, SCALE_DATATYPE, DataTypes::e4m3, DataTypes::int1
#define OUTPUT_DATATYPES INPUT_DATATYPE, VU_INPUT_TYPES
#define SPMM_META_DATATYPE DataTypes::int32

#define SA_INPUT_TYPE DataTypes::int6
#define SA_WEIGHT_TYPE DataTypes::int6

// Width of the widest data type in the list
#define INPUT_DTYPE_WIDTH 6
#define WEIGHT_DTYPE_WIDTH 6

// Number of bits used to represent the data type index
#define DTYPE_INDEX_WIDTH 2

#define IC_PORT_WIDTH (IC_DIMENSION * 4)
#define OC_PORT_WIDTH (OC_DIMENSION * 4)
#define VECTOR_UNIT_WIDTH OC_DIMENSION
#define MV_UNIT_WIDTH (OC_DIMENSION * 2)
#define SPMM_UNIT_WIDTH OC_DIMENSION

#define SUPPORT_MX true
#define SUPPORT_CODEBOOK_QUANT true

#elif defined(CFLOAT)

#define INPUT_DATATYPE CFloat
#define WEIGHT_DATATYPE CFloat
#define ACCUM_DATATYPE CFloat
#define VECTOR_DATATYPE CFloat

#else
#error "No datatype specified!"
#endif

// ================================================================
// Common Constants
// ================================================================

#ifndef DOUBLE_BUFFERED_ACCUM_BUFFER
#define DOUBLE_BUFFERED_ACCUM_BUFFER false
#endif

#ifndef IC_DIMENSION
#error "No IC dimension specified!"
#endif

#ifndef OC_DIMENSION
#error "No OC dimension specified!"
#endif

#define ADDRESS_WIDTH 32

#ifndef SUPPORT_MX
#define SUPPORT_MX false
#endif

#ifndef SUPPORT_MVM
#define SUPPORT_MVM false
#endif

#ifndef SUPPORT_SPMM
#define SUPPORT_SPMM false
#endif

#ifndef MV_UNIT_WIDTH
#define MV_UNIT_WIDTH OC_DIMENSION
#endif

#ifndef SPMM_UNIT_WIDTH
#define SPMM_UNIT_WIDTH OC_DIMENSION
#endif

#ifndef VECTOR_UNIT_WIDTH
#define VECTOR_UNIT_WIDTH OC_DIMENSION
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

#ifndef SCALE_DATATYPE
#define SCALE_DATATYPE DataTypes::fp8_e8m0
#endif

#ifndef VU_INPUT_TYPES
#if SUPPORT_MX
#define VU_INPUT_TYPES VECTOR_DATATYPE, SCALE_DATATYPE
#else
#define VU_INPUT_TYPES INPUT_DATATYPE, VECTOR_DATATYPE
#endif
#endif

#ifndef OUTPUT_DATATYPES
#if SUPPORT_MX
#define OUTPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE, SCALE_DATATYPE
#else
#define OUTPUT_DATATYPES INPUT_DATATYPE, VECTOR_DATATYPE
#endif
#endif

#ifndef SPMM_META_DATATYPE
#define SPMM_META_DATATYPE DataTypes::int16
#endif

// ================================================================
// Codebook Quantization
// ================================================================

#ifndef SUPPORT_CODEBOOK_QUANT
#define SUPPORT_CODEBOOK_QUANT false
#endif

#ifndef DECODED_INPUT_DTYPE_WIDTH
#define DECODED_INPUT_DTYPE_WIDTH SA_INPUT_TYPE::width
#endif

#ifndef DECODED_WEIGHT_DTYPE_WIDTH
#define DECODED_WEIGHT_DTYPE_WIDTH SA_WEIGHT_TYPE::width
#endif

#define MAX_DECODED_DTYPE_WIDTH                           \
  (DECODED_INPUT_DTYPE_WIDTH > DECODED_WEIGHT_DTYPE_WIDTH \
       ? DECODED_INPUT_DTYPE_WIDTH                        \
       : DECODED_WEIGHT_DTYPE_WIDTH)

#ifndef NUM_CODEBOOK_ENTRIES
#define NUM_CODEBOOK_ENTRIES 16
#endif

// ================================================================
// Datatype Width Configuration
// ================================================================

using InputTypeList = std::tuple<INPUT_DATATYPE>;
using WeightTypeList = std::tuple<WEIGHT_DATATYPE>;

#ifndef DTYPE_INDEX_WIDTH
#define DTYPE_INDEX_WIDTH 1
#endif

#ifndef INPUT_DTYPE_WIDTH
#define INPUT_DTYPE_WIDTH INPUT_DATATYPE::width
#endif

#ifndef WEIGHT_DTYPE_WIDTH
#define WEIGHT_DTYPE_WIDTH WEIGHT_DATATYPE::width
#endif

// ================================================================
// Port Width Definitions
// ================================================================

#ifndef IC_PORT_WIDTH
#define IC_PORT_WIDTH (IC_DIMENSION * INPUT_DTYPE_WIDTH)
#endif

#define IC_PORT_TYPE ac_int<IC_PORT_WIDTH, false>

#ifndef OC_PORT_WIDTH
#define OC_PORT_WIDTH (OC_DIMENSION * WEIGHT_DTYPE_WIDTH)
#endif

#define OC_PORT_TYPE ac_int<OC_PORT_WIDTH, false>

#define MVU_SCALE_PORT_WIDTH \
  (SCALE_DATATYPE::width * MV_UNIT_WIDTH / OC_DIMENSION)

// ================================================================
// Buffer Configurations
// ================================================================

#ifndef INPUT_BUFFER_SIZE
#define INPUT_BUFFER_SIZE 1024
#endif

#ifndef INPUT_BUFFER_WIDTH
#define INPUT_BUFFER_WIDTH (IC_DIMENSION * INPUT_DTYPE_WIDTH)
#endif

#ifndef WEIGHT_BUFFER_SIZE
#define WEIGHT_BUFFER_SIZE 1024
#endif

#ifndef WEIGHT_BUFFER_WIDTH
#define WEIGHT_BUFFER_WIDTH (OC_DIMENSION * WEIGHT_DTYPE_WIDTH)
#endif

#ifndef ACCUM_BUFFER_SIZE
#define ACCUM_BUFFER_SIZE 1024
#endif

// ================================================================
// DwC Configurations
// ================================================================

#ifndef SUPPORT_DWC
#define SUPPORT_DWC false
#endif

#ifndef DWC_WIDTH
#define DWC_WIDTH 40
#endif

#ifndef UNROLLFACTOR
#define UNROLLFACTOR OC_DIMENSION
#endif

#ifndef DWC_KERNEL_DIM
#define DWC_KERNEL_DIM 3
#endif
#define DWC_KERNEL_SIZE (DWC_KERNEL_DIM * DWC_KERNEL_DIM)

#ifndef DWC_DATATYPE
#define DWC_DATATYPE INPUT_DATATYPE
#endif

#ifndef DWC_PSUM
#define DWC_PSUM ACCUM_DATATYPE
#endif
