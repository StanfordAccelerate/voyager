#pragma once
#include <ac_std_float.h>

// forward declarations
template <int mantissa, int exp, bool useDWImpl,
          bool ieee_compliance, ac_q_mode Q>
class StdFloat;

template <int wdth, bool sgnd>
class Int;

// #include "FloatTypes.h"
#include "IntTypes.h"
#include "PositTypes.h"
#include "StdFloatTypes.h"

#ifndef __SYNTHESIS__
#include "CFloat.h"
#endif

namespace DataTypes {
typedef Int<8, true> int8;
typedef Int<16, true> int16;
typedef Int<32, true> int32;

typedef StdFloat<3, 4, false, true, AC_RND_CONV> e4m3;
typedef StdFloat<2, 5, false, true, AC_RND_CONV> e5m2;
typedef StdFloat<7, 8, false, true, AC_RND_CONV> bfloat16;
typedef StdFloat<23, 8, false, true, AC_RND_CONV> fp32;
};  // namespace DataTypes
