#pragma once
#include <ac_std_float.h>

#include <string>

// forward declarations
template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
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
typedef Int<24, true> int24;
typedef Int<32, true> int32;

typedef StdFloat<3, 4, false, true, AC_RND_CONV> e4m3;
typedef StdFloat<2, 5, false, true, AC_RND_CONV> e5m2;
typedef StdFloat<7, 8, false, true, AC_RND_CONV> bfloat16;
typedef StdFloat<23, 8, false, true, AC_RND_CONV> fp32;

template <typename T>
struct TypeName {
  static std::string name() { return "Unknown"; }
};

template <>
struct TypeName<int8> {
  static std::string name() { return "int8"; }
};

template <>
struct TypeName<int16> {
  static std::string name() { return "int16"; }
};

template <>
struct TypeName<int24> {
  static std::string name() { return "int24"; }
};

template <>
struct TypeName<int32> {
  static std::string name() { return "int32"; }
};

template <>
struct TypeName<e4m3> {
  static std::string name() { return "fp8_e4m3"; }
};

template <>
struct TypeName<e5m2> {
  static std::string name() { return "fp8_e5m2"; }
};

template <>
struct TypeName<bfloat16> {
  static std::string name() { return "bfloat16"; }
};

template <>
struct TypeName<fp32> {
  static std::string name() { return "fp32"; }
};
};  // namespace DataTypes
