#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

#include <ac_std_float.h>

#include <numeric>
#include <string>

// IWYU pragma: begin_exports
#include "IntTypes.h"
#include "NormalFloat.h"
#include "PositTypes.h"
#include "ScaleTypes.h"
#include "StdFloatTypes.h"

#ifndef __SYNTHESIS__
#include "CFloat.h"
#endif
// IWYU pragma: end_exports

namespace DataTypes {
typedef Int<1, true> int1;
typedef Int<2, false> uint2;
typedef Int<2, true> int2;
typedef Int<4, true> int4;
typedef Int<6, true> int6;
typedef Int<8, true> int8;
typedef Int<16, false> uint16;
typedef Int<16, true> int16;
typedef Int<18, true> int18;
typedef Int<24, true> int24;
typedef Int<32, true> int32;
typedef Int<64, true> int64;

typedef StdFloat<3, 4, false, true, AC_RND_CONV> e4m3;
typedef StdFloat<2, 5, false, true, AC_RND_CONV> e5m2;
typedef StdFloat<7, 8, false, true, AC_RND_CONV> bfloat16;
#if defined(CFLOAT)
typedef CFloat float32;
#else
typedef StdFloat<23, 8, false, true, AC_RND_CONV> float32;
#endif

typedef Posit<8, 1> posit8;

typedef UFloat<8, 8> fp8_e8m0;
typedef UFloat<8, 5> fp8_e5m3;

template <typename T>
struct TypeName {
  static std::string name() { return "Unknown"; }
};

template <>
struct TypeName<int1> {
  static std::string name() { return "int1"; }
};

template <>
struct TypeName<uint2> {
  static std::string name() { return "uint2"; }
};

template <>
struct TypeName<int2> {
  static std::string name() { return "int2"; }
};

template <>
struct TypeName<int4> {
  static std::string name() { return "int4"; }
};

template <>
struct TypeName<int6> {
  static std::string name() { return "int6"; }
};

template <>
struct TypeName<int8> {
  static std::string name() { return "int8"; }
};

template <>
struct TypeName<uint16> {
  static std::string name() { return "uint16"; }
};

template <>
struct TypeName<int16> {
  static std::string name() { return "int16"; }
};

template <>
struct TypeName<int18> {
  static std::string name() { return "int18"; }
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
struct TypeName<int64> {
  static std::string name() { return "int64"; }
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
struct TypeName<float32> {
  static std::string name() { return "float32"; }
};

template <>
struct TypeName<fp8_e8m0> {
  static std::string name() { return "fp8_e8m0"; }
};

template <>
struct TypeName<fp8_e5m3> {
  static std::string name() { return "fp8_e5m3"; }
};

template <>
struct TypeName<posit8> {
  static std::string name() { return "posit8_1"; }
};

};  // namespace DataTypes

// clang-format off
#define SUPPORTED_TYPES          \
  DataTypes::int1,               \
  DataTypes::uint2,              \
  DataTypes::int2,               \
  DataTypes::int4,               \
  DataTypes::int6,               \
  DataTypes::int8,               \
  DataTypes::uint16,             \
  DataTypes::int16,              \
  DataTypes::int18,              \
  DataTypes::int24,              \
  DataTypes::int32,              \
  DataTypes::int64,              \
  DataTypes::e4m3,               \
  DataTypes::e5m2,               \
  DataTypes::bfloat16,           \
  DataTypes::float32,            \
  DataTypes::posit8,             \
  DataTypes::fp8_e8m0,           \
  DataTypes::fp8_e5m3
// clang-format on

// ================================================================
// Datatype Indexing
// ================================================================

template <typename T, typename... Ts>
struct TypeIndex;

template <typename T, typename... Ts>
struct TypeIndex<T, T, Ts...> {
  static constexpr int value = 0;
};

template <typename T, typename U, typename... Ts>
struct TypeIndex<T, U, Ts...> {
  static constexpr int value = 1 + TypeIndex<T, Ts...>::value;
};

template <typename T, typename... Ts>
constexpr int get_type_index() {
  return TypeIndex<T, Ts...>::value;
}

/**
 * \brief Get the `::width` of a type, given its index into a type sequence.
 * \tparam Ts The sequence of types.
 * \param index The index of the type in the sequence.
 * \return The width of the type.
 */
template <typename... Ts>
size_t get_type_width(size_t index) {
  constexpr size_t widths[] = {Ts::width...};
  assert(index < sizeof...(Ts) && "Invalid type index");
  return widths[index];
}

template <typename... Ts>
int get_index_from_type_name(const std::string& dtype) {
  int index = -1;
  ((dtype == DataTypes::TypeName<Ts>::name()
        ? (index = get_type_index<Ts, Ts...>())
        : 0),
   ...);
  return index;
}

template <typename... Ts>
size_t get_type_width(std::string dtype) {
  int index = get_index_from_type_name<Ts...>(dtype);
  if (index == -1) {
    throw std::invalid_argument("Unsupported datatype: " + dtype);
  }
  return get_type_width<Ts...>(index);
}

template <typename T, size_t N, int port_width>
struct dtype_fetch_config {
  static constexpr int fetch_width = T::width * N;
  // Pack multiple blocks into a single fetch if possible. The fetch width is
  // the LCM of the data width and port width.
  static constexpr int packed_fetch_width =
      fetch_width * port_width / std::gcd(fetch_width, port_width);
  static constexpr int packing_factor = packed_fetch_width / fetch_width;
  static constexpr int max_fetch_width =
      packed_fetch_width > fetch_width ? packed_fetch_width : fetch_width;
};

// ================================================================
// Datatype Concatenation
// ================================================================

// Base case: single tuple
template <typename... Tuples>
struct TupleConcat;

template <typename... Ts>
struct TupleConcat<std::tuple<Ts...>> {
  using type = std::tuple<Ts...>;
};

// Recursive case: combine two or more
template <typename... Ts1, typename... Ts2, typename... Rest>
struct TupleConcat<std::tuple<Ts1...>, std::tuple<Ts2...>, Rest...> {
  using type = typename TupleConcat<std::tuple<Ts1..., Ts2...>, Rest...>::type;
};

template <typename... Tuples>
using TypeConcat = typename TupleConcat<Tuples...>::type;
