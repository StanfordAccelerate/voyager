#pragma once

// clang-format off
#include <systemc.h>
// clang-format on
#include <ac_int.h>
#include <ac_sc.h>
#include <ccs_types.h>
#include <mc_connections.h>

#include "DataTypes.h"
#include "Params.h"

#ifdef DEBUG_LOG
#define DLOG(x) CCS_LOG(x)
#else
#define DLOG(x)
#endif

struct MemoryRequest {
  int address;
  int burstSize;

  static const unsigned int width = 32 + 32;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & address;
    m & burstSize;
  }

  inline friend void sc_trace(sc_trace_file *tf,
                              const MemoryRequest &memRequest,
                              const std::string &name) {
    sc_trace(tf, memRequest.address, name + ".address");
    sc_trace(tf, memRequest.burstSize, name + ".burstSize");
  }

  inline friend std::ostream &operator<<(ostream &os,
                                         const MemoryRequest &memRequest) {
    os << memRequest.address << " ";
    os << memRequest.burstSize << " ";

    return os;
  }

  inline friend bool operator==(const MemoryRequest &lhs,
                                const MemoryRequest &rhs) {
    return lhs.address == rhs.address && lhs.burstSize == rhs.burstSize;
  }
};

template <typename TYPE>
struct PEInput {
  TYPE data;
  ac_int<1, false> swapWeights;

  // #ifdef HYBRID_FP8
  //   ac_int<1, false> castToE5M2;
  // #endif

  static const unsigned int width = TYPE::width + 1
      // #ifdef HYBRID_FP8
      //                                     + 1
      // #endif
      ;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & data;
    m & swapWeights;
    // #ifdef HYBRID_FP8
    //     m & castToE5M2;
    // #endif
  }

  inline friend void sc_trace(sc_trace_file *tf, const PEInput &peInput,
                              const std::string &name) {
    sc_trace(tf, peInput.data, name + ".data");
    sc_trace(tf, peInput.swapWeights, name + ".swapWeights");
  }

  inline friend std::ostream &operator<<(ostream &os, const PEInput &peInput) {
    os << peInput.data << " ";
    os << peInput.swapWeights << " ";

    return os;
  }

  inline friend bool operator==(const PEInput &lhs, const PEInput &rhs) {
    return lhs.data == rhs.data && lhs.swapWeights == rhs.swapWeights;
  }
};

template <typename TYPE>
struct PEWeight {
  TYPE data;

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int TAG_WIDTH = int_log2(IC_DIMENSION);
  ac_int<TAG_WIDTH, false> tag;
  static const unsigned int width = TYPE::width + TAG_WIDTH;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & data;
    m & tag;
  }

  inline friend void sc_trace(sc_trace_file *tf, const PEWeight &peWeight,
                              const std::string &name) {
    sc_trace(tf, peWeight.data, name + ".data");
    sc_trace(tf, peWeight.tag, name + ".tag");
  }

  inline friend std::ostream &operator<<(ostream &os,
                                         const PEWeight &peWeight) {
    os << peWeight.data << " ";
    os << peWeight.tag << " ";

    return os;
  }

  inline friend bool operator==(const PEWeight &lhs, const PEWeight &rhs) {
    return lhs.data == rhs.data && lhs.tag == rhs.tag;
  }
};

template <typename TYPE, size_t SIZE>
class Pack1D {
 public:
  TYPE value[SIZE];

  static const unsigned int width = TYPE::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<TYPE, SIZE>(); }

  TYPE &operator[](unsigned int i) { return this->value[i]; }
  const TYPE &operator[](unsigned int i) const { return this->value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i];
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

// TODO: is there a way to make this more generic?

template <size_t SIZE, int nbits, int es>
class Pack1D<PEInput<Posit<nbits, es> >, SIZE> {
 public:
  PEInput<Posit<nbits, es> > value[SIZE];

  static const unsigned int width = PEInput<Posit<nbits, es> >::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PEInput<Posit<nbits, es> >, SIZE>(); }

  PEInput<Posit<nbits, es> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEInput<Posit<nbits, es> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.bits;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].swapWeights;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t SIZE, int nbits, int es>
class Pack1D<PEWeight<Posit<nbits, es> >, SIZE> {
 public:
  PEWeight<Posit<nbits, es> > value[SIZE];

  static const unsigned int width = PEWeight<Posit<nbits, es> >::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PEWeight<Posit<nbits, es> >, SIZE>(); }

  PEWeight<Posit<nbits, es> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEWeight<Posit<nbits, es> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.bits;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t SIZE, int mantissa, int exp>
class Pack1D<StdFloat<mantissa, exp>, SIZE> {
 public:
  StdFloat<mantissa, exp> value[SIZE];
  static const unsigned int width = StdFloat<mantissa, exp>::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<StdFloat<mantissa, exp>, SIZE>(); }

  StdFloat<mantissa, exp> &operator[](size_t i) { return value[i]; }
  const StdFloat<mantissa, exp> &operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].float_val.d;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t SIZE, int mantissa, int exp>
class Pack1D<PEInput<StdFloat<mantissa, exp> >, SIZE> {
 public:
  PEInput<StdFloat<mantissa, exp> > value[SIZE];

  static const unsigned int width =
      PEInput<StdFloat<mantissa, exp> >::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEInput<StdFloat<mantissa, exp> >, SIZE>();
  }

  PEInput<StdFloat<mantissa, exp> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEInput<StdFloat<mantissa, exp> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.float_val.d;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].swapWeights;
    }

    // #ifdef HYBRID_FP8
    // #pragma hls_unroll yes
    //     for (unsigned int i = 0; i < SIZE; i++) {
    //       m &value[i].castToE5M2;
    //     }
    // #endif
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t SIZE, int mantissa, int exp>
class Pack1D<PEWeight<StdFloat<mantissa, exp> >, SIZE> {
 public:
  PEWeight<StdFloat<mantissa, exp> > value[SIZE];

  static const unsigned int width =
      PEWeight<StdFloat<mantissa, exp> >::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEWeight<StdFloat<mantissa, exp> >, SIZE>();
  }

  PEWeight<StdFloat<mantissa, exp> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEWeight<StdFloat<mantissa, exp> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.float_val.d;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t SIZE, int i_width, bool i_signed>
class Pack1D<Int<i_width, i_signed>, SIZE> {
 public:
  Int<i_width, i_signed> value[SIZE];
  static const unsigned int width = Int<i_width, i_signed>::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<Int<i_width, i_signed>, SIZE>(); }

  Int<i_width, i_signed> &operator[](size_t i) { return value[i]; }
  const Int<i_width, i_signed> &operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].int_val;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t SIZE, int i_width, bool i_signed>
class Pack1D<PEInput<Int<i_width, i_signed> >, SIZE> {
 public:
  PEInput<Int<i_width, i_signed> > value[SIZE];

  static const unsigned int width =
      PEInput<Int<i_width, i_signed> >::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEInput<Int<i_width, i_signed> >, SIZE>();
  }

  PEInput<Int<i_width, i_signed> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEInput<Int<i_width, i_signed> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.int_val;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].swapWeights;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t SIZE, int i_width, bool i_signed>
class Pack1D<PEWeight<Int<i_width, i_signed> >, SIZE> {
 public:
  PEWeight<Int<i_width, i_signed> > value[SIZE];

  static const unsigned int width =
      PEWeight<Int<i_width, i_signed> >::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEWeight<Int<i_width, i_signed> >, SIZE>();
  }

  PEWeight<Int<i_width, i_signed> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEWeight<Int<i_width, i_signed> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.int_val;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < SIZE; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <typename TYPE, size_t SIZE>
struct BufferWriteRequest {
  int address;
  Pack1D<TYPE, SIZE> data;

  static const unsigned int width = 32 + Pack1D<TYPE, SIZE>::width;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & address;
    m & data;
  }

  inline friend void sc_trace(sc_trace_file *tf,
                              const BufferWriteRequest &bufWrite,
                              const std::string &name) {
    sc_trace(tf, bufWrite.address, name + ".address");
    sc_trace(tf, bufWrite.data, name + ".data");
  }

  inline friend std::ostream &operator<<(ostream &os,
                                         const BufferWriteRequest &bufWrite) {
    os << bufWrite.address << " ";
    os << bufWrite.data << " ";

    return os;
  }

  inline friend bool operator==(const BufferWriteRequest &lhs,
                                const BufferWriteRequest &rhs) {
    return lhs.address == rhs.address && lhs.data == rhs.data;
  }
};

// template <typename TYPE, size_t SIZE>
// class Wrapped<Pack1D<TYPE, SIZE> > {
//  public:
//   Pack1D<TYPE, SIZE> val;
//   Wrapped() : val(0) {}
//   Wrapped(const Pack1D<TYPE, SIZE> &v) : val(v) {}
//   static const unsigned int width = TYPE::width * SIZE;
//   static const bool is_signed = false;
//   template <unsigned int Size>
//   void Marshall(Marshaller<Size> &m) {
//     m &val;
//   }
// };

// template <unsigned int Size, typename TYPE, size_t SIZE>
// Marshaller<Size> &operator&(Marshaller<Size> &m, Pack1D<TYPE, SIZE> &rhs) {
// #pragma hls_unroll yes
//   for (unsigned int i = 0; i < SIZE; i++) {
//     m &rhs.value[i];
//   }
// }

// template <unsigned int Size, size_t SIZE>
// Marshaller<Size> &operator&(Marshaller<Size> &m, Pack1D<PositFP, SIZE> &rhs)
// { #pragma hls_unroll yes
//   for (unsigned int i = 0; i < SIZE; i++) {
//     m &value[i].sign;
//     m &value[i].scale;
//     m &value[i].fraction;
//   }
// }

// template <size_t SIZE>
// template <unsigned int Size>
// void Pack1D<PositFP, SIZE>::Marshall(Marshaller<Size> &m) {
//   for (unsigned int i = 0; i < SIZE; i++) {
//     m &value[i].sign;
//   }
//   for (unsigned int i = 0; i < SIZE; i++) {
//     m &value[i].scale;
//   }
//   for (unsigned int i = 0; i < SIZE; i++) {
//     m &value[i].fraction;
//   }
// }

// template <typename TYPE, size_t SIZE>
// class Wrapped<Pack1D<TYPE, SIZE> > {
//  public:
//   typedef PositFP Type;
//   Type val;
//   Wrapped() {}
//   Wrapped(const Type &v) : val(v) {}
//   static const unsigned int width = Type::width * SIZE;
//   static const bool is_signed = 1;
//   template <unsigned int Size>
//   void Marshall(Marshaller<Size> &m) {
//     for (int i = 0; i < SIZE; i++) {
//     }
//     m &val.sign;
//     m &val.scale;
//     m &val.fraction;
//   }
// };
// template <unsigned int Size, size_t SIZE>
// Marshaller<Size> &operator&(Marshaller<Size> &m, Pack1D < PositFP & rhs) {
//   typedef PositFP Type;
//   m.template AddField<ac_int<1, false>, 1>(rhs.sign);
//   m.template AddField<ac_int<8, true>, 8>(rhs.scale);
//   m.template AddField<ac_int<16, false>, 16>(rhs.fraction);
//   return m;
// }

template <typename TYPE, size_t SIZE>
inline bool operator==(const Pack1D<TYPE, SIZE> &lhs,
                       const Pack1D<TYPE, SIZE> &rhs) {
  bool is_equal = true;
#pragma hls_unroll yes
  for (unsigned int i = 0; i < SIZE; i++) {
    is_equal &= (lhs.value[i] == rhs.value[i]);
  }
  return is_equal;
}

template <typename TYPE, size_t SIZE>
inline void sc_trace(sc_trace_file *tf, const Pack1D<TYPE, SIZE> &vec,
                     const std::string &name) {
  sc_trace(tf, vec.value, name);
}

template <typename TYPE, size_t SIZE>
inline std::ostream &operator<<(ostream &os, const Pack1D<TYPE, SIZE> &vec) {
  for (int i = 0; i < SIZE; i++) {
    os << vec[i] << " ";
  }
  return os;
}

// Convert lower precision Pack1D to higher precision Pack1D
template <typename LOWER_PRECISION_TYPE, typename HIGHER_PRECISION_TYPE,
          size_t SIZE>
void convertPack1D(Pack1D<LOWER_PRECISION_TYPE, SIZE>
                       lowerPrecision[HIGHER_PRECISION_TYPE::width /
                                      LOWER_PRECISION_TYPE::width],
                   Pack1D<HIGHER_PRECISION_TYPE, SIZE> &higherPrecision) {
  static_assert(
      LOWER_PRECISION_TYPE::width <= HIGHER_PRECISION_TYPE::width,
      "Lower precision type must be smaller than higher precision type.");

  constexpr int num_words =
      HIGHER_PRECISION_TYPE::width / LOWER_PRECISION_TYPE::width;
  ac_int<HIGHER_PRECISION_TYPE::width * SIZE, false> higherPrecisionBits;

#pragma hls_unroll yes
  for (int i = 0; i < num_words; i++) {
    higherPrecisionBits.set_slc(
        static_cast<unsigned int>(i * LOWER_PRECISION_TYPE::width * SIZE),
        BitsToType<ac_int<LOWER_PRECISION_TYPE::width * SIZE, false> >(
            TypeToBits(lowerPrecision[i])));
  }

  higherPrecision = BitsToType<Pack1D<HIGHER_PRECISION_TYPE, SIZE> >(
      TypeToBits(higherPrecisionBits));
}

// Convert higher precision Pack1D to lower precision Pack1D
template <typename LOWER_PRECISION_TYPE, typename HIGHER_PRECISION_TYPE,
          size_t SIZE>
void convertPack1D(Pack1D<HIGHER_PRECISION_TYPE, SIZE> &higherPrecision,
                   Pack1D<LOWER_PRECISION_TYPE, SIZE>
                       lowerPrecision[HIGHER_PRECISION_TYPE::width /
                                      LOWER_PRECISION_TYPE::width]) {
  static_assert(
      LOWER_PRECISION_TYPE::width <= HIGHER_PRECISION_TYPE::width,
      "Lower precision type must be smaller than higher precision type.");

  constexpr int num_words =
      HIGHER_PRECISION_TYPE::width / LOWER_PRECISION_TYPE::width;

  ac_int<HIGHER_PRECISION_TYPE::width * SIZE, false> higherPrecisionBits;
  higherPrecisionBits =
      BitsToType<decltype(higherPrecisionBits)>(TypeToBits(higherPrecision));

#pragma hls_unroll yes
  for (int i = 0; i < num_words; i++) {
    ac_int<LOWER_PRECISION_TYPE::width * SIZE, false> lowerPrecisionBits;
    lowerPrecisionBits =
        higherPrecisionBits.template slc<LOWER_PRECISION_TYPE::width * SIZE>(
            static_cast<unsigned int>(i * LOWER_PRECISION_TYPE::width * SIZE));
    lowerPrecision[i] = BitsToType<Pack1D<LOWER_PRECISION_TYPE, SIZE> >(
        TypeToBits(lowerPrecisionBits));
  }
}
