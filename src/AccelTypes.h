#pragma once

// clang-format off
#include <systemc.h>
// clang-format on
#include <ac_int.h>
#include <ac_sc.h>
#include <ccs_types.h>
#include <mc_connections.h>

#include "Params.h"
#include "PositTypes.h"

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
    m &address;
    m &burstSize;
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
};

template <typename TYPE>
struct PEInput {
  TYPE data;
  ac_int<1, false> swapWeights;

  static const unsigned int width = TYPE::width + 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &data;
    m &swapWeights;
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
};

template <typename TYPE>
struct PEWeight {
  TYPE data;
  ac_int<4, false> tag;

  static const unsigned int width = TYPE::width + 4;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &data;
    m &tag;
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
};

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
};

template <size_t SIZE, int sbits, int fbits>
class Pack1D<PEWeight<PositFP<sbits, fbits> >, SIZE> {
 public:
  PEWeight<PositFP<sbits, fbits> > value[SIZE];

  static const unsigned int width = PEWeight<PositFP<sbits, fbits> >::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PEWeight<PositFP<sbits, fbits> >, SIZE>(); }

  PEWeight<PositFP<sbits, fbits> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEWeight<PositFP<sbits, fbits> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.sign;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.scale;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data.fraction;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].data._zero;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].tag;
    }
  }
};

template <size_t SIZE, int sbits, int fbits>
class Pack1D<PositFP<sbits, fbits>, SIZE> {
 public:
  PositFP<sbits, fbits> value[SIZE];

  static const unsigned int width = PositFP<sbits, fbits>::width * SIZE;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PositFP<sbits, fbits>, SIZE>(); }

  PositFP<sbits, fbits> &operator[](unsigned int i) { return this->value[i]; }
  const PositFP<sbits, fbits> &operator[](unsigned int i) const {
    return this->value[i];
  }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].sign;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].scale;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i].fraction;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < SIZE; i++) {
      m &value[i]._zero;
    }
  }
};

template <typename TYPE, size_t SIZE>
struct BufferWriteRequest {
  int address;
  Pack1D<TYPE, SIZE> data;

  static const unsigned int width = 32 + Pack1D<TYPE, SIZE>::width;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &address;
    m &data;
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
