#pragma once

// clang-format off
#include <systemc.h>
// clang-format on
#include <ac_fixed.h>
#include <ac_int.h>
#include <ac_sc.h>
#include <ac_std_float.h>
#include <ccs_types.h>
#include <mc_connections.h>

#include "Params.h"

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

template <typename TYPE, size_t SIZE>
class Pack1D {
 public:
  TYPE value[SIZE];

  static const unsigned int width = TYPE::width * SIZE;

  Pack1D() {}

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
