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

#ifdef DEBUG
#define DLOG(x) CCS_LOG(x)
#else
#define DLOG(x)
#endif

struct MemoryRequest {
  ac_int<64, false> address;
  ac_int<32, false> burst_size;

  static const unsigned int width = 64 + 32;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & address;
    m & burst_size;
  }

  inline friend void sc_trace(sc_trace_file *tf, const MemoryRequest &request,
                              const std::string &name) {
    sc_trace(tf, request.address, name + ".address");
    sc_trace(tf, request.burst_size, name + ".burst_size");
  }

  inline friend std::ostream &operator<<(ostream &os,
                                         const MemoryRequest &request) {
    os << request.address << " ";
    os << request.burst_size << " ";
    return os;
  }

  inline friend bool operator==(const MemoryRequest &lhs,
                                const MemoryRequest &rhs) {
    return lhs.address == rhs.address && lhs.burst_size == rhs.burst_size;
  }
};

template <typename T>
struct PEInput {
  T data;
  ac_int<1, false> swapWeights;

  static const unsigned int width = T::width + 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & data;
    m & swapWeights;
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

template <typename T>
struct PEWeight {
  T data;

  static constexpr int int_log2(unsigned int n) {
    return (n <= 1) ? 0 : 1 + int_log2(n / 2);
  }

  static constexpr int TAG_WIDTH = int_log2(IC_DIMENSION);
  ac_int<TAG_WIDTH, false> tag;
  static const unsigned int width = T::width + TAG_WIDTH;

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

template <typename T, size_t Width>
class Pack1D {
 public:
  T value[Width];

  static const unsigned int width = T::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<T, Width>(); }

  T &operator[](unsigned int i) { return this->value[i]; }
  const T &operator[](unsigned int i) const { return this->value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i];
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

// TODO: is there a way to make this more generic?

template <size_t Width, int nbits, int es>
class Pack1D<PEInput<Posit<nbits, es> >, Width> {
 public:
  PEInput<Posit<nbits, es> > value[Width];

  static const unsigned int width = PEInput<Posit<nbits, es> >::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PEInput<Posit<nbits, es> >, Width>(); }

  PEInput<Posit<nbits, es> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEInput<Posit<nbits, es> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.bits;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].swapWeights;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int nbits, int es>
class Pack1D<PEWeight<Posit<nbits, es> >, Width> {
 public:
  PEWeight<Posit<nbits, es> > value[Width];

  static const unsigned int width = PEWeight<Posit<nbits, es> >::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PEWeight<Posit<nbits, es> >, Width>(); }

  PEWeight<Posit<nbits, es> > &operator[](unsigned int i) {
    return this->value[i];
  }
  const PEWeight<Posit<nbits, es> > &operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.bits;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int mantissa, int exp>
class Pack1D<StdFloat<mantissa, exp>, Width> {
 public:
  StdFloat<mantissa, exp> value[Width];
  static const unsigned int width = StdFloat<mantissa, exp>::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<StdFloat<mantissa, exp>, Width>(); }

  StdFloat<mantissa, exp> &operator[](size_t i) { return value[i]; }
  const StdFloat<mantissa, exp> &operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].float_val.d;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int mantissa, int exp>
class Pack1D<PEInput<StdFloat<mantissa, exp> >, Width> {
 public:
  PEInput<StdFloat<mantissa, exp> > value[Width];

  static const unsigned int width =
      PEInput<StdFloat<mantissa, exp> >::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEInput<StdFloat<mantissa, exp> >, Width>();
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
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.float_val.d;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].swapWeights;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int mantissa, int exp>
class Pack1D<PEWeight<StdFloat<mantissa, exp> >, Width> {
 public:
  PEWeight<StdFloat<mantissa, exp> > value[Width];

  static const unsigned int width =
      PEWeight<StdFloat<mantissa, exp> >::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEWeight<StdFloat<mantissa, exp> >, Width>();
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
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.float_val.d;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int i_width, bool i_signed>
class Pack1D<Int<i_width, i_signed>, Width> {
 public:
  Int<i_width, i_signed> value[Width];
  static const unsigned int width = Int<i_width, i_signed>::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<Int<i_width, i_signed>, Width>(); }

  Int<i_width, i_signed> &operator[](size_t i) { return value[i]; }
  const Int<i_width, i_signed> &operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].int_val;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int i_width, bool i_signed>
class Pack1D<PEInput<Int<i_width, i_signed> >, Width> {
 public:
  PEInput<Int<i_width, i_signed> > value[Width];

  static const unsigned int width =
      PEInput<Int<i_width, i_signed> >::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEInput<Int<i_width, i_signed> >, Width>();
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
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.int_val;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].swapWeights;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int i_width, bool i_signed>
class Pack1D<PEWeight<Int<i_width, i_signed> >, Width> {
 public:
  PEWeight<Int<i_width, i_signed> > value[Width];

  static const unsigned int width =
      PEWeight<Int<i_width, i_signed> >::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const {
    return Pack1D<PEWeight<Int<i_width, i_signed> >, Width>();
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
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.int_val;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width>
class Pack1D<NormalFloat4, Width> {
 public:
  NormalFloat4 value[Width];
  static const unsigned int width = NormalFloat4::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<NormalFloat4, Width>(); }

  NormalFloat4 &operator[](size_t i) { return value[i]; }
  const NormalFloat4 &operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].index;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width>
class Pack1D<PEInput<NormalFloat4>, Width> {
 public:
  PEInput<NormalFloat4> value[Width];

  static const unsigned int width = PEInput<NormalFloat4>::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PEInput<NormalFloat4>, Width>(); }

  PEInput<NormalFloat4> &operator[](unsigned int i) { return this->value[i]; }
  const PEInput<NormalFloat4> &operator[](unsigned int i) const {
    return this->value[i];
  }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.index;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].swapWeights;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width>
class Pack1D<PEWeight<NormalFloat4>, Width> {
 public:
  PEWeight<NormalFloat4> value[Width];

  static const unsigned int width = PEWeight<NormalFloat4>::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<PEWeight<NormalFloat4>, Width>(); }

  PEWeight<NormalFloat4> &operator[](unsigned int i) { return this->value[i]; }
  const PEWeight<NormalFloat4> &operator[](unsigned int i) const {
    return this->value[i];
  }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].data.index;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width, int W, int E>
class Pack1D<UFloat<W, E>, Width> {
 public:
  UFloat<W, E> value[Width];
  static const unsigned int width = UFloat<W, E>::width * Width;

  Pack1D() {}
  Pack1D(const int a) {}

  operator int() const { return Pack1D<UFloat<W, E>, Width>(); }

  UFloat<W, E> &operator[](size_t i) { return value[i]; }
  const UFloat<W, E> &operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < Width; i++) {
      m &value[i].d;
    }
  }

  inline friend bool operator==(const Pack1D &lhs, const Pack1D &rhs) {
    for (unsigned int i = 0; i < Width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t Width>
struct BufferWriteRequest {
  ac_int<16, false> address;
  ac_int<Width, false> data;

  static const unsigned int width = 16 + Width;

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

template <typename T, size_t Width>
inline bool operator==(const Pack1D<T, Width> &lhs,
                       const Pack1D<T, Width> &rhs) {
  bool is_equal = true;
#pragma hls_unroll yes
  for (unsigned int i = 0; i < Width; i++) {
    is_equal &= (lhs.value[i] == rhs.value[i]);
  }
  return is_equal;
}

template <typename T, size_t Width>
inline void sc_trace(sc_trace_file *tf, const Pack1D<T, Width> &vec,
                     const std::string &name) {
  sc_trace(tf, vec.value, name);
}

template <typename T, size_t Width>
inline std::ostream &operator<<(ostream &os, const Pack1D<T, Width> &vec) {
  for (int i = 0; i < Width; i++) {
    os << vec[i] << " ";
  }
  return os;
}
