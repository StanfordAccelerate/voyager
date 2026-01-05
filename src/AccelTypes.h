#pragma once

// clang-format off
#include <systemc.h>
// clang-format on
#include <ac_int.h>
#include <ac_sc.h>
#include <ccs_types.h>
#include <mc_connections.h>

#include "Params.h"
#include "datatypes/DataTypes.h"

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
  void Marshall(Marshaller<Size>& m) {
    m & address;
    m & burst_size;
  }

  inline friend void sc_trace(sc_trace_file* tf, const MemoryRequest& request,
                              const std::string& name) {
    sc_trace(tf, request.address, name + ".address");
    sc_trace(tf, request.burst_size, name + ".burst_size");
  }

  inline friend std::ostream& operator<<(ostream& os,
                                         const MemoryRequest& request) {
    os << request.address << " ";
    os << request.burst_size << " ";
    return os;
  }

  inline friend bool operator==(const MemoryRequest& lhs,
                                const MemoryRequest& rhs) {
    return lhs.address == rhs.address && lhs.burst_size == rhs.burst_size;
  }
};

template <typename T>
struct PEInput {
  T data;
  ac_int<1, false> swap_weights;

  static const unsigned int width = T::width + 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & data;
    m & swap_weights;
  }

  inline friend void sc_trace(sc_trace_file* tf, const PEInput& pe_input,
                              const std::string& name) {
    sc_trace(tf, pe_input.data, name + ".data");
    sc_trace(tf, pe_input.swap_weights, name + ".swap_weights");
  }

  inline friend std::ostream& operator<<(ostream& os, const PEInput& pe_input) {
    os << pe_input.data << " ";
    os << pe_input.swap_weights << " ";

    return os;
  }

  inline friend bool operator==(const PEInput& lhs, const PEInput& rhs) {
    return lhs.data == rhs.data && lhs.swap_weights == rhs.swap_weights;
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
  void Marshall(Marshaller<Size>& m) {
    m & data;
    m & tag;
  }

  inline friend void sc_trace(sc_trace_file* tf, const PEWeight& peWeight,
                              const std::string& name) {
    sc_trace(tf, peWeight.data, name + ".data");
    sc_trace(tf, peWeight.tag, name + ".tag");
  }

  inline friend std::ostream& operator<<(ostream& os,
                                         const PEWeight& peWeight) {
    os << peWeight.data << " ";
    os << peWeight.tag << " ";

    return os;
  }

  inline friend bool operator==(const PEWeight& lhs, const PEWeight& rhs) {
    return lhs.data == rhs.data && lhs.tag == rhs.tag;
  }
};

template <typename T, size_t pack_width>
class Pack1D {
 public:
  T value[pack_width];
  static const unsigned int width = T::width * pack_width;

  static Pack1D create(const T (&arr)[pack_width]) {
    Pack1D p;
#pragma hls_unroll yes
    for (unsigned i = 0; i < pack_width; ++i) {
      p.value[i] = arr[i];
    }
    return p;
  }

  T& operator[](unsigned int i) { return this->value[i]; }
  const T& operator[](unsigned int i) const { return this->value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i];
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

// TODO: is there a way to make this more generic?

template <size_t pack_width, int W, int es>
class Pack1D<PEInput<Posit<W, es>>, pack_width> {
 public:
  PEInput<Posit<W, es>> value[pack_width];
  static const unsigned int width = PEInput<Posit<W, es>>::width * pack_width;

  PEInput<Posit<W, es>>& operator[](unsigned int i) { return this->value[i]; }
  const PEInput<Posit<W, es>>& operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].data.bits;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].swap_weights;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int W, int es>
class Pack1D<PEWeight<Posit<W, es>>, pack_width> {
 public:
  PEWeight<Posit<W, es>> value[pack_width];
  static const unsigned int width = PEWeight<Posit<W, es>>::width * pack_width;

  PEWeight<Posit<W, es>>& operator[](unsigned int i) { return this->value[i]; }
  const PEWeight<Posit<W, es>>& operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].data.bits;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int mantissa, int exp>
class Pack1D<StdFloat<mantissa, exp>, pack_width> {
 public:
  StdFloat<mantissa, exp> value[pack_width];
  static const unsigned int width = StdFloat<mantissa, exp>::width * pack_width;

  static Pack1D zero() {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = StdFloat<mantissa, exp>::zero();
    }
    return p;
  }

  static Pack1D one() {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = StdFloat<mantissa, exp>::one();
    }
    return p;
  }

  static Pack1D fill(StdFloat<mantissa, exp> value) {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = value;
    }
    return p;
  }

  StdFloat<mantissa, exp>& operator[](size_t i) { return value[i]; }
  const StdFloat<mantissa, exp>& operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].float_val.d;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int mantissa, int exp>
class Pack1D<PEInput<StdFloat<mantissa, exp>>, pack_width> {
 public:
  PEInput<StdFloat<mantissa, exp>> value[pack_width];
  static const unsigned int width =
      PEInput<StdFloat<mantissa, exp>>::width * pack_width;

  PEInput<StdFloat<mantissa, exp>>& operator[](unsigned int i) {
    return this->value[i];
  }
  const PEInput<StdFloat<mantissa, exp>>& operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].data.float_val.d;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].swap_weights;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int mantissa, int exp>
class Pack1D<PEWeight<StdFloat<mantissa, exp>>, pack_width> {
 public:
  PEWeight<StdFloat<mantissa, exp>> value[pack_width];
  static const unsigned int width =
      PEWeight<StdFloat<mantissa, exp>>::width * pack_width;

  PEWeight<StdFloat<mantissa, exp>>& operator[](unsigned int i) {
    return this->value[i];
  }
  const PEWeight<StdFloat<mantissa, exp>>& operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].data.float_val.d;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int W, bool S>
class Pack1D<Int<W, S>, pack_width> {
 public:
  Int<W, S> value[pack_width];
  static const unsigned int width = Int<W, S>::width * pack_width;

  static Pack1D zero() {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = Int<W, S>::zero();
    }
    return p;
  }

  static Pack1D one() {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = Int<W, S>::one();
    }
    return p;
  }

  static Pack1D fill(Int<W, S> value) {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = value;
    }
    return p;
  }

  Int<W, S>& operator[](size_t i) { return value[i]; }
  const Int<W, S>& operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].int_val;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int W, bool S>
class Pack1D<PEInput<Int<W, S>>, pack_width> {
 public:
  PEInput<Int<W, S>> value[pack_width];
  static const unsigned int width = PEInput<Int<W, S>>::width * pack_width;

  PEInput<Int<W, S>>& operator[](unsigned int i) { return this->value[i]; }
  const PEInput<Int<W, S>>& operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].data.int_val;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].swap_weights;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int W, bool S>
class Pack1D<PEWeight<Int<W, S>>, pack_width> {
 public:
  PEWeight<Int<W, S>> value[pack_width];
  static const unsigned int width = PEWeight<Int<W, S>>::width * pack_width;

  PEWeight<Int<W, S>>& operator[](unsigned int i) { return this->value[i]; }
  const PEWeight<Int<W, S>>& operator[](unsigned int i) const {
    return this->value[i];
  }
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].data.int_val;
    }
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].tag;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <size_t pack_width, int W, int E>
class Pack1D<UFloat<W, E>, pack_width> {
 public:
  UFloat<W, E> value[pack_width];
  static const unsigned int width = UFloat<W, E>::width * pack_width;

  static Pack1D zero() {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = UFloat<W, E>::zero();
    }
    return p;
  }

  static Pack1D one() {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = UFloat<W, E>::one();
    }
    return p;
  }

  static Pack1D fill(UFloat<W, E> value) {
    Pack1D p;
#pragma hls_unroll yes
    for (int i = 0; i < pack_width; ++i) {
      p.value[i] = value;
    }
    return p;
  }

  UFloat<W, E>& operator[](size_t i) { return value[i]; }
  const UFloat<W, E>& operator[](size_t i) const { return value[i]; }

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
#pragma hls_unroll yes
    for (unsigned int i = 0; i < pack_width; i++) {
      m& value[i].d;
    }
  }

  inline friend bool operator==(const Pack1D& lhs, const Pack1D& rhs) {
    for (unsigned int i = 0; i < pack_width; i++) {
      if (!(lhs.value[i] == rhs.value[i])) {
        return false;
      }
    }

    return true;
  }
};

template <typename T, typename Meta, int pack_width>
struct CsrDataAndIndices {
  Pack1D<T, pack_width> data;
  Pack1D<Meta, pack_width> indices;
  bool is_last;

  static const unsigned int width =
      Pack1D<T, pack_width>::width + Pack1D<Meta, pack_width>::width + 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    for (int i = 0; i < pack_width; i++) {
      m& data[i];
    }
    for (int i = 0; i < pack_width; i++) {
      m& indices[i];
    }
    m & is_last;
  }

  inline friend void sc_trace(sc_trace_file* tf, const CsrDataAndIndices& csr,
                              const std::string& name) {
    sc_trace(tf, csr.data, name + ".data");
    sc_trace(tf, csr.indices, name + ".indices");
    sc_trace(tf, csr.is_last, name + ".is_last");
  }

  inline friend std::ostream& operator<<(ostream& os,
                                         const CsrDataAndIndices& csr) {
    return os << "data: " << csr.data << " indices: " << csr.indices
              << " is_last: " << csr.is_last;
  }

  inline friend bool operator==(const CsrDataAndIndices& lhs,
                                const CsrDataAndIndices& rhs) {
    return lhs.data == rhs.data && lhs.indices == rhs.indices &&
           lhs.is_last == rhs.is_last;
  }
};

template <typename T>
struct CsrWriteRequest {
  ac_int<ADDRESS_WIDTH, false> address;
  T data;

  static const unsigned int width = ADDRESS_WIDTH + T::width;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & address;
    m & data;
  }

  inline friend void sc_trace(sc_trace_file* tf, const CsrWriteRequest& request,
                              const std::string& name) {
    sc_trace(tf, request.address, name + ".address");
    sc_trace(tf, request.data, name + ".data");
  }

  inline friend std::ostream& operator<<(ostream& os,
                                         const CsrWriteRequest& request) {
    return os << request.address << " " << request.data << " ";
  }

  inline friend bool operator==(const CsrWriteRequest& lhs,
                                const CsrWriteRequest& rhs) {
    return lhs.address == rhs.address && lhs.data == rhs.data;
  }
};

template <typename T>
struct BufferWriteRequest {
  ac_int<16, false> address;
  T data;
  bool last;

  static const unsigned int width = 16 + T::width + 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & address;
    m & data;
    m & last;
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const BufferWriteRequest& request,
                              const std::string& name) {
    sc_trace(tf, request.address, name + ".address");
    sc_trace(tf, request.data, name + ".data");
    sc_trace(tf, request.last, name + ".last");
  }

  inline friend std::ostream& operator<<(ostream& os,
                                         const BufferWriteRequest& request) {
    os << request.address << " ";
    os << request.data << " ";
    os << request.last << " ";

    return os;
  }

  inline friend bool operator==(const BufferWriteRequest& lhs,
                                const BufferWriteRequest& rhs) {
    return lhs.address == rhs.address && lhs.data == rhs.data &&
           lhs.last == rhs.last;
  }
};

struct BufferReadRequest {
  ac_int<16, false> address;
  bool last;

  static const unsigned int width = 16 + 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & address;
    m & last;
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const BufferReadRequest& request,
                              const std::string& name) {
    sc_trace(tf, request.address, name + ".address");
    sc_trace(tf, request.last, name + ".last");
  }

  inline friend std::ostream& operator<<(ostream& os,
                                         const BufferReadRequest& request) {
    os << request.address << " ";
    os << request.last << " ";

    return os;
  }

  inline friend bool operator==(const BufferReadRequest& lhs,
                                const BufferReadRequest& rhs) {
    return lhs.address == rhs.address && lhs.last == rhs.last;
  }
};

template <typename T>
struct BufferReadResponse {
  T data;
  bool last;

  static const unsigned int width = T::width + 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & data;
    m & last;
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const BufferReadResponse& response,
                              const std::string& name) {
    sc_trace(tf, response.data, name + ".data");
    sc_trace(tf, response.last, name + ".last");
  }

  inline friend std::ostream& operator<<(ostream& os,
                                         const BufferReadResponse& response) {
    os << response.data << " ";
    os << response.last << " ";

    return os;
  }

  inline friend bool operator==(const BufferReadResponse& lhs,
                                const BufferReadResponse& rhs) {
    return lhs.data == rhs.data && lhs.last == rhs.last;
  }
};

template <typename T, size_t pack_width>
inline bool operator==(const Pack1D<T, pack_width>& lhs,
                       const Pack1D<T, pack_width>& rhs) {
  bool is_equal = true;
#pragma hls_unroll yes
  for (unsigned int i = 0; i < pack_width; i++) {
    is_equal &= (lhs.value[i] == rhs.value[i]);
  }
  return is_equal;
}

template <typename T, size_t pack_width>
inline void sc_trace(sc_trace_file* tf, const Pack1D<T, pack_width>& vec,
                     const std::string& name) {
  sc_trace(tf, vec.value, name);
}

template <typename T, size_t pack_width>
inline std::ostream& operator<<(ostream& os, const Pack1D<T, pack_width>& vec) {
  for (int i = 0; i < pack_width; i++) {
    os << vec[i] << " ";
  }
  return os;
}
