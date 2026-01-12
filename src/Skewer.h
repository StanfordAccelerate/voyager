#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "AccelTypes.h"

#ifndef PE_LATENCY
#define PE_LATENCY 2
#endif

#define REPEAT_IC(x) BOOST_PP_REPEAT(IC_DIMENSION, x, 0)
#define REPEAT_OC(x) BOOST_PP_REPEAT(OC_DIMENSION, x, 0)

// Helper: Takes a prefix (e.g., fifo) and an index (e.g., 0)
// Returns: fifo0{"fifo0"}
#define GEN_NAMED_VAR(prefix, i) \
  BOOST_PP_CAT(prefix, i) { BOOST_PP_STRINGIZE(BOOST_PP_CAT(prefix, i)) }

// A version of SC_THREAD that supports macro expansion for the function name
#define SC_THREAD_BOOST(func)                                     \
  declare_thread_process(                                         \
      BOOST_PP_CAT(func, _handle), /* Replaces func ## _handle */ \
      BOOST_PP_STRINGIZE(func),    /* Replaces #func */           \
                         SC_CURRENT_USER_MODULE, func)

template <typename T, int size>
SC_MODULE(InputSerializedSkewer) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<PEInput<T>, size>> CCS_INIT_S1(din);
  Connections::Out<PEInput<T>> dout[size];

#define DECL_FIFO(z, i, data)                                               \
  Connections::Fifo<PEInput<T>, i * PE_LATENCY + 2> GEN_NAMED_VAR(fifo, i); \
  Connections::Combinational<PEInput<T>> GEN_NAMED_VAR(fifo_din, i);        \
  Connections::Combinational<PEInput<T>> GEN_NAMED_VAR(fifo_dout, i);
  REPEAT_IC(DECL_FIFO)
#undef DECL_FIFO

  SC_CTOR(InputSerializedSkewer) {
    SC_THREAD(write_fifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define DECL_FIFO_CTOR(z, i, data)                      \
  BOOST_PP_CAT(fifo, i).clk(clk);                       \
  BOOST_PP_CAT(fifo, i).rst(rstn);                      \
  BOOST_PP_CAT(fifo, i).enq(BOOST_PP_CAT(fifo_din, i)); \
  BOOST_PP_CAT(fifo, i).deq(BOOST_PP_CAT(fifo_dout, i));
    REPEAT_IC(DECL_FIFO_CTOR)
#undef DECL_FIFO_CTOR

#define DECL_THREADS(z, i, data)               \
  SC_THREAD_BOOST(BOOST_PP_CAT(read_fifo, i)); \
  sensitive << clk.pos();                      \
  async_reset_signal_is(rstn, false);

    REPEAT_IC(DECL_THREADS)
#undef DECL_THREADS
  }

  void write_fifos() {
    din.Reset();
#define RESET_WRITE(z, i, data) BOOST_PP_CAT(fifo_din, i).ResetWrite();
    REPEAT_IC(RESET_WRITE)
#undef RESET_WRITE

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      auto input = din.Pop();
#define FIFO_WRITE(z, i, data) BOOST_PP_CAT(fifo_din, i).Push(input[i]);
      REPEAT_IC(FIFO_WRITE)
#undef FIFO_WRITE
    }
  }

#define DECL_FUNCS(z, i, data)                                  \
  void BOOST_PP_CAT(read_fifo, i)() {                           \
    dout[i].Reset();                                            \
    BOOST_PP_CAT(fifo_dout, i).ResetRead();                     \
    wait();                                                     \
    _Pragma("hls_pipeline_init_interval 1")                     \
        _Pragma("hls_pipeline_stall_mode flush") while (true) { \
      dout[i].Push(BOOST_PP_CAT(fifo_dout, i).Pop());           \
    }                                                           \
  }

  REPEAT_IC(DECL_FUNCS)
#undef DECL_FUNCS
};

template <typename T, int size>
SC_MODULE(WeightSerializedSkewer) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<PEWeight<T>, size>> CCS_INIT_S1(din);
  Connections::Out<PEWeight<T>> dout[size];

#define DECL_FIFO(z, i, data)                                         \
  Connections::Fifo<PEWeight<T>, i + 1> GEN_NAMED_VAR(fifo, i);       \
  Connections::Combinational<PEWeight<T>> GEN_NAMED_VAR(fifo_din, i); \
  Connections::Combinational<PEWeight<T>> GEN_NAMED_VAR(fifo_dout, i);
  REPEAT_OC(DECL_FIFO)
#undef DECL_FIFO

  SC_CTOR(WeightSerializedSkewer) {
    SC_THREAD(write_fifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define DECL_FIFO_CTOR(z, i, data)                      \
  BOOST_PP_CAT(fifo, i).clk(clk);                       \
  BOOST_PP_CAT(fifo, i).rst(rstn);                      \
  BOOST_PP_CAT(fifo, i).enq(BOOST_PP_CAT(fifo_din, i)); \
  BOOST_PP_CAT(fifo, i).deq(BOOST_PP_CAT(fifo_dout, i));
    REPEAT_OC(DECL_FIFO_CTOR)
#undef DECL_FIFO_CTOR

#define DECL_THREADS(z, i, data)               \
  SC_THREAD_BOOST(BOOST_PP_CAT(read_fifo, i)); \
  sensitive << clk.pos();                      \
  async_reset_signal_is(rstn, false);

    REPEAT_OC(DECL_THREADS)
#undef DECL_THREADS
  }

  void write_fifos() {
    din.Reset();
#define RESET_WRITE(z, i, data) BOOST_PP_CAT(fifo_din, i).ResetWrite();
    REPEAT_OC(RESET_WRITE)
#undef RESET_WRITE

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      auto input = din.Pop();
#define FIFO_WRITE(z, i, data) BOOST_PP_CAT(fifo_din, i).Push(input[i]);
      REPEAT_OC(FIFO_WRITE)
#undef FIFO_WRITE
    }
  }

#define DECL_FUNCS(z, i, data)                                  \
  void BOOST_PP_CAT(read_fifo, i)() {                           \
    dout[i].Reset();                                            \
    BOOST_PP_CAT(fifo_dout, i).ResetRead();                     \
    wait();                                                     \
    _Pragma("hls_pipeline_init_interval 1")                     \
        _Pragma("hls_pipeline_stall_mode flush") while (true) { \
      dout[i].Push(BOOST_PP_CAT(fifo_dout, i).Pop());           \
    }                                                           \
  }

  REPEAT_OC(DECL_FUNCS)
#undef DECL_FUNCS
};

template <typename T, int size>
SC_MODULE(DeserializedSkewer) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<T> din[size];
  Connections::Out<Pack1D<T, size>> CCS_INIT_S1(dout);

#define DECL_FIFO(z, i, data)                                        \
  Connections::Fifo<T, OC_DIMENSION - i + 2> GEN_NAMED_VAR(fifo, i); \
  Connections::Combinational<T> GEN_NAMED_VAR(fifo_din, i);          \
  Connections::Combinational<T> GEN_NAMED_VAR(fifo_dout, i);
  REPEAT_OC(DECL_FIFO)
#undef DECL_FIFO

  SC_CTOR(DeserializedSkewer) {
    SC_THREAD(read_fifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define DECL_FIFO_CTOR(z, i, data)                      \
  BOOST_PP_CAT(fifo, i).clk(clk);                       \
  BOOST_PP_CAT(fifo, i).rst(rstn);                      \
  BOOST_PP_CAT(fifo, i).enq(BOOST_PP_CAT(fifo_din, i)); \
  BOOST_PP_CAT(fifo, i).deq(BOOST_PP_CAT(fifo_dout, i));
    REPEAT_OC(DECL_FIFO_CTOR)
#undef DECL_FIFO_CTOR

#define DECL_THREADS(z, i, data)                \
  SC_THREAD_BOOST(BOOST_PP_CAT(write_fifo, i)); \
  sensitive << clk.pos();                       \
  async_reset_signal_is(rstn, false);

    REPEAT_OC(DECL_THREADS)
#undef DECL_THREADS
  }

  void read_fifos() {
    dout.Reset();
#define RESET_READ(z, i, data) BOOST_PP_CAT(fifo_dout, i).ResetRead();
    REPEAT_OC(RESET_READ)
#undef RESET_READ

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<T, size> output;
#define FIFO_READ(z, i, data) output[i] = BOOST_PP_CAT(fifo_dout, i).Pop();
      REPEAT_OC(FIFO_READ)
#undef FIFO_READ
      dout.Push(output);
    }
  }

#define DECL_FUNCS(z, i, data)                                  \
  void BOOST_PP_CAT(write_fifo, i)() {                          \
    din[i].Reset();                                             \
    BOOST_PP_CAT(fifo_din, i).ResetWrite();                     \
    wait();                                                     \
    _Pragma("hls_pipeline_init_interval 1")                     \
        _Pragma("hls_pipeline_stall_mode flush") while (true) { \
      BOOST_PP_CAT(fifo_din, i).Push(din[i].Pop());             \
    }                                                           \
  }

  REPEAT_OC(DECL_FUNCS)
#undef DECL_FUNCS
};
