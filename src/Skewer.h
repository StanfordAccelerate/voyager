#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

#ifndef PE_LATENCY
#define PE_LATENCY 2
#endif

#define REPEAT_IC(x) BOOST_PP_REPEAT(IC_DIMENSION, x, 0)
#define REPEAT_OC(x) BOOST_PP_REPEAT(OC_DIMENSION, x, 0)

template <typename T, int NUM_REGS>
class Fifo {
 public:
  Fifo() {}

#pragma hls_design interface ccore
  void run(T input, T &output) {
  SHIFT:
    for (int i = NUM_REGS - 1; i >= 0; i--) {
      if (i == 0) {
        regs[i] = input;
      } else {
        regs[i] = regs[i - 1];
      }

      output = regs[NUM_REGS - 1];
    }
  }

 private:
  T regs[NUM_REGS];
};

/*
 * Takes an input of Pack1D<IDTYPE, SIZE> and skews it to produce
 * n=SIZE outputs of ODTYPE
 */
template <typename IDTYPE, typename ODTYPE, int SIZE>
SC_MODULE(SerializedSkewer) {
 private:
#define DECL_FIFOS(z, i, unused) sc_fifo<IDTYPE> BOOST_PP_CAT(fifo, i);
  REPEAT_OC(DECL_FIFOS)
#undef DECL_FIFOS
  int dummy;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<IDTYPE, SIZE> > CCS_INIT_S1(din);
  Connections::Out<ODTYPE> dout[SIZE];

#ifdef P8_1
  // Temporary fix for Posit8 utilization problem
  // Need to add the latency of input conversion after read from fifo for
  // latency matching Latency is assumed to be 1 here, but it is decided by HLS
#define FIFO_SIZE_INIT(z, i, unused) BOOST_PP_CAT(fifo, i)(i * 1 + 1 + 1),
#else
#define FIFO_SIZE_INIT(z, i, unused) BOOST_PP_CAT(fifo, i)(i * PE_LATENCY + 1),
#endif

  SC_CTOR(SerializedSkewer) : REPEAT_OC(FIFO_SIZE_INIT) dummy(0) {
#undef FIFO_SIZE_INIT
    SC_THREAD(writeFifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define SC_THREAD_EXP(x) BOOST_PP_CAT(x, i)
#define SC_THREAD_2(x) SC_THREAD(x)

#define DECL_THREADS(z, i, unused)                                          \
  declare_thread_process(BOOST_PP_CAT(BOOST_PP_CAT(readFifos, i), _handle), \
                         BOOST_PP_STRINGIZE(BOOST_PP_CAT(readFifos, i)),    \
                                            SC_CURRENT_USER_MODULE,         \
                                            BOOST_PP_CAT(readFifos, i));    \
  sensitive << clk.pos();                                                   \
  async_reset_signal_is(rstn, false);

    REPEAT_OC(DECL_THREADS)
#undef DECL_THREADS
  }

  void writeFifos() {
    din.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<IDTYPE, SIZE> input = din.Pop();

      // CCS_LOG("psum input");
      // for (int i = 0; i < SIZE; i++) {
      //   std::cout << input[i].bits.to_string(AC_HEX) << " ";
      // }
      // std::cout << std::endl;

#define FIFO_WRITE(z, i, unused) BOOST_PP_CAT(fifo, i).write(input[i]);
      REPEAT_OC(FIFO_WRITE)
#undef FIFO_WRITE
    }
  }

#define DECL_FUNCS(z, i, unused)                                \
  void BOOST_PP_CAT(readFifos, i)() {                           \
    dout[i].Reset();                                            \
    wait();                                                     \
    _Pragma("hls_pipeline_init_interval 1")                     \
        _Pragma("hls_pipeline_stall_mode flush") while (true) { \
      dout[i].Push(BOOST_PP_CAT(fifo, i).read());               \
    }                                                           \
  }

  REPEAT_OC(DECL_FUNCS)
#undef DECL_FUNCS
};

/*
 * Takes an input of Pack1D<IDTYPE, SIZE> and skews it to produce
 * n=SIZE outputs of ODTYPE
 */
template <typename IDTYPE, typename ODTYPE, int SIZE>
SC_MODULE(MultiInputSerializedSkewer) {
 private:
#define DECL_FIFOS(z, i, unused) \
  sc_fifo<PEInput<IDTYPE> > BOOST_PP_CAT(fifo, i);
  REPEAT_IC(DECL_FIFOS)
#undef DECL_FIFOS
  int dummy;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<PEInput<IDTYPE>, SIZE> > CCS_INIT_S1(din);
  Connections::Out<PEInput<ODTYPE> > dout[SIZE];

#define FIFO_SIZE_INIT(z, i, unused) BOOST_PP_CAT(fifo, i)(i * PE_LATENCY + 1),

  SC_CTOR(MultiInputSerializedSkewer) : REPEAT_IC(FIFO_SIZE_INIT) dummy(0) {
#undef FIFO_SIZE_INIT
    SC_THREAD(writeFifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define SC_THREAD_EXP(x) BOOST_PP_CAT(x, i)
#define SC_THREAD_2(x) SC_THREAD(x)

#define DECL_THREADS(z, i, unused)                                          \
  declare_thread_process(BOOST_PP_CAT(BOOST_PP_CAT(readFifos, i), _handle), \
                         BOOST_PP_STRINGIZE(BOOST_PP_CAT(readFifos, i)),    \
                                            SC_CURRENT_USER_MODULE,         \
                                            BOOST_PP_CAT(readFifos, i));    \
  sensitive << clk.pos();                                                   \
  async_reset_signal_is(rstn, false);

    REPEAT_IC(DECL_THREADS)
#undef DECL_THREADS
  }

  void writeFifos() {
    din.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<PEInput<IDTYPE>, SIZE> input = din.Pop();

#define FIFO_WRITE(z, i, unused) BOOST_PP_CAT(fifo, i).write(input[i]);
      REPEAT_IC(FIFO_WRITE)
#undef FIFO_WRITE
    }
  }

#define DECL_FUNCS(z, i, unused)                                \
  void BOOST_PP_CAT(readFifos, i)() {                           \
    dout[i].Reset();                                            \
    wait();                                                     \
    _Pragma("hls_pipeline_init_interval 1")                     \
        _Pragma("hls_pipeline_stall_mode flush") while (true) { \
      PEInput<IDTYPE> inVal = BOOST_PP_CAT(fifo, i).read();     \
      PEInput<ODTYPE> outVal;                                   \
      outVal.data = inVal.data;                                 \
      outVal.swapWeights = inVal.swapWeights;                   \
      dout[i].Push(outVal);                                     \
    }                                                           \
  }

  REPEAT_IC(DECL_FUNCS)
#undef DECL_FUNCS
};

/*
 * Takes an input of Pack1D<IDTYPE, SIZE> and skews it to produce
 * n=SIZE outputs of ODTYPE
 */
template <typename IDTYPE, typename ODTYPE, int SIZE>
SC_MODULE(WeightSerializedSkewer) {
 private:
#define DECL_FIFOS(z, i, unused) \
  sc_fifo<PEWeight<IDTYPE> > BOOST_PP_CAT(fifo, i);
  REPEAT_OC(DECL_FIFOS)
#undef DECL_FIFOS
  int dummy;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<PEWeight<IDTYPE>, SIZE> > CCS_INIT_S1(din);
  Connections::Out<PEWeight<ODTYPE> > dout[SIZE];

#define FIFO_SIZE_INIT(z, i, unused) BOOST_PP_CAT(fifo, i)(i * 1 + 1),

  SC_CTOR(WeightSerializedSkewer) : REPEAT_OC(FIFO_SIZE_INIT) dummy(0) {
#undef FIFO_SIZE_INIT
    SC_THREAD(writeFifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define SC_THREAD_EXP(x) BOOST_PP_CAT(x, i)
#define SC_THREAD_2(x) SC_THREAD(x)

#define DECL_THREADS(z, i, unused)                                          \
  declare_thread_process(BOOST_PP_CAT(BOOST_PP_CAT(readFifos, i), _handle), \
                         BOOST_PP_STRINGIZE(BOOST_PP_CAT(readFifos, i)),    \
                                            SC_CURRENT_USER_MODULE,         \
                                            BOOST_PP_CAT(readFifos, i));    \
  sensitive << clk.pos();                                                   \
  async_reset_signal_is(rstn, false);

    REPEAT_OC(DECL_THREADS)
#undef DECL_THREADS
  }

  void writeFifos() {
    din.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<PEWeight<IDTYPE>, SIZE> input = din.Pop();

#define FIFO_WRITE(z, i, unused) BOOST_PP_CAT(fifo, i).write(input[i]);
      REPEAT_OC(FIFO_WRITE)
#undef FIFO_WRITE
    }
  }

#define DECL_FUNCS(z, i, unused)                                \
  void BOOST_PP_CAT(readFifos, i)() {                           \
    dout[i].Reset();                                            \
    wait();                                                     \
    _Pragma("hls_pipeline_init_interval 1")                     \
        _Pragma("hls_pipeline_stall_mode flush") while (true) { \
      PEWeight<IDTYPE> inVal = BOOST_PP_CAT(fifo, i).read();    \
      PEWeight<ODTYPE> outVal;                                  \
      outVal.data = inVal.data;                                 \
      outVal.tag = inVal.tag;                                   \
      dout[i].Push(outVal);                                     \
    }                                                           \
  }

  REPEAT_OC(DECL_FUNCS)
#undef DECL_FUNCS
};

/*
 * Takes an input of n=SIZE outputs of DTYPE and skews it to produce
 * an output Pack1D<DTYPE, SIZE>
 */
template <typename IDTYPE, typename ODTYPE, int SIZE>
SC_MODULE(DeserializedSkewer) {
 private:
#define DECL_FIFOS(z, i, unused) sc_fifo<ODTYPE> BOOST_PP_CAT(fifo, i);
  REPEAT_OC(DECL_FIFOS)
#undef DECL_FIFOS

  int dummy;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<IDTYPE> din[SIZE];
  Connections::Out<Pack1D<ODTYPE, SIZE> > CCS_INIT_S1(dout);

#define FIFO_SIZE_INIT(z, i, unused) \
  BOOST_PP_CAT(fifo, i)(1 * (OC_DIMENSION - i + 1)),

  SC_CTOR(DeserializedSkewer) : REPEAT_OC(FIFO_SIZE_INIT) dummy(0) {
#undef FIFO_SIZE_INIT
    SC_THREAD(readFifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define DECL_THREADS(z, i, unused)                                           \
  declare_thread_process(BOOST_PP_CAT(BOOST_PP_CAT(writeFifos, i), _handle), \
                         BOOST_PP_STRINGIZE(BOOST_PP_CAT(writeFifos, i)),    \
                                            SC_CURRENT_USER_MODULE,          \
                                            BOOST_PP_CAT(writeFifos, i));    \
  sensitive << clk.pos();                                                    \
  async_reset_signal_is(rstn, false);

    REPEAT_OC(DECL_THREADS)
#undef DECL_THREADS
  }

  void readFifos() {
    dout.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<ODTYPE, SIZE> output;

#define FIFO_READ(z, i, unused) output[i] = BOOST_PP_CAT(fifo, i).read();
      REPEAT_OC(FIFO_READ)
#undef FIFO_READ
      // for (int i = 0; i < SIZE; i++) {
      //   std::cout << output[i].bits.to_string(AC_HEX) << " ";
      // }
      // std::cout << std::endl;
      dout.Push(output);
    }
  }

#define DECL_FUNCS(z, i, unused)                                \
  void BOOST_PP_CAT(writeFifos, i)() {                          \
    din[i].Reset();                                             \
    wait();                                                     \
    _Pragma("hls_pipeline_init_interval 1")                     \
        _Pragma("hls_pipeline_stall_mode flush") while (true) { \
      BOOST_PP_CAT(fifo, i).write(din[i].Pop());                \
    }                                                           \
  }

  REPEAT_OC(DECL_FUNCS)
#undef DECL_FUNCS
};
