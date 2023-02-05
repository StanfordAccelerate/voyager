#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

#define REPEAT(x) BOOST_PP_REPEAT(DIMENSION, x, 0)

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
  REPEAT(DECL_FIFOS)
#undef DECL_FIFOS
  int dummy;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<IDTYPE, SIZE> > CCS_INIT_S1(din);
  Connections::Out<ODTYPE> dout[SIZE];

#define FIFO_SIZE_INIT(z, i, unused) BOOST_PP_CAT(fifo, i)(i * 1 + 1),

  SC_CTOR(SerializedSkewer) : REPEAT(FIFO_SIZE_INIT) dummy(0) {
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

    REPEAT(DECL_THREADS)
#undef DECL_THREADS
  }

  void writeFifos() {
    din.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<IDTYPE, SIZE> input = din.Pop();

#define FIFO_WRITE(z, i, unused) BOOST_PP_CAT(fifo, i).write(input[i]);
      REPEAT(FIFO_WRITE)
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

  REPEAT(DECL_FUNCS)
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
  REPEAT(DECL_FIFOS)
#undef DECL_FIFOS
  int dummy;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<PEInput<IDTYPE>, SIZE> > CCS_INIT_S1(din);
  Connections::Out<PEInput<ODTYPE> > dout[SIZE];

#define FIFO_SIZE_INIT(z, i, unused) BOOST_PP_CAT(fifo, i)(i * 3 + 1),

  SC_CTOR(MultiInputSerializedSkewer) : REPEAT(FIFO_SIZE_INIT) dummy(0) {
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

    REPEAT(DECL_THREADS)
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
      REPEAT(FIFO_WRITE)
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

  REPEAT(DECL_FUNCS)
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
  REPEAT(DECL_FIFOS)
#undef DECL_FIFOS

  int dummy;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<IDTYPE> din[SIZE];
  Connections::Out<Pack1D<ODTYPE, SIZE> > CCS_INIT_S1(dout);

#define FIFO_SIZE_INIT(z, i, unused) \
  BOOST_PP_CAT(fifo, i)(1 * (DIMENSION - i + 1)),

  SC_CTOR(DeserializedSkewer) : REPEAT(FIFO_SIZE_INIT) dummy(0) {
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

    REPEAT(DECL_THREADS)
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
      REPEAT(FIFO_READ)
#undef FIFO_READ
      // CCS_LOG("psum output");
      // for (int i = 0; i < SIZE; i++) {
      //   std::cout << output[i].bits.to_string(AC_HEX) << " ";
      // }
      // std::cout << std::endl;
      // CCS_LOG("psumOut: " << output);
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

  REPEAT(DECL_FUNCS)
#undef DECL_FUNCS
};
