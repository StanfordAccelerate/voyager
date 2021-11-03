#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include <boost/preprocessor/repetition/repeat.hpp>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

#define REPEAT(x) BOOST_PP_REPEAT(DIMENSION, x, 0)

template <typename T, int NUM_REGS>
class Fifo {
 public:
  Fifo() {}

  void run(T &input, T &output) {
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

template <typename DTYPE, int SIZE, bool reverse = false>
SC_MODULE(ToggleSkewer) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  sc_in<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(din);
  sc_in<bool> CCS_INIT_S1(din_toggle);

  sc_out<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(dout);
  sc_out<bool> CCS_INIT_S1(dout_valid);

  Pack1D<DTYPE, SIZE> fifoInput;
  Pack1D<DTYPE, SIZE> fifoOutput;

  bool old_toggle;

#define INIT_FIFOS(z, i, unused) Fifo<DTYPE, i + 1> BOOST_PP_CAT(fifo_, i);
  REPEAT(INIT_FIFOS)
#undef INIT_FIFOS

  SC_CTOR(ToggleSkewer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void connect(Pack1D<DTYPE, SIZE> & vecIn, Pack1D<DTYPE, SIZE> & vecOut) {
    if (reverse) {
#pragma hls_unroll yes
      for (int i = 0; i < SIZE; i++) {
        vecOut[i] = vecIn[SIZE - 1 - i];
      }
    } else {
#pragma hls_unroll yes
      for (int i = 0; i < SIZE; i++) {
        vecOut[i] = vecIn[i];
      }
    }
  }

  void run() {
    dout.write(Pack1D<DTYPE, SIZE>());
    dout_valid.write(false);
    old_toggle = false;

    wait();

    // #pragma hls_pipeline_init_interval 1
    while (true) {
      if (din_toggle != old_toggle) {
        Pack1D<DTYPE, SIZE> input = din.read();

        connect(input, fifoInput);

#define INPUT_FIFO_BODY(z, i, unused) \
  BOOST_PP_CAT(fifo_, i).run(fifoInput[i], fifoOutput[i]);

        REPEAT(INPUT_FIFO_BODY)
#undef INPUT_FIFO_BODY

        Pack1D<DTYPE, SIZE> output;
        connect(fifoOutput, output);
        dout.write(output);
        dout_valid.write(true);

      } else {
        dout_valid.write(false);
      }

      old_toggle = din_toggle;
      wait();
    }
  }
};

template <typename DTYPE, int SIZE, bool reverse = false>
SC_MODULE(ValidSkewer) {
 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  sc_in<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(din);
  sc_in<bool> CCS_INIT_S1(din_valid);

  sc_out<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(dout);
  sc_out<bool> CCS_INIT_S1(dout_valid);

  Pack1D<DTYPE, SIZE> fifoInput;
  Pack1D<DTYPE, SIZE> fifoOutput;

#define INIT_FIFOS(z, i, unused) Fifo<DTYPE, i + 1> BOOST_PP_CAT(fifo_, i);
  REPEAT(INIT_FIFOS)
#undef INIT_FIFOS

  SC_CTOR(ValidSkewer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void connect(Pack1D<DTYPE, SIZE> & vecIn, Pack1D<DTYPE, SIZE> & vecOut) {
    if (reverse) {
#pragma hls_unroll yes
      for (int i = 0; i < SIZE; i++) {
        vecOut[i] = vecIn[SIZE - 1 - i];
      }
    } else {
#pragma hls_unroll yes
      for (int i = 0; i < SIZE; i++) {
        vecOut[i] = vecIn[i];
      }
    }
  }

  void run() {
    dout.write(Pack1D<DTYPE, SIZE>());
    dout_valid.write(false);

    wait();

    // #pragma hls_pipeline_init_interval 1
    while (true) {
      if (din_valid) {
        Pack1D<DTYPE, SIZE> input = din.read();

        connect(input, fifoInput);

#define INPUT_FIFO_BODY(z, i, unused) \
  BOOST_PP_CAT(fifo_, i).run(fifoInput[i], fifoOutput[i]);

        REPEAT(INPUT_FIFO_BODY)
#undef INPUT_FIFO_BODY

        Pack1D<DTYPE, SIZE> output;
        connect(fifoOutput, output);
        dout.write(output);
      }
      dout_valid.write(din_valid);
      wait();
    }
  }
};
