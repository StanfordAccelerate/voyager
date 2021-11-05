#pragma once

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

#include <boost/preprocessor/repetition/repeat.hpp>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

#define REPEAT(x) BOOST_PP_REPEAT(DIMENSION, x, 0)

template <typename DTYPE, typename T>
class Skewer {
 public:
  Skewer() {}

#pragma hls_design interface ccore
  void run(DTYPE input, DTYPE &output) {
    // model 2 cycle latency caused by synchronization of CCORE
    // #ifndef __SYNTHESIS__
    //     delay[1] = delay[0];
    //     delay[0] = input;
    //     input = delay[1];
    // #endif

#define INPUT_FIFO_BODY(z, i, unused)                        \
  T BOOST_PP_CAT(fifo_output_, i);                           \
  T BOOST_PP_CAT(fifo_input_, i);                            \
  BOOST_PP_CAT(fifo_input_, i) = input[i];                   \
  BOOST_PP_CAT(fifo_, i).run(BOOST_PP_CAT(fifo_input_, i),   \
                             BOOST_PP_CAT(fifo_output_, i)); \
  output[i] = BOOST_PP_CAT(fifo_output_, i);

    REPEAT(INPUT_FIFO_BODY)
  }

 private:
#define INPUT_FIFOS_INIT(z, i, unused) Fifo<T, i + 1> BOOST_PP_CAT(fifo_, i);

  REPEAT(INPUT_FIFOS_INIT)
#ifndef __SYNTHESIS__
  DTYPE delay[2];
#endif
};

// #pragma once

// #include <mc_connections.h>
// #include <systemc.h>

// #include <boost/preprocessor/repetition/repeat.hpp>

// #include "AccelTypes.h"
// #include "ArchitectureParams.h"

// #define REPEAT(x) BOOST_PP_REPEAT(DIMENSION, x, 0)

// template <typename DTYPE>
// SC_MODULE(Fifo) {
//   sc_in<bool> CCS_INIT_S1(clk);
//   sc_in<bool> CCS_INIT_S1(rstn);

//   sc_in<DTYPE> CCS_INIT_S1(din);
//   sc_in<bool> CCS_INIT_S1(din_valid);
//   sc_out<DTYPE> CCS_INIT_S1(dout);

//   SC_HAS_PROCESS(Fifo);

//   Fifo(sc_module_name name, int size) : sc_module(name), size(size) {
//     SC_THREAD(run);
//     sensitive << clk.pos();
//     async_reset_signal_is(rstn, false);
//   }

//   void run() {
//     dout.write(0);

//     w_idx = 0;
//     r_idx = 1;

//     wait();

//     while (true) {
//       // if (size == 1 || size == 2) {
//       //   if (din_valid) dout.write(din);
//       // }
//       // else if (size == 2) {
//       //   if (din_valid) {
//       //     dout = regs[0];
//       //     regs[0] = din;
//       //   }
//       // }
//       // else {
//       if (din_valid) {
//         for (int i = size - 1; i >= 0; i--) {
//           if (i == 0) {
//             regs[i] = din;
//           } else {
//             regs[i] = regs[i - 1];
//           }
//         }
//         dout = regs[size - 1];
//         // dout = regs[r_idx];
//         // regs[w_idx] = din;

//         // r_idx++;
//         // if (r_idx == size) {
//         //   r_idx = 0;
//         // }
//         // w_idx++;
//         // if (w_idx == size) {
//         //   w_idx = 0;
//         // }
//       }
//       // }
//       wait();
//     }
//   }

//   // void run_1() {
//   //   dout.write(0);

//   //   wait();

//   //   while (true) {
//   //     dout.write(din);
//   //     wait();
//   //   }
//   // }

//  private:
//   DTYPE regs[32];
//   int size;
//   int w_idx;
//   int r_idx;
// };

// // template <typename T, int NUM_REGS>
// // class Fifo {
// //  public:
// //   Fifo() {}

// //   void run(T &input, T &output) {
// //     for (int i = NUM_REGS - 1; i >= 0; i--) {
// //       if (i == 0) {
// //         regs[i] = input;
// //       } else {
// //         regs[i] = regs[i - 1];
// //       }

// //       output = regs[NUM_REGS - 1];
// //     }
// //   }

// //  private:
// //   T regs[NUM_REGS];
// // };

// template <typename DTYPE, int SIZE, bool reverse = false>
// SC_MODULE(ToggleSkewer) {
//  public:
//   sc_in<bool> CCS_INIT_S1(clk);
//   sc_in<bool> CCS_INIT_S1(rstn);

//   sc_in<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(din);
//   sc_in<bool> CCS_INIT_S1(din_toggle);

//   sc_out<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(dout);
//   sc_out<bool> CCS_INIT_S1(dout_valid);

//   sc_signal<DTYPE> fifoInput[SIZE];
//   sc_signal<DTYPE> fifoOutput[SIZE];
//   sc_signal<bool> enable;

//   bool old_toggle;

//   // #define INIT_FIFOS(z, i, unused) Fifo<DTYPE, i + 1>
//   BOOST_PP_CAT(fifo_, i);
//   //   REPEAT(INIT_FIFOS)
//   // #undef INIT_FIFOS

//   SC_CTOR(ToggleSkewer) {
//     Fifo<DTYPE> *fifo[SIZE];
//     for (int i = 0 + 1; i < SIZE + 1; i++) {
//       fifo[i - 1] = new Fifo<DTYPE>(sc_gen_unique_name("fifo"), i);
//       fifo[i - 1]->clk(clk);
//       fifo[i - 1]->rstn(rstn);
//       fifo[i - 1]->din(fifoInput[i - 1]);
//       fifo[i - 1]->din_valid(enable);
//       fifo[i - 1]->dout(fifoOutput[i - 1]);
//     }

//     SC_THREAD(run);
//     sensitive << clk.pos();
//     async_reset_signal_is(rstn, false);
//   }

//   //   void connect(sc_signal<DTYPE> vecIn[SIZE], sc_signal<DTYPE>
//   vecOut[SIZE])
//   //   {
//   //     if (reverse) {
//   // #pragma hls_unroll yes
//   //       for (int i = 0; i < SIZE; i++) {
//   //         vecOut[i] = vecIn[SIZE - 1 - i];
//   //       }
//   //     } else {
//   // #pragma hls_unroll yes
//   //       for (int i = 0; i < SIZE; i++) {
//   //         vecOut[i] = vecIn[i];
//   //       }
//   //     }
//   //   }

//   void run() {
//     dout.write(Pack1D<DTYPE, SIZE>());
//     dout_valid.write(false);
//     old_toggle = false;
//     enable.write(false);
//     for (int i = 0; i < SIZE; i++) {
//       fifoInput[i].write(0);
//     }

//     wait();

//     // #pragma hls_pipeline_init_interval 1
//     while (true) {
//       enable.write(din_toggle != old_toggle);

//       if (din_toggle != old_toggle) {
//         Pack1D<DTYPE, SIZE> input = din.read();
//         std::cout << "Skewer In: " << std::endl;
//         std::cout << input << std::endl;

//         if (reverse) {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             fifoInput[i] = input[SIZE - 1 - i];
//           }
//         } else {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             fifoInput[i] = input[i];
//           }
//         }

//         Pack1D<DTYPE, SIZE> output;

//         if (reverse) {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             output[i] = fifoOutput[SIZE - 1 - i];
//           }
//         } else {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             output[i] = fifoOutput[i];
//           }
//         }

//         std::cout << "Skewer output: " << std::endl;
//         std::cout << output << std::endl;

//         dout.write(output);
//         dout_valid.write(true);
//       } else {
//         dout_valid.write(false);
//       }

//       old_toggle = din_toggle;
//       wait();
//     }
//   }
// };

// template <typename DTYPE, int SIZE, bool reverse = false>
// SC_MODULE(ValidSkewer) {
//  public:
//   sc_in<bool> CCS_INIT_S1(clk);
//   sc_in<bool> CCS_INIT_S1(rstn);

//   sc_in<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(din);
//   sc_in<bool> CCS_INIT_S1(din_valid);

//   sc_out<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(dout);
//   sc_out<bool> CCS_INIT_S1(dout_valid);

//   sc_signal<DTYPE> fifoInput[SIZE];
//   sc_signal<DTYPE> fifoOutput[SIZE];
//   sc_signal<bool> enable;

//   // #define INIT_FIFOS(z, i, unused) Fifo<DTYPE, i + 1>
//   BOOST_PP_CAT(fifo_, i);
//   //   REPEAT(INIT_FIFOS)
//   // #undef INIT_FIFOS

//   SC_CTOR(ValidSkewer) {
//     Fifo<DTYPE> *fifo[SIZE];
//     for (int i = 0 + 1; i < SIZE + 1; i++) {
//       fifo[i - 1] = new Fifo<DTYPE>(sc_gen_unique_name("fifo"), i);
//       fifo[i - 1]->clk(clk);
//       fifo[i - 1]->rstn(rstn);
//       fifo[i - 1]->din(fifoInput[i - 1]);
//       fifo[i - 1]->din_valid(enable);
//       fifo[i - 1]->dout(fifoOutput[i - 1]);
//     }

//     SC_THREAD(run);
//     sensitive << clk.pos();
//     async_reset_signal_is(rstn, false);
//   }

//   //   void connect(Pack1D<DTYPE, SIZE> & vecIn, Pack1D<DTYPE, SIZE> &
//   vecOut) {
//   //     if (reverse) {
//   // #pragma hls_unroll yes
//   //       for (int i = 0; i < SIZE; i++) {
//   //         vecOut[i] = vecIn[SIZE - 1 - i];
//   //       }
//   //     } else {
//   // #pragma hls_unroll yes
//   //       for (int i = 0; i < SIZE; i++) {
//   //         vecOut[i] = vecIn[i];
//   //       }
//   //     }
//   //   }

//   void run() {
//     dout.write(Pack1D<DTYPE, SIZE>());
//     dout_valid.write(false);
//     enable.write(false);
//     for (int i = 0; i < SIZE; i++) {
//       fifoInput[i].write(0);
//     }

//     wait();

//     // #pragma hls_pipeline_init_interval 1
//     while (true) {
//       enable.write(din_valid);

//       if (din_valid) {
//         Pack1D<DTYPE, SIZE> input = din.read();

//         if (reverse) {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             fifoInput[i] = input[SIZE - 1 - i];
//           }
//         } else {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             fifoInput[i] = input[i];
//           }
//         }

//         Pack1D<DTYPE, SIZE> output;

//         if (reverse) {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             output[i] = fifoOutput[SIZE - 1 - i];
//           }
//         } else {
// #pragma hls_unroll yes
//           for (int i = 0; i < SIZE; i++) {
//             output[i] = fifoOutput[i];
//           }
//         }

//         dout.write(output);
//       }
//       dout_valid.write(din_valid);
//       wait();
//     }
//   }
// };
