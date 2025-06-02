#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename DTYPE>
SC_MODULE(Broadcaster) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<DTYPE> CCS_INIT_S1(dataIn);
  Connections::In<ac_int<16, false> > CCS_INIT_S1(count);

  Connections::Out<DTYPE> CCS_INIT_S1(dataOut);

  SC_CTOR(Broadcaster) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    dataIn.Reset();
    count.Reset();
    dataOut.Reset();

    wait();

    while (true) {
      ac_int<16, false> broadcastCount = count.Pop();
      DTYPE data = dataIn.Pop();

      for (int i = 0; i < broadcastCount; i++) {
        dataOut.Push(data);

        if (i >= broadcastCount - 1) {
          break;
        }
      }
    }
  }
};
