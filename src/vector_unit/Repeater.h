#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename T>
SC_MODULE(Repeater) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<T> CCS_INIT_S1(data_in);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(count);

  Connections::Out<T> CCS_INIT_S1(data_out);

  SC_CTOR(Repeater) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    data_in.Reset();
    count.Reset();
    data_out.Reset();

    wait();

    while (true) {
      ac_int<16, false> repeat_count = count.Pop() - 1;
      T data = data_in.Pop();

      for (ac_int<16, false> i = 0;; i++) {
        data_out.Push(data);

        if (i == repeat_count) break;
      }
    }
  }
};
