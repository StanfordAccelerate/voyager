#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <class T>
SC_MODULE(Tieoff) {
 public:
  Connections::In<T> CCS_INIT_S1(in);

#ifdef CONNECTIONS_FAST_SIM
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);
#endif

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(Tieoff);
  Tieoff()
      : sc_module(sc_gen_unique_name("Tieoff"))
#else
  SC_CTOR(Tieoff)
#endif
  {
#ifdef CONNECTIONS_FAST_SIM
    SC_THREAD(drive_rdy);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#else
    SC_METHOD(drive_rdy);
    sensitive << in.rdy;

#ifdef CONNECTIONS_SIM_ONLY
    in.disable_spawn();
#endif

#endif
  }

  void drive_rdy() {
#ifdef CONNECTIONS_FAST_SIM
    in.Reset();

    wait();

    while (true) {
      in.Pop();
    }
#else
    in.rdy = 1;
#endif
  }
};

template <class T>
SC_MODULE(ZeroTieoff) {
 public:
  Connections::Out<T> CCS_INIT_S1(out);

#ifdef CONNECTIONS_FAST_SIM
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);
#endif

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(ZeroTieoff);
  ZeroTieoff()
      : sc_module(sc_gen_unique_name("ZeroTieoff"))
#else
  SC_CTOR(ZeroTieoff)
#endif
  {
#ifdef CONNECTIONS_FAST_SIM
    SC_THREAD(drive);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#else
    SC_METHOD(drive);
    sensitive << out.vld << out.dat;

#ifdef CONNECTIONS_SIM_ONLY
    out.disable_spawn();
#endif

#endif
  }

  void drive() {
    T zero = T::zero();
#ifdef CONNECTIONS_FAST_SIM
    out.Reset();

    wait();

    while (true) {
      out.Push(zero);
    }
#else
    out.vld = 1;
#ifdef __SYNTHESIS__
    out.dat = Connections::convert_to_lv(zero);
#else
    out.dat = zero;
#endif
#endif
  }
};
