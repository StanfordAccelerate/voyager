#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

#ifndef __SYNTHESIS__
#include "test/common/AccessCounter.h"
#endif

template <int depth, int width>
SC_MODULE(DoubleBuffer) {
 private:
  ac_int<width, false> mem0[depth];
  ac_int<width, false> mem1[depth];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<BufferWriteRequest<ac_int<width, false>>> write_request[2];
  Connections::In<BufferReadRequest> read_request[2];
  Connections::Combinational<BufferReadResponse<ac_int<width, false>>>
      read_data[2];
  Connections::Out<ac_int<width, false>> CCS_INIT_S1(output);

#ifndef __SYNTHESIS__
  AccessCounter* access_counter;
#endif

  SC_CTOR(DoubleBuffer) {
    SC_THREAD(mem0_run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(mem1_run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(output_data);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#ifndef __SYNTHESIS__
    access_counter = new AccessCounter();
#endif
  }

  template <int port>
  void mem_run() {
    write_request[port].Reset();
    read_data[port].ResetWrite();
    read_request[port].Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      bool done = false;
      while (!done) {
        BufferWriteRequest<ac_int<width, false>> req =
            write_request[port].Pop();
        if (req.last) {
          done = true;
        }

        ac_int<16, false> address = req.address;

#ifndef __SYNTHESIS__
        if (address > depth) {
          CCS_LOG("Address " << address << " is out of bounds!");
          throw std::runtime_error("Address out of bounds");
        }
#endif

        ac_int<width, false> data = req.data;

        if constexpr (port == 0) {
          mem0[address] = data;
        } else {
          mem1[address] = data;
        }
      }

      done = false;
      while (!done) {
        BufferReadRequest req = read_request[port].Pop();
        ac_int<16, false> address = req.address;
        if (req.last) {
          done = true;
        }

#ifndef __SYNTHESIS__
        access_counter->increment(name(), width);
#endif

        BufferReadResponse<ac_int<width, false>> response;
        response.last = req.last;

        if (address != 0xFFFF) {
          if constexpr (port == 0) {
            response.data = mem0[address];
          } else {
            response.data = mem1[address];
          }
        } else {
          response.data = 0;
        }
        read_data[port].Push(response);
      }
    }
  }

  void mem0_run() { mem_run<0>(); }

  void mem1_run() { mem_run<1>(); }

  void output_data() {
    bool bank_sel = 0;
    output.Reset();

    read_data[0].ResetRead();
    read_data[1].ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      bool done = false;
      while (!done) {
        BufferReadResponse<ac_int<width, false>> response =
            read_data[bank_sel].Pop();
        if (response.last) {
          done = true;
        }
        output.Push(response.data);
      }
      bank_sel = !bank_sel;
    }
  }
};
