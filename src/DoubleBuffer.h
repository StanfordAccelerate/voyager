#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

#ifndef __SYNTHESIS__
#include "test/common/AccessCounter.h"
#endif

template <int Depth, int Width>
SC_MODULE(DoubleBuffer) {
 private:
  ac_int<Width, false> mem0[Depth];
  ac_int<Width, false> mem1[Depth];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<BufferWriteRequest<ac_int<Width, false>>> writeRequest[2];
  Connections::In<BufferReadRequest> readAddress[2];
  Connections::Combinational<BufferReadResponse<ac_int<Width, false>>>
      readData[2];
  Connections::Out<ac_int<Width, false>> CCS_INIT_S1(output);

#ifndef __SYNTHESIS__
  AccessCounter *accessCounter;
#endif

  SC_CTOR(DoubleBuffer) {
    SC_THREAD(mem0Run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(mem1Run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(outputData);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#ifndef __SYNTHESIS__
    accessCounter = new AccessCounter();
#endif
  }

  template <int port>
  void mem_run() {
    writeRequest[port].Reset();
    readData[port].ResetWrite();
    readAddress[port].Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      bool done = false;
      while (!done) {
        BufferWriteRequest<ac_int<Width, false>> req = writeRequest[port].Pop();
        if (req.last) {
          done = true;
        }

        ac_int<16, false> address = req.address;

#ifndef __SYNTHESIS__
        if (address > Depth) {
          CCS_LOG("Address " << address << " is out of bounds!");
          throw std::runtime_error("Address out of bounds");
        }
#endif

        ac_int<Width, false> data = req.data;

        if constexpr (port == 0) {
          mem0[address] = data;
        } else {
          mem1[address] = data;
        }
      }

      done = false;
      while (!done) {
        BufferReadRequest req = readAddress[port].Pop();
        ac_int<16, false> address = req.address;
        if (req.last) {
          done = true;
        }

#ifndef __SYNTHESIS__
        accessCounter->increment(name(), Width);
#endif

        BufferReadResponse<ac_int<Width, false>> response;
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
        readData[port].Push(response);
      }
    }
  }

  void mem0Run() { mem_run<0>(); }

  void mem1Run() { mem_run<1>(); }

  void outputData() {
    bool bankSel = 0;
    output.Reset();

    readData[0].ResetRead();
    readData[1].ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      bool done = false;
      while (!done) {
        BufferReadResponse<ac_int<Width, false>> response =
            readData[bankSel].Pop();
        if (response.last) {
          done = true;
        }
        output.Push(response.data);
      }
      bankSel = !bankSel;
    }
  }
};
