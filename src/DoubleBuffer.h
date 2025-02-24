#pragma once

#include <mc_connections.h>
#include <systemc.h>

#ifndef __SYNTHESIS__
#include "test/common/AccessCounter.h"
#endif

template <typename DTYPE, int WIDTH, int BUFFER_SIZE>
SC_MODULE(DoubleBuffer) {
 private:
  Pack1D<DTYPE, WIDTH> mem0[BUFFER_SIZE];
  Pack1D<DTYPE, WIDTH> mem1[BUFFER_SIZE];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<BufferWriteRequest<DTYPE, WIDTH> > writeRequest[2];
  Connections::In<ac_int<32, false> > writeControl[2];

  Connections::Combinational<Pack1D<DTYPE, WIDTH> > readData[2];
  Connections::In<ac_int<16, false> > readAddress[2];
  Connections::In<ac_int<32, false> > readControl[2];

  Connections::Combinational<ac_int<32, false> > outputControl[2];
  Connections::Out<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(output);

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

  void mem0Run() {
    writeRequest[0].Reset();
    writeControl[0].Reset();

    readData[0].ResetWrite();
    readAddress[0].Reset();
    readControl[0].Reset();

    outputControl[0].ResetWrite();

    wait();

    while (true) {
      ac_int<32, false> wsize = writeControl[0].Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < wsize; i++) {
        BufferWriteRequest<DTYPE, WIDTH> req = writeRequest[0].Pop();

        ac_int<16, false> address = req.address;

#ifndef __SYNTHESIS__
        if (address > BUFFER_SIZE) {
          CCS_LOG("Address " << address << " is out of bounds!");
          throw std::runtime_error("Address out of bounds");
        }
#endif

        Pack1D<DTYPE, WIDTH> data = req.data;
        mem0[address] = data;

        if (i >= wsize - 1) {
          break;
        }
      }

      ac_int<32, false> rsize = readControl[0].Pop();
#ifndef __SYNTHESIS__
      accessCounter->increment(name(), rsize * WIDTH);
#endif
      outputControl[0].Push(rsize);
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < rsize; i++) {
        ac_int<16, false> address = readAddress[0].Pop();
        Pack1D<DTYPE, WIDTH> data;

        if (address != 0xFFFF) {
          data = mem0[address];
        } else {
#pragma hls_unroll yes
          for (int j = 0; j < WIDTH; j++) {
            data[j].set_zero();
          }
        }
        readData[0].Push(data);

        if (i >= rsize - 1) {
          break;
        }
      }
    }
  }

  void mem1Run() {
    writeRequest[1].Reset();
    writeControl[1].Reset();

    readData[1].ResetWrite();
    readAddress[1].Reset();
    readControl[1].Reset();

    outputControl[1].ResetWrite();

    wait();

    while (true) {
      ac_int<32, false> wsize = writeControl[1].Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < wsize; i++) {
        BufferWriteRequest<DTYPE, WIDTH> req = writeRequest[1].Pop();

        ac_int<16, false> address = req.address;

        Pack1D<DTYPE, WIDTH> data = req.data;
        mem1[address] = data;

        if (i >= wsize - 1) {
          break;
        }
      }

      ac_int<32, false> rsize = readControl[1].Pop();
#ifndef __SYNTHESIS__
      accessCounter->increment(name(), rsize * WIDTH);
#endif
      outputControl[1].Push(rsize);
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < rsize; i++) {
        ac_int<16, false> address = readAddress[1].Pop();
        Pack1D<DTYPE, WIDTH> data;

        if (address != 0xFFFF) {
          data = mem1[address];
        } else {
#pragma hls_unroll yes
          for (int j = 0; j < WIDTH; j++) {
            data[j].set_zero();
          }
        }

        readData[1].Push(data);

        if (i >= rsize - 1) {
          break;
        }
      }
    }
  }

  void outputData() {
    bool bankSel = 0;
    output.Reset();

    readData[0].ResetRead();
    readData[1].ResetRead();
    outputControl[0].ResetRead();
    outputControl[1].ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      ac_int<32, false> size = outputControl[bankSel].Pop();

      for (int i = 0; i < size; i++) {
        output.Push(readData[bankSel].Pop());

        if (i >= size - 1) {
          break;
        }
      }

      bankSel = !bankSel;
    }
  }
};
