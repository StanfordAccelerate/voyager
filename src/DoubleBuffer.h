#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename DTYPE, int WIDTH, int BUFFER_SIZE>
SC_MODULE(DoubleBuffer) {
 private:
  Pack1D<DTYPE, WIDTH> mem0[BUFFER_SIZE];
  Pack1D<DTYPE, WIDTH> mem1[BUFFER_SIZE];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> writeAddress[2];
  Connections::In<Pack1D<DTYPE, WIDTH> > writeData[2];
  Connections::In<int> writeControl[2];

  Connections::Combinational<Pack1D<DTYPE, WIDTH> > readData[2];
  Connections::In<int> readAddress[2];
  Connections::In<int> readControl[2];

  Connections::Combinational<int> outputControl[2];
  Connections::Out<Pack1D<DTYPE, WIDTH> > CCS_INIT_S1(output);

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
  }

  void mem0Run() {
    writeAddress[0].Reset();
    writeData[0].Reset();
    writeControl[0].Reset();

    readData[0].ResetWrite();
    readAddress[0].Reset();
    readControl[0].Reset();

    outputControl[0].ResetWrite();

    wait();

    while (true) {
      int wsize = writeControl[0].Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < wsize; i++) {
        int address = writeAddress[0].Pop();
        Pack1D<DTYPE, WIDTH> data = writeData[0].Pop();
        mem0[address] = data;

        if (i >= wsize - 1) {
          break;
        }
      }

      int rsize = readControl[0].Pop();
      outputControl[0].Push(rsize);
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < rsize; i++) {
        int address = readAddress[0].Pop();
        Pack1D<DTYPE, WIDTH> data;

        if (address != -1) {
          data = mem0[address];
        } else {
#pragma hls_unroll yes
          for (int j = 0; j < WIDTH; j++) {
            data[j] = 0;
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
    writeAddress[1].Reset();
    writeData[1].Reset();
    writeControl[1].Reset();

    readData[1].ResetWrite();
    readAddress[1].Reset();
    readControl[1].Reset();

    outputControl[1].ResetWrite();

    wait();

    while (true) {
      int wsize = writeControl[1].Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < wsize; i++) {
        int address = writeAddress[1].Pop();
        Pack1D<DTYPE, WIDTH> data = writeData[1].Pop();
        mem1[address] = data;

        if (i >= wsize - 1) {
          break;
        }
      }

      int rsize = readControl[1].Pop();
      outputControl[1].Push(rsize);
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < rsize; i++) {
        int address = readAddress[1].Pop();
        Pack1D<DTYPE, WIDTH> data;

        if (address != -1) {
          data = mem1[address];
        } else {
#pragma hls_unroll yes
          for (int j = 0; j < WIDTH; j++) {
            data[j] = 0;
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
      int size = outputControl[bankSel].Pop();

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
