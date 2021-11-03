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

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      int writeSize = writeControl[0].Pop();
      CCS_LOG(writeSize);
      for (int i = 0; i < writeSize; i++) {
        int address = writeAddress[0].Pop();
        Pack1D<DTYPE, WIDTH> data = writeData[0].Pop();
        mem0[address] = data;
      }

      int readSize = readControl[0].Pop();
      outputControl[0].Push(readSize);
      for (int i = 0; i < readSize; i++) {
        int address = readAddress[0].Pop();
        Pack1D<DTYPE, WIDTH> data = mem0[address];
        readData[0].Push(data);
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

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      int writeSize = writeControl[1].Pop();
      for (int i = 0; i < writeSize; i++) {
        int address = writeAddress[1].Pop();
        Pack1D<DTYPE, WIDTH> data = writeData[1].Pop();
        mem1[address] = data;
      }

      int readSize = readControl[1].Pop();
      outputControl[1].Push(readSize);
      for (int i = 0; i < readSize; i++) {
        int address = readAddress[1].Pop();
        Pack1D<DTYPE, WIDTH> data = mem1[address];
        readData[1].Push(data);
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
      int count = outputControl[bankSel].Pop();
      for (int i = 0; i < count; i++) {
        output.Push(readData[bankSel].Pop());
      }
      bankSel = !bankSel;
    }
  }
};
