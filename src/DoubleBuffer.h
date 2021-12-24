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

  void run(Connections::In<int> * wAddress,
           Connections::In<Pack1D<DTYPE, WIDTH> > * wData,
           Connections::In<int> * wControl,
           Connections::Combinational<Pack1D<DTYPE, WIDTH> > * rData,
           Connections::In<int> * rAddress, Connections::In<int> * rControl,
           Connections::Combinational<int> * oControl,
           Pack1D<DTYPE, WIDTH> mem[BUFFER_SIZE]) {
    bool swap = false;
    while (!swap) {
      if (wControl->Pop() == 1) {
        int address = wAddress->Pop();
        Pack1D<DTYPE, WIDTH> data = wData->Pop();
        mem[address] = data;
      } else {
        swap = true;
      }
    }

    CCS_LOG("contents:");
    for (int i = 0; i < 1024; i++) {
      std::cout << mem[i] << std::endl;
    }

    swap = false;
    while (!swap) {
      if (rControl->Pop() == 1) {
        int address = rAddress->Pop();
        Pack1D<DTYPE, WIDTH> data;

        if (address != -1) {
          data = mem[address];
        } else {
#pragma hls_unroll yes
          for (int j = 0; j < WIDTH; j++) {
            data[j] = 0;
          }
        }
        oControl->Push(1);
        rData->Push(data);
      } else {
        swap = true;
      }
    }
    oControl->Push(0);
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
      CCS_LOG("0");
      run(&writeAddress[0], &writeData[0], &writeControl[0], &readData[0],
          &readAddress[0], &readControl[0], &outputControl[0], mem0);
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
      CCS_LOG("1");
      run(&writeAddress[1], &writeData[1], &writeControl[1], &readData[1],
          &readAddress[1], &readControl[1], &outputControl[1], mem1);
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
      bool swap = false;
      while (!swap) {
        if (outputControl[bankSel].Pop() == 1) {
          output.Push(readData[bankSel].Pop());
        } else {
          swap = true;
        }
      }
      swap = false;
      bankSel = !bankSel;
    }
  }
};
