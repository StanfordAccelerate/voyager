#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename T, int BufferSize>
SC_MODULE(DualPortDoubleBuffer) {
 private:
  static const unsigned int NUM_BANKS = 2;
  static const unsigned int NUM_PORTS_PER_BANK = 2;

  T bank0[BufferSize];
  T bank1[BufferSize];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<16, false>>
      read_address[NUM_BANKS * NUM_PORTS_PER_BANK];
  Connections::Out<T> read_data[NUM_BANKS * NUM_PORTS_PER_BANK];
  Connections::In<BufferWriteRequest<T>>
      write_request[NUM_BANKS * NUM_PORTS_PER_BANK];
  Connections::SyncIn done[NUM_BANKS * NUM_PORTS_PER_BANK];

  SC_CTOR(DualPortDoubleBuffer) {
    SC_THREAD(bank0_run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(bank1_run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void bank0_run() {
    read_address[0].Reset();
    read_data[0].Reset();
    write_request[0].Reset();
    done[0].Reset();

    read_address[1].Reset();
    read_data[1].Reset();
    write_request[1].Reset();
    done[1].Reset();

    bool port_sel = false;

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      while (!done[port_sel].SyncPopNB()) {
#ifndef __SYNTHESIS__
        for (int i = 0; i < 100; i++)
#endif
        {
          BufferWriteRequest<T> req;

          if (write_request[port_sel].PopNB(req)) {
#ifndef __SYNTHESIS__
            if (req.address > BufferSize) {
              CCS_LOG("Address " << req.address << " is out of bounds!");
              throw std::runtime_error("Address out of bounds");
            }
#endif
          WRITE_BANK_0:
            bank0[req.address] = req.data;
          }

          ac_int<16, false> r_addr;
          if (read_address[port_sel].PopNB(r_addr)) {
#ifndef __SYNTHESIS__
            if (r_addr > BufferSize) {
              CCS_LOG("Address " << r_addr << " is out of bounds!");
              throw std::runtime_error("Address out of bounds");
            }
#endif

            T r_data;
          READ_BANK_0:
            r_data = bank0[r_addr];
            read_data[port_sel].Push(r_data);
          }

#ifndef __SYNTHESIS__
          wait();
#endif
        }
      }

      port_sel = !port_sel;
    }
  }

  void bank1_run() {
    read_address[2].Reset();
    read_data[2].Reset();
    write_request[2].Reset();
    done[2].Reset();

    read_address[3].Reset();
    read_data[3].Reset();
    write_request[3].Reset();
    done[3].Reset();

    bool port_sel = false;
    unsigned int port_idx = NUM_PORTS_PER_BANK;  // Start with port 2

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      port_idx =
          port_sel ? (NUM_PORTS_PER_BANK + 1) : NUM_PORTS_PER_BANK;  // 2 or 3

      while (!done[port_idx].SyncPopNB()) {
#ifndef __SYNTHESIS__
        for (int i = 0; i < 100; i++)
#endif
        {
          BufferWriteRequest<T> req;

          if (write_request[port_idx].PopNB(req)) {
#ifndef __SYNTHESIS__
            if (req.address > BufferSize) {
              CCS_LOG("Address " << req.address << " is out of bounds!");
              throw std::runtime_error("Address out of bounds");
            }
#endif

          WRITE_BANK_1:
            bank1[req.address] = req.data;
          }

          ac_int<16, false> r_addr;
          if (read_address[port_idx].PopNB(r_addr)) {
#ifndef __SYNTHESIS__
            if (r_addr > BufferSize) {
              CCS_LOG("Address " << r_addr << " is out of bounds!");
              throw std::runtime_error("Address out of bounds");
            }
#endif

            T r_data;
          READ_BANK_1:
            r_data = bank1[r_addr];
            read_data[port_idx].Push(r_data);
          }
#ifndef __SYNTHESIS__
          wait();
#endif
        }
      }
      port_sel = !port_sel;
    }
  }
};