#pragma once

#include <ac_shared.h>
#include <mc_connections.h>
#include <systemc.h>

#include "ArchitectureParams.h"
#include "PerfMonitor.h"

#ifndef __SYNTHESIS__
#include <stdexcept>
#endif

// Keep SRAM read and write backpressure independent.
// Clients must schedule dependent reads after the required writes commit.
template <typename T, int size>
SC_MODULE(DualPortBuffer) {
 private:
  static const unsigned int NUM_BANKS = DOUBLE_BUFFERED_ACCUM_BUFFER ? 2 : 1;
  static const unsigned int NUM_PORTS_PER_BANK =
      DOUBLE_BUFFERED_ACCUM_BUFFER ? 2 : 1;
  // Interface indices are bank-major: double buffering gives bank0 indices
  // 0/1 and bank1 indices 2/3. Single-buffer mode uses only index 0.
  ac_shared<T[size]> bank0;
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  ac_shared<T[size]> bank1;
  Connections::SyncChannel handoff[NUM_BANKS];
#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);
  Connections::In<ac_int<16, false>>
      read_address[NUM_BANKS * NUM_PORTS_PER_BANK];
  Connections::Out<T> read_data[NUM_BANKS * NUM_PORTS_PER_BANK];
  Connections::In<BufferWriteRequest<T>>
      write_request[NUM_BANKS * NUM_PORTS_PER_BANK];
#if ENABLE_PERF_COUNTERS
  sc_out<MatrixPerformance::Counter> perf_reads[NUM_BANKS],
      perf_writes[NUM_BANKS];
#endif
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::SyncIn done[NUM_BANKS * NUM_PORTS_PER_BANK];
#endif

  // Give each physical SRAM port its own process and stall condition
  SC_CTOR(DualPortBuffer) {
    SC_THREAD(bank0_write);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
    SC_THREAD(bank0_read);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    SC_THREAD(bank1_write);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
    SC_THREAD(bank1_read);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
  }

  // Drain the old owner's response before allowing either port to change owners
  template <int bank>
  void read_bank() {
#if ENABLE_PERF_COUNTERS
    MatrixPerformance::Counter reads = 0;
    perf_reads[bank].write(0);
#endif
#pragma hls_unroll yes
    for (int p = 0; p < NUM_PORTS_PER_BANK; ++p) {
      read_address[bank * NUM_PORTS_PER_BANK + p].Reset();
      read_data[bank * NUM_PORTS_PER_BANK + p].Reset();
    }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    handoff[bank].ResetRead();
#endif
    unsigned int port = bank * NUM_PORTS_PER_BANK;
    wait();
#if !DOUBLE_BUFFERED_ACCUM_BUFFER
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
#endif
    while (true) {
#if DOUBLE_BUFFERED_ACCUM_BUFFER
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!handoff[bank].vld.read())
#endif
      {
        ac_int<16, false> address;
        if (read_address[port].PopNB(address)) {
#ifndef __SYNTHESIS__
          if (address >= size) throw std::runtime_error("Address out of bounds");
#endif
          T value;
          if constexpr (bank == 0) {
            value = bank0[address];
          }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
          else {
            value = bank1[address];
          }
#endif
#if ENABLE_PERF_COUNTERS
          perf_reads[bank].write(++reads);
#endif
          read_data[port].Push(value);
        }
#ifndef __SYNTHESIS__
        wait();
#endif
      }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      handoff[bank].SyncPop();
      port ^= 1;
      // Let the sender lower valid before checking the next ownership phase.
      wait();
#endif
    }
  }

  // Accept writes independently and synchronize both ports at bank handoff
  template <int bank>
  void write_bank() {
#if ENABLE_PERF_COUNTERS
    MatrixPerformance::Counter writes = 0;
    perf_writes[bank].write(0);
#endif
#pragma hls_unroll yes
    for (int p = 0; p < NUM_PORTS_PER_BANK; ++p) {
      write_request[bank * NUM_PORTS_PER_BANK + p].Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      done[bank * NUM_PORTS_PER_BANK + p].Reset();
#endif
    }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    handoff[bank].ResetWrite();
#endif
    unsigned int port = bank * NUM_PORTS_PER_BANK;
    wait();
#if !DOUBLE_BUFFERED_ACCUM_BUFFER
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
#endif
    while (true) {
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      bool release_requested = false;
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (true)
#endif
      {
        BufferWriteRequest<T> request;
        const bool received_request = write_request[port].PopNB(request);
        if (received_request) {
#ifndef __SYNTHESIS__
          if (request.address >= size)
            throw std::runtime_error("Address out of bounds");
#endif
          if constexpr (bank == 0) {
            bank0[request.address] = request.data;
          }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
          else {
            bank1[request.address] = request.data;
          }
#endif
#if ENABLE_PERF_COUNTERS
          perf_writes[bank].write(++writes);
#endif
        }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
        // The sender holds done until acknowledgement, so later commands
        // cannot enter this port while its final buffered write drains.
        if (release_requested && !received_request) {
          break;
        }
        release_requested = done[port].vld.read();
#endif
#ifndef __SYNTHESIS__
        wait();
#endif
      }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      done[port].SyncPop();
      handoff[bank].SyncPush();
      port ^= 1;
#endif
    }
  }

  // Bind each port process to one shared SRAM bank
  void bank0_read() { read_bank<0>(); }
  // Bind the independent write process to bank zero
  void bank0_write() { write_bank<0>(); }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  // Bind the read process to bank one
  void bank1_read() { read_bank<1>(); }
  // Bind the independent write process to bank one
  void bank1_write() { write_bank<1>(); }
#endif
};
